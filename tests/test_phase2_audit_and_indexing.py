"""Phase 2 unit and integration test suite for SiteCompliance-Auditor.

Tests:
1. Zone-aware EHS prompt construction
2. Auditor multimodal extraction with PPE observation and critical NCR violation
3. Audit receipt updating NCR rollups and saving to disk
4. Vector index upsert, cosine search, and multi-attribute EHS filters (local JSON mode)
5. Vector index Pinecone mode with mock client (index creation, zero None metadata, querying)
6. End-to-end Phase 1 (Ingestion) -> Phase 2 (Audit + Indexing + Search) compliance pipeline
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from backend.app.auditor import SiteStoryboardVLMAuditor, build_ehs_audit_prompt
from backend.app.config import SITE_ZONE_PRESETS
from backend.app.indexer import ComplianceEmbedder, SieveAuditVectorIndex
from backend.app.ingestor import ingest_site_video
from backend.app.schemas import (
    AuditedWorkEpisodeRecord,
    SafetyViolationItem,
    SiteComplianceReceipt,
    WorkerPPEObservation,
    WorkEpisodeAuditSchema,
)

# ---------------------------------------------------------------------------
# Test Mocks for Google GenAI and Pinecone (Zero Live Network Calls)
# ---------------------------------------------------------------------------


class MockUsageMetadata:
    """Mock usage metadata matching Google GenAI SDK structure."""

    def __init__(self, prompt_tokens: int = 1350, candidate_tokens: int = 420):
        self.prompt_token_count = prompt_tokens
        self.candidates_token_count = candidate_tokens


class MockGenerateContentResponse:
    """Mock generate_content response matching Google GenAI SDK."""

    def __init__(
        self,
        audit_schema: WorkEpisodeAuditSchema,
        prompt_tokens: int = 1350,
        candidate_tokens: int = 420,
    ):
        self.parsed = audit_schema
        self.text = audit_schema.model_dump_json()
        self.usage_metadata = MockUsageMetadata(prompt_tokens, candidate_tokens)


class MockModelsService:
    """Mock models service recording call arguments and returning predefined responses."""

    def __init__(self, response: MockGenerateContentResponse | None = None):
        self.response = response or MockGenerateContentResponse(
            WorkEpisodeAuditSchema()
        )
        self.last_model: str | None = None
        self.last_contents: list[Any] = []
        self.last_config: Any | None = None

    def generate_content(
        self, model: str, contents: list[Any], config: Any = None
    ) -> MockGenerateContentResponse:
        self.last_model = model
        self.last_contents = contents
        self.last_config = config
        return self.response


class MockGeminiClient:
    """Mock Google GenAI Client with mock models service."""

    def __init__(self, response: MockGenerateContentResponse | None = None):
        self.models = MockModelsService(response)


class MockIndexDescription:
    """Mock Pinecone index description."""

    def __init__(self, name: str):
        self.name = name


class MockPineconeIndex:
    """Mock Pinecone index recording upserts, queries, and deletions."""

    def __init__(self, name: str):
        self.name = name
        self.upserted_vectors: list[dict[str, Any]] = []
        self.deleted_ids: list[str] = []

    def upsert(
        self, vectors: list[dict[str, Any]], namespace: str = "default"
    ) -> dict[str, int]:
        self.upserted_vectors.extend(vectors)
        return {"upserted_count": len(vectors)}

    def query(
        self,
        vector: list[float],
        top_k: int = 5,
        include_metadata: bool = True,
        namespace: str = "default",
        **kwargs: Any,
    ) -> Any:
        matches = []
        for v in self.upserted_vectors[:top_k]:
            matches.append(
                {
                    "id": v["id"],
                    "score": 0.942,
                    "metadata": v.get("metadata", {}),
                }
            )

        class QueryResponse:
            def __init__(self, m: list[dict[str, Any]]):
                self.matches = m

        return QueryResponse(matches)

    def delete(
        self,
        ids: list[str] | None = None,
        namespace: str = "default",
        **kwargs: Any,
    ) -> dict[str, Any]:
        if ids:
            self.deleted_ids.extend(ids)
        return {}


class MockPineconeClient:
    """Mock Pinecone client for serverless index creation and operations."""

    def __init__(self, existing_indexes: list[str] | None = None):
        self.indexes = list(existing_indexes or [])
        self.created_indexes: list[dict[str, Any]] = []
        self._instances: dict[str, MockPineconeIndex] = {}

    def list_indexes(self) -> list[MockIndexDescription]:
        return [MockIndexDescription(name=n) for n in self.indexes]

    def create_index(
        self, name: str, dimension: int, metric: str, spec: Any
    ) -> None:
        self.created_indexes.append(
            {
                "name": name,
                "dimension": dimension,
                "metric": metric,
                "spec": spec,
            }
        )
        self.indexes.append(name)

    def Index(self, name: str) -> MockPineconeIndex:
        if name not in self._instances:
            self._instances[name] = MockPineconeIndex(name)
        return self._instances[name]


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------


def test_zone_aware_ehs_prompt_construction() -> None:
    """Verifies build_ehs_audit_prompt contains zone-specific PPE, hazards, and adaptive grid counts."""
    record = AuditedWorkEpisodeRecord(
        video_id="vid_scaffold_test",
        video_filename="scaffold_test.mp4",
        site_zone="scaffolding_working_at_height",
        event_id=1,
        episode_index=0,
        start_time_sec=2.5,
        end_time_sec=7.5,
        duration_sec=5.0,
        keyframe_times_sec=[2.5, 4.5, 6.5],
        roi_bbox=(0, 0, 640, 480),
        grid_used=(1, 3),  # 1 row x 3 cols adaptive grid
        storyboard_rel_path="storyboards/event_001.jpg",
    )

    prompt = build_ehs_audit_prompt(record)

    # Context & role
    assert "Senior EHS" in prompt
    assert "External Scaffolding & Working at Height" in prompt
    assert "scaffolding_working_at_height" in prompt

    # Adaptive grid & tile counts
    assert "`3` populated keyframe tiles" in prompt or "3 populated keyframe tiles" in prompt
    assert "1x3" in prompt
    assert "1..3" in prompt
    assert "- Tile #1: +0.00s (abs 2.50s)" in prompt
    assert "- Tile #2: +2.00s (abs 4.50s)" in prompt
    assert "- Tile #3: +4.00s (abs 6.50s)" in prompt

    # Zone-specific PPE & hazard rules
    assert "safety_harness" in prompt
    assert "missing_harness_at_height" in prompt
    assert "A1..D4" in prompt
    assert "non_compliant_ncr" in prompt


def test_auditor_multimodal_extraction_with_ppe_and_ncr_violation(
    tmp_path: Path,
) -> None:
    """Verifies auditor passes JPEG/audio parts and applies structured audit with NCR violation."""
    # 1. Create dummy storyboard JPEG and dummy MP3 audio slice
    storyboard_file = tmp_path / "storyboards" / "event_001.jpg"
    storyboard_file.parent.mkdir(parents=True, exist_ok=True)
    # Write a minimal valid 10x10 JPEG
    dummy_img = np.full((10, 10, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(storyboard_file), dummy_img)

    audio_file = tmp_path / "audio" / "event_001.mp3"
    audio_file.parent.mkdir(parents=True, exist_ok=True)
    audio_file.write_bytes(b"DUMMY_MP3_STREAM_BYTES_FOR_TESTING")

    record = AuditedWorkEpisodeRecord(
        video_id="vid_tower_01",
        video_filename="tower_01.mp4",
        site_zone="scaffolding_working_at_height",
        event_id=1,
        episode_index=0,
        start_time_sec=10.0,
        end_time_sec=16.0,
        duration_sec=6.0,
        keyframe_times_sec=[10.0, 12.4, 15.0],
        roi_bbox=(50, 50, 300, 400),
        grid_used=(2, 3),
        storyboard_rel_path="storyboards/event_001.jpg",
        audio_rel_path="audio/event_001.mp3",
        sieve_input_tokens=1120,
    )

    # 2. Build mock Gemini response containing Worker #1 violation
    obs = WorkerPPEObservation(
        worker_label="Worker #1",
        tile_number=2,
        relative_time_sec=2.40,
        absolute_time_sec=12.40,
        grid_cells=["B2", "C2"],
        hardhat_worn=False,
        high_vis_vest_worn=True,
        safety_harness_worn=False,
        ppe_compliant=False,
        activity_description="Working on elevated platform without harness tethered",
    )
    violation = SafetyViolationItem(
        violation_type="missing_harness_at_height",
        severity="critical",
        tile_number=2,
        timestamp_sec=12.40,
        grid_cells=["B2", "C2"],
        description="Worker observed at 6m height without fall arrest harness anchored.",
        corrective_action_required="Stop work immediately; attach harness to certified lifeline.",
    )
    audit_schema = WorkEpisodeAuditSchema(
        episode_title="Elevated Scaffolding Fall Hazard",
        work_summary="Worker operating on outer scaffolding ring with no harness.",
        work_category="active_construction_work",
        workers_detected_count=1,
        machinery_detected=["material_hoist"],
        ppe_observations=[obs],
        safety_violations=[violation],
        audio_safety_cues="Wind rush and metal clicking.",
        episode_compliance_status="non_compliant_ncr",
        overall_severity="critical",
        safety_score_pct=15.0,
    )

    mock_resp = MockGenerateContentResponse(
        audit_schema=audit_schema,
        prompt_tokens=1350,
        candidate_tokens=420,
    )
    mock_client = MockGeminiClient(response=mock_resp)

    auditor = SiteStoryboardVLMAuditor(client=mock_client)
    audited_record = auditor.audit_episode(record, workspace_dir=tmp_path)

    # Verify multimodal parts were passed to Gemini
    contents = mock_client.models.last_contents
    assert len(contents) == 3  # Image JPEG Part, Audio MP3 Part, Prompt Text
    assert contents[0].inline_data.mime_type == "image/jpeg"
    assert contents[1].inline_data.mime_type == "audio/mpeg"
    assert isinstance(contents[2], str)
    assert "scaffolding_working_at_height" in contents[2]

    # Verify audit application
    assert audited_record.episode_title == "Elevated Scaffolding Fall Hazard"
    assert audited_record.episode_compliance_status == "non_compliant_ncr"
    assert audited_record.overall_severity == "critical"
    assert audited_record.safety_score_pct == 15.0
    assert audited_record.sieve_input_tokens == 1350
    assert audited_record.sieve_output_tokens == 420
    assert len(audited_record.ppe_observations) == 1
    assert audited_record.ppe_observations[0].grid_cells == ["B2", "C2"]
    assert len(audited_record.safety_violations) == 1
    assert (
        audited_record.safety_violations[0].violation_type
        == "missing_harness_at_height"
    )


def test_audit_receipt_updates_ncr_rollups_and_saves_to_disk(
    tmp_path: Path,
) -> None:
    """Verifies audit_receipt processes multiple episodes, recalculates rollups, and writes receipt."""
    # Create workspace with dummy images
    sb_dir = tmp_path / "storyboards"
    sb_dir.mkdir(parents=True, exist_ok=True)
    dummy_img = np.full((10, 10, 3), 100, dtype=np.uint8)
    cv2.imwrite(str(sb_dir / "event_001.jpg"), dummy_img)
    cv2.imwrite(str(sb_dir / "event_002.jpg"), dummy_img)

    ep1 = AuditedWorkEpisodeRecord(
        video_id="vid_multi_01",
        video_filename="multi_01.mp4",
        site_zone="general_site_floor",
        event_id=1,
        episode_index=0,
        start_time_sec=0.0,
        end_time_sec=5.0,
        duration_sec=5.0,
        keyframe_times_sec=[1.0, 3.0],
        storyboard_rel_path="storyboards/event_001.jpg",
        sieve_input_tokens=1120,
    )
    ep2 = AuditedWorkEpisodeRecord(
        video_id="vid_multi_01",
        video_filename="multi_01.mp4",
        site_zone="general_site_floor",
        event_id=2,
        episode_index=1,
        start_time_sec=8.0,
        end_time_sec=14.0,
        duration_sec=6.0,
        keyframe_times_sec=[9.0, 12.0],
        storyboard_rel_path="storyboards/event_002.jpg",
        sieve_input_tokens=1120,
    )

    receipt = SiteComplianceReceipt(
        video_id="vid_multi_01",
        video_filename="multi_01.mp4",
        site_zone="general_site_floor",
        duration_sec=20.0,
        active_work_duration_sec=11.0,
        idle_downtime_sec=9.0,
        site_utilization_pct=55.0,
        estimated_direct_video_tokens=6000,
        episodes=[ep1, ep2],
    )

    # Mock response returning a missing_hardhat violation
    violation = SafetyViolationItem(
        violation_type="missing_hardhat",
        severity="high",
        tile_number=1,
        timestamp_sec=2.0,
        grid_cells=["A1"],
        description="Worker on floor with missing hardhat.",
        corrective_action_required="Provide hardhat.",
    )
    audit = WorkEpisodeAuditSchema(
        episode_title="Floor Inspection",
        work_summary="Worker walking floor.",
        work_category="site_inspection_or_walkthrough",
        workers_detected_count=1,
        safety_violations=[violation],
        episode_compliance_status="non_compliant_ncr",
        overall_severity="high",
        safety_score_pct=60.0,
    )
    mock_client = MockGeminiClient(
        response=MockGenerateContentResponse(
            audit_schema=audit, prompt_tokens=1200, candidate_tokens=250
        )
    )

    auditor = SiteStoryboardVLMAuditor(client=mock_client)
    updated_receipt = auditor.audit_receipt(
        receipt, workspace_dir=tmp_path, save_updated_receipt=True
    )

    assert updated_receipt.ncr_triggered is True
    assert updated_receipt.total_violations_count == 2  # 1 per episode
    assert updated_receipt.overall_site_safety_score_pct == 60.0
    assert updated_receipt.actual_sieve_audit_tokens == (1200 + 250) * 2
    assert updated_receipt.tokens_saved_pct > 0.0

    # Verify compliance_receipt.json was written to disk and is valid JSON
    saved_receipt_path = tmp_path / "compliance_receipt.json"
    assert saved_receipt_path.is_file()
    saved_data = json.loads(saved_receipt_path.read_text(encoding="utf-8"))
    assert saved_data["ncr_triggered"] is True
    assert saved_data["total_violations_count"] == 2
    assert len(saved_data["episodes"]) == 2


def test_vector_index_upsert_search_and_ehs_filters(tmp_path: Path) -> None:
    """Verifies local vector store upserts, semantic search, and EHS attribute filtering."""
    index_file = tmp_path / "test_vector_index.json"
    embedder = ComplianceEmbedder(use_mock_embeddings=True)
    index = SieveAuditVectorIndex(
        embedder=embedder, local_index_path=index_file
    )

    # 1. Episode #1: Compliant concrete pouring
    rec1 = AuditedWorkEpisodeRecord(
        video_id="vid_site_A",
        video_filename="site_A.mp4",
        site_zone="general_site_floor",
        event_id=1,
        episode_index=0,
        start_time_sec=0.0,
        end_time_sec=10.0,
        duration_sec=10.0,
        keyframe_times_sec=[2.0, 5.0, 8.0],
        storyboard_rel_path="storyboards/event_001.jpg",
        episode_title="Concrete Pouring on Ground Slab",
        work_summary="Transit mixer unloading concrete cleanly with all workers wearing PPE.",
        work_category="active_construction_work",
        workers_detected_count=3,
        machinery_detected=["transit_mixer", "concrete_vibrator"],
        ppe_observations=[
            WorkerPPEObservation(
                worker_label="Finisher #1",
                tile_number=1,
                relative_time_sec=2.0,
                absolute_time_sec=2.0,
                grid_cells=["A2"],
                hardhat_worn=True,
                high_vis_vest_worn=True,
                safety_harness_worn=False,
                ppe_compliant=True,
                activity_description="Leveling concrete slurry",
            )
        ],
        safety_violations=[],
        episode_compliance_status="compliant",
        overall_severity="low",
        safety_score_pct=100.0,
    )

    # 2. Episode #2: Critical scaffolding fall hazard (NCR)
    rec2 = AuditedWorkEpisodeRecord(
        video_id="vid_site_A",
        video_filename="site_A.mp4",
        site_zone="scaffolding_working_at_height",
        event_id=2,
        episode_index=1,
        start_time_sec=15.0,
        end_time_sec=25.0,
        duration_sec=10.0,
        keyframe_times_sec=[16.0, 20.0, 24.0],
        storyboard_rel_path="storyboards/event_002.jpg",
        episode_title="Scaffolding Rigging without Safety Harness",
        work_summary="Worker standing on open ledger without safety harness attached.",
        work_category="active_construction_work",
        workers_detected_count=1,
        machinery_detected=["winch"],
        ppe_observations=[
            WorkerPPEObservation(
                worker_label="Rigger #1",
                tile_number=2,
                relative_time_sec=5.0,
                absolute_time_sec=20.0,
                grid_cells=["C2", "C3"],
                hardhat_worn=False,
                high_vis_vest_worn=True,
                safety_harness_worn=False,
                ppe_compliant=False,
                activity_description="Balancing on outer tubular pipe",
            )
        ],
        safety_violations=[
            SafetyViolationItem(
                violation_type="missing_harness_at_height",
                severity="critical",
                tile_number=2,
                timestamp_sec=20.0,
                grid_cells=["C2", "C3"],
                description="Worker at 5m height without safety harness.",
                corrective_action_required="Halt work; clip harness to anchor.",
            )
        ],
        episode_compliance_status="non_compliant_ncr",
        overall_severity="critical",
        safety_score_pct=20.0,
    )

    # 3. Episode #3: Excavator proximity hazard
    rec3 = AuditedWorkEpisodeRecord(
        video_id="vid_site_B",
        video_filename="site_B.mp4",
        site_zone="excavation_heavy_machinery",
        event_id=1,
        episode_index=0,
        start_time_sec=30.0,
        end_time_sec=42.0,
        duration_sec=12.0,
        keyframe_times_sec=[32.0, 36.0, 40.0],
        storyboard_rel_path="storyboards/event_003.jpg",
        episode_title="Trench Excavation Worker Proximity Hazard",
        work_summary="Worker walking into swing radius of rotating 20-ton excavator.",
        work_category="machinery_earthwork",
        workers_detected_count=2,
        machinery_detected=["crawler_excavator"],
        ppe_observations=[],
        safety_violations=[
            SafetyViolationItem(
                violation_type="unsafe_machinery_proximity",
                severity="high",
                tile_number=1,
                timestamp_sec=36.0,
                grid_cells=["B3"],
                description="Worker inside 2m excavator swing radius without spotter.",
                corrective_action_required="Barricade swing zone.",
            )
        ],
        episode_compliance_status="non_compliant_ncr",
        overall_severity="high",
        safety_score_pct=45.0,
    )

    # Upsert all 3 records
    upsert_res = index.upsert_episodes([rec1, rec2, rec3])
    assert upsert_res["upserted_count"] == 3
    assert index_file.is_file()

    # Search: Scaffolding without harness query -> rec2 should rank #1
    hits = index.search(
        "worker on scaffolding without safety harness", top_k=3
    )
    assert len(hits) == 3
    assert hits[0]["id"] == "vid_site_A_ep_001"
    assert (
        hits[0]["metadata"]["site_zone"] == "scaffolding_working_at_height"
    )
    assert hits[0]["score"] > hits[1]["score"]

    # Filter: ncr_only=True -> rec2 and rec3
    ncr_hits = index.search("construction work", ncr_only=True)
    assert len(ncr_hits) == 2
    ncr_ids = {h["id"] for h in ncr_hits}
    assert "vid_site_A_ep_001" in ncr_ids
    assert "vid_site_B_ep_000" in ncr_ids

    # Filter: site_zone="scaffolding_working_at_height" -> rec2 only
    scaffold_hits = index.search(
        "hazard", site_zone="scaffolding_working_at_height"
    )
    assert len(scaffold_hits) == 1
    assert scaffold_hits[0]["id"] == "vid_site_A_ep_001"

    # Filter: violation_type="missing_harness_at_height" -> rec2 only
    harness_hits = index.search(
        "safety issue", violation_type="missing_harness_at_height"
    )
    assert len(harness_hits) == 1
    assert harness_hits[0]["id"] == "vid_site_A_ep_001"
    assert len(harness_hits[0]["safety_violations"]) == 1

    # Filter: min_severity="critical" -> rec2 only (rec3 is high)
    crit_hits = index.search("work", min_severity="critical")
    assert len(crit_hits) == 1
    assert crit_hits[0]["id"] == "vid_site_A_ep_001"

    # Filter: video_id="vid_site_B" -> rec3 only
    site_b_hits = index.search("hazard", video_id="vid_site_B")
    assert len(site_b_hits) == 1
    assert site_b_hits[0]["id"] == "vid_site_B_ep_000"

    # Test delete_video: removes vid_site_A (rec1 and rec2)
    deleted_cnt = index.delete_video("vid_site_A")
    assert deleted_cnt == 2
    remaining_hits = index.search("work")
    assert len(remaining_hits) == 1
    assert remaining_hits[0]["id"] == "vid_site_B_ep_000"


def test_vector_index_pinecone_mode_with_mock_client(tmp_path: Path) -> None:
    """Verifies Pinecone serverless index creation, zero None metadata, and mock querying."""
    mock_pc = MockPineconeClient(existing_indexes=[])
    index_file = tmp_path / "mock_pc_index.json"
    embedder = ComplianceEmbedder(use_mock_embeddings=True)

    index = SieveAuditVectorIndex(
        embedder=embedder,
        index_name="sieveaudit-episodes",
        local_index_path=index_file,
        pinecone_client=mock_pc,
    )

    # Verify serverless index was created
    assert len(mock_pc.created_indexes) == 1
    assert mock_pc.created_indexes[0]["name"] == "sieveaudit-episodes"
    assert mock_pc.created_indexes[0]["dimension"] == 768
    assert mock_pc.created_indexes[0]["metric"] == "cosine"

    record = AuditedWorkEpisodeRecord(
        video_id="vid_excavation_01",
        video_filename="excavation_01.mp4",
        site_zone="excavation_heavy_machinery",
        event_id=1,
        episode_index=0,
        start_time_sec=0.0,
        end_time_sec=6.0,
        duration_sec=6.0,
        keyframe_times_sec=[1.0, 3.0, 5.0],
        storyboard_rel_path="storyboards/event_001.jpg",
        audio_rel_path=None,
    )

    res = index.upsert_episodes([record], namespace="test-site")
    assert res["upserted_count"] == 1
    assert res["backend"] == "pinecone"

    # Inspect the vectors upserted to Pinecone
    mock_idx = mock_pc.Index("sieveaudit-episodes")
    assert len(mock_idx.upserted_vectors) == 1
    upserted_entry = mock_idx.upserted_vectors[0]
    assert upserted_entry["id"] == "vid_excavation_01_ep_000"

    # Verify Pinecone metadata compliance: strictly zero None values!
    metadata = upserted_entry["metadata"]
    assert metadata["audio_rel_path"] == ""  # None mapped to empty string
    for k, v in metadata.items():
        assert v is not None, f"Metadata key '{k}' contains None!"
        if isinstance(v, list):
            for item in v:
                assert isinstance(
                    item, str
                ), f"List item in key '{k}' is not str!"
        else:
            assert isinstance(
                v, (str, int, float, bool)
            ), f"Key '{k}' has unsupported type {type(v)}"

    # Test querying in Pinecone mode
    hits = index.search("excavator motion", namespace="test-site")
    assert len(hits) == 1
    assert hits[0]["id"] == "vid_excavation_01_ep_000"
    assert hits[0]["score"] > 0.0


def test_end_to_end_phase1_to_phase2_site_compliance_pipeline(
    tmp_path: Path,
) -> None:
    """End-to-end integration: Video -> Ingestion -> VLM Audit -> Vector Index -> Search."""
    # 1. Generate synthetic construction video feed
    vid_path = tmp_path / "site_cctv.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(vid_path), fourcc, 10.0, (640, 480))

    # Static prefix
    for _ in range(10):
        out.write(np.full((480, 640, 3), 90, dtype=np.uint8))
    # Localized worker motion (100x120 block moving across)
    for i in range(25):
        frame = np.full((480, 640, 3), 90, dtype=np.uint8)
        x = 150 + i * 10
        cv2.rectangle(frame, (x, 200), (x + 100, 320), (255, 255, 255), -1)
        out.write(frame)
    # Static suffix
    for _ in range(30):
        out.write(np.full((480, 640, 3), 90, dtype=np.uint8))
    out.release()

    # 2. Ingestion (Phase 1)
    workspace_dir = tmp_path / "audit_workspace"
    raw_receipt = ingest_site_video(
        video_path=vid_path,
        output_dir=workspace_dir,
        video_id="site_e2e_001",
        site_zone="scaffolding_working_at_height",
        target_vlm="qwen-vl",
    )
    assert len(raw_receipt.episodes) >= 1

    # 3. VLM Audit (Phase 2)
    obs = WorkerPPEObservation(
        worker_label="Scaffold Worker #1",
        tile_number=1,
        relative_time_sec=1.5,
        absolute_time_sec=2.5,
        grid_cells=["B2", "C2"],
        hardhat_worn=True,
        high_vis_vest_worn=False,
        safety_harness_worn=False,
        ppe_compliant=False,
        activity_description="Securing plank without safety harness",
    )
    violation = SafetyViolationItem(
        violation_type="missing_harness_at_height",
        severity="critical",
        tile_number=1,
        timestamp_sec=2.5,
        grid_cells=["B2", "C2"],
        description="Worker on scaffolding missing safety harness.",
        corrective_action_required="Provide safety harness immediately.",
    )
    mock_audit = WorkEpisodeAuditSchema(
        episode_title="Scaffolding Rigging Incomplete PPE",
        work_summary="Active scaffolding assembly with missing fall arrest harness.",
        work_category="active_construction_work",
        workers_detected_count=1,
        ppe_observations=[obs],
        safety_violations=[violation],
        episode_compliance_status="non_compliant_ncr",
        overall_severity="critical",
        safety_score_pct=30.0,
    )
    mock_client = MockGeminiClient(
        response=MockGenerateContentResponse(
            audit_schema=mock_audit, prompt_tokens=1400, candidate_tokens=380
        )
    )

    auditor = SiteStoryboardVLMAuditor(client=mock_client)
    audited_receipt = auditor.audit_receipt(
        raw_receipt, workspace_dir=workspace_dir, save_updated_receipt=True
    )

    assert audited_receipt.ncr_triggered is True
    assert audited_receipt.overall_site_safety_score_pct == 30.0
    assert audited_receipt.total_violations_count >= 1

    # Verify updated compliance_receipt.json was written to disk
    receipt_file = workspace_dir / "compliance_receipt.json"
    assert receipt_file.is_file()
    saved_receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
    assert saved_receipt["ncr_triggered"] is True

    # 4. Vector Indexing & Search (Phase 2)
    embedder = ComplianceEmbedder(use_mock_embeddings=True)
    index = SieveAuditVectorIndex(
        embedder=embedder,
        local_index_path=workspace_dir / "vector_index.json",
    )
    upsert_out = index.upsert_episodes(audited_receipt.episodes)
    assert upsert_out["upserted_count"] == len(audited_receipt.episodes)

    # 5. Semantic Query
    results = index.search(
        "worker on scaffolding without safety harness",
        ncr_only=True,
        site_zone="scaffolding_working_at_height",
    )
    assert len(results) >= 1
    top_hit = results[0]
    assert top_hit["video_id"] == "site_e2e_001"
    assert top_hit["site_zone"] == "scaffolding_working_at_height"
    assert top_hit["overall_severity"] == "critical"
    assert top_hit["has_violations"] is True
    assert len(top_hit["safety_violations"]) >= 1
    assert (
        top_hit["safety_violations"][0]["violation_type"]
        == "missing_harness_at_height"
    )
    assert len(top_hit["ppe_observations"]) >= 1
