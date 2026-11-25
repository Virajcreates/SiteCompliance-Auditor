"""Phase 3 unit and integration test suite for SiteCompliance-Auditor.

Tests:
1. Health and Zones endpoints (/api/health and /api/zones)
2. Video Ingestion, NCR Dossier JSON, and Printable HTML generation
3. EHS Vector Search and RAG Copilot Question Answering with citations
4. Offline execution mode and Delete video lifecycle
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
from fastapi.testclient import TestClient
import numpy as np
import pytest

from backend.app.auditor import SiteStoryboardVLMAuditor
from backend.app.config import Settings
from backend.app.indexer import ComplianceEmbedder, SieveAuditVectorIndex
from backend.app.main import create_app
from backend.app.rag import SieveAuditRAGSynthesizer
from backend.app.schemas import (
    AuditedWorkEpisodeRecord,
    EHSCitation,
    EHSRAGAnswerSchema,
    SafetyViolationItem,
    WorkerPPEObservation,
    WorkEpisodeAuditSchema,
)

# ---------------------------------------------------------------------------
# Test Mocks for Google GenAI SDK (Zero Live Network Calls)
# ---------------------------------------------------------------------------


class MockUsageMetadata:
    def __init__(self, prompt_tokens: int = 1200, candidate_tokens: int = 350):
        self.prompt_token_count = prompt_tokens
        self.candidates_token_count = candidate_tokens


class MockGenerateContentResponse:
    def __init__(self, parsed_obj: Any):
        self.parsed = parsed_obj
        if hasattr(parsed_obj, "model_dump_json"):
            self.text = parsed_obj.model_dump_json()
        elif isinstance(parsed_obj, (dict, list)):
            self.text = json.dumps(parsed_obj)
        else:
            self.text = str(parsed_obj)
        self.usage_metadata = MockUsageMetadata()


class MockModelsService:
    def __init__(self, response: Any):
        self.response = response
        self.last_contents: list[Any] = []

    def generate_content(
        self, model: str, contents: list[Any], config: Any = None
    ) -> MockGenerateContentResponse:
        self.last_contents = contents
        return MockGenerateContentResponse(self.response)


class MockGeminiClient:
    def __init__(self, response: Any):
        self.models = MockModelsService(response)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_health_and_zones_endpoints(tmp_path: Path) -> None:
    """Verifies /api/health and /api/zones endpoints return expected system metadata."""
    app = create_app(workspace_dir=tmp_path)
    client = TestClient(app)

    # 1. Test /api/health
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["vlmsieve_version"] == "0.3.0"
    assert "SieveAudit" in data["app_name"]
    assert "vector_backend" in data
    assert "indexed_episodes_count" in data

    # 2. Test /api/zones
    resp_zones = client.get("/api/zones")
    assert resp_zones.status_code == 200
    zones = resp_zones.json()
    assert "general_site_floor" in zones
    assert "scaffolding_working_at_height" in zones
    assert "excavation_heavy_machinery" in zones
    assert "material_gate_logistics" in zones

    scaffold_zone = zones["scaffolding_working_at_height"]
    assert "safety_harness" in scaffold_zone["required_ppe"]
    assert "missing_harness_at_height" in scaffold_zone["primary_hazards"]


def test_ingest_site_video_and_generate_ncr_dossier_and_html(
    tmp_path: Path,
) -> None:
    """End-to-end test of ingestion, NCR dossier generation, and printable HTML export."""
    # 1. Create a synthetic video with active worker motion
    vid_path = tmp_path / "scaffolding_feed.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(vid_path), fourcc, 10.0, (640, 480))
    for _ in range(10):
        out.write(np.full((480, 640, 3), 90, dtype=np.uint8))
    for i in range(25):
        frame = np.full((480, 640, 3), 90, dtype=np.uint8)
        x = 150 + i * 10
        cv2.rectangle(frame, (x, 200), (x + 100, 320), (255, 255, 255), -1)
        out.write(frame)
    for _ in range(30):
        out.write(np.full((480, 640, 3), 90, dtype=np.uint8))
    out.release()

    # 2. Setup mock VLM auditor returning a critical missing-harness violation
    obs = WorkerPPEObservation(
        worker_label="Rigger #1",
        tile_number=1,
        relative_time_sec=1.5,
        absolute_time_sec=2.5,
        grid_cells=["B2", "C2"],
        hardhat_worn=True,
        high_vis_vest_worn=True,
        safety_harness_worn=False,
        ppe_compliant=False,
        activity_description="Securing scaffolding ledger without safety harness",
    )
    violation = SafetyViolationItem(
        violation_type="missing_harness_at_height",
        severity="critical",
        tile_number=1,
        timestamp_sec=2.5,
        grid_cells=["B2", "C2"],
        description="Worker at 5m height without fall arrest harness anchored.",
        corrective_action_required="Halt work immediately; attach harness to lifeline.",
    )
    mock_audit = WorkEpisodeAuditSchema(
        episode_title="Scaffolding Rigging Near Open Edge",
        work_summary="Active scaffolding assembly observed without required safety harness.",
        work_category="active_construction_work",
        workers_detected_count=1,
        ppe_observations=[obs],
        safety_violations=[violation],
        episode_compliance_status="non_compliant_ncr",
        overall_severity="critical",
        safety_score_pct=25.0,
    )

    mock_client = MockGeminiClient(response=mock_audit)
    mock_auditor = SiteStoryboardVLMAuditor(client=mock_client)

    app = create_app(
        workspace_dir=tmp_path / "storage",
        auditor=mock_auditor,
    )
    client = TestClient(app)

    # 3. Upload video via POST /api/videos/ingest
    with open(str(vid_path), "rb") as f:
        resp = client.post(
            "/api/videos/ingest",
            data={
                "site_zone": "scaffolding_working_at_height",
                "target_vlm": "qwen-vl",
                "run_vlm_audit": "true",
                "index_episodes": "true",
            },
            files={"file": ("scaffolding_feed.mp4", f, "video/mp4")},
        )

    assert resp.status_code == 200
    data = resp.json()
    video_id = data["video_id"]
    assert data["ncr_triggered"] is True
    assert data["total_violations_count"] >= 1
    assert data["overall_site_safety_score_pct"] == 25.0
    assert "video_url" in data
    assert "ncr_url" in data
    assert "ncr_html_url" in data

    # 4. Test GET /api/videos
    list_resp = client.get("/api/videos")
    assert list_resp.status_code == 200
    videos = list_resp.json()
    assert len(videos) >= 1
    assert any(v["video_id"] == video_id for v in videos)

    # 5. Test GET /api/videos/{video_id}
    detail_resp = client.get(f"/api/videos/{video_id}")
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert detail["video_id"] == video_id
    assert len(detail["episodes"]) >= 1
    assert detail["episodes"][0]["storyboard_url"].startswith("/media/")

    # 6. Test GET /api/stats
    stats_resp = client.get("/api/stats")
    assert stats_resp.status_code == 200
    stats = stats_resp.json()
    assert stats["total_videos"] >= 1
    assert stats["total_ncr_videos"] >= 1
    assert stats["total_violations_count"] >= 1

    # 7. Test GET /api/videos/{video_id}/ncr (JSON Dossier)
    ncr_resp = client.get(f"/api/videos/{video_id}/ncr")
    assert ncr_resp.status_code == 200
    ncr_json = ncr_resp.json()
    assert ncr_json["ncr_status"] == "NON_CONFORMANCE_NCR_ISSUED"
    assert ncr_json["site_zone"] == "scaffolding_working_at_height"
    assert len(ncr_json["violation_records"]) >= 1
    assert (
        ncr_json["violation_records"][0]["violation_type"]
        == "missing_harness_at_height"
    )
    assert len(ncr_json["episode_summaries"]) >= 1

    # 8. Test GET /api/videos/{video_id}/ncr/html (Printable HTML5)
    html_resp = client.get(f"/api/videos/{video_id}/ncr/html")
    assert html_resp.status_code == 200
    assert "text/html" in html_resp.headers["content-type"]
    html_text = html_resp.text
    assert ncr_json["ncr_reference_id"] in html_text
    assert "Non-Conformance Report" in html_text
    assert "missing_harness_at_height" in html_text or "Missing Harness" in html_text
    assert "utilization" in html_text.lower()
    assert "safety_harness" in html_text


def test_ehs_search_and_ask_endpoints_with_violation_citations(
    tmp_path: Path,
) -> None:
    """Verifies vector search filtering and RAG Copilot synthesis with grounded citations."""
    storage_dir = tmp_path / "storage"
    storage_dir.mkdir(parents=True, exist_ok=True)

    embedder = ComplianceEmbedder(use_mock_embeddings=True)
    index = SieveAuditVectorIndex(
        embedder=embedder,
        local_index_path=storage_dir / "vector_index.json",
    )

    # Seed 2 distinct episodes
    ep1 = AuditedWorkEpisodeRecord(
        video_id="vid_scaffold_99",
        video_filename="scaffold_99.mp4",
        site_zone="scaffolding_working_at_height",
        event_id=1,
        episode_index=0,
        start_time_sec=10.0,
        end_time_sec=18.0,
        duration_sec=8.0,
        storyboard_rel_path="storyboards/event_001.jpg",
        episode_title="Working at Height without Safety Harness",
        work_summary="Rigger observed fastening pipe without safety harness anchored.",
        work_category="active_construction_work",
        workers_detected_count=1,
        safety_violations=[
            SafetyViolationItem(
                violation_type="missing_harness_at_height",
                severity="critical",
                tile_number=2,
                timestamp_sec=14.0,
                grid_cells=["B2", "C2"],
                description="Worker balancing on external ledger without fall arrest harness.",
                corrective_action_required="Halt work; clip harness to lifeline.",
            )
        ],
        episode_compliance_status="non_compliant_ncr",
        overall_severity="critical",
        safety_score_pct=20.0,
    )

    ep2 = AuditedWorkEpisodeRecord(
        video_id="vid_gate_01",
        video_filename="gate_01.mp4",
        site_zone="material_gate_logistics",
        event_id=1,
        episode_index=0,
        start_time_sec=0.0,
        end_time_sec=12.0,
        duration_sec=12.0,
        storyboard_rel_path="storyboards/event_001.jpg",
        episode_title="Material Delivery at Logistics Gate",
        work_summary="Flatbed truck unloading steel reinforcement bundles safely.",
        work_category="material_handling_or_crane",
        workers_detected_count=3,
        safety_violations=[],
        episode_compliance_status="compliant",
        overall_severity="low",
        safety_score_pct=100.0,
    )

    index.upsert_episodes([ep1, ep2])

    # Setup mock Gemini RAG response
    mock_rag_answer = EHSRAGAnswerSchema(
        answer="A critical fall hazard was identified on the external scaffolding where a rigger operated without fall protection.",
        confidence="high",
        ncr_escalation_recommended=True,
        recommended_corrective_actions=[
            "Halt all work at height on external scaffolding immediately.",
            "Verify 100% harness tie-off to certified lifelines.",
        ],
        citations=[
            EHSCitation(
                video_id="vid_scaffold_99",
                episode_index=0,
                site_zone="scaffolding_working_at_height",
                start_time_sec=10.0,
                end_time_sec=18.0,
                tile_number=2,
                timestamp_sec=14.0,
                grid_cells=["B2", "C2"],
                violation_type="missing_harness_at_height",
                severity="critical",
                evidence="Worker balancing on external ledger without fall arrest harness.",
                corrective_action="Halt work; clip harness to lifeline.",
                storyboard_url="storyboards/event_001.jpg",
            )
        ],
    )

    mock_client = MockGeminiClient(response=mock_rag_answer)
    synthesizer = SieveAuditRAGSynthesizer(
        index=index,
        workspace_dir=storage_dir,
        client=mock_client,
    )

    app = create_app(
        workspace_dir=storage_dir,
        index=index,
        synthesizer=synthesizer,
    )
    client = TestClient(app)

    # 1. Test POST /api/search with NCR & violation_type filter
    search_resp = client.post(
        "/api/search",
        json={
            "query": "scaffolding fall hazard",
            "ncr_only": True,
            "violation_type": "missing_harness_at_height",
        },
    )
    assert search_resp.status_code == 200
    hits = search_resp.json()
    assert len(hits) == 1
    assert hits[0]["video_id"] == "vid_scaffold_99"
    assert hits[0]["site_zone"] == "scaffolding_working_at_height"
    assert hits[0]["overall_severity"] == "critical"

    # 2. Test POST /api/ask (RAG Copilot)
    ask_resp = client.post(
        "/api/ask",
        json={
            "question": "Are any workers on scaffolding missing required fall arrest harnesses?",
            "site_zone": "scaffolding_working_at_height",
            "ncr_only": True,
        },
    )
    assert ask_resp.status_code == 200
    ask_data = ask_resp.json()
    assert ask_data["confidence"] == "high"
    assert ask_data["ncr_escalation_recommended"] is True
    assert len(ask_data["recommended_corrective_actions"]) == 2
    assert len(ask_data["citations"]) == 1

    citation = ask_data["citations"][0]
    assert citation["video_id"] == "vid_scaffold_99"
    assert citation["violation_type"] == "missing_harness_at_height"
    assert citation["storyboard_url"].startswith("/media/")


def test_offline_mode_and_delete_video_lifecycle(tmp_path: Path) -> None:
    """Verifies safe offline execution (API key empty) and video deletion lifecycle."""
    storage_dir = tmp_path / "offline_storage"
    storage_dir.mkdir(parents=True, exist_ok=True)

    # Force empty API key and None client
    test_settings = Settings(
        BASE_DIR=tmp_path,
        DATA_DIR=storage_dir,
        GEMINI_API_KEY="",
        PINECONE_API_KEY="",
    )
    offline_auditor = SiteStoryboardVLMAuditor(api_key="", client=None)
    offline_index = SieveAuditVectorIndex(
        embedder=ComplianceEmbedder(use_mock_embeddings=True),
        local_index_path=storage_dir / "vector_index.json",
    )
    offline_synth = SieveAuditRAGSynthesizer(
        index=offline_index,
        workspace_dir=storage_dir,
        api_key="",
        client=None,
    )

    app = create_app(
        workspace_dir=storage_dir,
        settings=test_settings,
        auditor=offline_auditor,
        index=offline_index,
        synthesizer=offline_synth,
    )
    client = TestClient(app)

    # 1. Create a synthetic video
    vid_path = tmp_path / "offline_feed.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(vid_path), fourcc, 10.0, (640, 480))
    for _ in range(10):
        out.write(np.full((480, 640, 3), 90, dtype=np.uint8))
    for i in range(25):
        frame = np.full((480, 640, 3), 90, dtype=np.uint8)
        x = 150 + i * 10
        cv2.rectangle(frame, (x, 200), (x + 100, 320), (255, 255, 255), -1)
        out.write(frame)
    for _ in range(30):
        out.write(np.full((480, 640, 3), 90, dtype=np.uint8))
    out.release()

    # 2. Ingest offline -> must return 200 without throwing 500
    with open(str(vid_path), "rb") as f:
        resp = client.post(
            "/api/videos/ingest",
            data={
                "site_zone": "general_site_floor",
                "run_vlm_audit": "true",
                "index_episodes": "true",
            },
            files={"file": ("offline_feed.mp4", f, "video/mp4")},
        )
    assert resp.status_code == 200
    vid_data = resp.json()
    video_id = vid_data["video_id"]
    assert vid_data["overall_site_safety_score_pct"] == 100.0

    # 3. Ask RAG in offline mode -> must return deterministic answer without throwing 500
    ask_resp = client.post(
        "/api/ask",
        json={"question": "What is the site utilization and safety score?"},
    )
    assert ask_resp.status_code == 200
    ask_data = ask_resp.json()
    assert "safety score" in ask_data["answer"].lower()
    assert len(ask_data["retrieved_episodes"]) >= 1

    # 4. Delete video lifecycle via DELETE /api/videos/{video_id}
    del_resp = client.delete(f"/api/videos/{video_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["deleted"] is True

    # 5. Confirm video receipt is removed
    get_resp = client.get(f"/api/videos/{video_id}")
    assert get_resp.status_code == 404
