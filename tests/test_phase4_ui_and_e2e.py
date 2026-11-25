"""Phase 4 E2E, UI, Static Assets, and DOM Integrity Test Suite for SiteCompliance-Auditor.

Tests:
1. Command Center UI and static assets served without errors and free of banned emojis
2. Complete DOM ID integrity check between frontend/app.js and frontend/index.html
3. End-to-end local video ingest, streaming media routes, official NCR dossier, and stats
4. Video deletion lifecycle and storage/index cleanup
"""

from __future__ import annotations

import json
from pathlib import Path
import re
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

BANNED_EMOJIS = [
    "↑", "↓", "🔍", "✓", "✔", "▷", "▶", "🗑", "⚠", "⚡", "↻", "↺", "🛡", "📁", "🎥",
]


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

    def generate_content(
        self, model: str, contents: list[Any], config: Any = None
    ) -> MockGenerateContentResponse:
        return MockGenerateContentResponse(self.response)


class MockGeminiClient:
    def __init__(self, response: Any):
        self.models = MockModelsService(response)


def create_synthetic_ehs_video(
    output_path: Path,
    duration_sec: float = 12.0,
    fps: int = 10,
    width: int = 640,
    height: int = 480,
) -> Path:
    """Generates a synthetic construction video with localized motion episodes."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    num_frames = int(duration_sec * fps)

    for i in range(num_frames):
        frame = np.full((height, width, 3), 40, dtype=np.uint8)

        # Draw static background grid simulating scaffolding / floor
        for x in range(0, width, 80):
            cv2.line(frame, (x, 0), (x, height), (60, 60, 60), 1)
        for y in range(0, height, 80):
            cv2.line(frame, (0, y), (width, y), (60, 60, 60), 1)

        # Episode 1: 0.5s - 4.5s (Frames 5 to 45): Dynamic block motion (worker moving)
        if 5 <= i <= 45:
            offset = (i - 5) * 4
            cv2.rectangle(
                frame,
                (100 + offset, 120),
                (200 + offset, 240),
                (0, 200, 255),
                -1,
            )
            cv2.circle(frame, (150 + offset, 100), 18, (255, 255, 255), -1)

        # Episode 2: 7.0s - 11.0s (Frames 70 to 110): Dynamic block motion in another sector
        elif 70 <= i <= 110:
            offset = (i - 70) * 3
            cv2.rectangle(
                frame,
                (350, 150 + offset),
                (470, 280 + offset),
                (0, 165, 255),
                -1,
            )

        out.write(frame)

    out.release()
    return output_path


def test_command_center_ui_and_static_assets_served(tmp_path: Path) -> None:
    """Verifies GET /, GET /static/styles.css, and GET /static/app.js return 200 OK and have zero banned emojis."""
    project_root = Path(__file__).resolve().parent.parent
    frontend_dir = project_root / "frontend"

    app = create_app(workspace_dir=tmp_path, frontend_dir=frontend_dir)
    client = TestClient(app)

    # 1. Test GET / (index.html)
    resp_index = client.get("/")
    assert resp_index.status_code == 200
    html_text = resp_index.text
    assert "SIEVEAUDIT" in html_text
    assert "Tactical EHS & Site Compliance Command Center" in html_text
    assert "main-video-player" in html_text
    assert "btn-ask-rag" in html_text

    for emoji in BANNED_EMOJIS:
        assert emoji not in html_text, f"Found banned emoji '{emoji}' in index.html"

    # 2. Test GET /static/styles.css
    resp_css = client.get("/static/styles.css")
    assert resp_css.status_code == 200
    css_text = resp_css.text
    assert "--bg-canvas: #090A0C;" in css_text
    assert "--signal-amber: #F59E0B;" in css_text
    assert "--signal-emerald: #10B981;" in css_text
    assert "--signal-crimson: #EF4444;" in css_text

    for emoji in BANNED_EMOJIS:
        assert emoji not in css_text, f"Found banned emoji '{emoji}' in styles.css"

    # 3. Test GET /static/app.js
    resp_js = client.get("/static/app.js")
    assert resp_js.status_code == 200
    js_text = resp_js.text
    assert "fetchStats" in js_text
    assert "selectVideo" in js_text
    assert "selectEpisode" in js_text

    for emoji in BANNED_EMOJIS:
        assert emoji not in js_text, f"Found banned emoji '{emoji}' in app.js"


def test_dom_id_integrity_between_html_and_js() -> None:
    """Verifies that all elements accessed via document.getElementById in app.js exist in index.html."""
    project_root = Path(__file__).resolve().parent.parent
    frontend_dir = project_root / "frontend"

    js_file = frontend_dir / "app.js"
    html_file = frontend_dir / "index.html"

    assert js_file.is_file(), "frontend/app.js must exist"
    assert html_file.is_file(), "frontend/index.html must exist"

    js_content = js_file.read_text(encoding="utf-8")
    html_content = html_file.read_text(encoding="utf-8")

    # Extract all getElementById calls
    js_ids = set(re.findall(r'document\.getElementById\([\'"]([^\'"]+)[\'"]\)', js_content))
    assert len(js_ids) >= 25, f"Expected at least 25 DOM IDs in app.js, found {len(js_ids)}"

    # Extract all id="..." attributes in HTML
    html_ids = set(re.findall(r'id=[\'"]([^\'"]+)[\'"]', html_content))

    missing_in_html = js_ids - html_ids
    assert not missing_in_html, (
        f"The following IDs accessed in app.js are missing from index.html: {missing_in_html}"
    )


def test_e2e_local_video_ingest_streaming_ncr_and_stats(tmp_path: Path) -> None:
    """Comprehensive E2E test verifying ingestion, media serving, NCR generation, and statistics."""
    project_root = Path(__file__).resolve().parent.parent
    frontend_dir = project_root / "frontend"

    # 1. Create synthetic video
    video_path = tmp_path / "site_e2e_feed.mp4"
    create_synthetic_ehs_video(video_path, duration_sec=12.0)

    # 2. Setup mock VLM audit output
    mock_audit = WorkEpisodeAuditSchema(
        episode_title="Concrete Placement & Pouring Session",
        work_category="active_construction_work",
        work_summary="Concrete placement and formwork inspection underway.",
        workers_detected_count=1,
        machinery_detected=["Transit Mixer", "Concrete Pump"],
        ppe_observations=[
            WorkerPPEObservation(
                worker_label="Worker #1",
                tile_number=1,
                relative_time_sec=1.0,
                absolute_time_sec=1.5,
                grid_cells=["B2"],
                hardhat_worn=False,  # Violation!
                high_vis_vest_worn=True,
                safety_harness_worn=False,
                ppe_compliant=False,
                activity_description="Guiding concrete chute near pour edge",
            )
        ],
        safety_violations=[
            SafetyViolationItem(
                violation_type="missing_hardhat",
                severity="critical",
                tile_number=1,
                timestamp_sec=1.5,
                grid_cells=["B2"],
                description="Worker observed guiding concrete chute with no protective headwear",
                corrective_action_required="Immediately issue Class E hardhat and conduct toolbox talk",
            )
        ],
        audio_safety_cues="Engine Rumble, Hydraulic Whine",
        episode_compliance_status="non_compliant_ncr",
        overall_severity="critical",
        safety_score_pct=65.0,
    )

    mock_auditor = SiteStoryboardVLMAuditor(api_key="mock_key")
    mock_auditor.client = MockGeminiClient(mock_audit)

    class MockEmbedder(ComplianceEmbedder):
        def embed_text(self, text: str) -> list[float]:
            return [0.08] * 768

    mock_index = SieveAuditVectorIndex(
        local_index_path=tmp_path / "vector_index.json",
        embedder=MockEmbedder(api_key="mock_key"),
    )

    app = create_app(
        workspace_dir=tmp_path / "storage",
        frontend_dir=frontend_dir,
        auditor=mock_auditor,
        index=mock_index,
    )
    client = TestClient(app)

    # 3. Ingest video via local path
    ingest_payload = {
        "local_video_path": str(video_path),
        "site_zone": "general_site_floor",
        "target_vlm": "qwen-vl",
        "run_audit": "true",
        "index_vectors": "true",
    }
    resp = client.post("/api/videos/ingest", data=ingest_payload)
    assert resp.status_code == 200, resp.text
    receipt = resp.json()

    vid = receipt["video_id"]
    assert vid.startswith("vid_")
    assert len(receipt["episodes"]) >= 1
    assert receipt["site_utilization_pct"] > 0
    assert receipt["total_violations_count"] >= 1

    # 4. Verify Media Streaming routes
    # Raw video
    resp_video_media = client.get(receipt["video_url"])
    assert resp_video_media.status_code == 200
    assert resp_video_media.headers.get("content-type") == "video/mp4"

    # Storyboard image from first episode
    ep0 = receipt["episodes"][0]
    sb_url = ep0.get("storyboard_url") or ""
    if not sb_url:
        sb = (ep0.get("storyboard_rel_path") or "").lstrip("/")
        sb_url = f"/media/{vid}/{sb}"
    assert sb_url.startswith("/media/")
    resp_sb = client.get(sb_url)
    assert resp_sb.status_code == 200
    assert resp_sb.headers.get("content-type") == "image/jpeg"

    # 5. Verify Video Detail route
    resp_detail = client.get(f"/api/videos/{vid}")
    assert resp_detail.status_code == 200
    detail_data = resp_detail.json()
    assert detail_data["video_id"] == vid
    assert detail_data["ncr_url"] == f"/api/videos/{vid}/ncr"
    assert detail_data["ncr_html_url"] == f"/api/videos/{vid}/ncr/html"

    # 6. Verify Official NCR JSON Dossier
    resp_ncr_json = client.get(f"/api/videos/{vid}/ncr")
    assert resp_ncr_json.status_code == 200
    ncr_data = resp_ncr_json.json()
    assert ncr_data["ncr_status"] == "NON_CONFORMANCE_NCR_ISSUED"
    assert ncr_data["total_violations_count"] >= 1
    assert len(ncr_data["violation_records"]) >= 1
    assert ncr_data["violation_records"][0]["violation_type"] == "missing_hardhat"
    assert len(ncr_data["episode_summaries"]) >= 1

    # 7. Verify Official NCR Printable HTML route
    resp_ncr_html = client.get(f"/api/videos/{vid}/ncr/html")
    assert resp_ncr_html.status_code == 200
    assert "text/html" in resp_ncr_html.headers.get("content-type", "")
    html_report = resp_ncr_html.text
    assert ncr_data["ncr_reference_id"] in html_report
    assert "Missing Hardhat" in html_report or "missing_hardhat" in html_report

    # 8. Verify /api/stats reflects ingested feed
    resp_stats = client.get("/api/stats")
    assert resp_stats.status_code == 200
    stats = resp_stats.json()
    assert stats["total_videos"] >= 1
    assert stats["total_violations_count"] >= 1


def test_delete_video_cleans_storage_and_index(tmp_path: Path) -> None:
    """Verifies that DELETE /api/videos/{video_id} completely removes local files and index records."""
    project_root = Path(__file__).resolve().parent.parent
    frontend_dir = project_root / "frontend"

    video_path = tmp_path / "feed_to_delete.mp4"
    create_synthetic_ehs_video(video_path, duration_sec=6.0)

    class MockEmbedder(ComplianceEmbedder):
        def embed_text(self, text: str) -> list[float]:
            return [0.05] * 768

    storage_root = tmp_path / "storage"
    mock_index = SieveAuditVectorIndex(
        local_index_path=storage_root / "vector_index.json",
        embedder=MockEmbedder(api_key="mock_key"),
    )

    app = create_app(
        workspace_dir=storage_root,
        frontend_dir=frontend_dir,
        index=mock_index,
    )
    client = TestClient(app)

    # Ingest without VLM audit to make it fast
    ingest_payload = {
        "local_video_path": str(video_path),
        "site_zone": "scaffolding_working_at_height",
        "run_audit": "false",
        "index_vectors": "true",
    }
    resp = client.post("/api/videos/ingest", data=ingest_payload)
    assert resp.status_code == 200
    vid = resp.json()["video_id"]

    video_dir = storage_root / vid
    assert video_dir.is_dir()

    # Check it appears in /api/videos
    list_resp = client.get("/api/videos")
    videos_list = list_resp.json()
    if isinstance(videos_list, dict):
        videos_list = videos_list.get("videos", [])
    assert any(v["video_id"] == vid for v in videos_list)

    # Delete video
    del_resp = client.delete(f"/api/videos/{vid}")
    assert del_resp.status_code == 200
    del_data = del_resp.json()
    assert del_data["video_id"] == vid
    assert del_data["deleted"] is True

    # Verify storage directory removed
    assert not video_dir.exists()

    # Verify it is gone from /api/videos
    list_resp_after = client.get("/api/videos")
    videos_after = list_resp_after.json()
    if isinstance(videos_after, dict):
        videos_after = videos_after.get("videos", [])
    assert not any(v["video_id"] == vid for v in videos_after)

    # Verify 404 on subsequent get
    get_after = client.get(f"/api/videos/{vid}")
    assert get_after.status_code == 404
