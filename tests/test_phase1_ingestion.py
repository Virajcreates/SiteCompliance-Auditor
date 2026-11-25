"""Phase 1 unit and integration test suite for SiteCompliance-Auditor.

Tests:
1. Zone presets & Settings exports
2. Worker PPE observation & 4x4 spatial grid validation
3. Audited work episode record & NCR rollup calculation
4. Duty-cycle utilization calculation
5. FFmpeg resolution & silent vs audio slicing
6. End-to-end site video ingestion with Qwen-VL patch alignment (28px)
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import cv2
import numpy as np
from pydantic import ValidationError
import pytest
from vlmsieve import SieveConfig

from backend.app.config import (
    AUDIO_BITRATE,
    AUDIO_CHANNELS,
    AUDIO_PAD_SEC,
    AUDIO_SAMPLE_RATE,
    DEFAULT_TARGET_VLM,
    DIRECT_VIDEO_TOKENS_PER_SEC,
    GEMINI_EMBEDDING_MODEL,
    GEMINI_VLM_MODEL,
    PINECONE_INDEX_NAME,
    SITE_ZONE_PRESETS,
    Settings,
    settings,
)
from backend.app.ingestor import (
    compute_duty_cycle_metrics,
    extract_episode_audio,
    get_ffmpeg_binary,
    has_audio_stream,
    ingest_site_video,
)
from backend.app.schemas import (
    AuditedWorkEpisodeRecord,
    SafetyViolationItem,
    SiteComplianceReceipt,
    WorkerPPEObservation,
    WorkEpisodeAuditSchema,
)


def test_zone_presets_and_settings_exports() -> None:
    """Verifies all 4 construction zone presets and settings exports."""
    expected_zones = {
        "general_site_floor",
        "scaffolding_working_at_height",
        "excavation_heavy_machinery",
        "material_gate_logistics",
    }
    assert set(SITE_ZONE_PRESETS.keys()) == expected_zones

    for zone_key, preset in SITE_ZONE_PRESETS.items():
        assert "label" in preset and isinstance(preset["label"], str)
        assert "required_ppe" in preset and isinstance(preset["required_ppe"], list)
        assert len(preset["required_ppe"]) >= 2
        assert "primary_hazards" in preset and isinstance(preset["primary_hazards"], list)
        assert "motion_threshold" in preset and isinstance(preset["motion_threshold"], float)
        assert 0.005 <= preset["motion_threshold"] <= 0.030

    # Test Settings instance and class
    custom_settings = Settings()
    assert "SieveAudit" in custom_settings.APP_NAME
    assert custom_settings.APP_VERSION == "0.1.0"
    assert custom_settings.BASE_DIR.exists()
    assert custom_settings.SIEVE_SAMPLE_FPS == 10.0
    assert custom_settings.SIEVE_SUPPRESS_DISTURBANCES is True
    assert custom_settings.SIEVE_TILE_MAX_DIM == 640
    assert custom_settings.SIEVE_TOKENS_PER_IMAGE == 1120
    assert custom_settings.DIRECT_VIDEO_TOKENS_PER_SEC == 300.0

    # Test module-level constant bindings
    assert settings.APP_NAME == custom_settings.APP_NAME
    assert DEFAULT_TARGET_VLM == "qwen-vl"
    assert AUDIO_SAMPLE_RATE == 16000
    assert AUDIO_CHANNELS == 1
    assert AUDIO_BITRATE == "32k"
    assert AUDIO_PAD_SEC == 0.5
    assert GEMINI_VLM_MODEL == "gemini-3.8-flash"
    assert GEMINI_EMBEDDING_MODEL == "gemini-embedding-001"
    assert PINECONE_INDEX_NAME == "sieveaudit-episodes"
    assert DIRECT_VIDEO_TOKENS_PER_SEC == 300.0


def test_worker_ppe_observation_and_grid_validation() -> None:
    """Verifies spatial 4x4 grid cell normalization and validation."""
    # Valid observation
    obs = WorkerPPEObservation(
        worker_label="Scaffolding Rig Worker #1",
        tile_number=2,
        relative_time_sec=1.5,
        absolute_time_sec=12.5,
        grid_cells=["a1", "b2", "c3", "d4"],
        hardhat_worn=True,
        high_vis_vest_worn=True,
        safety_harness_worn=True,
        ppe_compliant=True,
        activity_description="Fastening scaffolding ledger clamp",
    )
    # Check normalized uppercase cells
    assert obs.grid_cells == ["A1", "B2", "C3", "D4"]
    # Check action alias property
    assert obs.action == "Fastening scaffolding ledger clamp"

    # Invalid grid cell out of bounds (e.g. Z9, E1, A5)
    with pytest.raises(ValidationError):
        WorkerPPEObservation(
            tile_number=1,
            relative_time_sec=0.5,
            absolute_time_sec=1.0,
            grid_cells=["Z9"],
            activity_description="Working",
        )

    with pytest.raises(ValidationError):
        WorkerPPEObservation(
            tile_number=1,
            relative_time_sec=0.5,
            absolute_time_sec=1.0,
            grid_cells=["A5"],
            activity_description="Working",
        )

    # Tile number must be between 1 and 6
    with pytest.raises(ValidationError):
        WorkerPPEObservation(
            tile_number=0,
            relative_time_sec=0.5,
            absolute_time_sec=1.0,
            grid_cells=["A1"],
            activity_description="Working",
        )

    with pytest.raises(ValidationError):
        WorkerPPEObservation(
            tile_number=7,
            relative_time_sec=0.5,
            absolute_time_sec=1.0,
            grid_cells=["A1"],
            activity_description="Working",
        )


def test_work_episode_audit_and_ncr_rollup() -> None:
    """Verifies episode audit application, embedding text, vector metadata, and NCR rollup."""
    record = AuditedWorkEpisodeRecord(
        video_id="vid_site_floor_001",
        video_filename="site_floor_001.mp4",
        site_zone="scaffolding_working_at_height",
        event_id=1,
        episode_index=0,
        start_time_sec=10.0,
        end_time_sec=18.5,
        duration_sec=8.5,
        keyframe_times_sec=[10.5, 12.0, 14.0, 16.0, 18.0],
        roi_bbox=(100, 150, 400, 450),
        grid_used=(2, 3),
        storyboard_rel_path="storyboards/event_001.jpg",
        audio_rel_path=None,
        sieve_input_tokens=1120,
        sieve_output_tokens=0,
        qwen_vl_patch_tokens=320,
        qwen7b_kv_cache_mb=1.25,
    )

    # Apply audit with a critical safety violation
    obs = WorkerPPEObservation(
        worker_label="Steel Erector #1",
        tile_number=3,
        relative_time_sec=3.5,
        absolute_time_sec=13.5,
        grid_cells=["B2", "B3"],
        hardhat_worn=False,
        high_vis_vest_worn=True,
        safety_harness_worn=False,
        ppe_compliant=False,
        activity_description="Standing on open scaffolding edge without harness",
    )
    violation = SafetyViolationItem(
        violation_type="missing_harness_at_height",
        severity="critical",
        tile_number=3,
        timestamp_sec=13.5,
        grid_cells=["B2", "B3"],
        description="Worker observed at 4m height without fall arrest harness tethered.",
        corrective_action_required="Immediate work stop; harness must be anchored to lifeline.",
    )
    audit = WorkEpisodeAuditSchema(
        episode_title="Scaffolding Assembly Near Unprotected Edge",
        work_summary="Worker erecting outer framework without safety harness secured.",
        work_category="active_construction_work",
        workers_detected_count=1,
        machinery_detected=["scaffold_winch"],
        ppe_observations=[obs],
        safety_violations=[violation],
        audio_safety_cues="Metal clatter; no audible warning.",
        episode_compliance_status="non_compliant_ncr",
        overall_severity="critical",
        safety_score_pct=25.0,
    )

    record.apply_audit(audit, input_tokens=1120, output_tokens=180)

    assert record.episode_title == "Scaffolding Assembly Near Unprotected Edge"
    assert record.episode_compliance_status == "non_compliant_ncr"
    assert record.sieve_output_tokens == 180
    assert record.safety_score_pct == 25.0

    # Semantic text generation
    embedding_text = record.to_embedding_text()
    assert "Zone: scaffolding_working_at_height" in embedding_text
    assert "missing_harness_at_height [CRITICAL]" in embedding_text
    assert "Steel Erector #1" in embedding_text
    assert "Scaffolding Assembly Near Unprotected Edge" in embedding_text

    # Vector metadata generation (Pinecone-safe primitives, zero None values)
    meta = record.to_vector_metadata()
    assert isinstance(meta, dict)
    assert meta["audio_rel_path"] == ""  # None mapped to empty string
    assert meta["has_violations"] is True
    assert "missing_harness_at_height" in meta["violation_types"]

    # Verify Pinecone type compliance: strings, numbers, bools, or list[str] only
    for key, val in meta.items():
        assert val is not None, f"Metadata key '{key}' has None value!"
        if isinstance(val, list):
            for item in val:
                assert isinstance(item, str), f"List metadata key '{key}' contains non-str item '{item}'!"
        else:
            assert isinstance(val, (str, int, float, bool)), f"Metadata key '{key}' has illegal type {type(val)}!"

    # Verify JSON strings decode properly
    loaded_ppe = json.loads(meta["ppe_observations_json"])
    assert len(loaded_ppe) == 1
    assert loaded_ppe[0]["worker_label"] == "Steel Erector #1"

    loaded_viols = json.loads(meta["safety_violations_json"])
    assert len(loaded_viols) == 1
    assert loaded_viols[0]["violation_type"] == "missing_harness_at_height"

    # Test SiteComplianceReceipt recalculate_rollups
    receipt = SiteComplianceReceipt(
        video_id="vid_site_floor_001",
        video_filename="site_floor_001.mp4",
        site_zone="scaffolding_working_at_height",
        duration_sec=30.0,
        estimated_direct_video_tokens=2745,
        episodes=[record],
    )
    receipt.recalculate_rollups()

    assert receipt.ncr_triggered is True
    assert receipt.total_violations_count == 1
    assert receipt.overall_site_safety_score_pct == 25.0
    assert receipt.actual_sieve_audit_tokens == (1120 + 180)
    assert receipt.total_qwen_vl_patch_tokens == 320
    assert receipt.total_qwen7b_kv_cache_mb == 1.25
    assert receipt.tokens_saved_pct > 0.0


def test_duty_cycle_utilization_calculation() -> None:
    """Verifies interval merging and duty-cycle / utilization calculation."""
    # Case 1: Simple non-overlapping intervals
    eps1 = [
        {"start_time_sec": 10.0, "end_time_sec": 20.0},
        {"start_time_sec": 30.0, "end_time_sec": 45.0},
    ]
    active, idle, util = compute_duty_cycle_metrics(eps1, total_duration_sec=100.0)
    assert active == 25.0
    assert idle == 75.0
    assert util == 25.0

    # Case 2: Overlapping and contiguous intervals
    # [10, 25], [20, 35], [35, 40] -> merged into [10, 40] = 30s
    # [50, 60] = 10s
    # Total active = 40s
    eps2 = [
        {"start_sec": 10.0, "end_sec": 25.0},
        {"start_sec": 20.0, "end_sec": 35.0},
        {"start_sec": 35.0, "end_sec": 40.0},
        {"start_sec": 50.0, "end_sec": 60.0},
    ]
    active2, idle2, util2 = compute_duty_cycle_metrics(eps2, total_duration_sec=80.0)
    assert active2 == 40.0
    assert idle2 == 40.0
    assert util2 == 50.0

    # Case 3: Out-of-bounds intervals clamped to total duration
    eps3 = [
        {"start_time_sec": 80.0, "end_time_sec": 120.0},
    ]
    active3, idle3, util3 = compute_duty_cycle_metrics(eps3, total_duration_sec=100.0)
    assert active3 == 20.0
    assert idle3 == 80.0
    assert util3 == 20.0

    # Case 4: Edge cases (empty, zero duration)
    assert compute_duty_cycle_metrics([], total_duration_sec=50.0) == (0.0, 50.0, 0.0)
    assert compute_duty_cycle_metrics(eps1, total_duration_sec=0.0) == (0.0, 0.0, 0.0)


def test_ffmpeg_resolution_and_silent_vs_audio_slicing(tmp_path: Path) -> None:
    """Verifies FFmpeg binary location, audio stream detection, and MP3 slicing."""
    ffmpeg_bin = get_ffmpeg_binary()
    assert Path(ffmpeg_bin).is_file()

    # 1. Create a synthetic silent video (2 seconds at 10 fps)
    silent_vid = tmp_path / "silent_cctv.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(silent_vid), fourcc, 10.0, (320, 240))
    for _ in range(20):
        out.write(np.zeros((240, 320, 3), dtype=np.uint8))
    out.release()

    assert silent_vid.is_file()
    assert has_audio_stream(silent_vid) is False

    # Slicing silent video must return None gracefully
    audio_slice_silent = extract_episode_audio(
        video_path=silent_vid,
        output_audio_path=tmp_path / "silent_slice.mp3",
        start_sec=0.5,
        end_sec=1.5,
    )
    assert audio_slice_silent is None

    # 2. Create a synthetic video with an audio track using ffmpeg lavfi
    audio_vid = tmp_path / "audio_cctv.mp4"
    gen_cmd = [
        ffmpeg_bin,
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=2:size=320x240:rate=10",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=1000:duration=2",
        "-c:v",
        "mpeg4",
        "-c:a",
        "aac",
        "-shortest",
        str(audio_vid),
    ]
    proc = subprocess.run(gen_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    assert proc.returncode == 0
    assert audio_vid.is_file()
    assert has_audio_stream(audio_vid) is True

    # Slicing audio video must produce a valid MP3 file
    target_mp3 = tmp_path / "audio" / "event_001.mp3"
    extracted_audio = extract_episode_audio(
        video_path=audio_vid,
        output_audio_path=target_mp3,
        start_sec=0.5,
        end_sec=1.2,
        pad_sec=0.2,
        sample_rate=16000,
        channels=1,
        bitrate="32k",
    )
    assert extracted_audio is not None
    assert extracted_audio.is_file()
    assert extracted_audio.stat().st_size > 0


def test_ingest_site_video_end_to_end_with_qwen_vl_alignment(tmp_path: Path) -> None:
    """Verifies end-to-end ingestion with disturbance suppression and 28px storyboard divisibility."""
    # Generate synthetic construction video (640x480, 10 fps, 100 frames = 10s)
    # Sequence:
    # 1. 10 static background frames
    # 2. 5 full-frame flash / shake frames (simulated camera disturbance)
    # 3. 60 static frames (6.0s gap allowing disturbance event separation with 5.0s merge_gap_sec)
    # 4. 20 frames of localized moving box (simulated worker motion in lower quadrant)
    # 5. 30 static cooldown frames
    vid_path = tmp_path / "synthetic_site_feed.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(vid_path), fourcc, 10.0, (640, 480))

    # Phase 1: Static
    for _ in range(10):
        out.write(np.full((480, 640, 3), 90, dtype=np.uint8))

    # Phase 2: Full-frame disturbance
    for _ in range(5):
        out.write(np.full((480, 640, 3), 245, dtype=np.uint8))

    # Phase 3: Static gap (> 5.0s merge_gap_sec)
    for _ in range(60):
        out.write(np.full((480, 640, 3), 90, dtype=np.uint8))

    # Phase 4: Localized worker motion (60x60 white block)
    for i in range(20):
        frame = np.full((480, 640, 3), 90, dtype=np.uint8)
        x = 220 + i * 10
        cv2.rectangle(frame, (x, 260), (x + 60, 320), (255, 255, 255), -1)
        out.write(frame)

    # Phase 5: Cooldown static
    for _ in range(30):
        out.write(np.full((480, 640, 3), 90, dtype=np.uint8))

    out.release()
    assert vid_path.is_file()

    output_dir = tmp_path / "ingest_output"
    receipt = ingest_site_video(
        video_path=vid_path,
        output_dir=output_dir,
        video_id="test_scaffolding_clip",
        site_zone="scaffolding_working_at_height",
        target_vlm="qwen-vl",
    )

    # Verify compliance receipt
    assert receipt.video_id == "test_scaffolding_clip"
    assert receipt.site_zone == "scaffolding_working_at_height"
    assert receipt.duration_sec >= 9.0

    # Verify disturbance suppression
    assert receipt.disturbances_suppressed >= 1
    assert len(receipt.disturbances) >= 1
    assert receipt.disturbances[0]["event_type"] == "camera_disturbance"

    # Verify work episode emitted
    assert len(receipt.episodes) >= 1
    ep = receipt.episodes[0]
    assert ep.event_id == 1
    assert ep.duration_sec > 0.0

    # Verify storyboard was generated and conforms to Qwen-VL 28px divisibility
    sb_path = output_dir / ep.storyboard_rel_path
    assert sb_path.is_file()
    img = cv2.imread(str(sb_path))
    assert img is not None
    h, w, _ = img.shape
    assert w % 28 == 0, f"Storyboard width {w} is not divisible by 28 for Qwen-VL"
    assert h % 28 == 0, f"Storyboard height {h} is not divisible by 28 for Qwen-VL"

    # Verify telemetry calculations
    assert receipt.active_work_duration_sec > 0.0
    assert receipt.idle_downtime_sec > 0.0
    assert 0.0 < receipt.site_utilization_pct <= 100.0
    assert receipt.estimated_direct_video_tokens > 0
    assert receipt.actual_sieve_audit_tokens > 0
    assert receipt.tokens_saved_pct >= 0.0

    # Verify compliance receipt JSON was written to disk
    receipt_json_path = output_dir / "compliance_receipt.json"
    assert receipt_json_path.is_file()
    saved_data = json.loads(receipt_json_path.read_text(encoding="utf-8"))
    assert saved_data["video_id"] == "test_scaffolding_clip"
    assert len(saved_data["episodes"]) >= 1
    assert saved_data["disturbances_suppressed"] >= 1
