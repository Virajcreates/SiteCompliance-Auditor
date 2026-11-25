"""Site Video Ingestor, Duty-Cycle Calculator & Audio Slicer for SieveAudit."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

import cv2
from vlmsieve import SieveConfig, VideoSieve

from backend.app.config import SITE_ZONE_PRESETS, settings
from backend.app.schemas import (
    AuditedWorkEpisodeRecord,
    SiteComplianceReceipt,
    SiteZoneType,
)


def get_ffmpeg_binary() -> str:
    """Resolves the FFmpeg executable path via imageio_ffmpeg or system PATH."""
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).is_file():
            return str(exe)
    except Exception:
        pass

    found = shutil.which("ffmpeg")
    if found:
        return found
    raise RuntimeError(
        "FFmpeg binary not found. Please ensure imageio-ffmpeg is installed or ffmpeg is on system PATH."
    )


def has_audio_stream(video_path: str | Path) -> bool:
    """Probes whether a video file contains at least one audio stream."""
    path = Path(video_path)
    if not path.is_file():
        return False
    try:
        ffmpeg_bin = get_ffmpeg_binary()
        cmd = [ffmpeg_bin, "-i", str(path)]
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        combined_output = (proc.stderr or "") + (proc.stdout or "")
        return "Audio:" in combined_output
    except Exception:
        return False


def extract_episode_audio(
    video_path: str | Path,
    output_audio_path: str | Path,
    start_sec: float,
    end_sec: float,
    pad_sec: float = 0.5,
    sample_rate: int = 16000,
    channels: int = 1,
    bitrate: str = "32k",
) -> Path | None:
    """Slices a bounded audio segment from a video with boundary padding.

    Returns the output audio Path if successful, or None if the video is silent or slicing fails.
    """
    if not has_audio_stream(video_path):
        return None

    try:
        ffmpeg_bin = get_ffmpeg_binary()
        out_path = Path(output_audio_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        effective_start = max(0.0, float(start_sec) - float(pad_sec))
        effective_end = float(end_sec) + float(pad_sec)
        effective_duration = max(0.05, effective_end - effective_start)

        cmd = [
            ffmpeg_bin,
            "-y",
            "-ss",
            f"{effective_start:.3f}",
            "-t",
            f"{effective_duration:.3f}",
            "-i",
            str(video_path),
            "-vn",
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            "-b:a",
            str(bitrate),
            str(out_path),
        ]
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if (
            proc.returncode == 0
            and out_path.is_file()
            and out_path.stat().st_size > 0
        ):
            return out_path
        return None
    except Exception:
        return None


def compute_duty_cycle_metrics(
    episodes: list[Any], total_duration_sec: float
) -> tuple[float, float, float]:
    """Merges active episode intervals, clamps them, and computes duty-cycle telemetry.

    Returns:
        (active_work_duration_sec, idle_downtime_sec, site_utilization_pct)
    """
    total_sec = max(0.0, float(total_duration_sec))
    if total_sec == 0.0:
        return 0.0, 0.0, 0.0

    raw_intervals: list[tuple[float, float]] = []
    for ep in episodes:
        if isinstance(ep, dict):
            s = float(ep.get("start_time_sec", ep.get("start_sec", 0.0)))
            e = float(ep.get("end_time_sec", ep.get("end_sec", 0.0)))
        else:
            s = float(getattr(ep, "start_time_sec", getattr(ep, "start_sec", 0.0)))
            e = float(getattr(ep, "end_time_sec", getattr(ep, "end_sec", 0.0)))

        if e > s:
            clamped_s = max(0.0, min(total_sec, s))
            clamped_e = max(0.0, min(total_sec, e))
            if clamped_e > clamped_s:
                raw_intervals.append((clamped_s, clamped_e))

    if not raw_intervals:
        idle_sec = round(total_sec, 2)
        return 0.0, idle_sec, 0.0

    raw_intervals.sort(key=lambda x: (x[0], x[1]))

    merged: list[list[float]] = []
    for start, end in raw_intervals:
        if not merged:
            merged.append([start, end])
        else:
            prev_start, prev_end = merged[-1]
            if start <= prev_end:
                merged[-1][1] = max(prev_end, end)
            else:
                merged.append([start, end])

    active_work_duration_sec = round(
        sum(interval[1] - interval[0] for interval in merged), 2
    )
    # Ensure active does not exceed total_sec
    active_work_duration_sec = min(active_work_duration_sec, total_sec)
    idle_downtime_sec = round(max(0.0, total_sec - active_work_duration_sec), 2)
    site_utilization_pct = round(
        min(100.0, (active_work_duration_sec / total_sec) * 100.0), 2
    )

    return active_work_duration_sec, idle_downtime_sec, site_utilization_pct


class SiteVideoIngestor:
    """Ingests construction site videos, manages ROI/storyboard generation and audio extraction."""

    def __init__(
        self,
        target_vlm: str = "qwen-vl",
        base_output_dir: str | Path | None = None,
    ):
        self.target_vlm = target_vlm
        self.base_output_dir = Path(base_output_dir or settings.DATA_DIR)

    def ingest_video(
        self,
        video_path: str | Path,
        output_dir: str | Path | None = None,
        video_id: str | None = None,
        site_zone: SiteZoneType = "general_site_floor",
        target_vlm: str | None = None,
        sieve_config: SieveConfig | None = None,
    ) -> SiteComplianceReceipt:
        """Processes a construction CCTV video through vlm-sieve and audio slicer."""
        v_path = Path(video_path).resolve()
        if not v_path.is_file():
            raise FileNotFoundError(f"Video file not found at: {v_path}")

        v_id = video_id or f"vid_{v_path.stem}"
        out_dir = Path(output_dir or (self.base_output_dir / v_id)).resolve()
        storyboards_dir = out_dir / "storyboards"
        audio_dir = out_dir / "audio"
        storyboards_dir.mkdir(parents=True, exist_ok=True)
        audio_dir.mkdir(parents=True, exist_ok=True)

        chosen_target_vlm = target_vlm or self.target_vlm or settings.DEFAULT_TARGET_VLM

        if sieve_config is None:
            preset = SITE_ZONE_PRESETS.get(
                site_zone, SITE_ZONE_PRESETS["general_site_floor"]
            )
            motion_thresh = preset.get(
                "motion_threshold", settings.SIEVE_MOTION_THRESHOLD
            )
            sieve_config = SieveConfig(
                sample_fps=settings.SIEVE_SAMPLE_FPS,
                pre_roll_sec=settings.SIEVE_PRE_ROLL_SEC,
                merge_gap_sec=settings.SIEVE_MERGE_GAP_SEC,
                tile_max_dim=settings.SIEVE_TILE_MAX_DIM,
                spatial_grid_overlay=settings.SIEVE_SPATIAL_GRID_OVERLAY,
                burn_timestamps=settings.SIEVE_BURN_TIMESTAMPS,
                suppress_disturbances=settings.SIEVE_SUPPRESS_DISTURBANCES,
                motion_threshold=motion_thresh,
                target_vlm=chosen_target_vlm,
            )

        # Run vlm-sieve processing
        try:
            sieve = VideoSieve(source=str(v_path), config=sieve_config)
            events = sieve.process_video_to_dir(storyboards_dir)
        except TypeError:
            sieve = VideoSieve(config=sieve_config)
            events = sieve.process_video_to_dir(str(v_path), storyboards_dir)

        # Measure video total duration
        total_duration_sec = 0.0
        try:
            cap = cv2.VideoCapture(str(v_path))
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                if fps and fps > 0:
                    total_duration_sec = frame_count / fps
                cap.release()
        except Exception:
            pass

        if events:
            last_end = max(
                float(getattr(ev, "end_time_sec", getattr(ev, "end_sec", 0.0)))
                for ev in events
            )
            total_duration_sec = max(total_duration_sec, last_end)

        audio_available = has_audio_stream(v_path)
        episodes: list[AuditedWorkEpisodeRecord] = []

        for idx, ev in enumerate(events):
            ev_id = getattr(ev, "event_id", idx + 1)
            img_name = f"event_{ev_id:03d}.jpg"
            storyboard_rel = f"storyboards/{img_name}"

            audio_rel: str | None = None
            if audio_available:
                audio_filename = f"event_{ev_id:03d}.mp3"
                audio_full_path = audio_dir / audio_filename
                start_t = float(
                    getattr(ev, "start_time_sec", getattr(ev, "start_sec", 0.0))
                )
                end_t = float(
                    getattr(ev, "end_time_sec", getattr(ev, "end_sec", 0.0))
                )
                extracted = extract_episode_audio(
                    video_path=v_path,
                    output_audio_path=audio_full_path,
                    start_sec=start_t,
                    end_sec=end_t,
                    pad_sec=settings.AUDIO_PAD_SEC,
                    sample_rate=settings.AUDIO_SAMPLE_RATE,
                    channels=settings.AUDIO_CHANNELS,
                    bitrate=settings.AUDIO_BITRATE,
                )
                if extracted:
                    audio_rel = f"audio/{audio_filename}"

            record = AuditedWorkEpisodeRecord.from_sieve_event(
                event=ev,
                video_id=v_id,
                video_filename=v_path.name,
                site_zone=site_zone,
                episode_index=idx,
                storyboard_rel_path=storyboard_rel,
                audio_rel_path=audio_rel,
                default_tokens=settings.SIEVE_TOKENS_PER_IMAGE,
            )
            episodes.append(record)

        # Duty cycle calculation
        active_sec, idle_sec, util_pct = compute_duty_cycle_metrics(
            episodes, total_duration_sec
        )

        # Token savings report
        report = (
            sieve.get_savings_report()
            if hasattr(sieve, "get_savings_report")
            else None
        )
        raw_frames = getattr(report, "raw_frames_inspected", 0) if report else 0
        sampled_frames = (
            getattr(report, "sampled_frames_decoded", 0) if report else 0
        )
        disturbances_suppressed = (
            getattr(report, "disturbances_suppressed", 0) if report else 0
        )
        processing_wall_sec = round(
            float(getattr(sieve, "_last_processing_sec", 0.0)), 3
        )

        direct_tokens = int(
            round(total_duration_sec * settings.DIRECT_VIDEO_TOKENS_PER_SEC)
        )
        if direct_tokens == 0 and report:
            direct_tokens = int(getattr(report, "estimated_naive_tokens", 0))

        disturbances_raw = []
        if hasattr(sieve, "get_disturbances") and callable(sieve.get_disturbances):
            disturbances_raw = sieve.get_disturbances()
        elif hasattr(sieve, "_disturbances"):
            disturbances_raw = sieve._disturbances

        disturbances_data = [
            d.model_dump() if hasattr(d, "model_dump") else dict(d)
            for d in disturbances_raw
        ]

        receipt = SiteComplianceReceipt(
            video_id=v_id,
            video_filename=v_path.name,
            site_zone=site_zone,
            duration_sec=round(total_duration_sec, 2),
            active_work_duration_sec=active_sec,
            idle_downtime_sec=idle_sec,
            site_utilization_pct=util_pct,
            raw_frames_inspected=raw_frames,
            sampled_frames_decoded=sampled_frames,
            storyboards_emitted=len(episodes),
            disturbances_suppressed=disturbances_suppressed,
            processing_wall_sec=processing_wall_sec,
            estimated_direct_video_tokens=direct_tokens,
            episodes=episodes,
            disturbances=disturbances_data,
        )
        receipt.recalculate_rollups()

        # Serialize receipt to disk
        receipt_path = out_dir / "compliance_receipt.json"
        receipt_path.write_text(receipt.model_dump_json(indent=2), encoding="utf-8")

        return receipt


def ingest_site_video(
    video_path: str | Path,
    output_dir: str | Path,
    video_id: str | None = None,
    site_zone: SiteZoneType = "general_site_floor",
    target_vlm: str = "qwen-vl",
    sieve_config: SieveConfig | None = None,
) -> SiteComplianceReceipt:
    """Convenience function to ingest a site video and return its SiteComplianceReceipt."""
    ingestor = SiteVideoIngestor(target_vlm=target_vlm)
    return ingestor.ingest_video(
        video_path=video_path,
        output_dir=output_dir,
        video_id=video_id,
        site_zone=site_zone,
        target_vlm=target_vlm,
        sieve_config=sieve_config,
    )

