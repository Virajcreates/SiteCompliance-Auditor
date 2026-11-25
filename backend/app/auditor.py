"""Zone-Aware Multimodal EHS VLM Auditor for SiteCompliance-Auditor.

Inspects visual storyboards and synchronized audio using Google GenAI SDK (google-genai).
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from google import genai
from google.genai import types

from backend.app.config import GEMINI_VLM_MODEL, SITE_ZONE_PRESETS, settings
from backend.app.schemas import (
    VALID_GRID_CELLS,
    AuditedWorkEpisodeRecord,
    SiteComplianceReceipt,
    WorkEpisodeAuditSchema,
)


def build_ehs_audit_prompt(record: AuditedWorkEpisodeRecord) -> str:
    """Constructs a strict domain-specific EHS engineering prompt for Gemini VLM."""
    zone_cfg = SITE_ZONE_PRESETS.get(
        record.site_zone, SITE_ZONE_PRESETS["general_site_floor"]
    )
    zone_label = zone_cfg["label"]
    required_ppe = ", ".join(zone_cfg["required_ppe"])
    primary_hazards = ", ".join(zone_cfg["primary_hazards"])
    rows, cols = record.grid_used
    num_tiles = max(1, len(record.keyframe_times_sec))

    # Formats exact keyframe timestamps
    timestamp_lines: list[str] = []
    if record.keyframe_times_sec:
        for idx, abs_t in enumerate(record.keyframe_times_sec):
            tile_num = idx + 1
            rel_t = max(0.0, abs_t - record.start_time_sec)
            timestamp_lines.append(
                f"- Tile #{tile_num}: +{rel_t:.2f}s (abs {abs_t:.2f}s)"
            )
    else:
        timestamp_lines.append(
            f"- Tile #1: +0.00s (abs {record.start_time_sec:.2f}s)"
        )
    keyframe_timestamps_str = "\n".join(timestamp_lines)

    prompt = f"""You are an Senior EHS (Environment, Health & Safety) & Construction Site Compliance Auditor inspecting a composite storyboard from zone: `{zone_label}` (`{record.site_zone}`). The storyboard has `{num_tiles}` populated keyframe tiles arranged in a `{rows}x{cols}` grid (`{record.start_time_sec:.2f}s` to `{record.end_time_sec:.2f}s`, duration `{record.duration_sec:.2f}s`).

Populated Keyframe Timestamps:
{keyframe_timestamps_str}

Mandatory Zone PPE Checklist:
- Required PPE for `{zone_label}`: `{required_ppe}`.
- Primary monitored hazards: `{primary_hazards}`.

Spatial & Temporal Grounding Rules:
1. Each tile has a top-left badge `#tile | +offset_sec` and spatial 4x4 grid coordinates (`A1..D4`, where rows `A, B, C, D` run top-to-bottom and columns `1, 2, 3, 4` run left-to-right).
2. Inspect every worker and piece of machinery across all `{num_tiles}` tiles (`1..{num_tiles}`). Do NOT invent observations for unpopulated tile numbers above `{num_tiles}`.
3. For every worker observed, record a `WorkerPPEObservation` in `ppe_observations` with exact `tile_number` (`1..{num_tiles}`), `relative_time_sec`, `absolute_time_sec`, `grid_cells` (`A1..D4`), `hardhat_worn`, `high_vis_vest_worn`, `safety_harness_worn`, `ppe_compliant`, and `activity_description`.
4. If any worker is missing required PPE for this zone (e.g., missing hardhat, missing high-vis vest, or missing safety harness when in `scaffolding_working_at_height`) or if an environmental hazard exists (`unbarricaded_edge_or_shaft`, `unsafe_machinery_proximity`, `unauthorized_zone_entry`, `housekeeping_or_debris_hazard`), add a `SafetyViolationItem` in `safety_violations` with exact `tile_number`, `timestamp_sec`, `grid_cells`, `severity` (`low | medium | high | critical`), `description`, and `corrective_action_required`.
5. Set `episode_compliance_status` to `'non_compliant_ncr'` if any `high` or `critical` violation is present, `'minor_observation'` if only `low`/`medium` issues exist, or `'compliant'` if 100% compliant.
6. If an audio track is attached, transcribe any acoustic safety events (reverse alarms, crane horns, machinery grinding, shouted warnings) in `audio_safety_cues`. Otherwise set to `'No audio stream or silent episode.'`
"""
    return prompt


class SiteStoryboardVLMAuditor:
    """Zone-aware multimodal EHS auditor utilizing Gemini VLM with structured outputs."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = GEMINI_VLM_MODEL,
        client: Any | None = None,
        max_retries: int = 2,
        retry_delay_sec: float = 1.5,
    ) -> None:
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model_name = model_name
        self.client = client
        self.max_retries = max_retries
        self.retry_delay_sec = retry_delay_sec

    def _get_client(self) -> Any:
        if self.client is not None:
            return self.client
        if not self.api_key:
            raise ValueError(
                "Gemini API key is required when no client is provided. "
                "Set GEMINI_API_KEY environment variable or pass api_key."
            )
        self.client = genai.Client(api_key=self.api_key)
        return self.client

    def _parse_and_sanitize_audit(self, response: Any) -> WorkEpisodeAuditSchema:
        """Parses and sanitizes VLM structured response, ensuring tile bounds and grid cell validity."""
        raw_dict: dict[str, Any] = {}

        # 1. Attempt extracting dictionary from response.parsed or response.text
        if hasattr(response, "parsed") and response.parsed is not None:
            if isinstance(response.parsed, WorkEpisodeAuditSchema):
                raw_dict = response.parsed.model_dump()
            elif isinstance(response.parsed, dict):
                raw_dict = dict(response.parsed)
            else:
                try:
                    raw_dict = response.parsed.model_dump()
                except Exception:
                    raw_dict = {}

        if not raw_dict and hasattr(response, "text") and response.text:
            try:
                raw_dict = json.loads(response.text)
            except Exception:
                raw_dict = {}

        if not raw_dict:
            # Fallback default compliant schema
            return WorkEpisodeAuditSchema()

        # 2. Sanitize and clamp ppe_observations
        sanitized_ppe: list[dict[str, Any]] = []
        for obs in raw_dict.get("ppe_observations", []):
            if isinstance(obs, dict):
                obs_copy = dict(obs)
                tile_n = int(obs_copy.get("tile_number", 1))
                obs_copy["tile_number"] = max(1, min(6, tile_n))
                cells = obs_copy.get("grid_cells", [])
                if isinstance(cells, list):
                    obs_copy["grid_cells"] = [
                        c.strip().upper()
                        for c in cells
                        if isinstance(c, str) and c.strip().upper() in VALID_GRID_CELLS
                    ]
                sanitized_ppe.append(obs_copy)
        raw_dict["ppe_observations"] = sanitized_ppe

        # 3. Sanitize and clamp safety_violations
        sanitized_viols: list[dict[str, Any]] = []
        for v in raw_dict.get("safety_violations", []):
            if isinstance(v, dict):
                v_copy = dict(v)
                tile_n = int(v_copy.get("tile_number", 1))
                v_copy["tile_number"] = max(1, min(6, tile_n))
                cells = v_copy.get("grid_cells", [])
                if isinstance(cells, list):
                    v_copy["grid_cells"] = [
                        c.strip().upper()
                        for c in cells
                        if isinstance(c, str) and c.strip().upper() in VALID_GRID_CELLS
                    ]
                sanitized_viols.append(v_copy)
        raw_dict["safety_violations"] = sanitized_viols

        return WorkEpisodeAuditSchema.model_validate(raw_dict)

    def audit_episode(
        self,
        record: AuditedWorkEpisodeRecord,
        workspace_dir: str | Path,
    ) -> AuditedWorkEpisodeRecord:
        """Audits a single work episode storyboard + synchronized audio with Gemini VLM."""
        ws_path = Path(workspace_dir).resolve()

        # 1. Resolve storyboard JPEG path
        storyboard_path = ws_path / record.storyboard_rel_path
        if not storyboard_path.is_file():
            alt_path = ws_path / record.video_id / record.storyboard_rel_path
            if alt_path.is_file():
                storyboard_path = alt_path
            else:
                raise FileNotFoundError(
                    f"Storyboard image not found at '{storyboard_path}' or '{alt_path}'"
                )

        img_bytes = storyboard_path.read_bytes()
        parts: list[Any] = [
            types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")
        ]

        # 2. Check for optional synchronized audio slice
        if record.audio_rel_path:
            audio_path = ws_path / record.audio_rel_path
            if not audio_path.is_file():
                alt_audio = ws_path / record.video_id / record.audio_rel_path
                if alt_audio.is_file():
                    audio_path = alt_audio
            if audio_path.is_file() and audio_path.stat().st_size > 0:
                audio_bytes = audio_path.read_bytes()
                parts.append(
                    types.Part.from_bytes(data=audio_bytes, mime_type="audio/mpeg")
                )

        # 3. Append zone-aware prompt
        prompt_text = build_ehs_audit_prompt(record)
        parts.append(prompt_text)

        # 4. Generate structured audit response via Gemini VLM
        client = self._get_client()
        last_exc: Exception | None = None
        response = None

        for attempt in range(self.max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=parts,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        response_mime_type="application/json",
                        response_schema=WorkEpisodeAuditSchema,
                    ),
                )
                break
            except Exception as e:
                last_exc = e
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay_sec)

        if response is None:
            raise RuntimeError(
                f"Failed to generate EHS audit content after {self.max_retries + 1} attempts: {last_exc}"
            )

        # 5. Parse and sanitize structured output
        audit = self._parse_and_sanitize_audit(response)

        # 6. Extract token telemetry
        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = (
            getattr(usage, "prompt_token_count", None)
            or record.sieve_input_tokens
        )
        output_tokens = getattr(usage, "candidates_token_count", None) or 0

        # 7. Apply audit in-place to record
        record.apply_audit(
            audit, input_tokens=int(prompt_tokens), output_tokens=int(output_tokens)
        )
        return record

    def audit_receipt(
        self,
        receipt: SiteComplianceReceipt,
        workspace_dir: str | Path,
        save_updated_receipt: bool = True,
    ) -> SiteComplianceReceipt:
        """Audits all episodes in a SiteComplianceReceipt and recalculates rollups."""
        ws_path = Path(workspace_dir).resolve()

        for ep in receipt.episodes:
            self.audit_episode(ep, ws_path)

        receipt.recalculate_rollups()

        if save_updated_receipt:
            vid_dir = ws_path / receipt.video_id
            if vid_dir.is_dir():
                out_path = vid_dir / "compliance_receipt.json"
            else:
                out_path = ws_path / "compliance_receipt.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(receipt.model_dump_json(indent=2), encoding="utf-8")

        return receipt

