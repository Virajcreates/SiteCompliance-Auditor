"""Pydantic v2 schemas for EHS, PPE observations, duty-cycle and site compliance receipts."""

from __future__ import annotations
from pydantic import model_validator

import json
from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator

SiteZoneType = Literal[
    "general_site_floor",
    "scaffolding_working_at_height",
    "excavation_heavy_machinery",
    "material_gate_logistics",
]

VALID_GRID_CELLS: set[str] = {
    f"{row}{col}" for row in ("A", "B", "C", "D") for col in (1, 2, 3, 4)
}


class WorkerPPEObservation(BaseModel):
    """Observation of a worker, PPE adherence, and localized activity within a storyboard tile."""

    worker_label: str = Field(
        default="Worker #1",
        description="Identifier such as Worker #1, Excavator Operator, Supervisor",
    )
    tile_number: int = Field(..., ge=1, le=6)
    relative_time_sec: float = Field(..., ge=0.0)
    absolute_time_sec: float = Field(..., ge=0.0)
    grid_cells: list[str] = Field(
        default_factory=list,
        description="Spatial 4x4 grid cells e.g. ['B2', 'C2']",
    )
    hardhat_worn: bool = True
    high_vis_vest_worn: bool = True
    safety_harness_worn: bool = False
    ppe_compliant: bool = True
    activity_description: str = Field(..., min_length=1)

    @field_validator("grid_cells")
    @classmethod
    def validate_grid_cells(cls, cells: list[str]) -> list[str]:
        normalized: list[str] = []
        for cell in cells:
            clean = cell.strip().upper()
            if clean not in VALID_GRID_CELLS:
                raise ValueError(
                    f"Invalid grid cell '{cell}'. Must be one of A1..D4 (4x4 spatial grid)."
                )
            normalized.append(clean)
        return normalized

    @property
    def action(self) -> str:
        """Alias for activity_description."""
        return self.activity_description


class SafetyViolationItem(BaseModel):
    """Specific EHS violation or Non-Conformance Report (NCR) line item."""

    violation_type: Literal[
        "missing_hardhat",
        "missing_high_vis_vest",
        "missing_harness_at_height",
        "unbarricaded_edge_or_shaft",
        "unsafe_machinery_proximity",
        "unauthorized_zone_entry",
        "housekeeping_or_debris_hazard",
        "other_ehs_hazard",
    ]
    severity: Literal["low", "medium", "high", "critical"] = "medium"
    tile_number: int = Field(..., ge=1, le=6)
    timestamp_sec: float = Field(..., ge=0.0)
    grid_cells: list[str] = Field(default_factory=list)
    description: str
    corrective_action_required: str

    @field_validator("grid_cells")
    @classmethod
    def validate_grid_cells(cls, cells: list[str]) -> list[str]:
        normalized: list[str] = []
        for cell in cells:
            clean = cell.strip().upper()
            if clean not in VALID_GRID_CELLS:
                raise ValueError(
                    f"Invalid grid cell '{cell}'. Must be one of A1..D4."
                )
            normalized.append(clean)
        return normalized


class WorkEpisodeAuditSchema(BaseModel):
    """Structured audit response schema returned by Gemini VLM for a single work episode."""

    episode_title: str = "Active Site Work Episode"
    work_summary: str = "Detected active construction or machinery movement."
    work_category: Literal[
        "active_construction_work",
        "material_handling_or_crane",
        "machinery_earthwork",
        "site_inspection_or_walkthrough",
        "idle_or_non_productive_movement",
    ] = "active_construction_work"
    workers_detected_count: int = Field(default=0, ge=0)
    machinery_detected: list[str] = Field(default_factory=list)
    ppe_observations: list[WorkerPPEObservation] = Field(default_factory=list)
    safety_violations: list[SafetyViolationItem] = Field(default_factory=list)
    audio_safety_cues: str = "No audio stream or silent episode."
    episode_compliance_status: Literal[
        "compliant", "minor_observation", "non_compliant_ncr"
    ] = "compliant"
    overall_severity: Literal["low", "medium", "high", "critical"] = "low"
    safety_score_pct: float = Field(default=100.0, ge=0.0, le=100.0)


class AuditedWorkEpisodeRecord(BaseModel):
    """Unified work episode record combining Sieve video telemetry with EHS audit results."""

    # Physical / Video Telemetry
    video_id: str
    video_filename: str
    site_zone: SiteZoneType = "general_site_floor"
    event_id: int
    episode_index: int
    start_time_sec: float
    end_time_sec: float
    duration_sec: float
    keyframe_times_sec: list[float] = Field(default_factory=list)
    roi_bbox: tuple[int, int, int, int] | list[int] = (0, 0, 0, 0)
    grid_used: tuple[int, int] = (2, 3)
    storyboard_rel_path: str
    audio_rel_path: str | None = None
    sieve_input_tokens: int = 1120
    sieve_output_tokens: int = 0
    qwen_vl_patch_tokens: int = 0
    qwen7b_kv_cache_mb: float = 0.0

    @field_validator("roi_bbox", mode="before")
    @classmethod
    def normalize_roi_bbox(cls, v: Any) -> tuple[int, int, int, int]:
        if hasattr(v, "x") and hasattr(v, "y"):
            w = getattr(v, "width", 0)
            h = getattr(v, "height", 0)
            return (int(v.x), int(v.y), int(w), int(h))
        if isinstance(v, dict):
            return (
                int(v.get("x", 0)),
                int(v.get("y", 0)),
                int(v.get("width", 0)),
                int(v.get("height", 0)),
            )
        if isinstance(v, (tuple, list)):
            if len(v) > 0 and isinstance(v[0], (tuple, list)):
                d = dict(v)
                return (
                    int(d.get("x", 0)),
                    int(d.get("y", 0)),
                    int(d.get("width", 0)),
                    int(d.get("height", 0)),
                )
            if len(v) >= 4:
                return (int(v[0]), int(v[1]), int(v[2]), int(v[3]))
        return (0, 0, 0, 0)

    @field_validator("grid_used", mode="before")
    @classmethod
    def normalize_grid_used(cls, v: Any) -> tuple[int, int]:
        if isinstance(v, (tuple, list)) and len(v) >= 2:
            return (int(v[0]), int(v[1]))
        return (2, 3)

    # Audit fields
    episode_title: str = "Active Site Work Episode"
    work_summary: str = "Detected active construction or machinery movement."
    work_category: Literal[
        "active_construction_work",
        "material_handling_or_crane",
        "machinery_earthwork",
        "site_inspection_or_walkthrough",
        "idle_or_non_productive_movement",
    ] = "active_construction_work"
    workers_detected_count: int = Field(default=0, ge=0)
    machinery_detected: list[str] = Field(default_factory=list)
    ppe_observations: list[WorkerPPEObservation] = Field(default_factory=list)
    safety_violations: list[SafetyViolationItem] = Field(default_factory=list)
    audio_safety_cues: str = "No audio stream or silent episode."
    episode_compliance_status: Literal[
        "compliant", "minor_observation", "non_compliant_ncr"
    ] = "compliant"
    overall_severity: Literal["low", "medium", "high", "critical"] = "low"
    safety_score_pct: float = Field(default=100.0, ge=0.0, le=100.0)

    @classmethod
    def from_sieve_event(
        cls,
        event: Any,
        video_id: str,
        video_filename: str,
        site_zone: SiteZoneType = "general_site_floor",
        episode_index: int = 0,
        storyboard_rel_path: str = "",
        audio_rel_path: str | None = None,
        default_tokens: int = 1120,
    ) -> AuditedWorkEpisodeRecord:
        """Constructs an AuditedWorkEpisodeRecord from a vlm-sieve Event / StoryboardResult."""
        event_id = getattr(event, "event_id", getattr(event, "id", episode_index + 1))
        start_sec = float(
            getattr(event, "start_time_sec", getattr(event, "start_sec", 0.0))
        )
        end_sec = float(
            getattr(event, "end_time_sec", getattr(event, "end_sec", 0.0))
        )
        duration_sec = float(
            getattr(event, "duration_sec", max(0.0, end_sec - start_sec))
        )
        keyframe_times = list(
            getattr(event, "keyframe_times_sec", getattr(event, "keyframe_timestamps", []))
        )
        roi_bbox = getattr(event, "roi_bbox", (0, 0, 0, 0))
        grid_used = getattr(event, "grid_used", getattr(event, "grid", (2, 3)))

        # Extract provider token estimates if available from vlm-sieve v0.3.0
        estimates: dict[str, Any] = {}
        if hasattr(event, "provider_token_estimates") and callable(
            event.provider_token_estimates
        ):
            try:
                estimates = event.provider_token_estimates() or {}
            except Exception:
                estimates = {}
        elif hasattr(event, "token_estimates") and isinstance(event.token_estimates, dict):
            estimates = event.token_estimates

        qwen_vl_tokens = int(
            estimates.get("qwen_vl_tokens")
            or estimates.get("qwen_vl_patch_tokens")
            or estimates.get("qwen_vl", {}).get("patch_tokens", 0)
            or estimates.get("qwen_tokens", 0)
        )
        qwen_kv_cache = float(
            estimates.get("qwen7b_kv_cache_mb")
            or estimates.get("qwen_vl", {}).get("kv_cache_mb", 0.0)
            or estimates.get("kv_cache_mb", 0.0)
        )
        input_tokens = int(
            estimates.get("gemini_tokens")
            or estimates.get("sieve_input_tokens")
            or estimates.get("gemini", {}).get("input_tokens", default_tokens)
            or estimates.get("input_tokens", default_tokens)
        )

        return cls(
            video_id=video_id,
            video_filename=video_filename,
            site_zone=site_zone,
            event_id=int(event_id),
            episode_index=episode_index,
            start_time_sec=round(start_sec, 3),
            end_time_sec=round(end_sec, 3),
            duration_sec=round(duration_sec, 3),
            keyframe_times_sec=[round(float(t), 3) for t in keyframe_times],
            roi_bbox=roi_bbox,
            grid_used=tuple(grid_used),
            storyboard_rel_path=storyboard_rel_path,
            audio_rel_path=audio_rel_path,
            sieve_input_tokens=input_tokens,
            sieve_output_tokens=0,
            qwen_vl_patch_tokens=qwen_vl_tokens,
            qwen7b_kv_cache_mb=round(qwen_kv_cache, 3),
        )

    def apply_audit(
        self,
        audit: WorkEpisodeAuditSchema,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> AuditedWorkEpisodeRecord:
        """Applies Gemini VLM structured audit results to this episode record."""
        self.episode_title = audit.episode_title
        self.work_summary = audit.work_summary
        self.work_category = audit.work_category
        self.workers_detected_count = audit.workers_detected_count
        self.machinery_detected = list(audit.machinery_detected)
        self.ppe_observations = list(audit.ppe_observations)
        self.safety_violations = list(audit.safety_violations)
        self.audio_safety_cues = audit.audio_safety_cues
        self.episode_compliance_status = audit.episode_compliance_status
        self.overall_severity = audit.overall_severity
        self.safety_score_pct = audit.safety_score_pct
        if input_tokens is not None:
            self.sieve_input_tokens = input_tokens
        if output_tokens is not None:
            self.sieve_output_tokens = output_tokens
        return self

    def to_embedding_text(self) -> str:
        """Builds a deterministic semantic text representation for vector embedding."""
        parts = [
            f"Zone: {self.site_zone}",
            f"Title: {self.episode_title}",
            f"Category: {self.work_category}",
            f"Summary: {self.work_summary}",
            f"Workers Count: {self.workers_detected_count}",
            f"Machinery: {', '.join(self.machinery_detected) if self.machinery_detected else 'None'}",
            f"Compliance Status: {self.episode_compliance_status} (Severity: {self.overall_severity}, Safety Score: {self.safety_score_pct}%)",
        ]
        if self.ppe_observations:
            ppe_str = "; ".join(
                f"{obs.worker_label} (Tile {obs.tile_number}, Cells: {obs.grid_cells}, Action: {obs.activity_description}, Hardhat: {obs.hardhat_worn}, Vest: {obs.high_vis_vest_worn}, Harness: {obs.safety_harness_worn})"
                for obs in self.ppe_observations
            )
            parts.append(f"PPE Observations: {ppe_str}")
        else:
            parts.append("PPE Observations: None")

        if self.safety_violations:
            viol_str = "; ".join(
                f"{v.violation_type} [{v.severity.upper()}] (Tile {v.tile_number}, Time {v.timestamp_sec}s): {v.description} - Action Required: {v.corrective_action_required}"
                for v in self.safety_violations
            )
            parts.append(f"Safety Violations: {viol_str}")
        else:
            parts.append("Safety Violations: None")

        parts.append(f"Audio Safety Cues: {self.audio_safety_cues}")
        return "\n".join(parts)

    def to_vector_metadata(self) -> dict[str, Any]:
        """Returns a flat dictionary of Pinecone-safe primitives (str, int, float, bool, list[str]).

        Guarantees zero None values and no unsupported nested structures.
        """
        roi_str = (
            ",".join(str(x) for x in self.roi_bbox)
            if isinstance(self.roi_bbox, (tuple, list))
            else str(self.roi_bbox)
        )
        return {
            "video_id": self.video_id,
            "video_filename": self.video_filename,
            "site_zone": str(self.site_zone),
            "event_id": int(self.event_id),
            "episode_index": int(self.episode_index),
            "start_time_sec": float(self.start_time_sec),
            "end_time_sec": float(self.end_time_sec),
            "duration_sec": float(self.duration_sec),
            "keyframe_times_sec": [str(t) for t in self.keyframe_times_sec],
            "roi_bbox": roi_str,
            "grid_used": f"{self.grid_used[0]}x{self.grid_used[1]}",
            "storyboard_rel_path": str(self.storyboard_rel_path),
            "audio_rel_path": str(self.audio_rel_path or ""),
            "sieve_input_tokens": int(self.sieve_input_tokens),
            "sieve_output_tokens": int(self.sieve_output_tokens),
            "qwen_vl_patch_tokens": int(self.qwen_vl_patch_tokens),
            "qwen7b_kv_cache_mb": float(self.qwen7b_kv_cache_mb),
            "episode_title": str(self.episode_title),
            "work_summary": str(self.work_summary),
            "work_category": str(self.work_category),
            "workers_detected_count": int(self.workers_detected_count),
            "machinery_detected": [str(m) for m in self.machinery_detected],
            "audio_safety_cues": str(self.audio_safety_cues),
            "episode_compliance_status": str(self.episode_compliance_status),
            "overall_severity": str(self.overall_severity),
            "safety_score_pct": float(self.safety_score_pct),
            "violation_types": [str(v.violation_type) for v in self.safety_violations],
            "has_violations": bool(len(self.safety_violations) > 0),
            "ppe_observations_json": json.dumps(
                [obs.model_dump() for obs in self.ppe_observations]
            ),
            "safety_violations_json": json.dumps(
                [v.model_dump() for v in self.safety_violations]
            ),
        }


class SiteComplianceReceipt(BaseModel):
    """Complete ingestion, duty-cycle, token compression & EHS safety receipt."""

    video_id: str
    video_filename: str
    site_zone: SiteZoneType = "general_site_floor"
    duration_sec: float

    # Duty-Cycle & Productivity Telemetry
    active_work_duration_sec: float = 0.0
    idle_downtime_sec: float = 0.0
    site_utilization_pct: float = 0.0

    # Sieve & Token Compression Telemetry
    raw_frames_inspected: int = 0
    sampled_frames_decoded: int = 0
    storyboards_emitted: int = 0
    disturbances_suppressed: int = 0
    processing_wall_sec: float = 0.0
    estimated_direct_video_tokens: int = 0
    actual_sieve_audit_tokens: int = 0
    tokens_saved_pct: float = 0.0
    total_qwen_vl_patch_tokens: int = 0
    total_qwen7b_kv_cache_mb: float = 0.0

    # EHS & NCR Rollup
    overall_site_safety_score_pct: float = 100.0
    total_violations_count: int = 0
    ncr_triggered: bool = False

    episodes: list[AuditedWorkEpisodeRecord] = Field(default_factory=list)
    disturbances: list[dict[str, Any]] = Field(default_factory=list)

    def recalculate_rollups(self) -> None:
        """Recalculates rollup metrics across all audited episodes."""
        self.actual_sieve_audit_tokens = sum(
            ep.sieve_input_tokens + ep.sieve_output_tokens for ep in self.episodes
        )
        if self.estimated_direct_video_tokens > 0:
            saved = (
                self.estimated_direct_video_tokens - self.actual_sieve_audit_tokens
            )
            self.tokens_saved_pct = round(
                max(
                    0.0,
                    (saved / self.estimated_direct_video_tokens) * 100.0,
                ),
                2,
            )
        else:
            self.tokens_saved_pct = 0.0

        self.total_qwen_vl_patch_tokens = sum(
            ep.qwen_vl_patch_tokens for ep in self.episodes
        )
        self.total_qwen7b_kv_cache_mb = round(
            sum(ep.qwen7b_kv_cache_mb for ep in self.episodes), 3
        )
        self.total_violations_count = sum(
            len(ep.safety_violations) for ep in self.episodes
        )

        # NCR is triggered if any episode is marked non_compliant_ncr or has high/critical severity,
        # or if any violation item is high or critical severity.
        has_ncr_episode = any(
            ep.episode_compliance_status == "non_compliant_ncr"
            or ep.overall_severity in ("high", "critical")
            for ep in self.episodes
        )
        has_critical_violation = any(
            v.severity in ("high", "critical")
            for ep in self.episodes
            for v in ep.safety_violations
        )
        self.ncr_triggered = has_ncr_episode or has_critical_violation

        if self.episodes:
            avg_score = sum(ep.safety_score_pct for ep in self.episodes) / len(
                self.episodes
            )
            self.overall_site_safety_score_pct = round(avg_score, 2)
        else:
            self.overall_site_safety_score_pct = 100.0


# ---------------------------------------------------------------------------
# Phase 3: EHS Multimodal RAG Copilot & Official NCR Dossier Schemas
# ---------------------------------------------------------------------------


class EHSCitation(BaseModel):
    """Specific ground-truth visual/audio episode citation for RAG answers."""

    video_id: str
    episode_index: int
    site_zone: str = "general_site_floor"
    start_time_sec: float
    end_time_sec: float
    tile_number: int | None = None
    timestamp_sec: float | None = None
    grid_cells: list[str] = Field(default_factory=list)
    violation_type: str = ""
    severity: str = "low"
    evidence: str
    corrective_action: str = ""
    storyboard_url: str = ""
    audio_url: str = ""


class EHSRAGAnswerSchema(BaseModel):
    """Structured response schema returned by Gemini VLM during EHS RAG synthesis."""

    answer: str
    confidence: Literal["low", "medium", "high"] = "high"
    ncr_escalation_recommended: bool = False
    recommended_corrective_actions: list[str] = Field(default_factory=list)
    citations: list[EHSCitation] = Field(default_factory=list)


class EHSSearchRequest(BaseModel):
    @model_validator(mode="before")
    @classmethod
    def _empty_strings_to_none(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for k in ("video_id", "site_zone", "violation_type", "work_category", "min_severity"):
                if k in data and data[k] in ("", "all", "ALL"):
                    data[k] = None
            if "question" not in data and "query" in data:
                data["question"] = data["query"]
        return data

    """Request schema for semantic vector similarity search across audited episodes."""

    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=25)
    video_id: str | None = None
    site_zone: str | None = None
    ncr_only: bool = False
    violation_type: str | None = None
    work_category: str | None = None
    min_severity: Literal["low", "medium", "high", "critical"] | None = None


class EHSAskRequest(BaseModel):
    @model_validator(mode="before")
    @classmethod
    def _empty_strings_to_none(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for k in ("video_id", "site_zone", "violation_type", "work_category", "min_severity"):
                if k in data and data[k] in ("", "all", "ALL"):
                    data[k] = None
            if "question" not in data and "query" in data:
                data["question"] = data["query"]
        return data

    """Request schema for multimodal EHS RAG Copilot question answering."""

    question: str = Field(..., min_length=1)
    top_k: int = Field(default=4, ge=1, le=10)
    video_id: str | None = None
    site_zone: str | None = None
    ncr_only: bool = False
    violation_type: str | None = None
    work_category: str | None = None
    min_severity: Literal["low", "medium", "high", "critical"] | None = None
    include_storyboards: bool = True


class EHSAskResponse(BaseModel):
    """Enriched response returned by EHS RAG Copilot."""

    question: str
    answer: str
    confidence: Literal["low", "medium", "high"]
    ncr_escalation_recommended: bool
    recommended_corrective_actions: list[str]
    citations: list[EHSCitation]
    retrieved_episodes: list[dict[str, Any]]
    prompt_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0


class OfficialNCRDossier(BaseModel):
    """Enterprise Non-Conformance Report (NCR) & Duty-Cycle Telemetry Dossier."""

    ncr_reference_id: str
    video_id: str
    video_filename: str
    site_zone: str
    site_zone_label: str
    required_ppe: list[str]
    ncr_status: Literal[
        "COMPLIANT_PASS", "OBSERVATION_ISSUED", "NON_CONFORMANCE_NCR_ISSUED"
    ]
    overall_site_safety_score_pct: float
    total_duration_sec: float
    active_work_duration_sec: float
    idle_downtime_sec: float
    site_utilization_pct: float
    storyboards_emitted: int
    disturbances_suppressed: int
    tokens_saved_pct: float
    total_violations_count: int
    executive_summary: str
    violation_records: list[dict[str, Any]] = Field(default_factory=list)
    episode_summaries: list[dict[str, Any]] = Field(default_factory=list)

