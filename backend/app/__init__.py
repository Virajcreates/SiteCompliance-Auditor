"""App package for SiteCompliance-Auditor (SieveAudit)."""

from backend.app.auditor import (
    SiteStoryboardVLMAuditor,
    build_ehs_audit_prompt,
)
from backend.app.config import (
    SITE_ZONE_PRESETS,
    Settings,
    settings,
)
from backend.app.indexer import (
    ComplianceEmbedder,
    SieveAuditVectorIndex,
)
from backend.app.ingestor import (
    SiteVideoIngestor,
    compute_duty_cycle_metrics,
    extract_episode_audio,
    get_ffmpeg_binary,
    has_audio_stream,
    ingest_site_video,
)
from backend.app.main import (
    app,
    create_app,
)
from backend.app.rag import (
    SieveAuditRAGSynthesizer,
    build_ehs_rag_prompt,
    generate_official_ncr_dossier,
    render_ncr_html,
)
from backend.app.schemas import (
    AuditedWorkEpisodeRecord,
    EHSAskRequest,
    EHSAskResponse,
    EHSCitation,
    EHSRAGAnswerSchema,
    EHSSearchRequest,
    OfficialNCRDossier,
    SafetyViolationItem,
    SiteComplianceReceipt,
    SiteZoneType,
    WorkerPPEObservation,
    WorkEpisodeAuditSchema,
)

__all__ = [
    # Config
    "Settings",
    "settings",
    "SITE_ZONE_PRESETS",
    # Schemas
    "SiteZoneType",
    "WorkerPPEObservation",
    "SafetyViolationItem",
    "WorkEpisodeAuditSchema",
    "AuditedWorkEpisodeRecord",
    "SiteComplianceReceipt",
    "EHSCitation",
    "EHSRAGAnswerSchema",
    "EHSSearchRequest",
    "EHSAskRequest",
    "EHSAskResponse",
    "OfficialNCRDossier",
    # Ingestor
    "SiteVideoIngestor",
    "ingest_site_video",
    "compute_duty_cycle_metrics",
    "extract_episode_audio",
    "has_audio_stream",
    "get_ffmpeg_binary",
    # Auditor
    "SiteStoryboardVLMAuditor",
    "build_ehs_audit_prompt",
    # Indexer
    "ComplianceEmbedder",
    "SieveAuditVectorIndex",
    # RAG & NCR
    "SieveAuditRAGSynthesizer",
    "build_ehs_rag_prompt",
    "generate_official_ncr_dossier",
    "render_ncr_html",
    # Main App
    "create_app",
    "app",
]
