"""Configuration settings and construction site zone presets for SieveAudit."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env if present
load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

SITE_ZONE_PRESETS: dict[str, dict[str, Any]] = {
    "general_site_floor": {
        "label": "General Construction Floor / RCC Slab",
        "required_ppe": ["hardhat", "high_vis_vest", "safety_boots"],
        "primary_hazards": [
            "missing_hardhat",
            "missing_high_vis_vest",
            "housekeeping_or_debris_hazard",
            "unbarricaded_edge_or_shaft",
        ],
        "motion_threshold": 0.012,
    },
    "scaffolding_working_at_height": {
        "label": "External Scaffolding & Working at Height",
        "required_ppe": ["hardhat", "high_vis_vest", "safety_harness"],
        "primary_hazards": [
            "missing_harness_at_height",
            "missing_hardhat",
            "unbarricaded_edge_or_shaft",
        ],
        "motion_threshold": 0.010,
    },
    "excavation_heavy_machinery": {
        "label": "Excavation Pit, Tower Crane & Earthmoving Bay",
        "required_ppe": ["hardhat", "high_vis_vest"],
        "primary_hazards": [
            "unsafe_machinery_proximity",
            "missing_hardhat",
            "missing_high_vis_vest",
            "unauthorized_zone_entry",
        ],
        "motion_threshold": 0.015,
    },
    "material_gate_logistics": {
        "label": "Material Gate, Transit Mixer & Loading Bay",
        "required_ppe": ["hardhat", "high_vis_vest", "gloves"],
        "primary_hazards": [
            "unsafe_machinery_proximity",
            "missing_high_vis_vest",
            "unauthorized_zone_entry",
        ],
        "motion_threshold": 0.014,
    },
}


class Settings(BaseModel):
    """Application settings for SiteCompliance-Auditor."""

    APP_NAME: str = "SieveAudit // Construction & Industrial Site Compliance Engine"
    APP_VERSION: str = "0.1.0"
    BASE_DIR: Path = REPO_ROOT
    DATA_DIR: Path = REPO_ROOT / "backend" / "data" / "storage"

    # vlm-sieve parameters optimized for construction CCTV
    SIEVE_SAMPLE_FPS: float = 10.0
    SIEVE_PRE_ROLL_SEC: float = 3.0
    SIEVE_MERGE_GAP_SEC: float = 5.0
    SIEVE_TILE_MAX_DIM: int = 640
    SIEVE_SPATIAL_GRID_OVERLAY: bool = True
    SIEVE_BURN_TIMESTAMPS: bool = True
    SIEVE_SUPPRESS_DISTURBANCES: bool = True
    SIEVE_MOTION_THRESHOLD: float = 0.012
    SIEVE_TOKENS_PER_IMAGE: int = 1120
    DIRECT_VIDEO_TOKENS_PER_SEC: float = 300.0
    DEFAULT_TARGET_VLM: str = Field(
        default_factory=lambda: os.getenv("DEFAULT_TARGET_VLM", "qwen-vl")
    )

    # Audio slicing parameters
    AUDIO_SAMPLE_RATE: int = 16000
    AUDIO_CHANNELS: int = 1
    AUDIO_BITRATE: str = "32k"
    AUDIO_PAD_SEC: float = 0.5

    # VLM & Vector Index settings
    GEMINI_API_KEY: str = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    GEMINI_VLM_MODEL: str = Field(
        default_factory=lambda: os.getenv("GEMINI_VLM_MODEL", "gemini-3.8-flash")
    )
    GEMINI_EMBEDDING_MODEL: str = Field(
        default_factory=lambda: os.getenv(
            "GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"
        )
    )
    EMBEDDING_DIMENSION: int = Field(
        default_factory=lambda: int(os.getenv("EMBEDDING_DIMENSION", "768"))
    )
    PINECONE_API_KEY: str = Field(
        default_factory=lambda: os.getenv("PINECONE_API_KEY", "")
    )
    PINECONE_INDEX_NAME: str = Field(
        default_factory=lambda: os.getenv(
            "PINECONE_INDEX_NAME", "sieveaudit-episodes"
        )
    )
    PINECONE_NAMESPACE: str = Field(
        default_factory=lambda: os.getenv("PINECONE_NAMESPACE", "default")
    )


settings = Settings()

# Module-level exports
APP_NAME = settings.APP_NAME
APP_VERSION = settings.APP_VERSION
BASE_DIR = settings.BASE_DIR
DATA_DIR = settings.DATA_DIR

SIEVE_SAMPLE_FPS = settings.SIEVE_SAMPLE_FPS
SIEVE_PRE_ROLL_SEC = settings.SIEVE_PRE_ROLL_SEC
SIEVE_MERGE_GAP_SEC = settings.SIEVE_MERGE_GAP_SEC
SIEVE_TILE_MAX_DIM = settings.SIEVE_TILE_MAX_DIM
SIEVE_SPATIAL_GRID_OVERLAY = settings.SIEVE_SPATIAL_GRID_OVERLAY
SIEVE_BURN_TIMESTAMPS = settings.SIEVE_BURN_TIMESTAMPS
SIEVE_SUPPRESS_DISTURBANCES = settings.SIEVE_SUPPRESS_DISTURBANCES
SIEVE_MOTION_THRESHOLD = settings.SIEVE_MOTION_THRESHOLD
SIEVE_TOKENS_PER_IMAGE = settings.SIEVE_TOKENS_PER_IMAGE
DIRECT_VIDEO_TOKENS_PER_SEC = settings.DIRECT_VIDEO_TOKENS_PER_SEC
DEFAULT_TARGET_VLM = settings.DEFAULT_TARGET_VLM

AUDIO_SAMPLE_RATE = settings.AUDIO_SAMPLE_RATE
AUDIO_CHANNELS = settings.AUDIO_CHANNELS
AUDIO_BITRATE = settings.AUDIO_BITRATE
AUDIO_PAD_SEC = settings.AUDIO_PAD_SEC

GEMINI_API_KEY = settings.GEMINI_API_KEY
GEMINI_VLM_MODEL = settings.GEMINI_VLM_MODEL
GEMINI_EMBEDDING_MODEL = settings.GEMINI_EMBEDDING_MODEL
EMBEDDING_DIMENSION = settings.EMBEDDING_DIMENSION
PINECONE_API_KEY = settings.PINECONE_API_KEY
PINECONE_INDEX_NAME = settings.PINECONE_INDEX_NAME
PINECONE_NAMESPACE = settings.PINECONE_NAMESPACE

