# SieveAudit — Construction & Industrial Site Compliance Engine

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![vlm-sieve v0.3.0](https://img.shields.io/badge/vlm--sieve-v0.3.0-F59E0B.svg)](https://pypi.org/project/vlm-sieve/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-10B981.svg)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/tests-20%2F20%20passed-10B981.svg)](#testing)
[![License: MIT](https://img.shields.io/badge/License-MIT-9CA3AF.svg)](#license)

**SieveAudit** (`SiteCompliance-Auditor`) is an end-to-end construction and industrial Environment, Health & Safety (EHS) video compliance engine, duty-cycle productivity analyzer, and multimodal RAG copilot. Built on top of [`vlm-sieve` v0.3.0](https://github.com/Virajcreates/vlm-sieve) ([PyPI](https://pypi.org/project/vlm-sieve/)), it eliminates the token blowup and temporal blindness of raw full-frame VLM video ingestion while delivering spatially and temporally grounded Non-Conformance Reports (NCRs).

---

## Key Capabilities

- **Zero-GPU CPU Video Gatekeeper (`vlm-sieve v0.3.0`):**
  - Samples CCTV streams at `10.0 FPS` (`SIEVE_SAMPLE_FPS`), suppresses global camera disturbances (pan-tilt sweeps, lighting shifts, lens shake), and prunes idle dead-air while preserving `3.0s` of pre-roll safety context (`SIEVE_PRE_ROLL_SEC`) and `5.0s` merge gaps (`SIEVE_MERGE_GAP_SEC`).
  - Automatically crops localized motion Regions of Interest (ROIs) and compiles `28px` NaViT-patch-aligned (`qwen-vl`, `640px` max tile dimension) composite keyframe storyboards stamped with relative time badges (`#tile | +offset_sec`) and a `4x4` spatial coordinate grid (`A1..D4`).
- **Synchronized Acoustic Extraction:**
  - Uses FFmpeg (`imageio-ffmpeg` or system `ffmpeg`) to slice `16 kHz`, `1-channel` (`32k` bitrate) mono MP3 audio segments bounded to each active episode (`0.5s` boundary padding), surfacing reverse alarms, crane horns, and heavy machinery acoustic cues.
- **Zone-Aware Multimodal EHS Auditing (`gemini-3.8-flash`):**
  - Evaluates storyboard images alongside synchronized audio using zone-specific engineering prompts via the `google-genai` SDK.
  - Emits structured `WorkEpisodeAuditSchema` records containing a **Worker PPE Adherence Matrix** (`hardhat_worn`, `high_vis_vest_worn`, `safety_harness_worn`, `ppe_compliant`) and itemized **Safety Violations** anchored to exact `Tile #`, `Grid Cell (A1..D4)`, and second-level timestamps.
- **Duty-Cycle Productivity & Token Compression Telemetry:**
  - Merges active work intervals to compute `active_work_duration_sec`, `idle_downtime_sec`, and `site_utilization_pct`, alongside per-feed and per-episode token reduction ratios (`tokens_saved_pct`), `qwen_vl_patch_tokens`, and `qwen7b_kv_cache_mb` savings.
- **Grounded EHS Copilot & Hybrid Vector Search (`gemini-embedding-001`):**
  - Indexes episode summaries, PPE observations, and violations into a hybrid vector store (Pinecone v5 Serverless when `PINECONE_API_KEY` is configured, with automatic deterministic local JSON/NumPy cosine fallback).
  - Supports multi-attribute filtering (`site_zone`, `violation_type`, `work_category`, `min_severity`, `ncr_only`) with click-to-seek video citations and expandable **Tall View**, **Wide View**, and **Full-Screen Modal** readers.
- **In-Page Printable A4 Enterprise NCR PDF Dossier:**
  - Generates deterministic Non-Conformance Reports (`GET /api/videos/{video_id}/ncr` JSON and `GET /api/videos/{video_id}/ncr/html` printable HTML5/A4 layout) complete with executive summaries, hazard severity badges, grid coordinates, and sign-off signature blocks.

---

## System Architecture

```text
+-----------------------------------------------------------------------------------+
| PHASE 1: VIDEO INGESTION & CPU GATEKEEPER (backend/app/ingestor.py)               |
|                                                                                   |
|  Raw CCTV Video (.mp4/.mov)                                                       |
|        |                                                                          |
|        +---> VideoSieve (10.0 FPS stride, disturbance suppression, 3.0s pre-roll) |
|        |        +---> Motion ROI Auto-Crop & 28px NaViT Composite Storyboards     |
|        |        +---> Spatial 4x4 Grid Overlay (A1..D4) & Tile Timestamps         |
|        +---> FFmpeg Audio Slicer (16 kHz mono MP3, 32k bitrate, 0.5s padding)     |
|        +---> Duty-Cycle Calculator (Active vs. Idle intervals -> Utilization %)   |
+-----------------------------------------------------------------------------------+
                                        |
                                        v
+-----------------------------------------------------------------------------------+
| PHASE 2: ZONE-AWARE VLM AUDITOR & VECTOR INDEXER (auditor.py, indexer.py)         |
|                                                                                   |
|  AuditedWorkEpisodeRecord + Storyboard JPEG + Audio MP3                           |
|        |                                                                          |
|        +---> build_ehs_audit_prompt() <--- SITE_ZONE_PRESETS (config.py)          |
|        +---> SiteStoryboardVLMAuditor (google-genai / gemini-3.8-flash)           |
|        |        +---> Worker PPE Matrix + Safety Violations (Tile #, A1..D4, ts)  |
|        +---> ComplianceEmbedder (gemini-embedding-001, 768-dim / hash fallback)   |
|        +---> SieveAuditVectorIndex (Pinecone v5 Serverless + Local JSON/NumPy)    |
+-----------------------------------------------------------------------------------+
                                        |
                                        v
+-----------------------------------------------------------------------------------+
| PHASES 3 & 4: RAG COPILOT, NCR ENGINE & COMMAND CENTER (rag.py, main.py, frontend)|
|                                                                                   |
|  FastAPI Backend (backend/app/main.py)                                            |
|        +---> /api/search & /api/ask ---> SieveAuditRAGSynthesizer (Citations)     |
|        +---> /api/videos/{id}/ncr[/html] ---> OfficialNCRDossier (Printable A4)   |
|        +---> /static/* & /media/* ---> Tactical Command Center UI (frontend/)     |
|                 |-- Left Pane: Upload/Local Path Ingest + Live Progress Poller    |
|                 |-- Center Pane: HTML5 Monitor, Duty-Cycle Bar, Episode Scrubber  |
|                 +-- Right Pane: EHS Copilot, Tall/Wide/Modal Readers, Click-Seek  |
+-----------------------------------------------------------------------------------+
```

---

## Empirical Benchmark (12-Minute Heavy Machinery Audit)

Evaluated on a continuous **12-minute (`715.7s`)** industrial heavy machinery CCTV feed running `vlm-sieve` (`10 FPS` motion stride, `28px` NaViT alignment) paired with `gemini-3.8-flash`:

| Metric | Raw 1-FPS Multimodal Baseline | SieveAudit (`vlm-sieve v0.3.0`) | Delta / Impact |
| :--- | :---: | :---: | :---: |
| **Feed Duration Analyzed** | `715.7s` | `715.7s` (`15` episodes) | `92.3%` active work utilization |
| **Multimodal Input Tokens** | `400,803` tokens | `58,602` tokens | **`85.38%` token reduction** |
| **Grounded EHS Findings** | Unstructured prose | `24` localized violations | Anchored to Tile `#` + `A1..D4` cell |

> **Note on Fixed CCTV Intervals:** This `715.7s` heavy machinery clip maintained `92.3%` near-continuous motion (`660.6s` active). On typical fixed construction site CCTV cameras—where long overnight shifts, lunch breaks, and crane wait cycles produce `50%–80%` static idle intervals—dead-air pruning compounds with storyboard compression to push total token savings to **`90%–95%+`**.

---

## Construction Zone Presets

Defined in [`backend/app/config.py`](backend/app/config.py) (`SITE_ZONE_PRESETS`):

| Zone Key (`site_zone`) | Display Label | Required PPE | Primary Monitored Hazards | Motion Threshold |
| :--- | :--- | :--- | :--- | :---: |
| `general_site_floor` | General Construction Floor / RCC Slab | `hardhat`, `high_vis_vest`, `safety_boots` | `missing_hardhat`, `missing_high_vis_vest`, `housekeeping_or_debris_hazard`, `unbarricaded_edge_or_shaft` | `0.012` |
| `scaffolding_working_at_height` | External Scaffolding & Working at Height | `hardhat`, `high_vis_vest`, `safety_harness` | `missing_harness_at_height`, `missing_hardhat`, `unbarricaded_edge_or_shaft` | `0.010` |
| `excavation_heavy_machinery` | Excavation Pit, Tower Crane & Earthmoving Bay | `hardhat`, `high_vis_vest` | `unsafe_machinery_proximity`, `missing_hardhat`, `missing_high_vis_vest`, `unauthorized_zone_entry` | `0.015` |
| `material_gate_logistics` | Material Gate, Transit Mixer & Loading Bay | `hardhat`, `high_vis_vest`, `gloves` | `unsafe_machinery_proximity`, `missing_high_vis_vest`, `unauthorized_zone_entry` | `0.014` |

---

## Repository Layout

```text
SiteCompliance-Auditor/
├── .env.example                       # Safe environment template
├── .gitignore                         # Excludes .venv, .env, runtime storage, media
├── README.md
├── requirements.txt                   # Production & test dependencies
├── backend/
│   ├── __init__.py
│   ├── requirements.txt
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py                  # Settings & SITE_ZONE_PRESETS
│   │   ├── schemas.py                 # Pydantic v2 EHS, PPE, NCR & RAG schemas
│   │   ├── ingestor.py                # vlm-sieve wrapper, duty-cycle & FFmpeg audio
│   │   ├── auditor.py                 # Zone-aware Gemini VLM multimodal auditor
│   │   ├── indexer.py                 # Hybrid Pinecone v5 + local JSON/NumPy store
│   │   ├── rag.py                     # EHS RAG Copilot & printable A4 NCR builder
│   │   └── main.py                    # FastAPI server, media routes & static mount
│   └── data/
│       └── storage/
│           └── .gitkeep               # Preserves runtime storage directory structure
├── frontend/
│   ├── index.html                     # Tactical 3-pane Command Center layout
│   ├── styles.css                     # Carbon dark palette (#090A0C / #1E222A)
│   └── app.js                         # Player sync, Tall/Wide/Modal readers, NCR PDF
└── tests/
    ├── test_phase1_ingestion.py       # Gatekeeper, ROI crop, duty-cycle & audio tests
    ├── test_phase2_audit_and_indexing.py # Prompting, VLM audit & hybrid index tests
    ├── test_phase3_rag_ncr_and_api.py # Vector search, RAG citations & NCR API tests
    └── test_phase4_ui_and_e2e.py      # Static UI, DOM ID integrity & E2E lifecycle
```

---

## Quickstart Guide (Windows PowerShell)

### 1. Clone & Create Virtual Environment
```powershell
git clone https://github.com/Virajcreates/SiteCompliance-Auditor.git
cd SiteCompliance-Auditor

python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Configure Environment Variables
```powershell
Copy-Item .env.example .env
```
Open `.env` and populate `GEMINI_API_KEY` (required for live VLM auditing, `gemini-embedding-001` embeddings, and RAG synthesis). Leave `PINECONE_API_KEY` blank to use the built-in local JSON/NumPy vector index, or provide a Pinecone key to enable serverless hybrid indexing.

### 3. Launch SieveAudit Server
```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```
Open the Tactical Command Center in your browser at **`http://localhost:8000`**.

---

## REST API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the Tactical EHS & Site Compliance Command Center (`frontend/index.html`). |
| `GET` | `/static/{path}` | Serves frontend static assets (`styles.css`, `app.js`). |
| `GET` | `/api/health` | Returns service health, `vlm-sieve` version, default VLM, vector backend (`local` or `pinecone`), and indexed episode count. |
| `GET` | `/api/ingest/progress` | Live polling endpoint reporting storyboard JPEG and MP3 audio slice counts during active ingestion. |
| `GET` | `/api/zones` | Returns all 4 `SITE_ZONE_PRESETS` configurations, required PPE, and monitored hazards. |
| `POST` | `/api/videos/ingest` | Ingests a video via multipart upload (`file`) or local path (`local_video_path`), runs `vlm-sieve`, optional VLM audit, and vector indexing. |
| `GET` | `/api/videos` | Lists summary compliance cards for all ingested site feeds. |
| `GET` | `/api/videos/{video_id}` | Retrieves the full `SiteComplianceReceipt`, duty-cycle telemetry, and audited episodes for a video. |
| `DELETE` | `/api/videos/{video_id}` | Purges a video's stored media, storyboards, receipts, and indexed vectors. |
| `GET` | `/api/videos/{video_id}/ncr` | Compiles and returns the structured `OfficialNCRDossier` JSON payload. |
| `GET` | `/api/videos/{video_id}/ncr/html` | Renders the printable HTML5/A4 Non-Conformance Report certificate. |
| `GET` | `/api/stats` | Aggregates site-wide utilization, average safety score, total violations, token reduction, and KV-cache savings. |
| `POST` | `/api/search` | Executes semantic vector search (`EHSSearchRequest`) with zone, violation type, work category, severity, and NCR filters. |
| `POST` | `/api/ask` | Runs the Multimodal EHS RAG Copilot (`EHSAskRequest`) returning grounded answers, corrective actions, and clickable citations. |
| `GET` | `/media/{video_id}/{filename}` | Streams ingested video feeds, composite storyboard JPEGs, and `16 kHz` MP3 audio slices. |

---

## Testing

All 4 phases are covered by an automated offline test suite (20 unit, integration, DOM integrity, and E2E tests) using deterministic mocks with zero external network calls:

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

Expected output: `20 passed`.

---

## License

Released under the [MIT License](https://opensource.org/licenses/MIT).
