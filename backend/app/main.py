"""FastAPI REST Application & Media Server for SiteCompliance-Auditor (SieveAudit)."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import time
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from backend.app.auditor import SiteStoryboardVLMAuditor
from backend.app.config import (
    SITE_ZONE_PRESETS,
    Settings,
    settings as default_settings,
)
from backend.app.indexer import SieveAuditVectorIndex
from backend.app.ingestor import ingest_site_video
from backend.app.rag import (
    SieveAuditRAGSynthesizer,
    generate_official_ncr_dossier,
    render_ncr_html,
)
from backend.app.schemas import (
    EHSAskRequest,
    EHSAskResponse,
    EHSSearchRequest,
    OfficialNCRDossier,
    SiteComplianceReceipt,
    SiteZoneType,
)


def create_app(
    workspace_dir: str | Path | None = None,
    settings: Settings | None = None,
    auditor: SiteStoryboardVLMAuditor | None = None,
    index: SieveAuditVectorIndex | None = None,
    synthesizer: SieveAuditRAGSynthesizer | None = None,
    frontend_dir: str | Path | None = None,
) -> FastAPI:
    """Creates and configures the FastAPI application with injected or default services."""
    app_settings = settings or default_settings
    storage_dir = Path(workspace_dir or app_settings.DATA_DIR).resolve()
    storage_dir.mkdir(parents=True, exist_ok=True)
    resolved_frontend_dir = Path(
        frontend_dir or (app_settings.BASE_DIR / "frontend")
    ).resolve()
    resolved_frontend_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Indexer
    app_index = index or SieveAuditVectorIndex(
        local_index_path=storage_dir / "vector_index.json"
    )

    # Initialize Auditor
    if auditor is not None:
        app_auditor = auditor
    else:
        app_auditor = SiteStoryboardVLMAuditor(
            api_key=app_settings.GEMINI_API_KEY
        )

    # Initialize Synthesizer
    if synthesizer is not None:
        app_synthesizer = synthesizer
    else:
        app_synthesizer = SieveAuditRAGSynthesizer(
            index=app_index,
            workspace_dir=storage_dir,
            api_key=app_settings.GEMINI_API_KEY,
        )

    # Offline safety flag
    has_live_vlm = bool(
        app_auditor.client is not None
        or (
            getattr(app_auditor, "api_key", "")
            and getattr(app_auditor, "api_key", "").strip()
        )
    )

    app = FastAPI(
        title=app_settings.APP_NAME,
        version=app_settings.APP_VERSION,
        description="Core Construction Site Video Ingestion, ROI Auto-Cropping, Multimodal EHS Auditor & RAG Copilot",
    )

    # Enable CORS for frontend clients
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount static assets for frontend command center
    app.mount(
        "/static",
        StaticFiles(directory=str(resolved_frontend_dir)),
        name="static",
    )

    @app.get("/", response_class=FileResponse)
    def serve_index() -> FileResponse:
        """Serves the Tactical EHS & Site Compliance Command Center index.html."""
        index_file = resolved_frontend_dir / "index.html"
        if not index_file.is_file():
            raise HTTPException(
                status_code=404, detail="Frontend index.html not found."
            )
        return FileResponse(str(index_file))

    # -----------------------------------------------------------------------
    # System & Zone Endpoints
    # -----------------------------------------------------------------------

    @app.get("/v1/models", include_in_schema=False)
    def openai_models_stub() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": "sieveaudit-ehs-v0.3.0", "object": "model"}]}

    @app.get("/health", include_in_schema=False)
    @app.get("/api/ingest/progress", include_in_schema=False)
    def ingest_live_progress() -> dict[str, Any]:
        """Returns real-time file counts in storage_dir while a long video is being ingested/audited."""
        folders = []
        if storage_dir.exists():
            for d in sorted(storage_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                if d.is_dir():
                    sbs = list((d / "storyboards").glob("*.jpg")) if (d / "storyboards").exists() else []
                    auds = list((d / "audio").glob("*.mp3")) if (d / "audio").exists() else []
                    has_receipt = (d / "compliance_receipt.json").exists()
                    folders.append({
                        "video_id": d.name,
                        "storyboards_count": len(sbs),
                        "audio_slices_count": len(auds),
                        "receipt_ready": has_receipt,
                    })
        return {"active_folders": folders[:3]}

    @app.get("/api/health")
    def get_health() -> dict[str, Any]:
        """Returns health status, active models, vector backend, and indexed episode counts."""
        vector_backend = "pinecone" if getattr(app_index, "_pc_index", None) else "local"
        indexed_count = len(getattr(app_index, "_local_records", {}))
        return {
            "status": "ok",
            "app_name": app_settings.APP_NAME,
            "version": app_settings.APP_VERSION,
            "vlmsieve_version": "0.3.0",
            "target_vlm_default": app_settings.DEFAULT_TARGET_VLM,
            "gemini_configured": bool(app_settings.GEMINI_API_KEY),
            "vector_backend": vector_backend,
            "indexed_episodes_count": indexed_count,
            "storage_dir": str(storage_dir),
        }

    @app.get("/api/zones")
    def get_zones() -> dict[str, dict[str, Any]]:
        """Returns the 4 specialized construction zone presets, required PPE, and monitored hazards."""
        return SITE_ZONE_PRESETS

    # -----------------------------------------------------------------------
    # Ingestion & Video Endpoints
    # -----------------------------------------------------------------------

    @app.post("/api/videos/ingest")
    async def ingest_video(
        file: UploadFile | None = File(default=None),
        local_video_path: str | None = Form(default=None),
        site_zone: str = Form(default="general_site_floor"),
        target_vlm: str = Form(default="qwen-vl"),
        run_vlm_audit: bool = Form(default=True),
        index_episodes: bool = Form(default=True),
    ) -> dict[str, Any]:
        """Ingests a site CCTV feed, performs ROI auto-cropping, VLM auditing, and vector indexing."""
        if file is None and not local_video_path:
            raise HTTPException(
                status_code=400,
                detail="Either 'file' or 'local_video_path' must be provided.",
            )

        # 1. Determine video_id and storage paths
        if file is not None:
            filename = file.filename or "uploaded_video.mp4"
            stem = Path(filename).stem
            video_id = f"vid_{stem}"
            vid_dir = storage_dir / video_id
            vid_dir.mkdir(parents=True, exist_ok=True)
            target_path = vid_dir / filename
            content = await file.read()
            target_path.write_bytes(content)
        else:
            assert local_video_path is not None
            source_path = Path(local_video_path).resolve()
            if not source_path.is_file():
                raise HTTPException(
                    status_code=404,
                    detail=f"Local video file not found at: {local_video_path}",
                )
            filename = source_path.name
            stem = source_path.stem
            video_id = f"vid_{stem}"
            vid_dir = storage_dir / video_id
            vid_dir.mkdir(parents=True, exist_ok=True)
            target_path = vid_dir / filename
            if source_path != target_path:
                shutil.copy2(source_path, target_path)

        # 2. Execute Phase 1 Video Sieve Ingestion
        try:
            zone_key: SiteZoneType = (
                site_zone if site_zone in SITE_ZONE_PRESETS else "general_site_floor"  # type: ignore[assignment]
            )
            receipt = ingest_site_video(
                video_path=target_path,
                output_dir=vid_dir,
                video_id=video_id,
                site_zone=zone_key,
                target_vlm=target_vlm,
            )
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Video ingestion failed: {e}"
            )

        # 3. Optional VLM Auditing
        if run_vlm_audit:
            if has_live_vlm:
                try:
                    res = app_auditor.audit_receipt(
                        receipt, workspace_dir=vid_dir, save_updated_receipt=True
                    )
                    if isinstance(res, SiteComplianceReceipt):
                        receipt = res
                    else:
                        receipt.recalculate_rollups()
                except Exception:
                    receipt.recalculate_rollups()
            else:
                receipt.recalculate_rollups()
                (vid_dir / "compliance_receipt.json").write_text(
                    receipt.model_dump_json(indent=2), encoding="utf-8"
                )

        # 4. Optional Vector Indexing
        upserted_count = 0
        if index_episodes and receipt.episodes:
            try:
                upsert_res = app_index.upsert_episodes(receipt.episodes)
                upserted_count = upsert_res.get("upserted_count", len(receipt.episodes))
            except Exception:
                upserted_count = 0

        # 5. Enrich receipt response with accessible URLs
        res_dict = receipt.model_dump()
        res_dict["video_url"] = f"/media/{video_id}/{filename}"
        res_dict["ncr_url"] = f"/api/videos/{video_id}/ncr"
        res_dict["ncr_html_url"] = f"/api/videos/{video_id}/ncr/html"
        res_dict["upserted_vectors"] = upserted_count

        if receipt.episodes:
            first_ep = receipt.episodes[0]
            clean_sb = (first_ep.storyboard_rel_path or "").lstrip("/")
            res_dict["storyboard_url"] = f"/media/{video_id}/{clean_sb}"
            if first_ep.audio_rel_path:
                clean_aud = (first_ep.audio_rel_path or "").lstrip("/")
                res_dict["audio_url"] = f"/media/{video_id}/{clean_aud}"

        return res_dict

    @app.get("/api/videos")
    def list_videos() -> list[dict[str, Any]]:
        """Returns summary compliance cards for all processed site videos."""
        cards: list[dict[str, Any]] = []

        if not storage_dir.is_dir():
            return cards

        for vid_folder in storage_dir.iterdir():
            if not vid_folder.is_dir():
                continue
            receipt_file = vid_folder / "compliance_receipt.json"
            if receipt_file.is_file():
                try:
                    data = json.loads(receipt_file.read_text(encoding="utf-8"))
                    vid_id = data.get("video_id", vid_folder.name)
                    filename = data.get("video_filename", "video.mp4")
                    cards.append(
                        {
                            "video_id": vid_id,
                            "video_filename": filename,
                            "site_zone": data.get("site_zone", "general_site_floor"),
                            "duration_sec": data.get("duration_sec", 0.0),
                            "active_work_duration_sec": data.get(
                                "active_work_duration_sec", 0.0
                            ),
                            "idle_downtime_sec": data.get("idle_downtime_sec", 0.0),
                            "site_utilization_pct": data.get(
                                "site_utilization_pct", 0.0
                            ),
                            "overall_site_safety_score_pct": data.get(
                                "overall_site_safety_score_pct", 100.0
                            ),
                            "total_violations_count": data.get(
                                "total_violations_count", 0
                            ),
                            "ncr_triggered": data.get("ncr_triggered", False),
                            "storyboards_emitted": data.get(
                                "storyboards_emitted", len(data.get("episodes", []))
                            ),
                            "tokens_saved_pct": data.get("tokens_saved_pct", 0.0),
                            "video_url": f"/media/{vid_id}/{filename}",
                            "ncr_url": f"/api/videos/{vid_id}/ncr",
                            "ncr_html_url": f"/api/videos/{vid_id}/ncr/html",
                        }
                    )
                except Exception:
                    pass

        return cards

    @app.get("/api/videos/{video_id}")
    def get_video_receipt(video_id: str) -> dict[str, Any]:
        """Returns the full compliance receipt and telemetry for a specific video."""
        receipt_file = storage_dir / video_id / "compliance_receipt.json"
        if not receipt_file.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"Compliance receipt not found for video '{video_id}'",
            )

        try:
            data = json.loads(receipt_file.read_text(encoding="utf-8"))
            filename = data.get("video_filename", "video.mp4")
            data["video_url"] = f"/media/{video_id}/{filename}"
            data["ncr_url"] = f"/api/videos/{video_id}/ncr"
            data["ncr_html_url"] = f"/api/videos/{video_id}/ncr/html"

            for ep in data.get("episodes", []):
                sb = (ep.get("storyboard_rel_path") or "").lstrip("/")
                ep["storyboard_url"] = f"/media/{video_id}/{sb}" if sb else ""
                aud = (ep.get("audio_rel_path") or "").lstrip("/")
                ep["audio_url"] = f"/media/{video_id}/{aud}" if aud else ""

            return data
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to read receipt: {e}"
            )

    @app.delete("/api/videos/{video_id}")
    def delete_video(video_id: str) -> dict[str, Any]:
        """Deletes a video, its storyboards, receipts, and vector embeddings."""
        # 1. Purge from vector store
        deleted_vectors = app_index.delete_video(video_id)

        # 2. Purge files on disk
        vid_dir = storage_dir / video_id
        if vid_dir.is_dir():
            shutil.rmtree(vid_dir, ignore_errors=True)

        return {
            "deleted": True,
            "video_id": video_id,
            "purged_vectors": deleted_vectors,
        }

    # -----------------------------------------------------------------------
    # Official Non-Conformance Report (NCR) Dossier Endpoints
    # -----------------------------------------------------------------------

    @app.get("/api/videos/{video_id}/ncr")
    def get_video_ncr_json(video_id: str) -> OfficialNCRDossier:
        """Generates and returns the official enterprise Non-Conformance Report (NCR) JSON."""
        receipt_file = storage_dir / video_id / "compliance_receipt.json"
        if not receipt_file.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"Compliance receipt not found for video '{video_id}'",
            )

        try:
            data = json.loads(receipt_file.read_text(encoding="utf-8"))
            receipt = SiteComplianceReceipt.model_validate(data)
            return generate_official_ncr_dossier(receipt)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to compile NCR dossier: {e}"
            )

    @app.get("/api/videos/{video_id}/ncr/html")
    def get_video_ncr_html(video_id: str) -> HTMLResponse:
        """Renders the official enterprise Non-Conformance Report as printable HTML5."""
        receipt_file = storage_dir / video_id / "compliance_receipt.json"
        if not receipt_file.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"Compliance receipt not found for video '{video_id}'",
            )

        try:
            data = json.loads(receipt_file.read_text(encoding="utf-8"))
            receipt = SiteComplianceReceipt.model_validate(data)
            dossier = generate_official_ncr_dossier(receipt)
            html_content = render_ncr_html(dossier)
            return HTMLResponse(content=html_content, status_code=200)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to render NCR HTML: {e}"
            )

    # -----------------------------------------------------------------------
    # Corpus Telemetry & Stats
    # -----------------------------------------------------------------------

    @app.get("/api/stats")
    def get_corpus_stats() -> dict[str, Any]:
        """Calculates site-wide productivity, duty-cycle, token compression, and safety metrics."""
        receipts: list[dict[str, Any]] = []

        if storage_dir.is_dir():
            for folder in storage_dir.iterdir():
                r_file = folder / "compliance_receipt.json"
                if r_file.is_file():
                    try:
                        receipts.append(
                            json.loads(r_file.read_text(encoding="utf-8"))
                        )
                    except Exception:
                        pass

        total_videos = len(receipts)
        total_duration = sum(float(r.get("duration_sec", 0.0)) for r in receipts)
        total_active = sum(
            float(r.get("active_work_duration_sec", 0.0)) for r in receipts
        )
        total_idle = sum(float(r.get("idle_downtime_sec", 0.0)) for r in receipts)

        avg_util = (
            sum(float(r.get("site_utilization_pct", 0.0)) for r in receipts)
            / total_videos
            if total_videos > 0
            else 0.0
        )
        avg_score = (
            sum(
                float(r.get("overall_site_safety_score_pct", 100.0))
                for r in receipts
            )
            / total_videos
            if total_videos > 0
            else 100.0
        )
        total_violations = sum(
            int(r.get("total_violations_count", 0)) for r in receipts
        )
        ncr_videos = sum(1 for r in receipts if r.get("ncr_triggered"))

        total_storyboards = sum(
            int(r.get("storyboards_emitted", len(r.get("episodes", []))))
            for r in receipts
        )
        total_disturbances = sum(
            int(r.get("disturbances_suppressed", 0)) for r in receipts
        )
        avg_token_savings = (
            sum(float(r.get("tokens_saved_pct", 0.0)) for r in receipts)
            / total_videos
            if total_videos > 0
            else 0.0
        )
        total_kv_cache = sum(
            float(r.get("total_qwen7b_kv_cache_mb", 0.0)) for r in receipts
        )

        return {
            "total_videos": total_videos,
            "total_duration_sec": round(total_duration, 2),
            "total_active_work_sec": round(total_active, 2),
            "total_idle_downtime_sec": round(total_idle, 2),
            "average_utilization_pct": round(avg_util, 2),
            "average_safety_score_pct": round(avg_score, 2),
            "total_violations_count": total_violations,
            "total_ncr_videos": ncr_videos,
            "total_storyboards_emitted": total_storyboards,
            "total_disturbances_suppressed": total_disturbances,
            "overall_tokens_saved_pct": round(avg_token_savings, 2),
            "total_qwen7b_kv_cache_mb": round(total_kv_cache, 2),
            "indexed_episodes_count": len(getattr(app_index, "_local_records", {})),
        }

    # -----------------------------------------------------------------------
    # EHS Vector Search & RAG Copilot Endpoints
    # -----------------------------------------------------------------------

    @app.post("/api/search")
    def search_episodes(req: EHSSearchRequest) -> list[dict[str, Any]]:
        """Executes multi-attribute semantic search across audited site episodes."""
        return app_index.search(
            query=req.query,
            top_k=req.top_k,
            video_id=req.video_id,
            site_zone=req.site_zone,
            ncr_only=req.ncr_only,
            violation_type=req.violation_type,
            work_category=req.work_category,
            min_severity=req.min_severity,
        )

    @app.post("/api/ask")
    def ask_rag_copilot(req: EHSAskRequest) -> EHSAskResponse:
        """Multimodal EHS RAG Copilot synthesizing grounded answers with citations."""
        return app_synthesizer.answer_question(req)

    # -----------------------------------------------------------------------
    # Static Media Streaming Route
    # -----------------------------------------------------------------------

    @app.get("/media/{video_id}/{filename:path}")
    def serve_media(video_id: str, filename: str) -> FileResponse:
        """Serves video files, storyboard JPEGs, and audio MP3 slices."""
        clean_name = filename.lstrip("/")
        candidate_paths = [
            storage_dir / video_id / clean_name,
            storage_dir / clean_name,
        ]

        found_path: Path | None = None
        for p in candidate_paths:
            if p.is_file():
                found_path = p
                break

        if not found_path:
            raise HTTPException(
                status_code=404,
                detail=f"Media file '{filename}' not found for video '{video_id}'",
            )

        suffix = found_path.suffix.lower()
        media_type = "application/octet-stream"
        if suffix in (".jpg", ".jpeg"):
            media_type = "image/jpeg"
        elif suffix == ".png":
            media_type = "image/png"
        elif suffix == ".mp3":
            media_type = "audio/mpeg"
        elif suffix in (".mp4", ".mov", ".webm"):
            media_type = "video/mp4"

        return FileResponse(str(found_path), media_type=media_type)

    return app


# Default application instance for Uvicorn
app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
