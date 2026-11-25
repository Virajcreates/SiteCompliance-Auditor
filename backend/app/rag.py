"""EHS Multimodal RAG Copilot & Official Non-Conformance Report (NCR) Dossier Generator."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from google import genai
from google.genai import types

from backend.app.config import GEMINI_VLM_MODEL, SITE_ZONE_PRESETS, settings
from backend.app.indexer import SieveAuditVectorIndex
from backend.app.schemas import (
    EHSAskRequest,
    EHSAskResponse,
    EHSCitation,
    EHSRAGAnswerSchema,
    OfficialNCRDossier,
    SiteComplianceReceipt,
)


def build_ehs_rag_prompt(question: str, hits: list[dict[str, Any]]) -> str:
    """Builds a grounded EHS engineering synthesis prompt incorporating retrieved episodes and evidence."""
    episodes_context_blocks: list[str] = []

    for idx, hit in enumerate(hits):
        vid = hit.get("video_id", "unknown")
        ep_idx = hit.get("episode_index", idx)
        zone = hit.get("site_zone", "general_site_floor")
        start_t = hit.get("start_time_sec", 0.0)
        end_t = hit.get("end_time_sec", 0.0)
        duration = hit.get("duration_sec", max(0.0, end_t - start_t))
        title = hit.get("episode_title", "Site Episode")
        category = hit.get("work_category", "active_construction_work")
        summary = hit.get("work_summary", "")
        status = hit.get("episode_compliance_status", "compliant")
        severity = hit.get("overall_severity", "low")
        score = hit.get("safety_score_pct", 100.0)
        workers_cnt = hit.get("workers_detected_count", 0)
        machinery = hit.get("machinery_detected", [])
        audio_cues = hit.get("audio_safety_cues", "No audio cues")

        # Format itemized PPE observations
        ppe_lines: list[str] = []
        raw_ppes = hit.get("ppe_observations", [])
        for p in raw_ppes:
            lbl = p.get("worker_label", "Worker")
            t_num = p.get("tile_number", 1)
            rel_t = p.get("relative_time_sec", 0.0)
            abs_t = p.get("absolute_time_sec", 0.0)
            cells = p.get("grid_cells", [])
            act = p.get("activity_description", "")
            hd = p.get("hardhat_worn", True)
            vest = p.get("high_vis_vest_worn", True)
            harn = p.get("safety_harness_worn", False)
            ppe_lines.append(
                f"    - {lbl} (Tile #{t_num}, +{rel_t:.2f}s/abs {abs_t:.2f}s, Cells: {cells}): "
                f"Action='{act}', Hardhat={hd}, Vest={vest}, Harness={harn}"
            )
        ppe_str = "\n".join(ppe_lines) if ppe_lines else "    - None"

        # Format itemized safety violations
        viol_lines: list[str] = []
        raw_viols = hit.get("safety_violations", [])
        for v in raw_viols:
            v_type = v.get("violation_type", "hazard")
            v_sev = v.get("severity", "medium").upper()
            t_num = v.get("tile_number", 1)
            t_sec = v.get("timestamp_sec", start_t)
            cells = v.get("grid_cells", [])
            desc = v.get("description", "")
            action = v.get("corrective_action_required", "")
            viol_lines.append(
                f"    - [{v_sev}] {v_type} (Tile #{t_num}, {t_sec:.2f}s, Cells: {cells}): "
                f"{desc} | Action Required: {action}"
            )
        viol_str = "\n".join(viol_lines) if viol_lines else "    - None"

        block = f"""[Retrieved Episode #{idx + 1}]
- Video ID: {vid} | Episode Index: {ep_idx} | Zone: {zone}
- Time Bounds: {start_t:.2f}s to {end_t:.2f}s (Duration: {duration:.2f}s)
- Title: {title} | Category: {category}
- Summary: {summary}
- Compliance Status: {status} | Overall Severity: {severity} | Safety Score: {score}%
- Workers Detected: {workers_cnt} | Machinery: {', '.join(machinery) if machinery else 'None'}
- Audio Safety Cues: {audio_cues}
- PPE Observations:
{ppe_str}
- Safety Violations:
{viol_str}"""
        episodes_context_blocks.append(block)

    all_context = "\n\n".join(episodes_context_blocks)

    prompt = f"""You are an Senior EHS (Environment, Health & Safety) Compliance Officer and Construction Superintendent.
Answer the following user question strictly based on the retrieved audited construction site episodes and attached visual storyboards.

Retrieved Audited Episodes Context:
====================================
{all_context}
====================================

User Question:
"{question}"

Instructions & Grounding Rules:
1. Provide a direct, professional, and definitive answer to the user's inquiry based strictly on the factual evidence in the retrieved episodes.
2. If any severe EHS hazard or Non-Conformance (NCR) is detected (missing harness at height, missing hardhat, excavator swing radius breach, unbarricaded shaft), set `ncr_escalation_recommended=True` and provide itemized `recommended_corrective_actions`.
3. Provide precise citations in `citations` with `video_id`, `episode_index`, `tile_number`, `timestamp_sec`, `grid_cells`, `evidence`, and `corrective_action`.
4. If no relevant evidence exists in the retrieved context to answer the question, state that clearly and set `confidence="low"`.
"""
    return prompt


class SieveAuditRAGSynthesizer:
    """Multimodal EHS RAG copilot synthesizing grounded answers from vector search and storyboards."""

    def __init__(
        self,
        index: SieveAuditVectorIndex,
        workspace_dir: str | Path | None = None,
        api_key: str | None = None,
        model_name: str = GEMINI_VLM_MODEL,
        client: Any | None = None,
    ) -> None:
        self.index = index
        self.workspace_dir = Path(workspace_dir or settings.DATA_DIR).resolve()
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model_name = model_name
        self.client = client

    def _get_client(self) -> Any:
        if self.client is not None:
            return self.client
        if not self.api_key:
            return None
        self.client = genai.Client(api_key=self.api_key)
        return self.client

    def answer_question(self, request: EHSAskRequest) -> EHSAskResponse:
        """Executes vector retrieval, attaches storyboard evidence, and synthesizes grounded EHS answers."""
        t0 = time.time()

        # 1. Search vector index with filters
        hits = self.index.search(
            query=request.question,
            top_k=request.top_k,
            video_id=request.video_id,
            site_zone=request.site_zone,
            ncr_only=request.ncr_only,
            violation_type=request.violation_type,
            work_category=request.work_category,
            min_severity=request.min_severity,
        )

        if not hits:
            return EHSAskResponse(
                question=request.question,
                answer="No matching site compliance episodes were found for the specified query and filters.",
                confidence="low",
                ncr_escalation_recommended=False,
                recommended_corrective_actions=[],
                citations=[],
                retrieved_episodes=[],
                prompt_tokens=0,
                output_tokens=0,
                latency_ms=round((time.time() - t0) * 1000, 2),
            )

        # 2. Normalize media URLs on hits
        for h in hits:
            vid = h.get("video_id", "")
            sb_rel = str(h.get("storyboard_rel_path", "")).lstrip("/")
            if sb_rel and not sb_rel.startswith("media/"):
                h["storyboard_url"] = f"/media/{vid}/{sb_rel}"
            else:
                h["storyboard_url"] = f"/{sb_rel}" if sb_rel else ""

            audio_rel = str(h.get("audio_rel_path", "")).lstrip("/")
            if audio_rel and not audio_rel.startswith("media/"):
                h["audio_url"] = f"/media/{vid}/{audio_rel}"
            else:
                h["audio_url"] = f"/{audio_rel}" if audio_rel else ""

        # 3. Resolve storyboards if requested and workspace directory is accessible
        parts: list[Any] = []
        if request.include_storyboards and self.workspace_dir:
            for hit in hits[:3]:
                vid = hit.get("video_id", "")
                sb_rel = hit.get("storyboard_rel_path", "")
                if sb_rel:
                    candidate_paths = [
                        self.workspace_dir / sb_rel,
                        self.workspace_dir / vid / sb_rel,
                    ]
                    for p in candidate_paths:
                        if p.is_file():
                            try:
                                img_bytes = p.read_bytes()
                                parts.append(
                                    types.Part.from_bytes(
                                        data=img_bytes, mime_type="image/jpeg"
                                    )
                                )
                                break
                            except Exception:
                                pass

        prompt_text = build_ehs_rag_prompt(request.question, hits)
        parts.append(prompt_text)

        # 4. Attempt Gemini VLM structured generation if configured
        client = self._get_client()
        if client is not None:
            try:
                res = client.models.generate_content(
                    model=self.model_name,
                    contents=parts,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        response_mime_type="application/json",
                        response_schema=EHSRAGAnswerSchema,
                    ),
                )
                raw_schema: EHSRAGAnswerSchema
                if hasattr(res, "parsed") and isinstance(
                    res.parsed, EHSRAGAnswerSchema
                ):
                    raw_schema = res.parsed
                elif hasattr(res, "text") and res.text:
                    raw_schema = EHSRAGAnswerSchema.model_validate_json(
                        res.text
                    )
                else:
                    raw_schema = EHSRAGAnswerSchema(
                        answer="Unable to parse structured response from VLM."
                    )

                # Normalize citation URLs
                normalized_citations: list[EHSCitation] = []
                for cit in raw_schema.citations:
                    c_vid = cit.video_id or hits[0].get("video_id", "")
                    sb_url = cit.storyboard_url.lstrip("/")
                    if sb_url and not sb_url.startswith("media/"):
                        cit.storyboard_url = f"/media/{c_vid}/{sb_url}"
                    elif not sb_url and hits[0].get("storyboard_url"):
                        cit.storyboard_url = hits[0]["storyboard_url"]

                    aud_url = cit.audio_url.lstrip("/")
                    if aud_url and not aud_url.startswith("media/"):
                        cit.audio_url = f"/media/{c_vid}/{aud_url}"

                    normalized_citations.append(cit)

                usage = getattr(res, "usage_metadata", None)
                p_tokens = getattr(usage, "prompt_token_count", 0) or 0
                o_tokens = getattr(usage, "candidates_token_count", 0) or 0

                return EHSAskResponse(
                    question=request.question,
                    answer=raw_schema.answer,
                    confidence=raw_schema.confidence,
                    ncr_escalation_recommended=raw_schema.ncr_escalation_recommended,
                    recommended_corrective_actions=raw_schema.recommended_corrective_actions,
                    citations=normalized_citations,
                    retrieved_episodes=hits,
                    prompt_tokens=p_tokens,
                    output_tokens=o_tokens,
                    latency_ms=round((time.time() - t0) * 1000, 2),
                )
            except Exception:
                # Fall back gracefully to deterministic offline synthesis
                pass

        # 5. Deterministic Offline Fallback Synthesis
        top_hit = hits[0]
        vid = top_hit.get("video_id", "unknown")
        zone = top_hit.get("site_zone", "general_site_floor")
        status = top_hit.get("episode_compliance_status", "compliant")
        score = float(top_hit.get("safety_score_pct", 100.0))
        title = top_hit.get("episode_title", "Site Activity")
        summary = top_hit.get("work_summary", "")
        sev = str(top_hit.get("overall_severity", "low")).lower()
        viols = top_hit.get("safety_violations", [])
        ppes = top_hit.get("ppe_observations", [])

        ncr_recommended = (
            status == "non_compliant_ncr"
            or sev in ("high", "critical")
            or len(viols) > 0
        )

        corrective_actions: list[str] = []
        citations: list[EHSCitation] = []

        if viols:
            for v in viols:
                act = v.get("corrective_action_required", "")
                if act and act not in corrective_actions:
                    corrective_actions.append(act)
                citations.append(
                    EHSCitation(
                        video_id=vid,
                        episode_index=int(top_hit.get("episode_index", 0)),
                        site_zone=zone,
                        start_time_sec=float(top_hit.get("start_time_sec", 0.0)),
                        end_time_sec=float(top_hit.get("end_time_sec", 0.0)),
                        tile_number=v.get("tile_number", 1),
                        timestamp_sec=float(
                            v.get(
                                "timestamp_sec",
                                top_hit.get("start_time_sec", 0.0),
                            )
                        ),
                        grid_cells=v.get("grid_cells", []),
                        violation_type=v.get("violation_type", ""),
                        severity=v.get("severity", "medium"),
                        evidence=v.get("description", "Safety violation observed."),
                        corrective_action=act,
                        storyboard_url=top_hit.get("storyboard_url", ""),
                        audio_url=top_hit.get("audio_url", ""),
                    )
                )
        elif ppes:
            p = ppes[0]
            citations.append(
                EHSCitation(
                    video_id=vid,
                    episode_index=int(top_hit.get("episode_index", 0)),
                    site_zone=zone,
                    start_time_sec=float(top_hit.get("start_time_sec", 0.0)),
                    end_time_sec=float(top_hit.get("end_time_sec", 0.0)),
                    tile_number=p.get("tile_number", 1),
                    timestamp_sec=float(
                        p.get(
                            "absolute_time_sec",
                            top_hit.get("start_time_sec", 0.0),
                        )
                    ),
                    grid_cells=p.get("grid_cells", []),
                    violation_type="",
                    severity="low",
                    evidence=f"Worker activity: {p.get('activity_description', '')}",
                    corrective_action="",
                    storyboard_url=top_hit.get("storyboard_url", ""),
                    audio_url=top_hit.get("audio_url", ""),
                )
            )

        ans = (
            f"Based on retrieved site episode '{title}' ({vid}, Zone: {zone}): "
            f"{summary} Compliance status is '{status}' with an overall safety score of {score:.1f}% "
            f"(Severity: {sev.upper()})."
        )
        if viols:
            ans += f" Detected {len(viols)} safety violation(s) requiring immediate intervention: {viols[0].get('description', '')}"

        return EHSAskResponse(
            question=request.question,
            answer=ans,
            confidence="high" if top_hit.get("score", 0.0) > 0.05 else "medium",
            ncr_escalation_recommended=ncr_recommended,
            recommended_corrective_actions=corrective_actions,
            citations=citations,
            retrieved_episodes=hits,
            prompt_tokens=0,
            output_tokens=0,
            latency_ms=round((time.time() - t0) * 1000, 2),
        )


def generate_official_ncr_dossier(
    receipt: SiteComplianceReceipt,
) -> OfficialNCRDossier:
    """Compiles an enterprise Non-Conformance Report (NCR) dossier from a SiteComplianceReceipt."""
    zone_preset = SITE_ZONE_PRESETS.get(
        receipt.site_zone, SITE_ZONE_PRESETS["general_site_floor"]
    )
    zone_label = str(zone_preset.get("label", receipt.site_zone))
    required_ppe = list(zone_preset.get("required_ppe", []))

    if receipt.ncr_triggered:
        ncr_status = "NON_CONFORMANCE_NCR_ISSUED"
    elif receipt.total_violations_count > 0:
        ncr_status = "OBSERVATION_ISSUED"
    else:
        ncr_status = "COMPLIANT_PASS"

    violation_records: list[dict[str, Any]] = []
    episode_summaries: list[dict[str, Any]] = []

    for ep in receipt.episodes:
        sb_url = f"/media/{receipt.video_id}/{ep.storyboard_rel_path.lstrip('/')}".replace(
            "//", "/"
        )
        audio_url = (
            f"/media/{receipt.video_id}/{ep.audio_rel_path.lstrip('/')}".replace(
                "//", "/"
            )
            if ep.audio_rel_path
            else ""
        )

        for v in ep.safety_violations:
            violation_records.append(
                {
                    "episode_index": ep.episode_index,
                    "storyboard_url": sb_url,
                    "audio_url": audio_url,
                    "tile_number": v.tile_number,
                    "timestamp_sec": v.timestamp_sec,
                    "grid_cells": v.grid_cells,
                    "violation_type": v.violation_type,
                    "severity": v.severity,
                    "description": v.description,
                    "corrective_action_required": v.corrective_action_required,
                }
            )

        episode_summaries.append(
            {
                "episode_index": ep.episode_index,
                "start_time_sec": ep.start_time_sec,
                "end_time_sec": ep.end_time_sec,
                "duration_sec": ep.duration_sec,
                "episode_title": ep.episode_title,
                "work_category": ep.work_category,
                "work_summary": ep.work_summary,
                "compliance_status": ep.episode_compliance_status,
                "severity": ep.overall_severity,
                "safety_score_pct": ep.safety_score_pct,
                "workers_count": ep.workers_detected_count,
                "machinery": ep.machinery_detected,
                "violations_count": len(ep.safety_violations),
                "storyboard_url": sb_url,
            }
        )

    exec_summary = (
        f"Official EHS Site Compliance Audit for video {receipt.video_filename} ({receipt.video_id}) "
        f"in monitored zone '{zone_label}'. Safety compliance status evaluated as {ncr_status} "
        f"with overall site safety score {receipt.overall_site_safety_score_pct:.1f}%. "
        f"Inspected {receipt.duration_sec:.1f}s of footage: {receipt.active_work_duration_sec:.1f}s active work "
        f"({receipt.site_utilization_pct:.1f}% utilization) and {receipt.idle_downtime_sec:.1f}s idle downtime. "
        f"Video sieve compression achieved {receipt.tokens_saved_pct:.1f}% token savings ({receipt.storyboards_emitted} "
        f"storyboards emitted, {receipt.disturbances_suppressed} camera disturbances suppressed). "
        f"Identified {receipt.total_violations_count} total safety violation(s)."
    )

    clean_vid_id = receipt.video_id.replace("vid_", "").upper()
    return OfficialNCRDossier(
        ncr_reference_id=f"NCR-{clean_vid_id}",
        video_id=receipt.video_id,
        video_filename=receipt.video_filename,
        site_zone=receipt.site_zone,
        site_zone_label=zone_label,
        required_ppe=required_ppe,
        ncr_status=ncr_status,
        overall_site_safety_score_pct=receipt.overall_site_safety_score_pct,
        total_duration_sec=receipt.duration_sec,
        active_work_duration_sec=receipt.active_work_duration_sec,
        idle_downtime_sec=receipt.idle_downtime_sec,
        site_utilization_pct=receipt.site_utilization_pct,
        storyboards_emitted=receipt.storyboards_emitted,
        disturbances_suppressed=receipt.disturbances_suppressed,
        tokens_saved_pct=receipt.tokens_saved_pct,
        total_violations_count=receipt.total_violations_count,
        executive_summary=exec_summary,
        violation_records=violation_records,
        episode_summaries=episode_summaries,
    )


def render_ncr_html(dossier: OfficialNCRDossier) -> str:
    """Renders a complete, printable HTML5 Non-Conformance Report (NCR) Dossier."""
    status_color = (
        "#dc2626"
        if dossier.ncr_status == "NON_CONFORMANCE_NCR_ISSUED"
        else "#d97706"
        if dossier.ncr_status == "OBSERVATION_ISSUED"
        else "#16a34a"
    )

    # Violations rows
    violation_rows = []
    if dossier.violation_records:
        for idx, v in enumerate(dossier.violation_records, 1):
            sev_badge = (
                '<span class="badge badge-critical">CRITICAL</span>'
                if v["severity"] == "critical"
                else '<span class="badge badge-high">HIGH</span>'
                if v["severity"] == "high"
                else '<span class="badge badge-med">MEDIUM</span>'
            )
            cells_str = ", ".join(v.get("grid_cells", [])) or "N/A"
            violation_rows.append(
                f"""<tr>
                <td><strong>#{idx}</strong></td>
                <td>{v['violation_type'].replace('_', ' ').title()}<br>{sev_badge}</td>
                <td>Episode #{v['episode_index']}<br>Tile #{v.get('tile_number', 1)} (+{v.get('timestamp_sec', 0.0):.2f}s)</td>
                <td><code>{cells_str}</code></td>
                <td>{v['description']}</td>
                <td><strong>{v['corrective_action_required']}</strong></td>
            </tr>"""
            )
    else:
        violation_rows.append(
            """<tr><td colspan="6" style="text-align: center; color: #16a34a; padding: 16px;">
            <strong>No Non-Conformance or Safety Violations Recorded. Full EHS Compliance Verified.</strong>
        </td></tr>"""
        )
    violations_tbody = "\n".join(violation_rows)

    # Episode summaries rows
    episode_rows = []
    for ep in dossier.episode_summaries:
        badge_cls = (
            "badge-critical"
            if ep["compliance_status"] == "non_compliant_ncr"
            else "badge-compliant"
        )
        episode_rows.append(
            f"""<tr>
            <td>Episode #{ep['episode_index']}</td>
            <td>{ep['start_time_sec']:.1f}s - {ep['end_time_sec']:.1f}s ({ep['duration_sec']:.1f}s)</td>
            <td><strong>{ep['episode_title']}</strong><br><small>{ep['work_summary']}</small></td>
            <td><span class="badge {badge_cls}">{ep['compliance_status'].upper()}</span></td>
            <td>{ep['safety_score_pct']:.0f}%</td>
            <td><a href="{ep['storyboard_url']}" target="_blank" class="link-btn">View Storyboard</a></td>
        </tr>"""
        )
    episodes_tbody = "\n".join(episode_rows)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>EHS Non-Conformance Report // {dossier.ncr_reference_id}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            color: #1e293b;
            background-color: #f8fafc;
            margin: 0;
            padding: 32px 24px;
            line-height: 1.5;
        }}
        .container {{
            max-width: 1050px;
            margin: 0 auto;
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            padding: 40px;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            border-bottom: 3px solid #0f172a;
            padding-bottom: 20px;
            margin-bottom: 24px;
        }}
        .title-block h1 {{
            margin: 0 0 6px 0;
            font-size: 26px;
            color: #0f172a;
            letter-spacing: -0.5px;
        }}
        .title-block p {{
            margin: 0;
            color: #64748b;
            font-size: 14px;
        }}
        .status-pill {{
            background: {status_color};
            color: #ffffff;
            font-weight: 700;
            font-size: 13px;
            padding: 8px 16px;
            border-radius: 6px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .scorecard-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
            margin-bottom: 30px;
        }}
        .card {{
            background: #f1f5f9;
            border: 1px solid #cbd5e1;
            border-radius: 6px;
            padding: 14px 18px;
        }}
        .card-label {{
            font-size: 12px;
            color: #64748b;
            text-transform: uppercase;
            font-weight: 600;
            margin-bottom: 4px;
        }}
        .card-val {{
            font-size: 22px;
            font-weight: 700;
            color: #0f172a;
        }}
        .card-sub {{
            font-size: 11px;
            color: #475569;
            margin-top: 4px;
        }}
        .section-title {{
            font-size: 17px;
            font-weight: 700;
            color: #0f172a;
            border-left: 4px solid #2563eb;
            padding-left: 10px;
            margin: 28px 0 14px 0;
        }}
        .exec-box {{
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            padding: 16px;
            font-size: 14px;
            color: #334155;
            margin-bottom: 24px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            margin-bottom: 24px;
        }}
        th, td {{
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
            vertical-align: top;
        }}
        th {{
            background: #f8fafc;
            color: #475569;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 11px;
            letter-spacing: 0.5px;
        }}
        .badge {{
            display: inline-block;
            font-size: 10px;
            font-weight: 700;
            padding: 2px 6px;
            border-radius: 4px;
            text-transform: uppercase;
        }}
        .badge-critical {{ background: #fef2f2; color: #dc2626; border: 1px solid #fecaca; }}
        .badge-high {{ background: #fff7ed; color: #ea580c; border: 1px solid #ffedd5; }}
        .badge-med {{ background: #fefce8; color: #ca8a04; border: 1px solid #fef08a; }}
        .badge-compliant {{ background: #f0fdf4; color: #16a34a; border: 1px solid #bbf7d0; }}
        .link-btn {{
            color: #2563eb;
            text-decoration: none;
            font-weight: 600;
        }}
        .link-btn:hover {{ text-decoration: underline; }}
        .signoff-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 40px;
            margin-top: 50px;
            border-top: 1px dashed #cbd5e1;
            padding-top: 30px;
        }}
        .sign-line {{
            border-bottom: 1px solid #0f172a;
            height: 40px;
            margin-bottom: 6px;
        }}
        .sign-label {{
            font-size: 12px;
            color: #64748b;
            text-transform: uppercase;
        }}
        @media print {{
            body {{ background: #ffffff; padding: 0; }}
            .container {{ border: none; box-shadow: none; padding: 0; }}
            .link-btn {{ display: none; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="title-block">
                <h1>Non-Conformance Report & EHS Telemetry Dossier</h1>
                <p>Ref ID: <strong>{dossier.ncr_reference_id}</strong> | Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}</p>
            </div>
            <div class="status-pill">{dossier.ncr_status.replace('_', ' ')}</div>
        </div>

        <div class="scorecard-grid">
            <div class="card">
                <div class="card-label">Safety Compliance</div>
                <div class="card-val" style="color: {status_color};">{dossier.overall_site_safety_score_pct:.1f}%</div>
                <div class="card-sub">{dossier.total_violations_count} violation(s) logged</div>
            </div>
            <div class="card">
                <div class="card-label">Site Utilization</div>
                <div class="card-val">{dossier.site_utilization_pct:.1f}%</div>
                <div class="card-sub">{dossier.active_work_duration_sec:.1f}s active / {dossier.idle_downtime_sec:.1f}s idle</div>
            </div>
            <div class="card">
                <div class="card-label">Monitored Zone</div>
                <div class="card-val" style="font-size: 16px;">{dossier.site_zone_label}</div>
                <div class="card-sub">Req PPE: {', '.join(dossier.required_ppe)}</div>
            </div>
            <div class="card">
                <div class="card-label">Token Savings</div>
                <div class="card-val">{dossier.tokens_saved_pct:.1f}%</div>
                <div class="card-sub">{dossier.disturbances_suppressed} disturbance(s) filtered</div>
            </div>
        </div>

        <div class="section-title">Executive Compliance Summary</div>
        <div class="exec-box">
            {dossier.executive_summary}
        </div>

        <div class="section-title">Itemized Non-Conformance & Safety Violations</div>
        <table>
            <thead>
                <tr>
                    <th style="width: 50px;">Item</th>
                    <th style="width: 180px;">Hazard / Severity</th>
                    <th style="width: 140px;">Temporal Grounding</th>
                    <th style="width: 90px;">Grid</th>
                    <th>Observed Evidence</th>
                    <th>Mandatory Corrective Action</th>
                </tr>
            </thead>
            <tbody>
                {violations_tbody}
            </tbody>
        </table>

        <div class="section-title">Audited Work Episode Summaries</div>
        <table>
            <thead>
                <tr>
                    <th style="width: 80px;">Episode</th>
                    <th style="width: 130px;">Time Bounds</th>
                    <th>Title & Activity Summary</th>
                    <th style="width: 130px;">Status</th>
                    <th style="width: 80px;">Score</th>
                    <th style="width: 120px;">Proof</th>
                </tr>
            </thead>
            <tbody>
                {episodes_tbody}
            </tbody>
        </table>

        <div class="signoff-grid">
            <div>
                <div class="sign-line"></div>
                <div class="sign-label">Senior EHS Site Officer Signature & Date</div>
            </div>
            <div>
                <div class="sign-line"></div>
                <div class="sign-label">Project Director / Contractor Superintendent Signature & Date</div>
            </div>
        </div>
    </div>
</body>
</html>"""
    return html
