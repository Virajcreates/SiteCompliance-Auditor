/**
 * SieveAudit Tactical EHS & Site Compliance Command Center
 * Tactical UI Controller (app.js)
 * Clean JS - Zero Emojis - Full Lucide SVG Icons - Strict DOM ID Matching
 */

// Application State
const state = {
  videos: [],
  currentVideoId: null,
  currentReceipt: null,
  currentEpisodeIndex: 0,
  zonePresets: {}
};

// Lucide Icon Strings (Reusable)
const ICONS = {
  check: '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>',
  cross: '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>',
  trash: '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/></svg>',
  alert: '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
  play: '<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="6 3 20 12 6 21 6 3"/></svg>',
  shield: '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>',
  chevronRight: '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"/></svg>'
};

// Utilities
function formatTime(sec) {
  if (sec == null || isNaN(sec)) return "00:00.00";
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  const ms = Math.floor((sec % 1) * 100);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}.${String(ms).padStart(2, "0")}`;
}

function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

let toastTimer = null;
function showToast(msg, durationMs = 2200) {
  const toast = document.getElementById("seek-toast-banner");
  if (!toast) return;
  toast.textContent = msg;
  toast.classList.add("visible");
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    toast.classList.remove("visible");
  }, durationMs);
}

function seekVideo(timestampSec) {
  const player = document.getElementById("main-video-player");
  if (!player) return;
  const target = Math.max(0, timestampSec || 0);
  player.currentTime = target;
  player.play().catch(() => {});
  showToast(`SEEK: ${formatTime(target)}`);
}

// Global Stats & Telemetry
async function fetchStats() {
  try {
    const res = await fetch("/api/stats");
    if (!res.ok) return;
    const stats = await res.json();

    const elTotalVideos = document.getElementById("stat-total-videos");
    const elAvgUtil = document.getElementById("stat-avg-utilization");
    const elAvgSafety = document.getElementById("stat-avg-safety-score");
    const elTotalViolations = document.getElementById("stat-total-violations");
    const elTokensSaved = document.getElementById("stat-tokens-saved");
    const elKvVram = document.getElementById("stat-kv-vram");

    const avgUtil = stats.average_utilization_pct ?? stats.avg_site_utilization_pct ?? 0;
    const avgScore = stats.average_safety_score_pct ?? stats.avg_safety_score ?? 100;
    const totalViol = stats.total_violations_count ?? stats.total_violations ?? 0;
    const tokenSav = stats.overall_tokens_saved_pct ?? stats.overall_token_reduction_pct ?? 0;
    const kvMb = stats.total_qwen7b_kv_cache_mb ?? stats.total_kv_vram_saved_mb ?? 0;

    if (elTotalVideos) elTotalVideos.textContent = stats.total_videos ?? 0;
    if (elAvgUtil) elAvgUtil.textContent = `${avgUtil.toFixed(1)}%`;
    if (elAvgSafety) elAvgSafety.textContent = `${avgScore.toFixed(1)}%`;
    if (elTotalViolations) elTotalViolations.textContent = totalViol;
    if (elTokensSaved) elTokensSaved.textContent = `${tokenSav.toFixed(1)}%`;
    if (elKvVram) elKvVram.textContent = `${kvMb.toFixed(1)} MB`;
  } catch (err) {
    console.error("fetchStats error:", err);
  }
}

// Zone Presets
async function fetchZones() {
  try {
    const res = await fetch("/api/zones");
    if (!res.ok) return;
    const data = await res.json();
    state.zonePresets = data.zones || {};
    renderZonePills();
  } catch (err) {
    console.error("fetchZones error:", err);
  }
}

function renderZonePills() {
  const select = document.getElementById("select-site-zone");
  const container = document.getElementById("zone-ppe-pills");
  if (!select || !container) return;

  const zoneKey = select.value;
  const cfg = state.zonePresets[zoneKey];
  if (!cfg) {
    container.innerHTML = "";
    return;
  }

  const ppeHtml = (cfg.required_ppe || [])
    .map(p => `<span class="ppe-pill required">${ICONS.check} ${escapeHtml(p.replace(/_/g, " "))}</span>`)
    .join("");

  const hazardHtml = (cfg.primary_hazards || [])
    .map(h => `<span class="hazard-tag">${ICONS.alert} ${escapeHtml(h.replace(/_/g, " "))}</span>`)
    .join("");

  container.innerHTML = `
    <div style="font-size: 10px; color: var(--text-muted); margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.04em;">Mandatory Zone PPE</div>
    <div style="display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 6px;">${ppeHtml}</div>
    <div style="font-size: 10px; color: var(--text-muted); margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.04em;">Primary Hazard Vectors</div>
    <div style="display: flex; flex-wrap: wrap; gap: 4px;">${hazardHtml}</div>
  `;
}

// Video Corpus Management
async function loadVideos(preferredVideoId = null) {
  try {
    const res = await fetch("/api/videos");
    if (!res.ok) return;
    const data = await res.json();
    state.videos = Array.isArray(data) ? data : (data.videos || []);

    renderVideoList();
    populateVideoFilterScope();

    if (preferredVideoId) {
      await selectVideo(preferredVideoId);
    } else if (state.currentVideoId && state.videos.some(v => v.video_id === state.currentVideoId)) {
      await selectVideo(state.currentVideoId);
    } else if (state.videos.length > 0) {
      await selectVideo(state.videos[0].video_id);
    } else {
      resetMonitorView();
    }
  } catch (err) {
    console.error("loadVideos error:", err);
  }
}

function renderVideoList() {
  const container = document.getElementById("video-corpus-list");
  if (!container) return;

  if (state.videos.length === 0) {
    container.innerHTML = `
      <div style="padding: 24px 12px; text-align: center; color: var(--text-muted); font-size: 11px;">
        No site feeds ingested yet. Use the ingestion form above to audit CCTV footage.
      </div>
    `;
    return;
  }

  container.innerHTML = state.videos.map(v => {
    const isActive = v.video_id === state.currentVideoId;
    const hasViolations = (v.total_violations || 0) > 0;
    const badgeHtml = hasViolations
      ? `<span class="feed-badge ncr">${ICONS.alert} NCR (${v.total_violations})</span>`
      : `<span class="feed-badge compliant">${ICONS.check} COMPLIANT</span>`;

    const utilPct = (v.site_utilization_pct || 0).toFixed(1);
    const durationSec = ((v.duration_sec ?? v.total_duration_sec ?? 0) || 0).toFixed(1);

    return `
      <div class="video-feed-card ${isActive ? 'active' : ''}" data-vid="${escapeHtml(v.video_id)}">
        <div class="feed-card-header">
          <span class="feed-title" title="${escapeHtml(v.video_id)}">${escapeHtml(v.video_id)}</span>
          ${badgeHtml}
        </div>
        <div class="feed-meta-row">
          <span>${escapeHtml(v.site_zone || "general_site_floor")}</span>
          <span>${durationSec}s | ${(v.storyboards_emitted ?? v.episodes_count ?? (v.episodes ? v.episodes.length : 0)) || 0} eps</span>
        </div>
        <div class="feed-stats-bar">
          <span>Utilization: <strong style="color: var(--signal-emerald);">${utilPct}%</strong></span>
          <button class="btn-delete-feed" data-vid="${escapeHtml(v.video_id)}" title="Delete Feed">
            ${ICONS.trash}
          </button>
        </div>
      </div>
    `;
  }).join("");

  // Attach card click handlers
  container.querySelectorAll(".video-feed-card").forEach(card => {
    card.addEventListener("click", (e) => {
      if (e.target.closest(".btn-delete-feed")) return;
      const vid = card.dataset.vid;
      if (vid) selectVideo(vid);
    });
  });

  // Attach delete handlers
  container.querySelectorAll(".btn-delete-feed").forEach(btn => {
    btn.addEventListener("click", async (e) => {
      e.stopPropagation();
      const vid = btn.dataset.vid;
      if (!vid) return;
      if (!confirm(`Delete feed ${vid} and remove its indexed vectors?`)) return;
      try {
        const delRes = await fetch(`/api/videos/${encodeURIComponent(vid)}`, { method: "DELETE" });
        if (delRes.ok) {
          if (state.currentVideoId === vid) {
            state.currentVideoId = null;
            state.currentReceipt = null;
          }
          await fetchStats();
          await loadVideos();
        } else {
          alert(`Failed to delete feed ${vid}`);
        }
      } catch (err) {
        console.error("Delete error:", err);
      }
    });
  });
}

function populateVideoFilterScope() {
  const select = document.getElementById("filter-video-scope");
  if (!select) return;

  const currentVal = select.value;
  select.innerHTML = '<option value="">All Site Feeds</option>';
  state.videos.forEach(v => {
    const opt = document.createElement("option");
    opt.value = v.video_id;
    opt.textContent = v.video_id;
    select.appendChild(opt);
  });
  if (currentVal && state.videos.some(v => v.video_id === currentVal)) {
    select.value = currentVal;
  }
}

function resetMonitorView() {
  const elTitle = document.getElementById("active-video-title");
  const elZone = document.getElementById("active-zone-badge");
  const elComp = document.getElementById("active-compliance-badge");
  const elNcrBtn = document.getElementById("btn-export-ncr");
  const player = document.getElementById("main-video-player");
  const track = document.getElementById("episode-timeline-track");

  if (elTitle) elTitle.textContent = "No feed selected";
  if (elZone) elZone.style.display = "none";
  if (elComp) elComp.style.display = "none";
  if (elNcrBtn) elNcrBtn.style.display = "none";
  if (player) {
    player.src = "";
    player.load();
  }
  if (track) track.innerHTML = "";

  resetEpisodeDossier();
}

function resetEpisodeDossier() {
  const epTitle = document.getElementById("ep-title");
  const epComp = document.getElementById("ep-compliance-badge");
  const epGrid = document.getElementById("ep-grid-badge");
  const epTime = document.getElementById("ep-time-range");
  const epScore = document.getElementById("ep-safety-score");
  const epImg = document.getElementById("ep-storyboard-img");
  const epAudio = document.getElementById("ep-audio-player");
  const epSummary = document.getElementById("ep-work-summary");
  const epMachinery = document.getElementById("ep-machinery-list");
  const epAcoustic = document.getElementById("ep-audio-cues");
  const epViolations = document.getElementById("ep-violations-list");
  const epPpe = document.getElementById("ep-ppe-matrix");

  if (epTitle) epTitle.textContent = "Select an episode";
  if (epComp) epComp.style.display = "none";
  if (epGrid) epGrid.style.display = "none";
  if (epTime) epTime.textContent = "";
  if (epScore) epScore.textContent = "";
  if (epImg) epImg.src = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='280' height='160'%3E%3Crect width='100%25' height='100%25' fill='%2314171D'/%3E%3Ctext x='50%25' y='50%25' dominant-baseline='middle' text-anchor='middle' fill='%236B7280' font-size='11'%3ENo Storyboard%3C/text%3E%3C/svg%3E";
  if (epAudio) {
    epAudio.src = "";
    epAudio.load();
  }
  if (epSummary) epSummary.textContent = "Select an episode to inspect work telemetry.";
  if (epMachinery) epMachinery.textContent = "None";
  if (epAcoustic) epAcoustic.textContent = "None";
  if (epViolations) epViolations.innerHTML = '<div style="font-size: 11px; color: var(--text-muted);">No violations in this episode.</div>';
  if (epPpe) epPpe.innerHTML = '<div style="font-size: 11px; color: var(--text-muted);">No workers detected.</div>';

  const telTokens = document.getElementById("telemetry-tokens");
  const telQwen = document.getElementById("telemetry-qwen");
  const telKv = document.getElementById("telemetry-kv");
  if (telTokens) telTokens.textContent = "--";
  if (telQwen) telQwen.textContent = "--";
  if (telKv) telKv.textContent = "--";
}

// Select & Load Video
async function selectVideo(videoId) {
  try {
    const res = await fetch(`/api/videos/${encodeURIComponent(videoId)}`);
    if (!res.ok) return;
    const receipt = await res.json();

    state.currentVideoId = videoId;
    state.currentReceipt = receipt;
    state.currentEpisodeIndex = 0;

    // Highlight active card
    document.querySelectorAll(".video-feed-card").forEach(card => {
      card.classList.toggle("active", card.dataset.vid === videoId);
    });

    // Auto-scope RAG filter
    const filterScope = document.getElementById("filter-video-scope");
    if (filterScope) {
      filterScope.value = videoId;
    }

    // Monitor Header
    const elTitle = document.getElementById("active-video-title");
    const elZone = document.getElementById("active-zone-badge");
    const elComp = document.getElementById("active-compliance-badge");
    const elNcrBtn = document.getElementById("btn-export-ncr");

    if (elTitle) elTitle.textContent = receipt.video_id;
    if (elZone) {
      elZone.textContent = receipt.site_zone || "general_site_floor";
      elZone.style.display = "inline-block";
    }
    // Total Violations
    const totalViolations = receipt.total_violations_count != null ? receipt.total_violations_count : (receipt.total_violations || 0);
    if (elComp) {
      elComp.style.display = "inline-block";
      if (totalViolations > 0) {
        elComp.className = "feed-badge ncr";
        elComp.innerHTML = `${ICONS.alert} NCR ESCALATED (${totalViolations} Violations)`;
      } else {
        elComp.className = "feed-badge compliant";
        elComp.innerHTML = `${ICONS.check} EHS COMPLIANT`;
      }
    }

    if (elNcrBtn) {
      elNcrBtn.style.display = "inline-flex";
      elNcrBtn.onclick = () => {
        window.open(receipt.ncr_html_url || `/api/videos/${encodeURIComponent(videoId)}/ncr/html`, "_blank");
      };
    }

    // Video Player
    const player = document.getElementById("main-video-player");
    if (player && receipt.video_url) {
      player.src = receipt.video_url;
      player.load();
    }

    // Duty-Cycle Metrics
    const duty = receipt.duty_cycle_metrics || {};
    const elUtil = document.getElementById("duty-utilization-pct");
    const elActive = document.getElementById("duty-active-sec");
    const elIdle = document.getElementById("duty-idle-sec");
    const elDist = document.getElementById("duty-disturbances-count");
    const elBar = document.getElementById("duty-cycle-bar");

    const utilPct = receipt.site_utilization_pct != null ? receipt.site_utilization_pct : (duty.site_utilization_pct || 0);
    const activeSec = receipt.active_work_duration_sec != null ? receipt.active_work_duration_sec : (duty.active_work_seconds || 0);
    const idleSec = receipt.idle_downtime_sec != null ? receipt.idle_downtime_sec : (duty.idle_downtime_seconds || 0);
    const distCount = receipt.disturbances_suppressed != null ? receipt.disturbances_suppressed : (duty.camera_disturbances_suppressed ?? 0);

    if (elUtil) elUtil.textContent = `${utilPct.toFixed(1)}%`;
    if (elActive) elActive.textContent = `${activeSec.toFixed(1)}s`;
    if (elIdle) elIdle.textContent = `${idleSec.toFixed(1)}s`;
    if (elDist) elDist.textContent = distCount;
    if (elBar) elBar.style.width = `${Math.min(100, Math.max(0, utilPct))}%`;

    // Render Timeline Track
    renderTimelineTrack();

    // Select first episode
    if (receipt.episodes && receipt.episodes.length > 0) {
      selectEpisode(0);
    } else {
      resetEpisodeDossier();
    }
  } catch (err) {
    console.error("selectVideo error:", err);
  }
}

function renderTimelineTrack() {
  const track = document.getElementById("episode-timeline-track");
  if (!track || !state.currentReceipt) return;

  const totalDur = Math.max(
    1,
    state.currentReceipt.duration_sec || state.currentReceipt.total_duration_sec || 1
  );
  const episodes = state.currentReceipt.episodes || [];

  if (episodes.length === 0) {
    track.innerHTML = '<div style="color: var(--text-muted); font-size: 11px; padding: 6px;">No localized work episodes detected (entire feed was idle or disturbed).</div>';
    return;
  }

  track.innerHTML = episodes.map((ep, idx) => {
    const leftPct = (ep.start_time_sec / totalDur) * 100;
    const widthPct = Math.max(1.8, (ep.duration_sec / totalDur) * 100);
    const violations = ep.safety_violations || ep.audit_result?.violations || [];
    const hasViolations = violations.length > 0;
    const blockClass = hasViolations ? "has-violations" : "compliant";

    return `
      <div class="timeline-block ${blockClass}"
           style="left: ${leftPct.toFixed(2)}%; width: ${widthPct.toFixed(2)}%;"
           data-index="${idx}"
           title="Episode #${ep.episode_index + 1}: ${formatTime(ep.start_time_sec)} - ${formatTime(ep.end_time_sec)} (${hasViolations ? 'NCR Violations' : 'Compliant'})">
      </div>
    `;
  }).join("");

  track.querySelectorAll(".timeline-block").forEach(block => {
    block.addEventListener("click", () => {
      const idx = parseInt(block.dataset.index, 10);
      selectEpisode(idx);
      if (state.currentReceipt?.episodes?.[idx]) {
        seekVideo(state.currentReceipt.episodes[idx].start_time_sec);
      }
    });
  });
}

function selectEpisode(index) {
  if (!state.currentReceipt?.episodes || !state.currentReceipt.episodes[index]) {
    resetEpisodeDossier();
    return;
  }

  state.currentEpisodeIndex = index;
  const ep = state.currentReceipt.episodes[index];
  const audit = ep.audit_result || {};

  // Highlight timeline block
  document.querySelectorAll(".timeline-block").forEach((blk, idx) => {
    blk.classList.toggle("active", idx === index);
  });

  // Dossier Header
  const epTitle = document.getElementById("ep-title");
  const epComp = document.getElementById("ep-compliance-badge");
  const epGrid = document.getElementById("ep-grid-badge");
  const epTime = document.getElementById("ep-time-range");
  const epScore = document.getElementById("ep-safety-score");

  const titleText = ep.episode_title || `Episode #${ep.episode_index + 1}: ${ep.work_category || "Work Session"}`;
  if (epTitle) epTitle.textContent = titleText;
  if (epTime) epTime.textContent = `${formatTime(ep.start_time_sec)} - ${formatTime(ep.end_time_sec)} (${ep.duration_sec.toFixed(1)}s)`;

  const violations = ep.safety_violations || audit.violations || [];
  if (epComp) {
    epComp.style.display = "inline-block";
    if (violations.length > 0) {
      epComp.className = "feed-badge ncr";
      epComp.innerHTML = `${ICONS.alert} NCR VIOLATIONS (${violations.length})`;
    } else {
      epComp.className = "feed-badge compliant";
      epComp.innerHTML = `${ICONS.check} EHS COMPLIANT`;
    }
  }

  if (epGrid) {
    epGrid.style.display = "inline-block";
    const g = ep.grid_used;
    epGrid.textContent = g ? `Grid: ${Array.isArray(g) ? g.join('x') : g}` : "Adaptive Grid";
  }

  if (epScore) {
    let scoreVal = 100;
    if (ep.safety_score_pct != null) {
      scoreVal = Math.round(ep.safety_score_pct);
    } else if (audit.safety_score != null) {
      scoreVal = Math.round(audit.safety_score * 100);
    }
    epScore.textContent = `${scoreVal}% Safety`;
    if (scoreVal >= 80) epScore.style.color = "var(--signal-emerald)";
    else if (scoreVal >= 50) epScore.style.color = "var(--signal-amber)";
    else epScore.style.color = "var(--signal-crimson)";
  }

  // Evidence
  const epImg = document.getElementById("ep-storyboard-img");
  const epAudio = document.getElementById("ep-audio-player");
  if (epImg) {
    epImg.src = ep.storyboard_url || "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='280' height='160'%3E%3Crect width='100%25' height='100%25' fill='%2314171D'/%3E%3Ctext x='50%25' y='50%25' dominant-baseline='middle' text-anchor='middle' fill='%236B7280' font-size='11'%3ENo Storyboard%3C/text%3E%3C/svg%3E";
  }
  if (epAudio) {
    if (ep.audio_url) {
      epAudio.src = ep.audio_url;
      epAudio.load();
    } else {
      epAudio.removeAttribute("src");
      epAudio.load();
    }
  }

  // Telemetry Strip
  const telTokens = document.getElementById("telemetry-tokens");
  const telQwen = document.getElementById("telemetry-qwen");
  const telKv = document.getElementById("telemetry-kv");
  const telem = ep.token_telemetry || {};

  const savedPct = ep.tokens_saved_pct ?? telem.tokens_saved_pct;
  const qwenTok = ep.qwen_vl_patch_tokens ?? telem.qwen_vl_tokens;
  const kvVram = ep.qwen7b_kv_cache_mb ?? telem.kv_cache_vram_saved_mb;

  if (telTokens) telTokens.textContent = savedPct != null ? `${savedPct.toFixed(0)}% saved` : "--";
  if (telQwen) telQwen.textContent = qwenTok != null ? `${qwenTok} tok` : "--";
  if (telKv) telKv.textContent = kvVram != null ? `${kvVram.toFixed(1)} MB` : "--";

  // Summary & Equipment
  const epSummary = document.getElementById("ep-work-summary");
  const epMachinery = document.getElementById("ep-machinery-list");
  const epAcoustic = document.getElementById("ep-audio-cues");

  const summaryText = ep.work_summary || audit.work_summary || "No automated summary provided.";
  if (epSummary) epSummary.textContent = summaryText;

  const machList = ep.machinery_detected || audit.machinery_detected || [];
  if (epMachinery) {
    epMachinery.textContent = machList.length ? machList.join(", ") : "None detected";
  }

  const audioCuesText = ep.audio_safety_cues || (audit.acoustic_cues && audit.acoustic_cues.length ? audit.acoustic_cues.join(", ") : "None detected");
  if (epAcoustic) epAcoustic.textContent = audioCuesText;

  // Render Violations
  renderViolations(violations);

  // Render PPE Matrix
  const ppeList = ep.ppe_observations || audit.worker_ppe_observations || [];
  renderPpeMatrix(ppeList);
}

function renderViolations(violations) {
  const container = document.getElementById("ep-violations-list");
  if (!container) return;

  if (!violations || violations.length === 0) {
    container.innerHTML = '<div style="font-size: 11px; color: var(--text-muted); padding: 6px 0;">No safety violations detected in this episode.</div>';
    return;
  }

  container.innerHTML = violations.map(v => {
    const sev = (v.severity || "low").toLowerCase();
    const cells = v.grid_cells ? (Array.isArray(v.grid_cells) ? v.grid_cells.join(",") : v.grid_cells) : v.grid_cell;
    const cellBadge = cells ? `<span class="badge-tag">Cell ${escapeHtml(cells)}</span>` : "";
    const tileBadge = v.tile_number != null ? `<span class="badge-tag">Tile #${v.tile_number}</span>` : "";
    const evidenceText = v.description || v.evidence || "";
    const actionText = v.corrective_action_required || v.corrective_action || "";

    return `
      <div class="violation-item" data-time="${v.timestamp_sec}">
        <div class="violation-header">
          <span class="severity-pill sev-${sev}">${escapeHtml(v.severity || "LOW")}</span>
          <span class="violation-type">${escapeHtml(v.violation_type || "Safety Violation")}</span>
          <span class="violation-time">${ICONS.play} ${formatTime(v.timestamp_sec)}</span>
          ${tileBadge}
          ${cellBadge}
        </div>
        <div class="violation-evidence">${escapeHtml(evidenceText)}</div>
        ${actionText ? `<div class="violation-corrective"><strong>Action:</strong> ${escapeHtml(actionText)}</div>` : ""}
      </div>
    `;
  }).join("");

  container.querySelectorAll(".violation-item").forEach(item => {
    item.addEventListener("click", () => {
      const t = parseFloat(item.dataset.time);
      seekVideo(t);
    });
  });
}

function renderPpeMatrix(observations) {
  const container = document.getElementById("ep-ppe-matrix");
  if (!container) return;

  if (!observations || observations.length === 0) {
    container.innerHTML = '<div style="font-size: 11px; color: var(--text-muted); padding: 6px 0;">No workers observed in this episode.</div>';
    return;
  }

  const ppeIcon = (val) => val
    ? `<span style="color: var(--signal-emerald);" title="Compliant">${ICONS.check}</span>`
    : `<span style="color: var(--signal-crimson);" title="Missing">${ICONS.cross}</span>`;

  container.innerHTML = observations.map(obs => {
    const cells = obs.grid_cells ? (Array.isArray(obs.grid_cells) ? obs.grid_cells.join(",") : obs.grid_cells) : obs.grid_cell;
    const cellBadge = cells ? `Cell ${escapeHtml(cells)}` : `Tile #${obs.tile_number ?? 1}`;
    const hardhat = obs.hardhat_worn ?? obs.hardhat_detected ?? true;
    const highVis = obs.high_vis_vest_worn ?? obs.high_vis_vest_detected ?? true;
    const harness = obs.safety_harness_worn ?? obs.harness_detected ?? false;
    const workerLbl = obs.worker_label || obs.worker_id || "Worker #1";

    const ppeChecklist = `
      <span class="ppe-check-item">Hardhat: ${ppeIcon(hardhat)}</span>
      <span class="ppe-check-item">High-Vis: ${ppeIcon(highVis)}</span>
      <span class="ppe-check-item">Harness: ${ppeIcon(harness)}</span>
    `;

    return `
      <div class="ppe-row" data-time="${obs.absolute_time_sec}">
        <div class="ppe-worker-id">
          <span>${escapeHtml(workerLbl)}</span>
          <span class="mono-data" style="color: var(--text-muted); font-size: 10px;">${cellBadge}</span>
          <span class="mono-data" style="color: var(--signal-cyan); font-size: 10px;">${ICONS.play} ${formatTime(obs.absolute_time_sec)}</span>
        </div>
        <div class="ppe-badges-cluster">${ppeChecklist}</div>
        <div style="font-size: 11px; color: var(--text-secondary); margin-top: 4px;">
          ${escapeHtml(obs.activity_description || "")}
        </div>
      </div>
    `;
  }).join("");

  container.querySelectorAll(".ppe-row").forEach(row => {
    row.addEventListener("click", () => {
      const t = parseFloat(row.dataset.time);
      seekVideo(t);
    });
  });
}

// Ingestion Form Controller
function setupIngestionForm() {
  const tabUpload = document.getElementById("tab-upload-file");
  const tabLocal = document.getElementById("tab-local-path");
  const panelUpload = document.getElementById("panel-upload-file");
  const panelLocal = document.getElementById("panel-local-path");
  const inputVideoFile = document.getElementById("input-video-file");
  const labelSelectedFile = document.getElementById("label-selected-file");
  const btnSubmit = document.getElementById("btn-ingest-submit");
  const statusMsg = document.getElementById("ingest-status-msg");
  const selectZone = document.getElementById("select-site-zone");
  const selectVlm = document.getElementById("select-target-vlm");
  const chkAudit = document.getElementById("chk-run-audit");
  const chkVectors = document.getElementById("chk-index-vectors");

  let activeMode = "upload"; // "upload" | "local"

  if (tabUpload && tabLocal) {
    tabUpload.addEventListener("click", () => {
      activeMode = "upload";
      tabUpload.classList.add("active");
      tabLocal.classList.remove("active");
      if (panelUpload) panelUpload.style.display = "block";
      if (panelLocal) panelLocal.style.display = "none";
    });

    tabLocal.addEventListener("click", () => {
      activeMode = "local";
      tabLocal.classList.add("active");
      tabUpload.classList.remove("active");
      if (panelUpload) panelUpload.style.display = "none";
      if (panelLocal) panelLocal.style.display = "block";
    });
  }

  if (inputVideoFile && labelSelectedFile) {
    inputVideoFile.addEventListener("change", () => {
      if (inputVideoFile.files && inputVideoFile.files[0]) {
        labelSelectedFile.textContent = inputVideoFile.files[0].name;
      } else {
        labelSelectedFile.textContent = "Choose Video File...";
      }
    });
  }

  if (selectZone) {
    selectZone.addEventListener("change", renderZonePills);
  }

  if (btnSubmit) {
    btnSubmit.addEventListener("click", async () => {
      if (statusMsg) {
        statusMsg.className = "";
        statusMsg.textContent = "";
      }

      const formData = new FormData();
      formData.append("site_zone", selectZone ? selectZone.value : "general_site_floor");
      formData.append("target_vlm", selectVlm ? selectVlm.value : "qwen-vl");
      formData.append("run_audit", chkAudit && chkAudit.checked ? "true" : "false");
      formData.append("index_vectors", chkVectors && chkVectors.checked ? "true" : "false");

      if (activeMode === "upload") {
        if (!inputVideoFile || !inputVideoFile.files || !inputVideoFile.files[0]) {
          if (statusMsg) {
            statusMsg.className = "error";
            statusMsg.textContent = "Please select a video file to upload.";
          }
          return;
        }
        formData.append("file", inputVideoFile.files[0]);
      } else {
        const inputLocal = document.getElementById("input-local-path");
        const pathVal = inputLocal ? inputLocal.value.trim() : "";
        if (!pathVal) {
          if (statusMsg) {
            statusMsg.className = "error";
            statusMsg.textContent = "Please provide an absolute path to a video file.";
          }
          return;
        }
        formData.append("local_video_path", pathVal);
      }

      btnSubmit.disabled = true;
      if (statusMsg) {
        statusMsg.className = "info";
        statusMsg.textContent = "Sieving video, cropping dynamic ROIs, and generating storyboards...";
      }

      try {
        const res = await fetch("/api/videos/ingest", {
          method: "POST",
          body: formData
        });

        const data = await res.json();
        if (!res.ok) {
          throw new Error((Array.isArray(data.detail) ? data.detail.map(e => e.msg || JSON.stringify(e)).join('; ') : (typeof data.detail === 'object' ? JSON.stringify(data.detail) : data.detail)) || "Ingestion failed");
        }

        if (statusMsg) {
          statusMsg.className = "success";
          statusMsg.textContent = `Audited ${data.video_id}: ${data.episodes_count} work episodes extracted (${data.duty_cycle_metrics?.site_utilization_pct?.toFixed(1) || 0}% active utilization).`;
        }

        await fetchStats();
        await loadVideos(data.video_id);
      } catch (err) {
        if (statusMsg) {
          statusMsg.className = "error";
          statusMsg.textContent = `Error: ${err.message}`;
        }
      } finally {
        btnSubmit.disabled = false;
      }
    });
  }
}

// EHS RAG Copilot & Vector Search Controller
function setupRAGController() {
  const inputQuery = document.getElementById("rag-query-input");
  const btnAsk = document.getElementById("btn-ask-rag");
  const btnSearch = document.getElementById("btn-vector-search");
  const resultsContainer = document.getElementById("rag-results-container");

  const filterScope = document.getElementById("filter-video-scope");
  const filterZone = document.getElementById("filter-site-zone");
  const filterType = document.getElementById("filter-violation-type");
  const filterSev = document.getElementById("filter-min-severity");
  const filterNcrOnly = document.getElementById("filter-ncr-only");
  const filterStoryboards = document.getElementById("filter-include-storyboards");

  // Quick Prompt Buttons
  document.querySelectorAll(".quick-prompt-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      if (inputQuery && btn.dataset.prompt) {
        inputQuery.value = btn.dataset.prompt;
        inputQuery.focus();
      }
    });
  });

  // Ask RAG Copilot
  if (btnAsk) {
    btnAsk.addEventListener("click", async () => {
      const q = inputQuery ? inputQuery.value.trim() : "";
      if (!q) {
        alert("Please enter an EHS query or click a quick prompt.");
        return;
      }

      if (!resultsContainer) return;
      resultsContainer.innerHTML = `
        <div style="text-align: center; padding: 24px; color: var(--text-muted); font-size: 11px;">
          Synthesizing EHS intelligence and cross-referencing site vector index...
        </div>
      `;

      btnAsk.disabled = true;

      const payload = {
        query: q,
        top_k: 5,
        video_id: filterScope && filterScope.value ? filterScope.value : null,
        site_zone: filterZone && filterZone.value ? filterZone.value : null,
        violation_type: filterType && filterType.value ? filterType.value : null,
        min_severity: filterSev && filterSev.value ? filterSev.value : null,
        ncr_only: filterNcrOnly ? filterNcrOnly.checked : false,
        include_storyboard_images: filterStoryboards ? filterStoryboards.checked : true
      };

      try {
        const res = await fetch("/api/ask", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });

        const data = await res.json();
        if (!res.ok) throw new Error((Array.isArray(data.detail) ? data.detail.map(e => e.msg || JSON.stringify(e)).join('; ') : (typeof data.detail === 'object' ? JSON.stringify(data.detail) : data.detail)) || "RAG synthesis failed");

        renderRAGResponse(data);
      } catch (err) {
        resultsContainer.innerHTML = `
          <div style="padding: 12px; background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); color: var(--signal-crimson); font-size: 11px; border-radius: var(--radius-sm);">
            ${escapeHtml(err.message)}
          </div>
        `;
      } finally {
        btnAsk.disabled = false;
      }
    });
  }

  // Search Vectors Directly
  if (btnSearch) {
    btnSearch.addEventListener("click", async () => {
      const q = inputQuery ? inputQuery.value.trim() : "";
      if (!q) {
        alert("Please enter a search query.");
        return;
      }

      if (!resultsContainer) return;
      resultsContainer.innerHTML = `
        <div style="text-align: center; padding: 24px; color: var(--text-muted); font-size: 11px;">
          Searching hybrid compliance vector index...
        </div>
      `;

      btnSearch.disabled = true;

      const payload = {
        query: q,
        top_k: 6,
        video_id: filterScope && filterScope.value ? filterScope.value : null,
        site_zone: filterZone && filterZone.value ? filterZone.value : null,
        violation_type: filterType && filterType.value ? filterType.value : null,
        min_severity: filterSev && filterSev.value ? filterSev.value : null,
        ncr_only: filterNcrOnly ? filterNcrOnly.checked : false
      };

      try {
        const res = await fetch("/api/search", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });

        const data = await res.json();
        if (!res.ok) throw new Error((Array.isArray(data.detail) ? data.detail.map(e => e.msg || JSON.stringify(e)).join('; ') : (typeof data.detail === 'object' ? JSON.stringify(data.detail) : data.detail)) || "Search failed");

        renderSearchResults(data);
      } catch (err) {
        resultsContainer.innerHTML = `
          <div style="padding: 12px; background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); color: var(--signal-crimson); font-size: 11px; border-radius: var(--radius-sm);">
            ${escapeHtml(err.message)}
          </div>
        `;
      } finally {
        btnSearch.disabled = false;
      }
    });
  }
}

function renderRAGResponse(data) {
  const container = document.getElementById("rag-results-container");
  if (!container) return;

  const ncrBadge = data.ncr_escalation_recommended
    ? `<span class="feed-badge ncr">${ICONS.alert} NCR ESCALATION RECOMMENDED</span>`
    : `<span class="feed-badge compliant">${ICONS.check} COMPLIANT SYNTHESIS</span>`;

  const confBadge = `<span class="badge-tag" style="background: rgba(255,255,255,0.06);">${escapeHtml(data.confidence || "high").toUpperCase()} CONFIDENCE</span>`;

  const actionsHtml = (data.recommended_corrective_actions || []).length > 0
    ? `
      <div style="margin-top: 10px; border-top: 1px solid var(--border-hairline); padding-top: 8px;">
        <div style="font-size: 10px; color: var(--text-muted); text-transform: uppercase; margin-bottom: 4px;">Corrective Actions</div>
        <ul style="margin: 0; padding-left: 16px; font-size: 11px; color: var(--signal-amber); line-height: 1.5;">
          ${data.recommended_corrective_actions.map(act => `<li>${escapeHtml(act)}</li>`).join("")}
        </ul>
      </div>
    `
    : "";

  const citationsHtml = (data.citations || []).length > 0
    ? `
      <div style="margin-top: 12px;">
        <div style="font-size: 10px; color: var(--text-muted); text-transform: uppercase; margin-bottom: 6px;">Grounding Citations (${data.citations.length})</div>
        <div style="display: flex; flex-direction: column; gap: 8px;">
          ${data.citations.map((c, i) => {
            const timeRange = `${formatTime(c.start_time_sec)} - ${formatTime(c.end_time_sec)}`;
            const seekTime = c.timestamp_sec != null ? c.timestamp_sec : c.start_time_sec;
            return `
              <div class="citation-card" data-vid="${escapeHtml(c.video_id)}" data-ep="${c.episode_index}" data-time="${seekTime}">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                  <strong style="color: var(--text-primary);">${escapeHtml(c.video_id)} #ep${c.episode_index + 1}</strong>
                  <span class="mono-data" style="color: var(--signal-cyan); font-size: 10px;">${ICONS.play} ${timeRange}</span>
                </div>
                ${c.violation_type ? `<div style="font-size: 11px; color: var(--signal-crimson); font-weight: 600; margin-bottom: 2px;">${escapeHtml(c.violation_type)}</div>` : ""}
                <div style="font-size: 11px; color: var(--text-secondary); line-height: 1.4;">${escapeHtml(c.evidence || "")}</div>
                ${c.storyboard_url ? `
                  <div style="margin-top: 6px;">
                    <img src="${c.storyboard_url}" alt="Citation Storyboard" style="width: 100%; border-radius: var(--radius-sm); border: 1px solid var(--border-hairline); max-height: 90px; object-fit: cover;">
                  </div>
                ` : ""}
              </div>
            `;
          }).join("")}
        </div>
      </div>
    `
    : '<div style="font-size: 11px; color: var(--text-muted); margin-top: 8px;">No explicit episode citations returned.</div>';

  container.innerHTML = `
    <div style="background: var(--bg-surface); border: 1px solid var(--border-hairline); border-radius: var(--radius-sm); padding: 12px; margin-bottom: 12px;">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
        ${ncrBadge}
        ${confBadge}
      </div>
      <div style="font-size: 12px; color: var(--text-primary); line-height: 1.5; white-space: pre-wrap;">${escapeHtml(data.answer)}</div>
      ${actionsHtml}
    </div>
    ${citationsHtml}
  `;

  // Attach citation seek handlers
  container.querySelectorAll(".citation-card").forEach(card => {
    card.addEventListener("click", async () => {
      const vid = card.dataset.vid;
      const epIdx = parseInt(card.dataset.ep, 10);
      const t = parseFloat(card.dataset.time);

      if (vid && vid !== state.currentVideoId) {
        await selectVideo(vid);
      }
      if (!isNaN(epIdx)) {
        selectEpisode(epIdx);
      }
      if (!isNaN(t)) {
        seekVideo(t);
      }
    });
  });
}

function renderSearchResults(data) {
  const container = document.getElementById("rag-results-container");
  if (!container) return;

  const results = data.results || [];
  if (results.length === 0) {
    container.innerHTML = `
      <div style="padding: 24px; text-align: center; color: var(--text-muted); font-size: 11px;">
        No vectors matched query. Try broadening your query or clearing filters.
      </div>
    `;
    return;
  }

  container.innerHTML = `
    <div style="font-size: 10px; color: var(--text-muted); text-transform: uppercase; margin-bottom: 8px;">
      Hybrid Vector Results (${results.length} matches, ${data.total_vectors_searched || 0} indexed)
    </div>
    <div style="display: flex; flex-direction: column; gap: 8px;">
      ${results.map(r => {
        const simScore = (r.similarity_score * 100).toFixed(1);
        const timeRange = `${formatTime(r.start_time_sec)} - ${formatTime(r.end_time_sec)}`;
        const hasViolations = (r.violations_count || 0) > 0;
        return `
          <div class="citation-card" data-vid="${escapeHtml(r.video_id)}" data-ep="${r.episode_index}" data-time="${r.start_time_sec}">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
              <strong style="color: var(--text-primary);">${escapeHtml(r.video_id)} #ep${r.episode_index + 1}</strong>
              <span class="mono-data" style="color: var(--signal-cyan); font-size: 10px;">${simScore}% match</span>
            </div>
            <div style="display: flex; gap: 6px; margin-bottom: 4px;">
              <span class="badge-tag">${escapeHtml(r.site_zone || "")}</span>
              <span class="badge-tag">${escapeHtml(r.work_category || "")}</span>
              ${hasViolations ? `<span class="badge-tag" style="color: var(--signal-crimson);">${r.violations_count} Violations</span>` : ""}
            </div>
            <div style="font-size: 11px; color: var(--text-secondary); line-height: 1.4;">${escapeHtml(r.work_summary || "")}</div>
            <div style="margin-top: 4px; display: flex; justify-content: flex-end;">
              <span class="mono-data" style="color: var(--signal-cyan); font-size: 10px;">${ICONS.play} Seek ${timeRange}</span>
            </div>
          </div>
        `;
      }).join("")}
    </div>
  `;

  container.querySelectorAll(".citation-card").forEach(card => {
    card.addEventListener("click", async () => {
      const vid = card.dataset.vid;
      const epIdx = parseInt(card.dataset.ep, 10);
      const t = parseFloat(card.dataset.time);

      if (vid && vid !== state.currentVideoId) {
        await selectVideo(vid);
      }
      if (!isNaN(epIdx)) {
        selectEpisode(epIdx);
      }
      if (!isNaN(t)) {
        seekVideo(t);
      }
    });
  });
}

// Storyboard Zoom Modal
function setupStoryboardModal() {
  const modal = document.getElementById("storyboard-modal");
  const modalImg = document.getElementById("modal-storyboard-img");
  const btnClose = document.getElementById("btn-close-modal");
  const container = document.getElementById("container-storyboard");
  const epImg = document.getElementById("ep-storyboard-img");

  function openModal() {
    if (!modal || !modalImg || !epImg) return;
    if (epImg.src && !epImg.src.startsWith("data:image/svg")) {
      modalImg.src = epImg.src;
      modal.classList.add("active");
    }
  }

  function closeModal() {
    if (modal) modal.classList.remove("active");
  }

  if (container) container.addEventListener("click", openModal);
  if (btnClose) btnClose.addEventListener("click", closeModal);
  if (modal) {
    modal.addEventListener("click", (e) => {
      if (e.target === modal) closeModal();
    });
  }
}

// Main Video Player Timecode Tracking
function setupPlayerTimecode() {
  const player = document.getElementById("main-video-player");
  const display = document.getElementById("video-timecode-display");
  if (!player || !display) return;

  player.addEventListener("timeupdate", () => {
    const cur = formatTime(player.currentTime);
    const dur = formatTime(player.duration || 0);
    display.textContent = `${cur} / ${dur}`;
  });
}

// Global Refresh Button
function setupGlobalRefresh() {
  const btnRefresh = document.getElementById("btn-refresh-global");
  if (!btnRefresh) return;

  btnRefresh.addEventListener("click", async () => {
    btnRefresh.style.transform = "rotate(180deg)";
    btnRefresh.style.transition = "transform 0.4s ease";
    setTimeout(() => {
      btnRefresh.style.transform = "none";
      btnRefresh.style.transition = "none";
    }, 450);

    await fetchStats();
    await loadVideos(state.currentVideoId);
  });
}

// Initial Boot
document.addEventListener("DOMContentLoaded", async () => {
  setupIngestionForm();
  setupRAGController();
  setupStoryboardModal();
  setupPlayerTimecode();
  setupGlobalRefresh();

  await fetchZones();
  await fetchStats();
  await loadVideos();
});


// === EHS Copilot Expandable Reader & Modal Controller ===
(function initCopilotReaderControls() {
  function setupReaderUI() {
    const resultsBox = document.getElementById("rag-results-container");
    if (!resultsBox || document.querySelector(".copilot-reader-toolbar")) return;

    // Locate the SYNTHESIS & GROUNDED CITATIONS header or create a bar right above resultsBox
    let headerEl = resultsBox.previousElementSibling;
    if (!headerEl) {
      headerEl = document.createElement("div");
      resultsBox.parentNode.insertBefore(headerEl, resultsBox);
    }
    headerEl.style.display = "flex";
    headerEl.style.alignItems = "center";
    headerEl.style.justifyContent = "space-between";
    headerEl.style.gap = "8px";

    const toolbar = document.createElement("div");
    toolbar.id = "copilot-reader-toolbar";
    toolbar.className = "copilot-reader-toolbar";
    toolbar.innerHTML = `
      <button type="button" data-reader-action="inputs" class="btn-reader-tool" title="Collapse prompt and filters to maximize reading height">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="18 15 12 9 6 15"></polyline></svg>
        <span>Tall View</span>
      </button>
      <button type="button" data-reader-action="wide" class="btn-reader-tool" title="Expand right panel width for side-by-side reading and video seeking">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="15 3 21 3 21 9"></polyline><polyline points="9 21 3 21 3 15"></polyline><line x1="21" y1="3" x2="14" y2="10"></line><line x1="3" y1="21" x2="10" y2="14"></line></svg>
        <span>Wide View</span>
      </button>
      <button type="button" data-reader-action="modal" class="btn-reader-tool" title="Pop out synthesis into full-screen reader modal">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"></rect><line x1="9" y1="3" x2="9" y2="21"></line></svg>
        <span>Modal</span>
      </button>
    `;
    headerEl.appendChild(toolbar);

    // Create full-screen modal backdrop
    const modal = document.createElement("div");
    modal.id = "copilot-reader-modal";
    modal.className = "copilot-modal-backdrop";
    modal.innerHTML = `
      <div class="copilot-modal-card">
        <div class="copilot-modal-header">
          <span>EHS COPILOT // SYNTHESIS & GROUNDED CITATIONS DOSSIER</span>
          <button type="button" data-reader-action="close-modal" class="btn-reader-tool">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
            <span>Close (Esc)</span>
          </button>
        </div>
        <div id="copilot-modal-body" class="copilot-modal-body"></div>
      </div>
    `;
    document.body.appendChild(modal);

    const btnInputs = toolbar.querySelector("[data-reader-action='inputs']");
    const btnWide = toolbar.querySelector("[data-reader-action='wide']");
    const btnModal = toolbar.querySelector("[data-reader-action='modal']");
    const btnCloseModal = modal.querySelector("[data-reader-action='close-modal']");
    const modalBody = modal.querySelector(".copilot-modal-body");

    function toggleInputs(forceState) {
      const active = typeof forceState === "boolean"
        ? forceState
        : !document.body.classList.contains("copilot-inputs-collapsed");
      document.body.classList.toggle("copilot-inputs-collapsed", active);
      btnInputs.classList.toggle("active", active);
      btnInputs.querySelector("span").textContent = active ? "Show Inputs" : "Tall View";
    }

    function toggleWide(forceState) {
      const active = typeof forceState === "boolean"
        ? forceState
        : !document.body.classList.contains("copilot-wide-mode");
      document.body.classList.toggle("copilot-wide-mode", active);
      btnWide.classList.toggle("active", active);
      if (active) toggleInputs(true);
    }

    btnInputs.addEventListener("click", () => toggleInputs());
    btnWide.addEventListener("click", () => toggleWide());

    btnModal.addEventListener("click", () => {
      modalBody.innerHTML = resultsBox.innerHTML;
      modal.classList.add("open");
      // Wire citation clicks inside modal to trigger original citations and close modal
      const modalInteractive = modalBody.querySelectorAll("[data-seek], [data-timestamp], button, .citation-chip, .rag-citation-card");
      const origInteractive = resultsBox.querySelectorAll("[data-seek], [data-timestamp], button, .citation-chip, .rag-citation-card");
      modalInteractive.forEach((el, idx) => {
        el.addEventListener("click", () => {
          modal.classList.remove("open");
          if (origInteractive[idx]) origInteractive[idx].click();
        });
      });
    });

    btnCloseModal.addEventListener("click", () => modal.classList.remove("open"));
    modal.addEventListener("click", (e) => {
      if (e.target === modal) modal.classList.remove("open");
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") modal.classList.remove("open");
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", setupReaderUI);
  } else {
    setupReaderUI();
  }
})();


// === Live Ingestion Progress Poller & Elapsed Timer ===
(function initIngestProgressMonitor() {
  function attachMonitor() {
    const submitBtn = document.querySelector("#btn-ingest-submit");
    if (!submitBtn || document.querySelector(".ingest-live-tracker")) return;

    const tracker = document.createElement("div");
    tracker.className = "ingest-live-tracker";
    tracker.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <span class="ingest-stage-label" style="color:#F59E0B;font-weight:600;">PROCESSING CCTV STREAM...</span>
        <span class="ingest-elapsed-timer">00:00</span>
      </div>
      <div class="ingest-progress-bar-well"><div class="ingest-progress-bar-fill"></div></div>
      <div class="ingest-sub-metrics" style="color:#9CA3AF;font-size:10px;">
        Scanning motion frames, slicing 16kHz audio & running Gemini EHS Audit...
      </div>
    `;
    submitBtn.parentNode.insertBefore(tracker, submitBtn.nextSibling);

    let timerId = null;
    let startSec = 0;

    const observer = new MutationObserver(() => {
      if (submitBtn.disabled) {
        if (!timerId) {
          startSec = Date.now();
          tracker.classList.add("active");
          const timerEl = tracker.querySelector(".ingest-elapsed-timer");
          const subEl = tracker.querySelector(".ingest-sub-metrics");
          const stageEl = tracker.querySelector(".ingest-stage-label");

          timerId = setInterval(async () => {
            const elapsed = Math.floor((Date.now() - startSec) / 1000);
            const mins = String(Math.floor(elapsed / 60)).padStart(2, "0");
            const secs = String(elapsed % 60).padStart(2, "0");
            timerEl.textContent = `${mins}:${secs}`;

            try {
              const res = await fetch("/api/ingest/progress");
              if (res.ok) {
                const data = await res.json();
                const latest = data.active_folders && data.active_folders[0];
                if (latest && latest.storyboards_count > 0) {
                  stageEl.textContent = `AUDITING EPISODES (${latest.storyboards_count} STORYBOARDS)...`;
                  subEl.textContent = `Feed: ${latest.video_id} | Storyboards: ${latest.storyboards_count} | Audio Slices: ${latest.audio_slices_count}`;
                }
              }
            } catch (_) {}
          }, 1000);
        }
      } else {
        if (timerId) {
          clearInterval(timerId);
          timerId = null;
          setTimeout(() => tracker.classList.remove("active"), 1500);
        }
      }
    });

    observer.observe(submitBtn, { attributes: true, attributeFilter: ["disabled"] });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", attachMonitor);
  } else {
    attachMonitor();
  }
})();


// === Interactive Multi-Episode Timeline, Storyboard Lightbox & In-Page NCR PDF Controller ===
(function initSieveAuditInteractiveEnhancements() {
  let cachedReceipt = null;
  let currentEpIdx = 0;

  function getActiveVideoId() {
    const scopeSel = document.querySelector("#filter-video-scope");
    if (scopeSel && scopeSel.value) return scopeSel.value;
    const activeCard = document.querySelector("#video-corpus-list .active, #video-corpus-list [data-video-id]");
    return activeCard ? activeCard.getAttribute("data-video-id") : null;
  }

  function getEpisodeColor(ep) {
    const status = (ep.episode_compliance_status || "").toLowerCase();
    const sev = (ep.overall_severity || "").toLowerCase();
    const vCount = (ep.safety_violations || []).length;
    if (status === "non_compliant_ncr" || sev === "critical" || sev === "high") {
      return "#EF4444"; // Crimson NCR
    }
    if (status === "minor_observation" || sev === "medium" || vCount > 0) {
      return "#F59E0B"; // Amber Observation
    }
    return "#10B981"; // Emerald Compliant
  }

  function formatMMSS(sec) {
    const s = Math.max(0, Number(sec) || 0);
    const m = Math.floor(s / 60);
    const rem = (s % 60).toFixed(1).padStart(4, "0");
    return `${String(m).padStart(2, "0")}:${rem}`;
  }

  function selectAndRenderEpisode(receipt, epArrayIndex, seekVideo = true) {
    if (!receipt || !receipt.episodes || !receipt.episodes[epArrayIndex]) return;
    currentEpIdx = epArrayIndex;
    const ep = receipt.episodes[epArrayIndex];
    const vidId = receipt.video_id;

    // Highlight active timeline block & pill
    document.querySelectorAll(".sieve-ep-timeline-seg").forEach((el, i) => {
      el.classList.toggle("active-ep", i === epArrayIndex);
    });
    document.querySelectorAll(".sieve-ep-pill").forEach((el, i) => {
      el.classList.toggle("active-ep", i === epArrayIndex);
    });

    // Seek video player if requested
    const player = document.querySelector("#main-video-player");
    if (seekVideo && player && typeof ep.start_time_sec === "number") {
      player.currentTime = ep.start_time_sec;
    }

    // Update Episode Dossier Header fields
    const titleEl = document.querySelector("#ep-title");
    if (titleEl) titleEl.textContent = `Ep #${ep.episode_index}: ${ep.episode_title || "Site Work Episode"}`;

    const scoreEl = document.querySelector("#ep-safety-score");
    if (scoreEl) scoreEl.textContent = `${Math.round(ep.safety_score_pct ?? 100)}% Safety`;

    const timeEl = document.querySelector("#ep-time-range");
    if (timeEl) {
      timeEl.textContent = `${formatMMSS(ep.start_time_sec)} - ${formatMMSS(ep.end_time_sec)} (${(ep.duration_sec || 0).toFixed(1)}s)`;
    }

    const gridEl = document.querySelector("#ep-grid-badge");
    if (gridEl && ep.grid_used) {
      gridEl.textContent = `GRID: ${ep.grid_used[0]}X${ep.grid_used[1]}`;
    }

    // Update Storyboard image
    const sbImg = document.querySelector("#ep-storyboard-img");
    if (sbImg) {
      const sbUrl = ep.storyboard_url || `/media/${vidId}/${ep.storyboard_rel_path}`;
      sbImg.src = sbUrl;
    }

    // Update Audio player
    const audPlayer = document.querySelector("#ep-audio-player");
    if (audPlayer) {
      const audUrl = ep.audio_url || (ep.audio_rel_path ? `/media/${vidId}/${ep.audio_rel_path}` : "");
      if (audUrl) {
        audPlayer.src = audUrl;
        audPlayer.style.display = "block";
      }
    }

    // Update Summary, Machinery, Acoustic Cues
    const sumEl = document.querySelector("#ep-work-summary");
    if (sumEl) sumEl.textContent = ep.work_summary || "";

    const machEl = document.querySelector("#ep-machinery-list");
    if (machEl) machEl.textContent = (ep.machinery_detected && ep.machinery_detected.length) ? ep.machinery_detected.join(", ") : "None detected";

    const cueEl = document.querySelector("#ep-audio-cues");
    if (cueEl) cueEl.textContent = ep.audio_safety_cues || "No audio stream or silent episode.";

    // Update Violations List
    const violContainer = document.querySelector("#ep-violations-list");
    if (violContainer) {
      const viols = ep.safety_violations || [];
      if (viols.length === 0) {
        violContainer.innerHTML = `<div style="color:#10B981;font-size:12px;padding:6px 0;">No safety violations detected in Episode #${ep.episode_index}.</div>`;
      } else {
        violContainer.innerHTML = viols.map(v => `
          <div class="violation-item-card" data-seek-ts="${v.timestamp_sec}" style="cursor:pointer;padding:8px 10px;margin-bottom:6px;background:#14171D;border-left:3px solid ${v.severity === 'critical' || v.severity === 'high' ? '#EF4444' : '#F59E0B'};border-radius:3px;">
            <div style="display:flex;justify-content:space-between;font-family:'JetBrains Mono',monospace;font-size:11px;margin-bottom:4px;">
              <span style="color:${v.severity === 'critical' || v.severity === 'high' ? '#EF4444' : '#F59E0B'};font-weight:700;text-transform:uppercase;">${v.severity} ? ${v.violation_type}</span>
              <span style="color:#9CA3AF;">? ${formatMMSS(v.timestamp_sec)} | Tile #${v.tile_number} | Cell ${(v.grid_cells || []).join(",")}</span>
            </div>
            <div style="font-size:12px;color:#F3F4F6;margin-bottom:3px;">${v.description}</div>
            <div style="font-size:11px;color:#9CA3AF;"><strong>Action:</strong> ${v.corrective_action_required}</div>
          </div>
        `).join("");
        violContainer.querySelectorAll("[data-seek-ts]").forEach(card => {
          card.addEventListener("click", () => {
            const ts = parseFloat(card.getAttribute("data-seek-ts"));
            if (player && !isNaN(ts)) {
              player.currentTime = ts;
              player.play().catch(() => {});
            }
          });
        });
      }
    }

    // Update Worker PPE Matrix
    const ppeContainer = document.querySelector("#ep-ppe-matrix");
    if (ppeContainer) {
      const obsList = ep.ppe_observations || [];
      if (obsList.length === 0) {
        ppeContainer.innerHTML = `<div style="color:#9CA3AF;font-size:12px;padding:6px 0;">No individual worker PPE observations logged in this episode.</div>`;
      } else {
        ppeContainer.innerHTML = obsList.map(o => `
          <div class="ppe-obs-row" data-seek-ts="${o.absolute_time_sec}" style="cursor:pointer;padding:7px 10px;margin-bottom:5px;background:#14171D;border:1px solid #282D37;border-radius:3px;">
            <div style="display:flex;justify-content:space-between;font-family:'JetBrains Mono',monospace;font-size:11px;margin-bottom:4px;">
              <strong style="color:#F3F4F6;">${o.worker_label}</strong>
              <span style="color:#9CA3AF;">? ${formatMMSS(o.absolute_time_sec)} | Tile #${o.tile_number} | Grid ${(o.grid_cells || []).join(",")}</span>
            </div>
            <div style="display:flex;gap:6px;font-family:'JetBrains Mono',monospace;font-size:10px;margin-bottom:4px;">
              <span style="padding:1px 6px;border-radius:2px;background:${o.hardhat_worn ? 'rgba(16,185,129,0.15)' : 'rgba(239,68,68,0.2)'};color:${o.hardhat_worn ? '#10B981' : '#EF4444'};">HARDHAT: ${o.hardhat_worn ? 'PASS' : 'FAIL'}</span>
              <span style="padding:1px 6px;border-radius:2px;background:${o.high_vis_vest_worn ? 'rgba(16,185,129,0.15)' : 'rgba(239,68,68,0.2)'};color:${o.high_vis_vest_worn ? '#10B981' : '#EF4444'};">VEST: ${o.high_vis_vest_worn ? 'PASS' : 'FAIL'}</span>
              <span style="padding:1px 6px;border-radius:2px;background:${o.safety_harness_worn ? 'rgba(16,185,129,0.15)' : 'rgba(156,163,175,0.15)'};color:${o.safety_harness_worn ? '#10B981' : '#9CA3AF'};">HARNESS: ${o.safety_harness_worn ? 'YES' : 'N/A'}</span>
            </div>
            <div style="font-size:11.5px;color:#D1D5DB;">${o.activity_description}</div>
          </div>
        `).join("");
        ppeContainer.querySelectorAll("[data-seek-ts]").forEach(row => {
          row.addEventListener("click", () => {
            const ts = parseFloat(row.getAttribute("data-seek-ts"));
            if (player && !isNaN(ts)) {
              player.currentTime = ts;
              player.play().catch(() => {});
            }
          });
        });
      }
    }
  }

  function renderTimelineAndPills(receipt) {
    const track = document.querySelector("#episode-timeline-track");
    if (!track || !receipt || !receipt.episodes) return;
    cachedReceipt = receipt;

    const totalDur = Math.max(1, Number(receipt.duration_sec || receipt.total_duration_sec || 1));
    track.innerHTML = "";

    receipt.episodes.forEach((ep, idx) => {
      const start = Math.max(0, Number(ep.start_time_sec) || 0);
      const end = Math.min(totalDur, Math.max(start + 1, Number(ep.end_time_sec) || (start + 5)));
      const leftPct = Math.min(98, (start / totalDur) * 100);
      const widthPct = Math.max(1.8, ((end - start) / totalDur) * 100);
      const color = getEpisodeColor(ep);

      const seg = document.createElement("div");
      seg.className = "sieve-ep-timeline-seg" + (idx === currentEpIdx ? " active-ep" : "");
      seg.style.left = `${leftPct}%`;
      seg.style.width = `${widthPct}%`;
      seg.style.background = color;
      seg.title = `Ep #${ep.episode_index}: ${ep.episode_title} (${formatMMSS(start)} - ${formatMMSS(end)}) | Safety: ${ep.safety_score_pct}%`;
      seg.textContent = `E${ep.episode_index}`;
      seg.addEventListener("click", (e) => {
        e.stopPropagation();
        selectAndRenderEpisode(receipt, idx, true);
      });
      track.appendChild(seg);
    });

    // Also render clickable Episode Pill Selector strip right below #episode-timeline-track
    let pillStrip = document.querySelector(".sieve-ep-pill-strip");
    if (!pillStrip) {
      pillStrip = document.createElement("div");
      pillStrip.className = "sieve-ep-pill-strip";
      track.parentNode.insertBefore(pillStrip, track.nextSibling);
    }
    pillStrip.innerHTML = receipt.episodes.map((ep, idx) => {
      const color = getEpisodeColor(ep);
      const vLen = (ep.safety_violations || []).length;
      return `
        <button type="button" class="sieve-ep-pill ${idx === currentEpIdx ? 'active-ep' : ''}" data-ep-idx="${idx}" title="${ep.episode_title}">
          <span class="sieve-ep-dot" style="background:${color};"></span>
          <span>EP #${ep.episode_index}</span>
          <span style="color:#9CA3AF;">${formatMMSS(ep.start_time_sec)}</span>
          ${vLen > 0 ? `<span style="color:${color};">(${vLen} Viol)</span>` : `<span style="color:#10B981;">OK</span>`}
        </button>
      `;
    }).join("");

    pillStrip.querySelectorAll(".sieve-ep-pill").forEach(btn => {
      btn.addEventListener("click", () => {
        const idx = parseInt(btn.getAttribute("data-ep-idx"), 10);
        selectAndRenderEpisode(receipt, idx, true);
      });
    });
  }

  async function syncActiveVideoTimeline() {
    const vidId = getActiveVideoId();
    if (!vidId) return;
    try {
      const res = await fetch(`/api/videos/${vidId}`);
      if (!res.ok) return;
      const receipt = await res.json();
      if (!cachedReceipt || cachedReceipt.video_id !== receipt.video_id || (document.querySelectorAll(".sieve-ep-timeline-seg").length === 0)) {
        currentEpIdx = 0;
        renderTimelineAndPills(receipt);
        selectAndRenderEpisode(receipt, 0, false);
      }
    } catch (_) {}
  }

  function setupStoryboardZoomAndNCRModal() {
    // 1. Storyboard Lightbox Zoom
    const sbModal = document.querySelector("#storyboard-modal");
    const sbModalImg = document.querySelector("#modal-storyboard-img");
    const sbCloseBtn = document.querySelector("#btn-close-modal");

    document.addEventListener("click", (e) => {
      const clickedSb = e.target.closest("#ep-storyboard-img, .storyboard-frame, .storyboard-container, .storyboard-wrapper");
      if (clickedSb && sbModal && sbModalImg) {
        const imgEl = document.querySelector("#ep-storyboard-img");
        if (imgEl && imgEl.src) {
          sbModalImg.src = imgEl.src;
          sbModal.classList.add("open", "active");
          sbModal.style.display = "flex";
        }
      }
    });

    if (sbCloseBtn && sbModal) {
      sbCloseBtn.addEventListener("click", () => {
        sbModal.classList.remove("open", "active");
        sbModal.style.display = "none";
      });
    }
    if (sbModal) {
      sbModal.addEventListener("click", (e) => {
        if (e.target === sbModal) {
          sbModal.classList.remove("open", "active");
          sbModal.style.display = "none";
        }
      });
    }

    // 2. Inject prominent "NCR DOSSIER PDF" button & In-Page PDF Preview/Print Modal
    if (!document.querySelector(".ncr-pdf-modal-backdrop")) {
      const ncrModal = document.createElement("div");
      ncrModal.className = "ncr-pdf-modal-backdrop";
      ncrModal.innerHTML = `
        <div class="ncr-pdf-modal-card">
          <div class="ncr-pdf-modal-header">
            <span style="font-weight:700;">OFFICIAL ENTERPRISE NON-CONFORMANCE REPORT (NCR) & DUTY-CYCLE DOSSIER</span>
            <div style="display:flex;gap:8px;align-items:center;">
              <button type="button" class="btn-ncr-pdf-action" data-ncr-action="print" style="margin:0;">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 6 2 18 2 18 9"></polyline><path d="M6 18H4a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-2"></path><rect x="6" y="14" width="12" height="8"></rect></svg>
                <span>Save / Print PDF (A4)</span>
              </button>
              <button type="button" class="btn-reader-tool" data-ncr-action="newtab">
                <span>Open Full Tab</span>
              </button>
              <button type="button" class="btn-reader-tool" data-ncr-action="close">
                <span>Close (Esc)</span>
              </button>
            </div>
          </div>
          <iframe class="ncr-pdf-iframe" title="NCR Dossier PDF Preview"></iframe>
        </div>
      `;
      document.body.appendChild(ncrModal);

      const iframe = ncrModal.querySelector(".ncr-pdf-iframe");
      ncrModal.querySelector("[data-ncr-action='print']").addEventListener("click", () => {
        if (iframe && iframe.contentWindow) {
          iframe.contentWindow.focus();
          iframe.contentWindow.print();
        }
      });
      ncrModal.querySelector("[data-ncr-action='newtab']").addEventListener("click", () => {
        const vidId = getActiveVideoId();
        if (vidId) window.open(`/api/videos/${vidId}/ncr/html`, "_blank");
      });
      ncrModal.querySelector("[data-ncr-action='close']").addEventListener("click", () => {
        ncrModal.classList.remove("open");
      });
      ncrModal.addEventListener("click", (e) => {
        if (e.target === ncrModal) ncrModal.classList.remove("open");
      });
      document.addEventListener("keydown", (e) => {
        if (e.key === "Escape") {
          ncrModal.classList.remove("open");
          if (sbModal) {
            sbModal.classList.remove("open", "active");
            sbModal.style.display = "none";
          }
        }
      });
    }

    function openNcrPdfModal() {
      const vidId = getActiveVideoId();
      if (!vidId) return;
      const ncrModal = document.querySelector(".ncr-pdf-modal-backdrop");
      const iframe = ncrModal ? ncrModal.querySelector(".ncr-pdf-iframe") : null;
      if (ncrModal && iframe) {
        iframe.src = `/api/videos/${vidId}/ncr/html`;
        ncrModal.classList.add("open");
      }
    }

    // Wire existing #btn-export-ncr AND inject a visible NCR DOSSIER PDF button right next to the Episode Title!
    const existingBtn = document.querySelector("#btn-export-ncr");
    if (existingBtn) {
      existingBtn.addEventListener("click", (e) => {
        e.preventDefault();
        openNcrPdfModal();
      });
    }

    const epTitle = document.querySelector("#ep-title");
    if (epTitle && epTitle.parentNode && !document.querySelector(".btn-inline-ncr-pdf")) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "btn-ncr-pdf-action btn-inline-ncr-pdf";
      btn.innerHTML = `
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line></svg>
        <span>NCR Dossier PDF</span>
      `;
      btn.addEventListener("click", openNcrPdfModal);
      epTitle.parentNode.appendChild(btn);
    }
  }

  function initAll() {
    setupStoryboardZoomAndNCRModal();
    syncActiveVideoTimeline();
    // Watch for video selection changes in the sidebar or scope dropdown
    const corpusList = document.querySelector("#video-corpus-list");
    if (corpusList) {
      corpusList.addEventListener("click", () => setTimeout(syncActiveVideoTimeline, 250));
    }
    const scopeSel = document.querySelector("#filter-video-scope");
    if (scopeSel) {
      scopeSel.addEventListener("change", () => {
        cachedReceipt = null;
        setTimeout(syncActiveVideoTimeline, 100);
      });
    }
    setInterval(syncActiveVideoTimeline, 2500);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initAll);
  } else {
    initAll();
  }
})();


// === Individual Feed & Episode Token Reduction Telemetry Controller ===
(function initIndividualTokenTelemetry() {
  let globalStatsCache = null;

  async function refreshFeedCardTokenBadges() {
    try {
      const [vRes, sRes] = await Promise.all([
        fetch("/api/videos"),
        fetch("/api/stats")
      ]);
      if (sRes.ok) globalStatsCache = await sRes.json();
      if (!vRes.ok) return;
      const vData = await vRes.json();
      const videos = Array.isArray(vData) ? vData : (vData.videos || []);
      const vMap = {};
      videos.forEach(v => { vMap[v.video_id] = v; });

      const cards = document.querySelectorAll("#video-corpus-list [data-video-id], #video-corpus-list > div");
      cards.forEach(card => {
        let vidId = card.getAttribute("data-video-id");
        if (!vidId) {
          const matched = videos.find(v => card.textContent.includes(v.video_id) || card.textContent.includes(v.video_filename));
          if (matched) vidId = matched.video_id;
        }
        if (!vidId || !vMap[vidId]) return;
        const vInfo = vMap[vidId];
        const savedPct = Number(vInfo.tokens_saved_pct ?? 0).toFixed(1);

        let badge = card.querySelector(".feed-token-badge");
        if (!badge) {
          badge = document.createElement("span");
          badge.className = "feed-token-badge";
          const utilNode = Array.from(card.querySelectorAll("span, div")).find(
            el => el.children.length === 0 && el.textContent.includes("Utilization:")
          );
          if (utilNode) {
            utilNode.appendChild(badge);
          } else {
            card.appendChild(badge);
          }
        }
        badge.textContent = `Saved: ${savedPct}%`;
      });
    } catch (_) {}
  }

  async function updateActiveVideoAndEpisodeTokens() {
    const scopeSel = document.querySelector("#filter-video-scope");
    const vidId = scopeSel ? scopeSel.value : null;
    if (!vidId) return;

    try {
      const res = await fetch(`/api/videos/${vidId}`);
      if (!res.ok) return;
      const receipt = await res.json();

      const feedSaved = Number(receipt.tokens_saved_pct ?? 0).toFixed(1);
      const actualTok = Number(receipt.actual_sieve_audit_tokens ?? 0);
      const baseTok = Number(receipt.estimated_direct_video_tokens ?? 0);
      const globalSaved = globalStatsCache
        ? Number(globalStatsCache.overall_tokens_saved_pct ?? 0).toFixed(1)
        : null;

      // 1. Top Ribbon: show Active Feed % + Combined Avg %
      const topTokenEl = document.querySelector("#stat-tokens-saved");
      if (topTokenEl) {
        if (globalSaved !== null) {
          topTokenEl.innerHTML = `${feedSaved}% <span class="top-stat-sub">FEED (${globalSaved}% ALL)</span>`;
        } else {
          topTokenEl.textContent = `${feedSaved}%`;
        }
      }

      // 2. Center Header: show Feed Token Breakdown Pill
      const epTitleEl = document.querySelector("#ep-title");
      const epHeader = epTitleEl ? epTitleEl.parentNode : null;
      if (epHeader) {
        let pill = epHeader.querySelector(".center-token-breakdown-pill");
        if (!pill) {
          pill = document.createElement("span");
          pill.className = "center-token-breakdown-pill";
          epHeader.insertBefore(pill, epHeader.querySelector(".btn-inline-ncr-pdf"));
        }
        pill.textContent = `Feed Tokens: ${actualTok.toLocaleString()} / ${baseTok.toLocaleString()} (-${feedSaved}%)`;
      }

      // 3. Bottom-Left Telemetry Box: show Individual Episode Token Reduction %
      const activePill = document.querySelector(".sieve-ep-pill.active-ep");
      const epIdx = activePill ? parseInt(activePill.getAttribute("data-ep-idx") || "0", 10) : 0;
      const ep = (receipt.episodes && receipt.episodes[epIdx]) ? receipt.episodes[epIdx] : (receipt.episodes && receipt.episodes[0]);

      if (ep) {
        const epUsed = (Number(ep.sieve_input_tokens) || 0) + (Number(ep.sieve_output_tokens) || 0);
        const epBase = Math.max(epUsed, Math.round((Number(ep.duration_sec) || 10) * 560.0));
        const epSavedPct = epBase > 0 ? Math.max(0, ((epBase - epUsed) / epBase) * 100).toFixed(1) : "0.0";

        const allLeafNodes = Array.from(document.querySelectorAll("#episode-dossier-panel div, #episode-dossier-panel span"));
        const tokenLeaf = allLeafNodes.find(el => (el.textContent || "").trim().startsWith("Tokens:") || (el.textContent || "").trim().startsWith("Ep Tokens:"));
        if (tokenLeaf) {
          tokenLeaf.innerHTML = `Ep Tokens: <strong style="color:#F3F4F6;">${epUsed.toLocaleString()}</strong> / ${epBase.toLocaleString()} <span style="color:#10B981;font-weight:700;">(-${epSavedPct}%)</span>`;
        }
      }
    } catch (_) {}
  }

  function init() {
    refreshFeedCardTokenBadges();
    setTimeout(updateActiveVideoAndEpisodeTokens, 300);

    document.addEventListener("click", (e) => {
      if (e.target.closest("#video-corpus-list, .sieve-ep-pill, .sieve-ep-timeline-seg, #btn-refresh-global")) {
        setTimeout(() => {
          refreshFeedCardTokenBadges();
          updateActiveVideoAndEpisodeTokens();
        }, 200);
      }
    });

    const scopeSel = document.querySelector("#filter-video-scope");
    if (scopeSel) {
      scopeSel.addEventListener("change", () => setTimeout(updateActiveVideoAndEpisodeTokens, 150));
    }

    setInterval(() => {
      refreshFeedCardTokenBadges();
      updateActiveVideoAndEpisodeTokens();
    }, 3000);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
