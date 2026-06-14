const mediaSelect = document.getElementById("media-select");
const skipRecommendedToggle = document.getElementById("skip-recommended-toggle");
const statusPill = document.getElementById("status-pill");
const videoViewer = document.getElementById("video-viewer");
const audioViewer = document.getElementById("audio-viewer");
const imageViewer = document.getElementById("image-viewer");
const textViewer = document.getElementById("text-viewer");
const brainFrame = document.getElementById("brain-frame");
const frameState = document.getElementById("frame-state");
const brainProfile = document.getElementById("brain-profile");
const warningState = document.getElementById("warning-state");
const riskSummary = document.getElementById("risk-summary");
const timelineTrack = document.getElementById("timeline-track");
const timelinePlayhead = document.getElementById("timeline-playhead");
const approveButton = document.getElementById("approve-button");
const disapproveButton = document.getElementById("disapprove-button");
const skipButton = document.getElementById("skip-button");
const analyzeButton = document.getElementById("analyze-button");
const analysisState = document.getElementById("analysis-state");
const feedbackState = document.getElementById("feedback-state");
const skipState = document.getElementById("skip-state");

let currentDetail = null;
let staticTimer = null;
let staticStartedAt = 0;

function preferredMediaIdFromPath() {
  const match = window.location.pathname.match(/^\/media\/([^/]+)$/);
  return match ? decodeURIComponent(match[1]) : undefined;
}

async function loadMediaList(preferredId) {
  const params = skipRecommendedToggle.checked ? "?exclude_skip_recommended=true" : "";
  const response = await fetch(`/api/media${params}`);
  const items = await response.json();
  mediaSelect.replaceChildren(...items.map((item) => {
    const option = document.createElement("option");
    option.value = item.id;
    option.textContent = item.path.split("/").pop();
    return option;
  }));
  if (items.length === 0) {
    clearReviewConsole("No media items match current queue filter.");
    return;
  }
  const selected = items.find((item) => item.id === preferredId) || items[0];
  mediaSelect.value = selected.id;
  await loadMedia(selected.id);
}

async function loadMedia(mediaId) {
  stopStaticPlaybackClock();
  const response = await fetch(`/api/media/${mediaId}`);
  currentDetail = await response.json();
  if (window.history && window.location.pathname !== `/media/${encodeURIComponent(mediaId)}`) {
    window.history.replaceState(null, "", `/media/${encodeURIComponent(mediaId)}`);
  }
  statusPill.textContent = currentDetail.frame_manifest.status;
  analysisState.textContent = currentDetail.frame_manifest.status;
  updateWarningControls(currentDetail.warning.decision);
  renderMedia(currentDetail.media);
  renderFrameTimeline(currentDetail.frame_manifest);
  updatePlaybackTime(0, mediaDurationSeconds());
  renderRisk(currentDetail);
}

function clearReviewConsole(message) {
  stopAllPlayback();
  currentDetail = null;
  statusPill.textContent = "empty";
  analysisState.textContent = "idle";
  frameState.textContent = "pending";
  feedbackState.textContent = "idle";
  skipState.textContent = "skip inactive";
  updateWarningControls("allow");
  [videoViewer, audioViewer, imageViewer, textViewer].forEach((node) => {
    node.style.display = "none";
    node.removeAttribute("src");
  });
  textViewer.textContent = "";
  brainFrame.removeAttribute("src");
  renderFrameTimeline({status: "pending", message, frames: []});
  renderRows(brainProfile, [["Attention", "pending"], ["Engagement", "pending"], ["Arousal", "pending"], ["Confidence", "pending"]]);
  renderRisk({vlm: null, vlm_status: null, warning: {thresholds: {}}});
}

function updateWarningControls(decision) {
  warningState.textContent = decision;
  skipButton.disabled = decision !== "skip_recommended";
  skipState.textContent = decision === "skip_recommended" ? "skip available" : "skip inactive";
}

function renderMedia(media) {
  [videoViewer, audioViewer, imageViewer, textViewer].forEach((node) => {
    node.style.display = "none";
    node.removeAttribute("src");
  });
  textViewer.textContent = "";
  if (media.kind === "video") {
    videoViewer.src = media.file_url;
    videoViewer.style.display = "block";
  } else if (media.kind === "audio") {
    audioViewer.src = media.file_url;
    audioViewer.style.display = "block";
  } else if (media.kind === "image") {
    imageViewer.src = media.file_url;
    imageViewer.alt = media.path.split("/").pop();
    imageViewer.style.display = "block";
    startStaticPlaybackClock();
  } else {
    fetch(media.file_url).then((response) => response.text()).then((text) => {
      textViewer.textContent = text;
      textViewer.style.display = "block";
      startStaticPlaybackClock();
    });
  }
}

function startStaticPlaybackClock() {
  stopStaticPlaybackClock();
  staticStartedAt = performance.now();
  const duration = mediaDurationSeconds();
  staticTimer = window.setInterval(() => {
    const seconds = Math.min(duration, (performance.now() - staticStartedAt) / 1000);
    updatePlaybackTime(seconds, duration);
    if (seconds >= duration) stopStaticPlaybackClock();
  }, 100);
}

function stopStaticPlaybackClock() {
  if (staticTimer !== null) {
    window.clearInterval(staticTimer);
    staticTimer = null;
  }
}

function stopAllPlayback() {
  stopStaticPlaybackClock();
  [videoViewer, audioViewer].forEach((node) => {
    node.pause();
    node.currentTime = 0;
  });
}

function mediaDurationSeconds() {
  const frames = (currentDetail && currentDetail.frame_manifest.frames) || [];
  const frameEnd = frames.length ? Math.max(...frames.map((frame) => frame.end_ms)) : 0;
  const mediaEnd = currentDetail && currentDetail.media.duration_ms ? currentDetail.media.duration_ms : 0;
  return Math.max(1, frameEnd, mediaEnd) / 1000;
}

function renderFrameTimeline(frameManifest) {
  const frames = frameManifest.frames || [];
  timelineTrack.replaceChildren();
  if (frames.length === 0) {
    const pending = document.createElement("div");
    pending.className = "timeline-empty";
    pending.textContent = frameManifest.message || "Waiting for PlotBrain frame artifacts.";
    timelineTrack.append(pending, timelinePlayhead);
    return;
  }
  const timelineEndMs = Math.max(...frames.map((frame) => frame.end_ms));
  const nodes = frames.map((frame) => {
    const node = document.createElement("img");
    node.className = "timeline-frame";
    node.src = frame.url;
    node.alt = `TRIBE-derived neural response proxy t=${frame.timestep}`;
    node.dataset.timestep = String(frame.timestep);
    node.style.left = `${Math.max(0, (frame.start_ms / timelineEndMs) * 100)}%`;
    node.style.width = `${Math.max(4, ((frame.end_ms - frame.start_ms) / timelineEndMs) * 100)}%`;
    return node;
  });
  timelineTrack.append(...nodes, timelinePlayhead);
}

function updatePlaybackTime(seconds, duration) {
  const timeMs = Math.max(0, Math.floor(seconds * 1000));
  const pct = duration > 0 ? Math.min(100, (seconds / duration) * 100) : 0;
  timelinePlayhead.style.left = `${pct}%`;
  renderBrainFrame(timeMs);
}

function renderBrainFrame(timeMs) {
  if (!currentDetail) return;
  const frames = currentDetail.frame_manifest.frames || [];
  const selected = frames.reduce((last, frame) => frame.start_ms <= timeMs ? frame : last, frames[0]);
  document.querySelectorAll(".timeline-frame").forEach((node) => {
    node.classList.toggle("is-active", selected && Number(node.dataset.timestep) === selected.timestep);
  });
  if (selected && selected.url) {
    brainFrame.src = selected.url;
  } else {
    brainFrame.removeAttribute("src");
  }
  frameState.textContent = currentDetail.frame_manifest.status;
  renderBrainProfile(timeMs);
}

function renderBrainProfile(timeMs) {
  const segments = currentDetail.segments || [];
  const selected = segments.reduce((last, segment) => segment.start_ms <= timeMs ? segment : last, segments[0]);
  if (!selected) {
    renderRows(brainProfile, [["Attention", "pending"], ["Engagement", "pending"], ["Arousal", "pending"], ["Confidence", "pending"]]);
    return;
  }
  renderRows(brainProfile, [
    ["Segment", `${selected.start_ms}-${selected.end_ms} ms`],
    ["Attention", score(selected.attention)],
    ["Engagement", score(selected.engagement)],
    ["Arousal", score(selected.arousal)],
    ["Confidence", score(selected.confidence)],
  ]);
}

function renderRisk(detail) {
  const vlm = detail.vlm || {};
  const vlmStatus = detail.vlm_status || {};
  const thresholds = (detail.warning && detail.warning.thresholds) || {};
  renderRows(riskSummary, [
    ["VLM status", vlmStatus.status || "pending"],
    ["Provider", vlmStatus.provider || "pending"],
    ["Provider error", vlmStatus.error || "none"],
    ["Theme", vlm.theme || "pending"],
    ["Risk", score(vlm.risk_score)],
    ["Education", score(vlm.educational_value)],
    ["Pacing", score(vlm.pacing_score)],
    ["Scene cadence", typeof vlm.scene_change_cadence_hz === "number" ? `${vlm.scene_change_cadence_hz.toFixed(2)} Hz` : "pending"],
    ["Contrast", score(vlm.contrast_score)],
    ["Sound effects", score(vlm.sound_effect_density)],
    ["Emotional hooks", score(vlm.emotional_hook_score)],
    ["Novelty", score(vlm.novelty_score)],
    ["Repetition", score(vlm.repetition_score)],
    ["Rationale", vlm.risk_rationale || "pending"],
    ["Engagement threshold", score(thresholds.engagement)],
    ["Risk threshold", score(thresholds.risk)],
    ["Beta disapproval", score(thresholds.beta_disapproval_mean)],
    ["GP disapproval", score(thresholds.gp_disapproval_mean)],
    ["Thompson sample", score(thresholds.thompson_sample)],
    ["Threshold source", thresholds.source || "pending"],
  ]);
}

function renderRows(target, rows) {
  target.replaceChildren(...rows.flatMap(([key, value]) => {
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    dt.textContent = key;
    dd.textContent = value;
    return [dt, dd];
  }));
}

function score(value) {
  return typeof value === "number" ? value.toFixed(2) : "pending";
}

[videoViewer, audioViewer].forEach((node) => {
  node.addEventListener("timeupdate", () => updatePlaybackTime(node.currentTime, node.duration || mediaDurationSeconds()));
  node.addEventListener("play", stopStaticPlaybackClock);
});

mediaSelect.addEventListener("change", () => loadMedia(mediaSelect.value));
skipRecommendedToggle.addEventListener("change", () => loadMediaList(currentDetail && currentDetail.media.id));
analyzeButton.addEventListener("click", runAnalysis);
approveButton.addEventListener("click", () => sendFeedback("approve"));
disapproveButton.addEventListener("click", () => sendFeedback("disapprove"));
skipButton.addEventListener("click", skipCurrentRecommended);

async function skipCurrentRecommended() {
  if (!currentDetail || currentDetail.warning.decision !== "skip_recommended") return;
  stopAllPlayback();
  skipButton.disabled = true;
  skipState.textContent = "skipping";
  skipRecommendedToggle.checked = true;
  await loadMediaList(currentDetail.media.id);
  skipState.textContent = "skip applied";
}

async function runAnalysis() {
  if (!currentDetail) return;
  analyzeButton.disabled = true;
  analysisState.textContent = "analyzing";
  statusPill.textContent = "analyzing";
  frameState.textContent = "pending";
  try {
    const durationMs = Math.max(1000, Math.round(mediaDurationSeconds() * 1000));
    const response = await fetch(`/api/media/${currentDetail.media.id}/analyze`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({duration_ms: durationMs}),
    });
    if (!response.ok) {
      const message = await responseMessage(response);
      throw new Error(message);
    }
    await loadMedia(currentDetail.media.id);
    analysisState.textContent = "complete";
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    analysisState.textContent = `analysis error: ${message}`;
    statusPill.textContent = "error";
    frameState.textContent = "error";
  } finally {
    analyzeButton.disabled = false;
  }
}

async function responseMessage(response) {
  try {
    const payload = await response.json();
    return payload.detail || response.statusText || `HTTP ${response.status}`;
  } catch (_error) {
    return response.statusText || `HTTP ${response.status}`;
  }
}

async function sendFeedback(label) {
  if (!currentDetail) return;
  approveButton.disabled = true;
  disapproveButton.disabled = true;
  feedbackState.textContent = "saving feedback";
  try {
    const response = await fetch(`/api/media/${currentDetail.media.id}/feedback`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({label}),
    });
    if (!response.ok) {
      const message = await responseMessage(response);
      throw new Error(message);
    }
    await loadMedia(currentDetail.media.id);
    feedbackState.textContent = `saved ${label}`;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    feedbackState.textContent = `feedback error: ${message}`;
  } finally {
    approveButton.disabled = false;
    disapproveButton.disabled = false;
  }
}

loadMediaList(preferredMediaIdFromPath()).catch((error) => {
  statusPill.textContent = "error";
  console.error(error);
});
