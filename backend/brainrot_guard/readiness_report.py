from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from brainrot_guard.learning_validation import (
    validate_caregiver_calibration_dataset,
    validate_learning_calibration,
)
from brainrot_guard.readiness import (
    validate_hardware_target,
    validate_local_media_folder,
    validate_local_tools,
    validate_tribe_plotbrain,
    validate_vlm_provider,
)


def validate_readiness_report(
    environ: Mapping[str, str] | None = None,
    *,
    repository=None,
    media_dir=None,
    artifacts_dir=None,
    runtime_analysis_report: Mapping[str, object] | None = None,
    browser_report: Mapping[str, object] | None = None,
    vlm_probe_report: Mapping[str, object] | None = None,
    force_live: bool = False,
    smoke_render: bool = False,
    force_vlm: bool = False,
    probe_vlm: bool = False,
    force_local: bool = False,
    force_hardware: bool = False,
    validate_image_conversion: bool = False,
    include_learning: bool = True,
) -> dict[str, Any]:
    env = environ or {}
    local_tools = validate_local_tools(env, force=force_local)
    tribe_plotbrain = validate_tribe_plotbrain(env, force=force_live, smoke_render=smoke_render)
    if vlm_probe_report is not None:
        vlm_provider = _saved_vlm_probe_report(vlm_probe_report)
    else:
        vlm_provider = validate_vlm_provider(env, force=force_vlm, probe=probe_vlm)
    target_hardware = validate_hardware_target(env, force=force_hardware)
    if media_dir is not None:
        media_folder = validate_local_media_folder(
            media_dir,
            artifacts_dir=artifacts_dir,
            validate_image_conversion=validate_image_conversion,
        )
        media_folder["configured"] = True
    else:
        media_folder = {
            "ready": False,
            "configured": False,
            "media_dir": None,
            "message": "media_dir is not configured",
        }
    caregiver_learning = _caregiver_learning_report(include_learning=include_learning)
    caregiver_calibration_dataset = _caregiver_calibration_dataset_report(repository)
    representative_live_analysis = _saved_analysis_report(
        runtime_analysis_report,
        missing_message="representative live analysis report was not provided",
    )
    browser_visual_proof = _saved_browser_report(
        browser_report,
        missing_message="browser Playwright visual proof report was not provided",
    )
    reports = {
        "media_folder": media_folder,
        "local_tools": local_tools,
        "tribe_plotbrain": tribe_plotbrain,
        "target_hardware": target_hardware,
        "vlm_provider": vlm_provider,
        "caregiver_learning": caregiver_learning,
        "caregiver_calibration_dataset": caregiver_calibration_dataset,
        "representative_live_analysis": representative_live_analysis,
        "browser_visual_proof": browser_visual_proof,
    }
    proofs = {
        "media_folder": _proof(
            "Local mixed-media folder is configured and contains supported media without YouTube/browser placeholders.",
            media_folder,
        ),
        "local_tools": _proof("Local tools can support static-image video conversion for TRIBE input.", local_tools),
        "tribe_plotbrain": _proof(
            "Live TRIBE/PlotBrain dependencies and fsaverage5 rendering are available.",
            tribe_plotbrain,
        ),
        "target_hardware": _proof("CUDA Ampere-class NVIDIA hardware with about 16GB VRAM is present.", target_hardware),
        "vlm_provider": _proof(
            "Configured cloud VLM adapter is ready behind the engagement gate.",
            vlm_provider,
        ),
        "caregiver_learning": _proof(
            "Binary caregiver approve/disapprove calibration moves Bayesian thresholds without child telemetry.",
            caregiver_learning,
        ),
        "caregiver_calibration_dataset": _proof(
            "Real local caregiver-labeled calibration dataset has enough balanced examples for personalization.",
            caregiver_calibration_dataset,
        ),
        "representative_live_analysis": _proof(
            "Representative local media has passed validate-analysis with real persisted segment/frame artifacts.",
            representative_live_analysis,
        ),
        "browser_visual_proof": _proof(
            "Browser-level review console proof shows playback, synchronized brain frames, and warning UI.",
            browser_visual_proof,
        ),
    }
    unmet = [name for name, proof in proofs.items() if not proof["ready"]]
    external_gaps = _external_proof_gaps(reports, probe_vlm=probe_vlm)
    return {
        "ready": not unmet and not external_gaps,
        "message": "ready" if not unmet and not external_gaps else "readiness proof is incomplete",
        "unmet_proofs": unmet,
        "external_proof_gaps": external_gaps,
        "proofs": proofs,
        **reports,
    }


def _caregiver_learning_report(*, include_learning: bool) -> dict[str, Any]:
    if not include_learning:
        return {
            "ready": False,
            "enabled": False,
            "message": "caregiver learning calibration was not checked",
        }
    report = validate_learning_calibration()
    report["enabled"] = True
    return report


def _caregiver_calibration_dataset_report(repository) -> dict[str, Any]:
    if repository is None:
        return {
            "ready": False,
            "enabled": False,
            "message": "repository was not provided for caregiver calibration dataset validation",
        }
    report = validate_caregiver_calibration_dataset(repository)
    report["enabled"] = True
    return report


def _saved_analysis_report(
    report: Mapping[str, object] | None, *, missing_message: str
) -> dict[str, Any]:
    if report is None:
        return {"ready": False, "provided": False, "message": missing_message}
    missing = _analysis_report_missing_fields(report)
    ready = bool(report.get("ready")) and not missing
    message = str(report.get("message", ""))
    if missing:
        message = "missing analysis proof fields: " + ", ".join(missing)
    return {**dict(report), "provided": True, "ready": ready, "message": message}


def _saved_browser_report(
    report: Mapping[str, object] | None, *, missing_message: str
) -> dict[str, Any]:
    if report is None:
        return {"ready": False, "provided": False, "message": missing_message}
    missing = _browser_report_missing_fields(report)
    ready = bool(report.get("ready")) and not missing
    message = str(report.get("message", ""))
    if missing:
        message = "missing browser proof fields: " + ", ".join(missing)
    return {**dict(report), "provided": True, "ready": ready, "message": message}


def _saved_vlm_probe_report(report: Mapping[str, object]) -> dict[str, Any]:
    missing = _vlm_probe_report_missing_fields(report)
    ready = bool(report.get("ready")) and not missing
    message = str(report.get("message", ""))
    if missing:
        message = "missing VLM proof fields: " + ", ".join(missing)
    return {**dict(report), "provided": True, "ready": ready, "message": message}


def _analysis_report_missing_fields(report: Mapping[str, object]) -> list[str]:
    missing = [
        key
        for key in ("media_count", "analyzed_count", "results")
        if key not in report
    ]
    results = report.get("results")
    if not isinstance(results, list) or not results:
        return missing + ["results[0]"]
    first = results[0]
    if not isinstance(first, Mapping):
        return missing + ["results[0]"]
    required_result = (
        "status",
        "segment_count",
        "npz_artifact_count",
        "frame_artifact_count",
        "frame_provenance_ready_count",
        "scalar_heatmap_bar_count",
        "artifact_integrity",
        "frame_manifest_status",
        "engagement_gate_crossed",
        "risk_gate_crossed",
        "warning_decision",
    )
    missing.extend(f"results[0].{key}" for key in required_result if key not in first)
    if first.get("status") != "ready":
        missing.append("results[0].status=ready")
    if first.get("artifact_integrity") != "ready":
        missing.append("results[0].artifact_integrity=ready")
    if first.get("frame_manifest_status") != "ready":
        missing.append("results[0].frame_manifest_status=ready")
    if int(first.get("segment_count") or 0) <= 0:
        missing.append("results[0].segment_count>0")
    if first.get("npz_artifact_count") != first.get("segment_count"):
        missing.append("results[0].npz_artifact_count=segment_count")
    if first.get("frame_artifact_count") != first.get("segment_count"):
        missing.append("results[0].frame_artifact_count=segment_count")
    if first.get("frame_provenance_ready_count") != first.get("segment_count"):
        missing.append("results[0].frame_provenance_ready_count=segment_count")
    if first.get("scalar_heatmap_bar_count") != 0:
        missing.append("results[0].scalar_heatmap_bar_count=0")
    return missing


def _vlm_probe_report_missing_fields(report: Mapping[str, object]) -> list[str]:
    required = (
        "provider",
        "model",
        "endpoint",
        "engagement_threshold",
        "engagement_threshold_source",
        "provider_probe",
        "probe_theme",
        "probe_risk_score",
    )
    missing = [key for key in required if key not in report]
    if report.get("provider") not in {"gemini", "xai", "minicpm"}:
        missing.append("provider=supported")
    if not str(report.get("model", "")).strip():
        missing.append("model")
    endpoint = str(report.get("endpoint", ""))
    if not endpoint:
        missing.append("endpoint")
    if _looks_like_unredacted_secret(endpoint):
        missing.append("endpoint redacted")
    if not isinstance(report.get("engagement_threshold"), (int, float)):
        missing.append("engagement_threshold")
    elif not 0 <= float(report["engagement_threshold"]) <= 1:
        missing.append("engagement_threshold in [0,1]")
    if report.get("provider_probe") != "ready":
        missing.append("provider_probe=ready")
    if not str(report.get("probe_theme", "")).strip():
        missing.append("probe_theme")
    if not isinstance(report.get("probe_risk_score"), (int, float)):
        missing.append("probe_risk_score")
    elif not 0 <= float(report["probe_risk_score"]) <= 1:
        missing.append("probe_risk_score in [0,1]")
    return missing


def _looks_like_unredacted_secret(endpoint: str) -> bool:
    lowered = endpoint.lower()
    if "key=redacted" in lowered:
        return False
    return "key=" in lowered or "api_key=" in lowered or "apikey=" in lowered


def _browser_report_missing_fields(report: Mapping[str, object]) -> list[str]:
    required = (
        "browser_status",
        "visible_media_stage",
        "visible_brain_frame",
        "visible_timeline_playhead",
        "playhead_moved",
        "brain_frame_synced",
        "active_timeline_frame_synced",
        "visible_feedback_controls",
        "visible_skip_controls",
        "visible_auto_close_controls",
        "visible_proxy_label",
        "screenshot_bytes",
        "missing_visible_checks",
    )
    missing = [key for key in required if key not in report]
    if report.get("browser_status") != "ready":
        missing.append("browser_status=ready")
    for key in required:
        if key not in {"browser_status", "screenshot_bytes", "missing_visible_checks"} and report.get(key) is not True:
            missing.append(f"{key}=true")
    if int(report.get("screenshot_bytes") or 0) <= 0:
        missing.append("screenshot_bytes>0")
    if report.get("missing_visible_checks") not in ([], ()):
        missing.append("missing_visible_checks=[]")
    return missing


def _proof(requirement: str, report: Mapping[str, object]) -> dict[str, Any]:
    return {
        "requirement": requirement,
        "ready": bool(report.get("ready")),
        "message": str(report.get("message", "")),
    }


def _external_proof_gaps(
    reports: Mapping[str, Mapping[str, object]], *, probe_vlm: bool
) -> list[str]:
    gaps: list[str] = []
    if not reports["tribe_plotbrain"].get("ready"):
        gaps.append("live_tribe_plotbrain")
    if reports["tribe_plotbrain"].get("plotbrain_smoke_render") != "ready":
        gaps.append("plotbrain_smoke_render")
    if not reports["target_hardware"].get("ready"):
        gaps.append("target_gpu_hardware")
    if not reports["vlm_provider"].get("ready"):
        gaps.append("vlm_provider_configuration")
    if not reports["vlm_provider"].get("ready") or reports["vlm_provider"].get("provider_probe") != "ready":
        gaps.append("credentialed_vlm_probe")
    if not reports["representative_live_analysis"].get("ready"):
        gaps.append("representative_live_analysis")
    if not reports["browser_visual_proof"].get("ready"):
        gaps.append("browser_playwright_visual_proof")
    if not reports["caregiver_calibration_dataset"].get("ready"):
        gaps.append("broader_caregiver_labeled_calibration")
    return gaps


__all__ = ["validate_readiness_report"]
