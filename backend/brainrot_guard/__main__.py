from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from brainrot_guard.analysis_factory import build_analysis_service_from_env
from brainrot_guard.analysis_validation import validate_runtime_analysis
from brainrot_guard.app import create_app
from brainrot_guard.browser_validation import validate_browser_review_console
from brainrot_guard.demo import generate_demo_all
from brainrot_guard.demo_validation import validate_demo_run
from brainrot_guard.learning_validation import validate_caregiver_calibration_dataset, validate_learning_calibration
from brainrot_guard.privacy_validation import validate_privacy_boundaries
from brainrot_guard.proofs import EVIDENCE_DIR_DEFAULT, load_all_proofs
from brainrot_guard.fixtures.caregiver_seed import seed_caregiver_calibration
from brainrot_guard.readiness import (
    validate_hardware_target,
    validate_local_media_folder,
    validate_local_tools,
    validate_tribe_plotbrain,
    validate_vlm_provider,
)
from brainrot_guard.readiness_report import validate_readiness_report
from brainrot_guard.repository import Repository
from brainrot_guard.review_validation import validate_review_console


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo")
    demo.add_argument("--media-dir", required=True)
    demo.add_argument("--db-path", required=True)
    demo.add_argument("--artifacts-dir", required=True)
    demo.add_argument("--duration-ms", type=int, default=3000)
    validate_demo = sub.add_parser("validate-demo")
    validate_demo.add_argument("--media-dir", required=True)
    validate_demo.add_argument("--db-path", required=True)
    validate_demo.add_argument("--artifacts-dir", required=True)
    validate_demo.add_argument("--duration-ms", type=int, default=3000)
    validate_demo.add_argument("--include-browser", action="store_true")
    validate_demo.add_argument("--browser-screenshot-path")
    serve = sub.add_parser("serve")
    serve.add_argument("--media-dir")
    serve.add_argument("--db-path", required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8787)
    serve.add_argument("--artifacts-dir")
    serve.add_argument("--enable-live-analysis", action="store_true")
    validate = sub.add_parser("validate-live")
    validate.add_argument("--force", action="store_true")
    validate.add_argument("--smoke-render", action="store_true")
    validate_vlm = sub.add_parser("validate-vlm-live")
    validate_vlm.add_argument("--force", action="store_true")
    validate_vlm.add_argument("--probe", action="store_true")
    validate_local = sub.add_parser("validate-local")
    validate_local.add_argument("--force", action="store_true")
    validate_hardware = sub.add_parser("validate-hardware")
    validate_hardware.add_argument("--force", action="store_true")
    validate_readiness = sub.add_parser("validate-readiness")
    validate_readiness.add_argument("--media-dir")
    validate_readiness.add_argument("--db-path")
    validate_readiness.add_argument("--artifacts-dir")
    validate_readiness.add_argument("--analysis-report-json")
    validate_readiness.add_argument("--browser-report-json")
    validate_readiness.add_argument("--vlm-report-json")
    validate_readiness.add_argument("--evidence-dir", default=None)
    validate_readiness.add_argument("--force-live", action="store_true")
    validate_readiness.add_argument("--smoke-render", action="store_true")
    validate_readiness.add_argument("--force-vlm", action="store_true")
    validate_readiness.add_argument("--probe-vlm", action="store_true")
    validate_readiness.add_argument("--force-local", action="store_true")
    validate_readiness.add_argument("--force-hardware", action="store_true")
    validate_readiness.add_argument("--validate-image-conversion", action="store_true")
    validate_media = sub.add_parser("validate-media")
    validate_media.add_argument("--media-dir", required=True)
    validate_media.add_argument("--artifacts-dir")
    validate_media.add_argument("--validate-image-conversion", action="store_true")
    validate_analysis = sub.add_parser("validate-analysis")
    validate_analysis.add_argument("--media-dir", required=True)
    validate_analysis.add_argument("--db-path", required=True)
    validate_analysis.add_argument("--artifacts-dir", required=True)
    validate_analysis.add_argument("--enable-live-analysis", action="store_true")
    validate_analysis.add_argument("--limit", type=int, default=1)
    validate_analysis.add_argument("--duration-ms", type=int)
    validate_learning = sub.add_parser("validate-learning")
    validate_learning.add_argument("--random-seed", type=int, default=7)
    validate_calibration = sub.add_parser("validate-calibration")
    validate_calibration.add_argument("--db-path", required=True)
    validate_calibration.add_argument("--min-examples", type=int, default=6)
    validate_calibration.add_argument("--min-approve", type=int, default=2)
    validate_calibration.add_argument("--min-disapprove", type=int, default=2)
    validate_calibration.add_argument("--min-segment-response", type=int, default=1)
    validate_calibration.add_argument("--min-full-decomposition", type=int, default=2)
    seed_calibration = sub.add_parser("seed-calibration")
    seed_calibration.add_argument("--db-path", required=True)
    seed_calibration.add_argument("--min-per-label", type=int, default=3)
    validate_ui = sub.add_parser("validate-ui")
    validate_ui.add_argument("--db-path", required=True)
    validate_ui.add_argument("--media-id")
    validate_ui.add_argument("--limit", type=int, default=5)
    validate_privacy = sub.add_parser("validate-privacy")
    validate_privacy.add_argument("--db-path", required=True)
    validate_privacy.add_argument("--media-dir")
    validate_browser = sub.add_parser("validate-browser")
    validate_browser.add_argument("--db-path", required=True)
    validate_browser.add_argument("--media-dir")
    validate_browser.add_argument("--artifacts-dir")
    validate_browser.add_argument("--media-id")
    validate_browser.add_argument("--screenshot-path")
    validate_browser.add_argument("--duration-ms", type=int, default=3000)
    args = parser.parse_args(argv)
    if args.command == "validate-live":
        print(
            json.dumps(
                validate_tribe_plotbrain(os.environ, force=args.force, smoke_render=args.smoke_render),
                sort_keys=True,
            )
        )
        return 0
    if args.command == "validate-vlm-live":
        print(
            json.dumps(
                validate_vlm_provider(os.environ, force=args.force, probe=args.probe),
                sort_keys=True,
            )
        )
        return 0
    if args.command == "validate-local":
        print(json.dumps(validate_local_tools(os.environ, force=args.force), sort_keys=True))
        return 0
    if args.command == "validate-hardware":
        print(json.dumps(validate_hardware_target(os.environ, force=args.force), sort_keys=True))
        return 0
    if args.command == "validate-readiness":
        readiness_repository = None
        if args.db_path:
            readiness_repository = Repository(Path(args.db_path))
            readiness_repository.initialize()
        evidence_root = Path(args.evidence_dir) if args.evidence_dir else EVIDENCE_DIR_DEFAULT
        loaded = load_all_proofs(evidence_root)
        explicit_analysis = _read_json_report(args.analysis_report_json)
        explicit_browser = _read_json_report(args.browser_report_json)
        explicit_vlm = _read_json_report(args.vlm_report_json)
        print(
            json.dumps(
                validate_readiness_report(
                    os.environ,
                    repository=readiness_repository,
                    media_dir=Path(args.media_dir) if args.media_dir else None,
                    artifacts_dir=Path(args.artifacts_dir) if args.artifacts_dir else None,
                    runtime_analysis_report=explicit_analysis or loaded.get("analysis"),
                    browser_report=explicit_browser or loaded.get("browser"),
                    vlm_probe_report=explicit_vlm or loaded.get("vlm_probe"),
                    force_live=args.force_live,
                    smoke_render=args.smoke_render,
                    force_vlm=args.force_vlm,
                    probe_vlm=args.probe_vlm,
                    force_local=args.force_local,
                    force_hardware=args.force_hardware,
                    validate_image_conversion=args.validate_image_conversion,
                ),
                sort_keys=True,
            )
        )
        return 0
    if args.command == "validate-media":
        report = validate_local_media_folder(
            Path(args.media_dir),
            artifacts_dir=Path(args.artifacts_dir) if args.artifacts_dir else None,
            validate_image_conversion=args.validate_image_conversion,
        )
        print(json.dumps(report, sort_keys=True))
        return 0
    if args.command == "validate-learning":
        print(json.dumps(validate_learning_calibration(random_seed=args.random_seed), sort_keys=True))
        return 0
    repo = Repository(Path(args.db_path))
    repo.initialize()
    if args.command == "validate-calibration":
        print(
            json.dumps(
                validate_caregiver_calibration_dataset(
                    repo,
                    min_examples=args.min_examples,
                    min_approve=args.min_approve,
                    min_disapprove=args.min_disapprove,
                    min_segment_response=args.min_segment_response,
                    min_full_decomposition=args.min_full_decomposition,
                ),
                sort_keys=True,
            )
        )
        return 0
    if args.command == "seed-calibration":
        result = seed_caregiver_calibration(repo, min_per_label=args.min_per_label)
        print(json.dumps({"db_path": str(Path(args.db_path)), **result}, sort_keys=True))
        return 0
    if args.command == "validate-ui":
        print(
            json.dumps(
                validate_review_console(repository=repo, media_id=args.media_id, limit=args.limit),
                sort_keys=True,
            )
        )
        return 0
    if args.command == "validate-privacy":
        print(
            json.dumps(
                validate_privacy_boundaries(
                    repository=repo,
                    media_dir=Path(args.media_dir) if args.media_dir else None,
                ),
                sort_keys=True,
            )
        )
        return 0
    if args.command == "validate-browser":
        print(
            json.dumps(
                validate_browser_review_console(
                    repository=repo,
                    media_dir=Path(args.media_dir) if args.media_dir else None,
                    artifacts_dir=Path(args.artifacts_dir) if args.artifacts_dir else None,
                    media_id=args.media_id,
                    screenshot_path=Path(args.screenshot_path) if args.screenshot_path else None,
                    duration_ms=args.duration_ms,
                ),
                sort_keys=True,
            )
        )
        return 0
    if args.command == "validate-analysis":
        env = dict(os.environ)
        env["BRAINROT_GUARD_ARTIFACTS_DIR"] = args.artifacts_dir
        if args.enable_live_analysis:
            env["BRAINROT_GUARD_ENABLE_LIVE_ANALYSIS"] = "1"
        analysis_service = build_analysis_service_from_env(repo, env)
        report = validate_runtime_analysis(
            media_dir=Path(args.media_dir),
            repository=repo,
            analysis_service=analysis_service,
            limit=args.limit,
            duration_ms=args.duration_ms,
        )
        print(json.dumps(report, sort_keys=True))
        return 0
    if args.command == "validate-demo":
        report = validate_demo_run(
            repository=repo,
            media_dir=Path(args.media_dir),
            artifacts_dir=Path(args.artifacts_dir),
            duration_ms=args.duration_ms,
            include_browser=args.include_browser,
            browser_screenshot_path=Path(args.browser_screenshot_path) if args.browser_screenshot_path else None,
        )
        print(json.dumps(report, sort_keys=True))
        return 0
    if args.command == "demo":
        media_ids = generate_demo_all(repo, Path(args.media_dir), Path(args.artifacts_dir), args.duration_ms)
        detail = {"media_id": media_ids[0], "media_count": len(media_ids), "frame_manifest_status": "ready"}
        print(json.dumps(detail, sort_keys=True))
        return 0
    if args.command == "serve":
        import uvicorn

        env = dict(os.environ)
        if args.artifacts_dir:
            env["BRAINROT_GUARD_ARTIFACTS_DIR"] = args.artifacts_dir
        if args.enable_live_analysis:
            env["BRAINROT_GUARD_ENABLE_LIVE_ANALYSIS"] = "1"
        analysis_service = build_analysis_service_from_env(repo, env)
        app = create_app(
            repository=repo,
            media_dir=Path(args.media_dir) if args.media_dir else None,
            analysis_service=analysis_service,
            environ=env,
        )
        uvicorn.run(app, host=args.host, port=args.port)
        return 0
    return 1


def _read_json_report(path: str | None) -> dict | None:
    if path is None:
        return None
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON report must be an object: {path}")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
