from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from brainrot_guard.artifacts import write_segment_npz
from brainrot_guard.artifacts import render_demo_frame
from brainrot_guard.ingestion import scan_media_folder
from brainrot_guard.ingestion.scanner import SUPPORTED, _looks_like_url_file
from brainrot_guard.models import MediaKind, SegmentSignal
from brainrot_guard.plotbrain_adapter import TribePlotBrainRenderer
from brainrot_guard.runtime import load_npz_prediction
from brainrot_guard.tribe_adapter import generate_static_image_video
from brainrot_guard.vlm import VLMDecompositionRequest
from brainrot_guard.vlm.adapters import HTTPVLMDecomposer
from brainrot_guard.vlm.config import provider_config_from_env


def validate_tribe_plotbrain(
    environ: Mapping[str, str] | None = None,
    *,
    force: bool = False,
    smoke_render: bool = False,
    tribe_loader: Callable[[Path], object] | None = None,
    plotbrain_loader: Callable[[], object] | None = None,
) -> dict:
    env = environ or {}
    enabled = force or _truthy(env.get("BRAINROT_GUARD_LIVE_VALIDATE"))
    should_smoke_render = smoke_render or _truthy(env.get("BRAINROT_GUARD_PLOTBRAIN_SMOKE"))
    checkpoint_raw = env.get("BRAINROT_GUARD_TRIBE_CKPT")
    fsaverage5_raw = env.get("BRAINROT_GUARD_FSAVERAGE5_DIR")
    report = {
        "enabled": enabled,
        "ready": False,
        "checkpoint_dir": checkpoint_raw,
        "fsaverage5_dir": fsaverage5_raw,
        "plotbrain_smoke_render": "not_checked",
        "message": "Live TRIBE/PlotBrain validation is opt-in; set BRAINROT_GUARD_LIVE_VALIDATE=1 or use --force.",
    }
    if not enabled:
        return report
    missing_env = [name for name, value in {
        "BRAINROT_GUARD_TRIBE_CKPT": checkpoint_raw,
        "BRAINROT_GUARD_FSAVERAGE5_DIR": fsaverage5_raw,
    }.items() if not value]
    if missing_env:
        report["message"] = "Missing " + ", ".join(missing_env)
        return report
    checkpoint = Path(str(checkpoint_raw))
    fsaverage5 = Path(str(fsaverage5_raw))
    missing_paths = [str(path) for path in (checkpoint, fsaverage5) if not path.exists()]
    if missing_paths:
        report["message"] = "Missing local TRIBE/PlotBrain assets: " + ", ".join(missing_paths)
        return report
    plotter = None
    try:
        _load_tribe(checkpoint, tribe_loader)
        plotter = _load_plotbrain(plotbrain_loader)
        if should_smoke_render:
            _validate_plotbrain_smoke_render(lambda: plotter)
            report["plotbrain_smoke_render"] = "ready"
    except Exception as exc:
        if should_smoke_render:
            report["plotbrain_smoke_render"] = "error"
        report["message"] = str(exc)
        return report
    report["ready"] = True
    report["message"] = "ready"
    return report


def validate_vlm_provider(
    environ: Mapping[str, str] | None = None,
    *,
    force: bool = False,
    probe: bool = False,
    client: object | None = None,
) -> dict:
    env = environ or {}
    provider = env.get("VLM_PROVIDER", "minicpm").strip().lower()
    enabled = force or _truthy(env.get("BRAINROT_GUARD_VLM_LIVE_VALIDATE"))
    should_probe = probe or _truthy(env.get("BRAINROT_GUARD_VLM_PROBE"))
    report = {
        "enabled": enabled,
        "ready": False,
        "provider": provider,
        "model": _model_for(provider, env),
        "provider_probe": "not_checked",
        "message": "Live VLM validation is opt-in; set BRAINROT_GUARD_VLM_LIVE_VALIDATE=1 or use --force.",
    }
    if not enabled:
        return report
    required = _required_provider_env(provider)
    if not required:
        report["message"] = f"Unsupported VLM_PROVIDER '{provider}'. Supported providers: minicpm, gemini, xai."
        return report
    missing = [key for key in required if not env.get(key)]
    if missing:
        report["message"] = "Missing " + ", ".join(missing)
        return report
    if should_probe:
        try:
            decomposition = _probe_vlm_provider(env, client=client)
        except Exception as exc:
            report["provider_probe"] = "error"
            report["message"] = str(exc)
            return report
        report["provider_probe"] = "ready"
        report["probe_theme"] = decomposition.theme
        report["probe_risk_score"] = decomposition.risk_score
    report["ready"] = True
    report["message"] = "configured"
    return report


def validate_local_tools(
    environ: Mapping[str, str] | None = None,
    *,
    force: bool = False,
    image_video_validator: Callable[[], object] | None = None,
) -> dict:
    env = environ or {}
    enabled = force or _truthy(env.get("BRAINROT_GUARD_LOCAL_VALIDATE"))
    report = {
        "enabled": enabled,
        "ready": False,
        "ffmpeg_image_video": "not_checked",
        "message": "Local tool validation is opt-in; set BRAINROT_GUARD_LOCAL_VALIDATE=1 or use --force.",
    }
    if not enabled:
        return report
    try:
        (image_video_validator or _validate_image_video_tool)()
    except Exception as exc:
        report["ffmpeg_image_video"] = "error"
        report["message"] = str(exc)
        return report
    report["ready"] = True
    report["ffmpeg_image_video"] = "ready"
    report["message"] = "ready"
    return report


def validate_hardware_target(
    environ: Mapping[str, str] | None = None,
    *,
    force: bool = False,
    gpu_probe: Callable[[], str] | None = None,
) -> dict:
    env = environ or {}
    enabled = force or _truthy(env.get("BRAINROT_GUARD_HARDWARE_VALIDATE"))
    report = {
        "enabled": enabled,
        "ready": False,
        "target": "CUDA Ampere-class NVIDIA GPU with about 16GB VRAM",
        "minimum_memory_mb": 15000,
        "minimum_compute_capability": 8.0,
        "gpu_count": 0,
        "gpus": [],
        "message": "Hardware validation is opt-in; set BRAINROT_GUARD_HARDWARE_VALIDATE=1 or use --force.",
    }
    if not enabled:
        return report
    try:
        raw = (gpu_probe or _nvidia_smi_probe)()
    except FileNotFoundError:
        report["message"] = "nvidia-smi is not available; CUDA NVIDIA hardware was not detected."
        return report
    except Exception as exc:
        report["message"] = str(exc)
        return report

    gpus = [_parse_gpu_line(line) for line in raw.splitlines() if line.strip()]
    report["gpus"] = gpus
    report["gpu_count"] = len(gpus)
    ready_gpus = [gpu for gpu in gpus if gpu["memory_ok"] and gpu["ampere_or_newer"]]
    if ready_gpus:
        report["ready"] = True
        report["message"] = "ready"
        return report
    if not gpus:
        report["message"] = "No CUDA NVIDIA GPUs were reported by nvidia-smi."
    else:
        report["message"] = "No GPU met the target: Ampere-or-newer compute capability and about 16GB VRAM."
    return report


def validate_local_media_folder(
    media_dir: Path,
    *,
    artifacts_dir: Path | None = None,
    validate_image_conversion: bool = False,
    image_video_factory: Callable[..., Path] | None = None,
) -> dict:
    root = media_dir.expanduser().resolve()
    report = {
        "ready": False,
        "media_dir": str(root),
        "artifacts_dir": str(artifacts_dir.expanduser().resolve()) if artifacts_dir else None,
        "media_count": 0,
        "kinds": {},
        "known_duration_count": 0,
        "unknown_duration_count": 0,
        "unknown_duration_media": [],
        "forbidden_source_count": 0,
        "forbidden_sources": [],
        "unsupported_count": 0,
        "image_conversion": "not_checked",
        "converted_image_count": 0,
        "message": "not checked",
    }
    if not root.is_dir():
        report["message"] = f"media folder does not exist: {root}"
        return report

    direct_files = [path for path in sorted(root.iterdir()) if path.is_file() and not path.is_symlink()]
    forbidden = [path.name for path in direct_files if _is_forbidden_source(path)]
    unsupported = [
        path.name
        for path in direct_files
        if path.suffix.lower() not in SUPPORTED and not _is_forbidden_source(path)
    ]
    items = scan_media_folder(root)
    kinds = Counter(item.kind.value for item in items)
    unknown_duration = [
        item.path.name
        for item in items
        if item.kind in {MediaKind.AUDIO, MediaKind.VIDEO} and item.duration_ms is None
    ]
    report.update(
        {
            "media_count": len(items),
            "kinds": dict(sorted(kinds.items())),
            "known_duration_count": sum(1 for item in items if item.duration_ms is not None),
            "unknown_duration_count": len(unknown_duration),
            "unknown_duration_media": unknown_duration,
            "forbidden_source_count": len(forbidden),
            "forbidden_sources": forbidden,
            "unsupported_count": len(unsupported),
        }
    )

    if validate_image_conversion:
        image_report = _validate_media_image_conversions(
            [item for item in items if item.kind == MediaKind.IMAGE],
            artifacts_dir=artifacts_dir,
            image_video_factory=image_video_factory,
        )
        report.update(image_report)

    problems = []
    if not items:
        problems.append("no supported local media files found")
    if forbidden:
        problems.append("YouTube/browser source placeholders are out of scope")
    if unknown_duration:
        problems.append("audio/video duration is unknown")
    if report["image_conversion"] == "error":
        problems.append("static image conversion failed")
    if problems:
        report["message"] = "; ".join(problems)
        return report

    report["ready"] = True
    report["message"] = "ready"
    return report


def _load_tribe(checkpoint: Path, loader: Callable[[Path], object] | None) -> object:
    if loader is not None:
        return loader(checkpoint)
    quantized_loader = checkpoint / "load_quantized_tribev2.py"
    if quantized_loader.exists():
        return object()
    try:
        import tribev2  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError("tribev2 package is not installed") from exc
    return object()


def _load_plotbrain(loader: Callable[[], object] | None) -> object:
    if loader is not None:
        return loader()
    from brainrot_guard.plotbrain_adapter import load_plotbrain_plotter

    return load_plotbrain_plotter()


def _required_provider_env(provider: str) -> tuple[str, ...]:
    if provider == "gemini":
        return ("GEMINI_API_KEY",)
    if provider == "xai":
        return ("XAI_API_KEY",)
    if provider == "minicpm":
        return ("MINICPM_API_KEY", "MINICPM_API_BASE_URL")
    return ()


def _model_for(provider: str, env: Mapping[str, str]) -> str | None:
    if provider == "gemini":
        return env.get("GEMINI_MODEL", "gemini-3.5-flash")
    if provider == "xai":
        return env.get("XAI_MODEL", "grok-vision")
    if provider == "minicpm":
        return env.get("MINICPM_MODEL", "minicpm-v")
    return None


def _validate_image_video_tool() -> None:
    with tempfile.TemporaryDirectory(prefix="brainrot-guard-local-") as tmp:
        tmp_path = Path(tmp)
        image_path = tmp_path / "frame.png"
        output_path = tmp_path / "frame.mp4"
        image_path.write_bytes(_PNG_BYTES)
        generate_static_image_video(image_path, output_path, duration_ms=1000)


def _nvidia_smi_probe() -> str:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "nvidia-smi failed to query GPU hardware")
    return result.stdout


def _parse_gpu_line(line: str) -> dict[str, object]:
    parts = [part.strip() for part in line.split(",")]
    name = parts[0] if parts else "unknown"
    memory_mb = _parse_int(parts[1]) if len(parts) > 1 else 0
    compute_capability = _parse_float(parts[2]) if len(parts) > 2 else 0.0
    return {
        "name": name,
        "memory_total_mb": memory_mb,
        "compute_capability": compute_capability,
        "memory_ok": memory_mb >= 15000,
        "ampere_or_newer": compute_capability >= 8.0,
    }


def _parse_int(value: str) -> int:
    try:
        return int(float(value))
    except ValueError:
        return 0


def _parse_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return 0.0


def _validate_plotbrain_smoke_render(plotter_loader: Callable[[], object]) -> None:
    with tempfile.TemporaryDirectory(prefix="brainrot-guard-plotbrain-") as tmp:
        tmp_path = Path(tmp)
        npz_path = write_segment_npz(
            tmp_path / "npz",
            media_id="plotbrain-smoke",
            timestep=0,
            start_ms=0,
            end_ms=1000,
            vertex_values=np.linspace(0.0, 1.0, 20484, dtype=np.float32),
        )
        prediction = load_npz_prediction(npz_path)
        renderer = TribePlotBrainRenderer(plotter_loader=plotter_loader)
        renderer.render_png(prediction, tmp_path / "frames" / "000000.png")


def _probe_vlm_provider(env: Mapping[str, str], *, client: object | None):
    config = provider_config_from_env(env)
    decomposer = HTTPVLMDecomposer(config, client=client)  # type: ignore[arg-type]
    with tempfile.TemporaryDirectory(prefix="brainrot-guard-vlm-probe-") as tmp:
        return decomposer.decompose(_sample_vlm_request(Path(tmp)))


def _sample_vlm_request(tmp_path: Path) -> VLMDecompositionRequest:
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    (media_dir / "probe.txt").write_text("counting by twos\n2 4 6 8\n", encoding="utf-8")
    media = scan_media_folder(media_dir)[0]
    npz_path = write_segment_npz(
        tmp_path / "npz",
        media_id=media.id,
        timestep=0,
        start_ms=0,
        end_ms=1000,
        vertex_values=np.ones(20484, dtype=np.float32),
    )
    frame_path = render_demo_frame(tmp_path / "frames", media_id=media.id, timestep=0)
    segment = SegmentSignal(
        timestep=0,
        start_ms=0,
        end_ms=1000,
        attention=0.6,
        engagement=0.9,
        arousal=0.5,
        confidence=0.95,
        npz_path=npz_path,
        frame_path=frame_path,
    )
    return VLMDecompositionRequest(media=media, segments=[segment], frame_paths=(frame_path,))


def _validate_media_image_conversions(
    image_items,
    *,
    artifacts_dir: Path | None,
    image_video_factory: Callable[..., Path] | None,
) -> dict:
    if not image_items:
        return {"image_conversion": "no_images", "converted_image_count": 0}
    output_root = (
        artifacts_dir.expanduser().resolve() / "validation" / "image-videos"
        if artifacts_dir
        else Path(tempfile.mkdtemp(prefix="brainrot-guard-media-images-"))
    )
    factory = image_video_factory or generate_static_image_video
    converted = 0
    try:
        for item in image_items:
            output_path = output_root / f"{item.path.stem}.mp4"
            factory(item.path, output_path, duration_ms=item.duration_ms or 3000)
            converted += 1
    except Exception as exc:
        return {
            "image_conversion": "error",
            "converted_image_count": converted,
            "image_conversion_error": str(exc),
        }
    return {"image_conversion": "ready", "converted_image_count": converted}


def _is_forbidden_source(path: Path) -> bool:
    return path.suffix.lower() == ".url" or _looks_like_url_file(path)


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc```\x00\x00"
    b"\x00\x04\x00\x01\xf6\x178U\x00\x00\x00\x00IEND\xaeB`\x82"
)
