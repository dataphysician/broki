import json
from pathlib import Path

import httpx

from brainrot_guard.__main__ import main
from brainrot_guard.app import create_app
from brainrot_guard.readiness import (
    validate_hardware_target,
    validate_local_media_folder,
    validate_local_tools,
    validate_tribe_plotbrain,
    validate_vlm_provider,
)
from brainrot_guard.repository import Repository


def test_tribe_plotbrain_validation_is_opt_in_by_default() -> None:
    report = validate_tribe_plotbrain({})

    assert report["enabled"] is False
    assert report["ready"] is False
    assert "opt-in" in report["message"]


def test_forced_tribe_plotbrain_validation_requires_local_paths(tmp_path: Path) -> None:
    report = validate_tribe_plotbrain({"BRAINROT_GUARD_TRIBE_CKPT": str(tmp_path / "missing")}, force=True)

    assert report["enabled"] is True
    assert report["ready"] is False
    assert "BRAINROT_GUARD_FSAVERAGE5_DIR" in report["message"]


def test_forced_tribe_plotbrain_validation_reports_ready_with_assets_and_loaders(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    fsaverage5 = tmp_path / "fsaverage5"
    checkpoint.mkdir()
    fsaverage5.mkdir()

    report = validate_tribe_plotbrain(
        {
            "BRAINROT_GUARD_TRIBE_CKPT": str(checkpoint),
            "BRAINROT_GUARD_FSAVERAGE5_DIR": str(fsaverage5),
        },
        force=True,
        tribe_loader=lambda path: object(),
        plotbrain_loader=lambda: object(),
    )

    assert report["enabled"] is True
    assert report["ready"] is True
    assert report["checkpoint_dir"] == str(checkpoint)
    assert report["fsaverage5_dir"] == str(fsaverage5)
    assert report["plotbrain_smoke_render"] == "not_checked"


def test_forced_tribe_plotbrain_validation_can_smoke_render_plotbrain_frame(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    fsaverage5 = tmp_path / "fsaverage5"
    checkpoint.mkdir()
    fsaverage5.mkdir()
    plotter = FakeSmokePlotBrain()

    report = validate_tribe_plotbrain(
        {
            "BRAINROT_GUARD_TRIBE_CKPT": str(checkpoint),
            "BRAINROT_GUARD_FSAVERAGE5_DIR": str(fsaverage5),
        },
        force=True,
        smoke_render=True,
        tribe_loader=lambda path: object(),
        plotbrain_loader=lambda: plotter,
    )

    assert report["ready"] is True
    assert report["plotbrain_smoke_render"] == "ready"
    assert plotter.neuro_shape == (1, 20484)
    assert plotter.timestamps == [0.0]


def test_forced_tribe_plotbrain_validation_reports_smoke_render_failure(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    fsaverage5 = tmp_path / "fsaverage5"
    checkpoint.mkdir()
    fsaverage5.mkdir()

    report = validate_tribe_plotbrain(
        {
            "BRAINROT_GUARD_TRIBE_CKPT": str(checkpoint),
            "BRAINROT_GUARD_FSAVERAGE5_DIR": str(fsaverage5),
        },
        force=True,
        smoke_render=True,
        tribe_loader=lambda path: object(),
        plotbrain_loader=lambda: object(),
    )

    assert report["ready"] is False
    assert report["plotbrain_smoke_render"] == "error"
    assert "plot_timesteps" in report["message"]


def test_vlm_live_validation_is_opt_in_by_default() -> None:
    report = validate_vlm_provider({"VLM_PROVIDER": "gemini", "GEMINI_API_KEY": "secret"})

    assert report["enabled"] is False
    assert report["ready"] is False
    assert "opt-in" in report["message"]


def test_forced_vlm_validation_reports_missing_provider_credentials() -> None:
    report = validate_vlm_provider({"VLM_PROVIDER": "gemini"}, force=True)

    assert report["enabled"] is True
    assert report["ready"] is False
    assert "GEMINI_API_KEY" in report["message"]


def test_forced_vlm_validation_reports_configured_provider_ready() -> None:
    report = validate_vlm_provider({"VLM_PROVIDER": "xai", "XAI_API_KEY": "secret"}, force=True)

    assert report["enabled"] is True
    assert report["ready"] is True
    assert report["provider"] == "xai"
    assert report["provider_probe"] == "not_checked"


def test_forced_vlm_validation_can_probe_provider_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer secret"
        assert "TRIBE output" in request.read().decode("utf-8")
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": '{"theme":"probe fixture","pacing_score":0.6,'
                            '"scene_change_cadence_hz":0.4,"contrast_score":0.5,'
                            '"sound_effect_density":0.2,"educational_value":0.7,'
                            '"emotional_hook_score":0.4,"novelty_score":0.3,'
                            '"repetition_score":0.2,"risk_score":0.25,'
                            '"risk_rationale":"probe normalized"}'
                        }
                    }
                ]
            },
        )

    report = validate_vlm_provider(
        {"VLM_PROVIDER": "xai", "XAI_API_KEY": "secret"},
        force=True,
        probe=True,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert report["ready"] is True
    assert report["provider_probe"] == "ready"
    assert report["probe_theme"] == "probe fixture"
    assert report["probe_risk_score"] == 0.25


def test_forced_vlm_validation_reports_provider_probe_error() -> None:
    report = validate_vlm_provider(
        {"VLM_PROVIDER": "gemini", "GEMINI_API_KEY": "secret"},
        force=True,
        probe=True,
        client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(503, text="down"))),
    )

    assert report["ready"] is False
    assert report["provider_probe"] == "error"
    assert "503" in report["message"]


def test_local_tools_validation_is_opt_in_by_default() -> None:
    report = validate_local_tools({})

    assert report["enabled"] is False
    assert report["ready"] is False
    assert "opt-in" in report["message"]


def test_forced_local_tools_validation_runs_image_video_smoke() -> None:
    calls = []

    def validator():
        calls.append("ffmpeg")

    report = validate_local_tools({}, force=True, image_video_validator=validator)

    assert calls == ["ffmpeg"]
    assert report["enabled"] is True
    assert report["ready"] is True
    assert report["ffmpeg_image_video"] == "ready"


def test_forced_local_tools_validation_reports_ffmpeg_error() -> None:
    def validator():
        raise RuntimeError("ffmpeg missing")

    report = validate_local_tools({}, force=True, image_video_validator=validator)

    assert report["enabled"] is True
    assert report["ready"] is False
    assert report["ffmpeg_image_video"] == "error"
    assert "ffmpeg missing" in report["message"]


def test_hardware_validation_is_opt_in_by_default() -> None:
    report = validate_hardware_target({})

    assert report["enabled"] is False
    assert report["ready"] is False
    assert "opt-in" in report["message"]


def test_forced_hardware_validation_accepts_ampere_16gb_gpu() -> None:
    def probe():
        return "NVIDIA RTX A4000, 16376, 8.6\n"

    report = validate_hardware_target({}, force=True, gpu_probe=probe)

    assert report["ready"] is True
    assert report["gpu_count"] == 1
    assert report["gpus"][0]["ampere_or_newer"] is True
    assert report["gpus"][0]["memory_total_mb"] == 16376


def test_forced_hardware_validation_reports_insufficient_vram() -> None:
    def probe():
        return "NVIDIA RTX 3060, 12288, 8.6\n"

    report = validate_hardware_target({}, force=True, gpu_probe=probe)

    assert report["ready"] is False
    assert report["gpus"][0]["memory_ok"] is False
    assert "16GB" in report["message"]


def test_forced_hardware_validation_reports_missing_nvidia_smi() -> None:
    def probe():
        raise FileNotFoundError("nvidia-smi")

    report = validate_hardware_target({}, force=True, gpu_probe=probe)

    assert report["ready"] is False
    assert report["gpu_count"] == 0
    assert "nvidia-smi" in report["message"]


def test_local_media_folder_validation_accepts_mixed_static_media(tmp_path: Path) -> None:
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    (media_dir / "lesson.txt").write_text("counting by twos", encoding="utf-8")
    (media_dir / "frame.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    artifacts_dir = tmp_path / "artifacts"
    conversions = []

    def image_video_factory(image_path: Path, output_path: Path, *, duration_ms: int):
        conversions.append((image_path.name, output_path.name, duration_ms))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"mp4")
        return output_path

    report = validate_local_media_folder(
        media_dir,
        artifacts_dir=artifacts_dir,
        validate_image_conversion=True,
        image_video_factory=image_video_factory,
    )

    assert report["ready"] is True
    assert report["media_count"] == 2
    assert report["kinds"] == {"image": 1, "text": 1}
    assert report["known_duration_count"] == 2
    assert report["image_conversion"] == "ready"
    assert conversions == [("frame.png", "frame.mp4", 3000)]


def test_local_media_folder_validation_rejects_youtube_link_files(tmp_path: Path) -> None:
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    (media_dir / "lesson.txt").write_text("counting by twos", encoding="utf-8")
    (media_dir / "youtube.md").write_text("https://youtube.com/watch?v=abc", encoding="utf-8")

    report = validate_local_media_folder(media_dir)

    assert report["ready"] is False
    assert report["forbidden_source_count"] == 1
    assert report["forbidden_sources"] == ["youtube.md"]
    assert "YouTube" in report["message"]


def test_local_media_folder_validation_reports_unknown_audio_video_durations(tmp_path: Path) -> None:
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    (media_dir / "sound.mp3").write_bytes(b"not a real mp3")

    report = validate_local_media_folder(media_dir)

    assert report["ready"] is False
    assert report["media_count"] == 1
    assert report["unknown_duration_count"] == 1
    assert report["unknown_duration_media"] == ["sound.mp3"]
    assert "duration" in report["message"]


def test_cli_validate_commands_print_json(capsys) -> None:
    assert main(["validate-live"]) == 0
    tribe = json.loads(capsys.readouterr().out)
    assert tribe["enabled"] is False

    assert main(["validate-vlm-live"]) == 0
    vlm = json.loads(capsys.readouterr().out)
    assert vlm["enabled"] is False

    assert main(["validate-local"]) == 0
    local = json.loads(capsys.readouterr().out)
    assert local["enabled"] is False

    assert main(["validate-hardware"]) == 0
    hardware = json.loads(capsys.readouterr().out)
    assert hardware["enabled"] is False


def test_cli_validate_media_prints_json(tmp_path: Path, capsys) -> None:
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    (media_dir / "lesson.txt").write_text("counting by twos", encoding="utf-8")

    assert main(["validate-media", "--media-dir", str(media_dir)]) == 0
    report = json.loads(capsys.readouterr().out)

    assert report["ready"] is True
    assert report["media_count"] == 1
    assert report["kinds"] == {"text": 1}


def test_api_exposes_readiness_reports(tmp_path: Path) -> None:
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    (media_dir / "lesson.txt").write_text("counting by twos", encoding="utf-8")
    app = create_app(
        repository=Repository(tmp_path / "state.sqlite3"),
        media_dir=media_dir,
        environ={"VLM_PROVIDER": "gemini"},
    )
    tribe_endpoint = _endpoint(app, "/api/config/tribe/live", {"GET"})
    vlm_endpoint = _endpoint(app, "/api/config/vlm/live", {"GET"})
    local_endpoint = _endpoint(app, "/api/config/local/live", {"GET"})
    hardware_endpoint = _endpoint(app, "/api/config/hardware/local", {"GET"})
    media_endpoint = _endpoint(app, "/api/config/media/local", {"GET"})

    assert tribe_endpoint()["enabled"] is False
    assert vlm_endpoint()["provider"] == "gemini"
    assert local_endpoint()["enabled"] is False
    assert hardware_endpoint()["enabled"] is False
    assert media_endpoint()["ready"] is True
    assert media_endpoint()["media_count"] == 1


def _endpoint(app, path: str, methods: set[str]):
    for route in app.routes:
        if getattr(route, "path", None) == path and getattr(route, "methods", set()) == methods:
            return route.endpoint
    raise AssertionError(f"route not found: {methods} {path}")


class FakeSmokePlotBrain:
    def __init__(self) -> None:
        self.neuro_shape = None
        self.timestamps = None

    def plot_timesteps(self, neuro, *, timestamps, **kwargs):
        self.neuro_shape = neuro.shape
        self.timestamps = timestamps
        return FakeSmokeFigure()


class FakeSmokeFigure:
    def savefig(self, output_path: Path, **kwargs) -> None:
        output_path.write_bytes(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc```\x00\x00"
            b"\x00\x04\x00\x01\xf6\x178U\x00\x00\x00\x00IEND\xaeB`\x82"
        )
