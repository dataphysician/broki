from pathlib import Path
import sys

import brainrot_guard.__main__ as cli
from brainrot_guard.analysis_factory import build_analysis_service_from_env, build_vlm_service_from_env
from brainrot_guard.repository import Repository
from brainrot_guard.runtime import AnalysisService
from brainrot_guard.vlm import GatedVLMService


def test_analysis_service_factory_is_disabled_by_default(tmp_path: Path) -> None:
    repo = Repository(tmp_path / "state.sqlite3")

    service = build_analysis_service_from_env(repo, {})

    assert service is None


def test_analysis_service_factory_requires_artifact_dir_when_enabled(tmp_path: Path) -> None:
    repo = Repository(tmp_path / "state.sqlite3")

    try:
        build_analysis_service_from_env(repo, {"BRAINROT_GUARD_ENABLE_LIVE_ANALYSIS": "1"})
    except RuntimeError as exc:
        assert "BRAINROT_GUARD_ARTIFACTS_DIR" in str(exc)
    else:
        raise AssertionError("live analysis must require an artifact directory")


def test_analysis_service_factory_wires_tribe_runtime_and_plotbrain_renderer(tmp_path: Path) -> None:
    repo = Repository(tmp_path / "state.sqlite3")
    artifacts_dir = tmp_path / "artifacts"

    service = build_analysis_service_from_env(
        repo,
        {
            "BRAINROT_GUARD_ENABLE_LIVE_ANALYSIS": "1",
            "BRAINROT_GUARD_ARTIFACTS_DIR": str(artifacts_dir),
            "BRAINROT_GUARD_TRIBE_STEP_MS": "500",
        },
        renderer_factory=lambda env: object(),
    )

    assert isinstance(service, AnalysisService)
    assert service.repository is repo
    assert service.runtime.npz_dir == artifacts_dir / "npz"
    assert service.runtime.step_ms == 500
    assert service.runtime.predictor.image_video_dir == artifacts_dir / "image-videos"
    assert service.frame_service.frames_dir == artifacts_dir / "frames"
    assert service.vlm_service is None


def test_vlm_service_factory_is_disabled_by_default() -> None:
    service = build_vlm_service_from_env({})

    assert service is None


def test_vlm_service_factory_requires_credentials_when_enabled() -> None:
    try:
        build_vlm_service_from_env({"BRAINROT_GUARD_ENABLE_VLM": "1", "VLM_PROVIDER": "gemini"})
    except RuntimeError as exc:
        assert "GEMINI_API_KEY" in str(exc)
    else:
        raise AssertionError("enabled VLM must require provider credentials")


def test_vlm_service_factory_wires_configured_decomposer() -> None:
    service = build_vlm_service_from_env(
        {
            "BRAINROT_GUARD_ENABLE_VLM": "1",
            "VLM_PROVIDER": "xai",
            "XAI_API_KEY": "secret",
            "BRAINROT_GUARD_VLM_ENGAGEMENT_THRESHOLD": "0.82",
        },
        decomposer_factory=lambda config: object(),
    )

    assert isinstance(service, GatedVLMService)
    assert service.provider == "xai"
    assert service.engagement_threshold == 0.82


def test_cli_serve_passes_live_analysis_service_to_app(monkeypatch, tmp_path: Path) -> None:
    calls = {}

    class FakeUvicorn:
        @staticmethod
        def run(app, *, host, port):
            calls["uvicorn"] = {"app": app, "host": host, "port": port}

    def fake_build_analysis_service(repository, env):
        calls["analysis_env"] = env
        calls["repository"] = repository
        return "analysis-service"

    def fake_create_app(**kwargs):
        calls["app_kwargs"] = kwargs
        return "app"

    monkeypatch.setitem(sys.modules, "uvicorn", FakeUvicorn)
    monkeypatch.setattr(cli, "build_analysis_service_from_env", fake_build_analysis_service)
    monkeypatch.setattr(cli, "create_app", fake_create_app)

    result = cli.main(
        [
            "serve",
            "--db-path",
            str(tmp_path / "state.sqlite3"),
            "--media-dir",
            str(tmp_path),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--enable-live-analysis",
            "--host",
            "127.0.0.1",
            "--port",
            "8999",
        ]
    )

    assert result == 0
    assert calls["analysis_env"]["BRAINROT_GUARD_ENABLE_LIVE_ANALYSIS"] == "1"
    assert calls["analysis_env"]["BRAINROT_GUARD_ARTIFACTS_DIR"] == str(tmp_path / "artifacts")
    assert calls["app_kwargs"]["analysis_service"] == "analysis-service"
    assert calls["uvicorn"] == {"app": "app", "host": "127.0.0.1", "port": 8999}
