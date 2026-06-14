from pathlib import Path

import numpy as np
import pytest

from brainrot_guard.ingestion import scan_media_folder
from brainrot_guard.runtime import SegmentWindow
from brainrot_guard.tribe_adapter import (
    TribeAdapterError,
    TribeV2Predictor,
    generate_static_image_video,
    load_tribe_model,
)


def test_tribe_v2_predictor_caches_full_media_predictions(tmp_path: Path) -> None:
    media = _media(tmp_path, "story.txt", "counting story")
    model = FakeTribeModel(preds=np.vstack([np.zeros(20484), np.ones(20484)]).astype(np.float32))
    predictor = TribeV2Predictor(model_loader=lambda: model)

    first = predictor.predict_window(media, SegmentWindow(timestep=0, start_ms=0, end_ms=1000))
    second = predictor.predict_window(media, SegmentWindow(timestep=1, start_ms=1000, end_ms=2000))

    assert len(model.events_calls) == 1
    assert len(model.predict_calls) == 1
    assert model.events_calls[0]["text_path"] == media.path
    assert first.vertex_values.shape == (20484,)
    assert second.vertex_values.shape == (20484,)
    assert second.engagement >= first.engagement


def test_tribe_v2_predictor_selects_video_audio_and_text_event_inputs(tmp_path: Path) -> None:
    video = _media(tmp_path, "clip.mp4", b"video")
    audio = _media(tmp_path, "sound.mp3", b"audio")
    text = _media(tmp_path, "notes.md", "text")
    model = FakeTribeModel(preds=np.ones((1, 20484), dtype=np.float32))
    predictor = TribeV2Predictor(model_loader=lambda: model)

    for item in (video, audio, text):
        predictor.predict_window(item, SegmentWindow(timestep=0, start_ms=0, end_ms=1000))

    assert model.events_calls[0]["video_path"] == video.path
    assert model.events_calls[1]["audio_path"] == audio.path
    assert model.events_calls[2]["text_path"] == text.path


def test_tribe_v2_predictor_converts_image_media_to_static_video_input(tmp_path: Path) -> None:
    image = _media(tmp_path, "frame.png", b"\x89PNG\r\n\x1a\n")
    model = FakeTribeModel(preds=np.ones((1, 20484), dtype=np.float32))
    created = []

    def image_video_factory(media, output_path):
        created.append((media, output_path))
        output_path.parent.mkdir(parents=True)
        output_path.write_bytes(b"mp4")
        return output_path

    predictor = TribeV2Predictor(
        model_loader=lambda: model,
        image_video_dir=tmp_path / "image-videos",
        image_video_factory=image_video_factory,
    )

    predictor.predict_window(image, SegmentWindow(timestep=0, start_ms=0, end_ms=1000))

    assert created[0][0].id == image.id
    assert created[0][1].name == f"{image.id}.mp4"
    assert model.events_calls[0]["video_path"] == created[0][1]


def test_tribe_v2_predictor_fails_clearly_for_image_without_artifact_dir(tmp_path: Path) -> None:
    image = _media(tmp_path, "frame.png", b"\x89PNG\r\n\x1a\n")
    predictor = TribeV2Predictor(model_loader=lambda: FakeTribeModel())

    with pytest.raises(TribeAdapterError, match="image_video_dir"):
        predictor.predict_window(image, SegmentWindow(timestep=0, start_ms=0, end_ms=1000))


def test_tribe_v2_predictor_fails_clearly_when_requested_timestep_is_missing(tmp_path: Path) -> None:
    media = _media(tmp_path, "story.txt", "counting story")
    model = FakeTribeModel(preds=np.ones((1, 20484), dtype=np.float32))
    predictor = TribeV2Predictor(model_loader=lambda: model)

    with pytest.raises(TribeAdapterError, match="did not return timestep 3"):
        predictor.predict_window(media, SegmentWindow(timestep=3, start_ms=3000, end_ms=4000))


def test_load_tribe_model_prefers_public_tribev2_api() -> None:
    calls = []

    def importer(name, globals=None, locals=None, fromlist=(), level=0):
        assert name == "tribev2"
        assert fromlist == ("TribeModel",)

        class Module:
            class TribeModel:
                @staticmethod
                def from_pretrained(model_ref, cache_folder=None):
                    calls.append((model_ref, cache_folder))
                    return "model"

        return Module()

    model = load_tribe_model("Jessylg27/tribev2-lite-qv", Path("/tmp/cache"), importer=importer)

    assert model == "model"
    assert calls == [("Jessylg27/tribev2-lite-qv", Path("/tmp/cache"))]


def test_load_tribe_model_fails_clearly_when_package_is_missing() -> None:
    def importer(name, globals=None, locals=None, fromlist=(), level=0):
        raise ModuleNotFoundError("No module named 'tribev2'")

    with pytest.raises(TribeAdapterError, match="tribev2 package is not installed"):
        load_tribe_model("facebook/tribev2", None, importer=importer)


def test_generate_static_image_video_uses_ffmpeg_loop_command(tmp_path: Path) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    output = tmp_path / "out" / "frame.mp4"
    calls = []

    def runner(cmd, **kwargs):
        calls.append(cmd)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"mp4")

        class Result:
            returncode = 0
            stderr = ""

        return Result()

    result = generate_static_image_video(image, output, duration_ms=2500, runner=runner)

    assert result == output
    assert calls[0][:4] == ["ffmpeg", "-y", "-loop", "1"]
    assert "-t" in calls[0]
    assert "2.500" in calls[0]
    assert "max(2" in calls[0][calls[0].index("-vf") + 1]


def test_generate_static_image_video_reports_ffmpeg_failure(tmp_path: Path) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")

    def runner(cmd, **kwargs):
        class Result:
            returncode = 1
            stderr = "bad image"

        return Result()

    with pytest.raises(TribeAdapterError, match="ffmpeg failed"):
        generate_static_image_video(image, tmp_path / "out.mp4", runner=runner)


class FakeTribeModel:
    def __init__(self, preds=None) -> None:
        self.preds = preds if preds is not None else np.ones((1, 20484), dtype=np.float32)
        self.events_calls = []
        self.predict_calls = []

    def get_events_dataframe(self, **kwargs):
        self.events_calls.append(kwargs)
        return {"events": kwargs}

    def predict(self, *, events):
        self.predict_calls.append(events)
        return self.preds, []


def _media(tmp_path: Path, name: str, contents):
    media_dir = tmp_path / "media"
    media_dir.mkdir(exist_ok=True)
    path = media_dir / name
    if isinstance(contents, bytes):
        path.write_bytes(contents)
    else:
        path.write_text(contents, encoding="utf-8")
    return [item for item in scan_media_folder(media_dir) if item.path == path][0]
