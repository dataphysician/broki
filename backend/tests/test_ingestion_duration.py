from pathlib import Path

from brainrot_guard.ingestion.scanner import _duration_ms
from brainrot_guard.models import MediaKind


def test_video_duration_uses_local_ffprobe_output(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fixture")
    calls = []

    def runner(cmd, **kwargs):
        calls.append(cmd)

        class Result:
            returncode = 0
            stdout = '{"format": {"duration": "2.75"}}'
            stderr = ""

        return Result()

    duration = _duration_ms(video, MediaKind.VIDEO, probe_runner=runner)

    assert duration == 2750
    assert calls[0][:5] == ["ffprobe", "-v", "error", "-show_entries", "format=duration"]
    assert str(video) in calls[0]


def test_non_wav_audio_duration_uses_local_ffprobe_output(tmp_path: Path) -> None:
    audio = tmp_path / "track.mp3"
    audio.write_bytes(b"fixture")

    def runner(cmd, **kwargs):
        class Result:
            returncode = 0
            stdout = '{"format": {"duration": "1.234"}}'
            stderr = ""

        return Result()

    assert _duration_ms(audio, MediaKind.AUDIO, probe_runner=runner) == 1234


def test_duration_probe_failure_keeps_duration_unknown(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fixture")

    def runner(cmd, **kwargs):
        class Result:
            returncode = 1
            stdout = ""
            stderr = "bad file"

        return Result()

    assert _duration_ms(video, MediaKind.VIDEO, probe_runner=runner) is None


def test_static_media_duration_remains_fixed_without_ffprobe(tmp_path: Path) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")

    def runner(cmd, **kwargs):
        raise AssertionError("static image duration should not call ffprobe")

    assert _duration_ms(image, MediaKind.IMAGE, probe_runner=runner) == 3000
