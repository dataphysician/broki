from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import socket
import threading
import time
from typing import Any
from urllib.request import urlopen
import errno

from brainrot_guard.app import create_app
from brainrot_guard.demo import generate_demo_all
from brainrot_guard.repository import Repository


BrowserRunner = Callable[..., dict[str, Any]]


def validate_browser_review_console(
    *,
    repository: Repository,
    media_dir: Path | None = None,
    artifacts_dir: Path | None = None,
    media_id: str | None = None,
    screenshot_path: Path | None = None,
    duration_ms: int = 3000,
    runner: BrowserRunner | None = None,
) -> dict[str, Any]:
    repository.initialize()
    selected_media_id = media_id or _first_media_id(repository)
    bootstrapped_demo = False
    media_count = len(repository.list_media())
    if selected_media_id is None and media_dir is not None and artifacts_dir is not None:
        try:
            media_ids = generate_demo_all(repository, media_dir, artifacts_dir, duration_ms=duration_ms)
        except ValueError as exc:
            return {
                "ready": False,
                "browser_status": "not_checked",
                "bootstrapped_demo": False,
                "media_count": 0,
                "message": str(exc),
            }
        selected_media_id = media_ids[0]
        media_count = len(media_ids)
        bootstrapped_demo = True
    if selected_media_id is None:
        return {
            "ready": False,
            "browser_status": "not_checked",
            "bootstrapped_demo": False,
            "media_count": media_count,
            "message": "no media items found",
        }
    app = create_app(repository=repository, media_dir=media_dir)
    screenshot = screenshot_path or Path("/tmp/brainrot-guard-browser-review.png")
    try:
        result = (runner or _run_playwright_review)(
            app,
            media_id=selected_media_id,
            screenshot_path=screenshot,
        )
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("playwright"):
            return {
                "ready": False,
                "browser_status": "not_available",
                "message": "playwright is not installed; install the dev extra and browser binaries to run validate-browser",
            }
        raise
    except Exception as exc:
        if _browser_not_available(exc):
            return {
                "ready": False,
                "browser_status": "not_available",
                "message": f"playwright browser automation is not available: {exc}",
            }
        return {
            "ready": False,
            "browser_status": "error",
            "message": str(exc),
        }

    required = [
        "visible_media_stage",
        "visible_brain_frame",
        "visible_timeline_playhead",
        "visible_feedback_controls",
        "visible_skip_controls",
        "visible_proxy_label",
    ]
    missing = [name for name in required if not result.get(name)]
    screenshot_bytes = int(result.get("screenshot_bytes") or 0)
    ready = not missing and screenshot_bytes > 0 and result.get("browser_status") == "ready"
    return {
        **result,
        "ready": ready,
        "media_id": selected_media_id,
        "media_count": media_count,
        "bootstrapped_demo": bootstrapped_demo,
        "missing_visible_checks": missing,
        "message": "ready" if ready else "browser review console validation failed",
    }


def _run_playwright_review(app, *, media_id: str, screenshot_path: Path | None) -> dict[str, Any]:
    from playwright.sync_api import sync_playwright

    host = "127.0.0.1"
    port = _free_port(host)
    server, thread = _start_uvicorn(app, host=host, port=port)
    try:
        _wait_for_health(host, port)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={"width": 1280, "height": 820})
            page.goto(f"http://{host}:{port}/media/{media_id}", wait_until="networkidle")
            page.wait_for_selector("#brain-frame", state="visible", timeout=5000)
            screenshot_path = screenshot_path or Path("/tmp/brainrot-guard-browser-review.png")
            screenshot_path.parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(screenshot_path), full_page=True)
            result = {
                "browser_status": "ready",
                "visible_media_stage": page.locator("#media-stage").is_visible(),
                "visible_brain_frame": page.locator("#brain-frame").is_visible(),
                "visible_timeline_playhead": page.locator("#timeline-playhead").is_visible(),
                "visible_warning_state": page.locator("#warning-state").inner_text(),
                "visible_feedback_controls": (
                    page.locator("#approve-button").is_visible()
                    and page.locator("#disapprove-button").is_visible()
                ),
                "visible_skip_controls": (
                    page.locator("#skip-button").is_visible()
                    and page.locator("#skip-state").is_visible()
                ),
                "visible_proxy_label": page.get_by_text("TRIBE-derived neural response proxy").is_visible(),
                "screenshot_path": str(screenshot_path),
                "screenshot_bytes": screenshot_path.stat().st_size,
            }
            browser.close()
            return result
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def _start_uvicorn(app, *, host: str, port: int):
    import uvicorn

    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    return server, thread


def _wait_for_health(host: str, port: int) -> None:
    deadline = time.monotonic() + 10
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urlopen(f"http://{host}:{port}/health", timeout=0.5) as response:
                if response.status == 200:
                    return
        except Exception as exc:
            last_error = exc
            time.sleep(0.05)
    raise RuntimeError(f"browser validation server did not become ready: {last_error}")


def _free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _first_media_id(repository: Repository) -> str | None:
    items = repository.list_media()
    return items[0].id if items else None


def _browser_not_available(exc: Exception) -> bool:
    if isinstance(exc, OSError) and getattr(exc, "errno", None) == errno.EPERM:
        return True
    text = str(exc).lower()
    return (
        "executable doesn't exist" in text
        or "playwright install" in text
        or "browser has been closed" in text
        or "operation not permitted" in text
    )


__all__ = ["validate_browser_review_console"]
