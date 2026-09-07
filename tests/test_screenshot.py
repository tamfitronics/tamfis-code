import asyncio

import pytest
from click import Group
from click.testing import CliRunner
from PIL import Image

from tamfis_code.screenshot import (
    ScreenshotTaker,
    add_screenshot_command,
    screenshot_cli,
)


def test_reusable_screenshot_command_registers_without_name_error():
    cli = add_screenshot_command(Group("test"))

    result = CliRunner().invoke(cli, ["screenshot", "--help"])

    assert result.exit_code == 0, result.output
    assert "URL_OR_PATH" in result.output


def test_screenshot_command_returns_failure_exit_code(monkeypatch):
    cli = add_screenshot_command(Group("test"))
    monkeypatch.setattr(
        ScreenshotTaker,
        "take_screenshot",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("capture failed")),
    )

    result = CliRunner().invoke(cli, ["screenshot", "https://example.test"])

    assert result.exit_code != 0
    assert "capture failed" in result.output


def test_local_image_uses_pillow_without_a_playwright_browser(tmp_path, monkeypatch):
    source = tmp_path / "source.png"
    Image.new("RGB", (4, 3), "red").save(source)
    output_dir = tmp_path / "output"
    taker = ScreenshotTaker(output_dir=output_dir)
    monkeypatch.setattr(taker, "backend", "playwright")

    result = taker.take_screenshot(source.as_posix(), filename="copy.png")

    assert result == output_dir / "copy.png"
    with Image.open(result) as copied:
        assert copied.size == (4, 3)


def test_explicit_backend_is_not_overridden_for_a_local_image(tmp_path, monkeypatch):
    source = tmp_path / "source.png"
    Image.new("RGB", (4, 3), "red").save(source)
    taker = ScreenshotTaker(output_dir=tmp_path / "output")
    expected = tmp_path / "output" / "explicit.png"
    called = []

    def fake_playwright(url_or_path, output_path, options):
        called.append((url_or_path, output_path, options))
        return output_path

    monkeypatch.setattr(taker, "_take_screenshot_playwright", fake_playwright)
    result = taker.take_screenshot(
        source.as_posix(), filename="explicit.png", backend="playwright",
    )

    assert result == expected
    assert called[0][0] == source.as_posix()


def test_async_wrapper_dispatches_sync_capture_to_a_worker(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ScreenshotTaker, "_detect_backend", lambda self: "playwright")
    dispatched = []

    async def fake_to_thread(function, *args, **kwargs):
        dispatched.append((function, args, kwargs))
        return tmp_path / "screenshot.png"

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

    result = asyncio.run(screenshot_cli("https://example.test"))

    assert result == tmp_path / "screenshot.png"
    assert dispatched[0][1] == ("https://example.test",)
