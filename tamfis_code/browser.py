"""Portable, workspace-independent Playwright browser tool."""
from __future__ import annotations

import asyncio
import ipaddress
import shutil
import socket
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .config import CONFIG_DIR


def _validate_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("browser requires an absolute http(s) URL")
    host = parsed.hostname.lower()
    if host == "localhost" or host.endswith(".localhost"):
        raise ValueError("browser blocks loopback/private destinations")
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(host, parsed.port or 443)}
    except socket.gaierror as exc:
        raise ValueError(f"could not resolve browser host: {host}") from exc
    for raw in addresses:
        address = ipaddress.ip_address(raw)
        if not address.is_global:
            raise ValueError("browser blocks loopback/private destinations")


class PortableBrowserTool:
    """Navigate, inspect and interact using a fresh headless browser context."""

    async def execute_async(self, **parameters: Any) -> dict[str, Any]:
        url = str(parameters.get("url") or "")
        action = str(parameters.get("action") or "navigate")
        _validate_public_url(url)
        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:
            raise RuntimeError("Browser support requires `pip install playwright`") from exc

        artifacts = CONFIG_DIR / "artifacts" / "browser"
        artifacts.mkdir(parents=True, exist_ok=True)
        timeout_ms = 30_000
        async with async_playwright() as playwright:
            launch_options: dict[str, Any] = {"headless": True}
            system_chromium = next(
                (path for name in ("chromium", "chromium-browser", "google-chrome")
                 if (path := shutil.which(name))),
                None,
            )
            if system_chromium:
                launch_options["executable_path"] = system_chromium
            browser = await playwright.chromium.launch(**launch_options)
            try:
                page = await browser.new_page(viewport={
                    "width": int(parameters.get("viewport_width") or 1440),
                    "height": int(parameters.get("viewport_height") or 900),
                })
                page.set_default_timeout(timeout_ms)
                response = await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                wait_ms = min(max(int(parameters.get("wait_after_load_ms") or 0), 0), 5000)
                if wait_ms:
                    await page.wait_for_timeout(wait_ms)
                wait_selector = parameters.get("wait_for_selector")
                if wait_selector:
                    await page.wait_for_selector(str(wait_selector))

                selector = str(parameters.get("selector") or "")
                if action == "click":
                    if not selector:
                        raise ValueError("click requires selector")
                    await page.click(selector)
                elif action == "fill_form":
                    form_data = parameters.get("form_data") or {}
                    for field_selector, value in form_data.items():
                        await page.fill(str(field_selector), str(value))
                    if parameters.get("submit_selector"):
                        await page.click(str(parameters["submit_selector"]))
                elif action == "scroll":
                    await page.evaluate("value => window.scrollBy(0, value)", int(parameters.get("scroll_y") or 700))
                elif action == "screenshot":
                    raw_name = Path(str(parameters.get("screenshot_name") or "browser.png")).name
                    if not raw_name.lower().endswith(".png"):
                        raw_name += ".png"
                    output = artifacts / raw_name
                    if parameters.get("screenshot_selector"):
                        await page.locator(str(parameters["screenshot_selector"])).screenshot(path=str(output))
                    else:
                        await page.screenshot(path=str(output), full_page=bool(parameters.get("full_page", True)))
                    return {
                        "success": True, "url": page.url, "screenshot_path": str(output),
                        "screenshot_url": str(output), "status_code": getattr(response, "status", None),
                    }
                elif action not in {"navigate", "extract"}:
                    raise ValueError(f"unsupported browser action: {action}")

                content = await (page.locator(selector).inner_text() if selector else page.locator("body").inner_text())
                return {
                    "success": True, "url": page.url, "title": await page.title(),
                    "content": content[:50_000], "status_code": getattr(response, "status", None),
                }
            finally:
                await browser.close()

    def execute(self, **parameters: Any) -> dict[str, Any]:
        return asyncio.run(self.execute_async(**parameters))
