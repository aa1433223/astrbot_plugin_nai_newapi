"""Local Playwright renderers for chat cards.

AstrBot's remote HTML renderer is useful for some built-in cards, but it is not
available in every deployment. These self-contained cards render locally and
are sent as normal AstrBot image components.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

_BROWSER_ARGS = (
    "--no-sandbox",
    "--disable-setuid-sandbox",
    "--disable-dev-shm-usage",
    "--disable-gpu",
    "--disable-crash-reporter",
    "--noerrdialogs",
)


def _find_chromium_executable() -> str | None:
    """Prefer a system Chrome when one exists, then let Playwright choose."""

    configured = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH", "").strip()
    candidates = [
        configured,
        "/usr/bin/google-chrome-stable",
        "/usr/bin/google-chrome",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def _render_template(template_path: Path, context: dict[str, Any]) -> str:
    environment = Environment(
        loader=FileSystemLoader(str(template_path.parent)),
        autoescape=select_autoescape(["html", "xml"]),
        undefined=StrictUndefined,
    )
    return environment.get_template(template_path.name).render(**context)


def _cleanup_old_files(
    output_dir: Path,
    max_age_seconds: int = 3600,
    patterns: tuple[str, ...] = ("style-gallery-*.jpg", "drawing-help-*.png"),
) -> None:
    cutoff = time.time() - max_age_seconds
    try:
        for pattern in patterns:
            for path in output_dir.glob(pattern):
                try:
                    if path.stat().st_mtime < cutoff:
                        path.unlink()
                except OSError:
                    continue
    except OSError:
        return


async def _render_html_image(
    template_path: Path,
    context: dict[str, Any],
    output_path: Path,
    *,
    viewport: dict[str, int],
    image_type: str,
    quality: int | None = None,
    wait_for_images: bool = False,
) -> Path:
    """Render one self-contained HTML document with the local browser."""
    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:  # pragma: no cover - depends on deployment
        raise RuntimeError(
            "未安装 Playwright，请在 AstrBot 使用的 Python 环境中安装 playwright"
        ) from exc

    rendered_html = _render_template(template_path, context)
    executable_path = _find_chromium_executable()
    async with async_playwright() as playwright:
        launch_kwargs: dict[str, Any] = {
            "headless": True,
            "args": list(_BROWSER_ARGS),
        }
        if executable_path:
            launch_kwargs["executable_path"] = executable_path

        browser = await asyncio.wait_for(
            playwright.chromium.launch(**launch_kwargs), timeout=20
        )
        try:
            page = await browser.new_page(
                viewport=viewport,
                device_scale_factor=1,
            )
            await page.set_content(rendered_html, wait_until="load", timeout=15000)
            if wait_for_images:
                await page.wait_for_function(
                    """() => Array.from(document.images).every((img) => img.complete)""",
                    timeout=15000,
                )
            screenshot_kwargs: dict[str, Any] = {
                "path": str(output_path),
                "full_page": True,
                "type": image_type,
            }
            if quality is not None:
                screenshot_kwargs["quality"] = quality
            await page.screenshot(**screenshot_kwargs)
        finally:
            await browser.close()

    return output_path


async def render_style_gallery_image(
    template_path: Path,
    context: dict[str, Any],
    *,
    output_dir: Path | None = None,
) -> Path:
    """Render a complete gallery HTML document to a JPEG using Playwright."""

    if output_dir is None:
        output_dir = Path(tempfile.gettempdir()) / "astrbot-novelai-style-gallery"
    output_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_old_files(output_dir)
    output_path = output_dir / f"style-gallery-{time.time_ns()}.jpg"
    return await _render_html_image(
        template_path,
        context,
        output_path,
        viewport={"width": 1200, "height": 800},
        image_type="jpeg",
        quality=88,
        wait_for_images=True,
    )


async def render_drawing_help_image(
    template_path: Path,
    context: dict[str, Any],
    *,
    output_dir: Path | None = None,
) -> Path:
    """Render the drawing command help as a PNG for chat delivery."""
    if output_dir is None:
        output_dir = Path(tempfile.gettempdir()) / "astrbot-novelai-drawing-help"
    output_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_old_files(output_dir)
    output_path = output_dir / f"drawing-help-{time.time_ns()}.png"
    return await _render_html_image(
        template_path,
        context,
        output_path,
        viewport={"width": 1200, "height": 900},
        image_type="png",
    )
