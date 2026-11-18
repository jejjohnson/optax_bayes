"""Render every diagram in this folder from its HTML source to a 2x PNG.

The HTML files are the editable source; the PNGs next to them are what
the READMEs and docs embed. Re-run after editing any ``*.html`` here::

    uv run --no-project --with playwright --with loguru \\
        python docs/assets/diagrams/render.py

Playwright uses its own Chromium (``playwright install chromium``); set
``CHROMIUM_PATH`` to point it at another Chromium build instead.
Pass file names to render a subset (``render.py stack-overview.html``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from loguru import logger
from playwright.sync_api import Browser, Playwright, sync_playwright


HERE = Path(__file__).resolve().parent


def _launch(p: Playwright) -> Browser:
    path = os.environ.get("CHROMIUM_PATH")
    if path:
        return p.chromium.launch(executable_path=path)
    return p.chromium.launch()


def render(sources: list[Path]) -> None:
    """Screenshot each source's ``#diagram`` element into ``<stem>.png``.

    Args:
        sources: The HTML diagram files to render.
    """
    with sync_playwright() as p:
        browser = _launch(p)
        page = browser.new_page(
            device_scale_factor=2, viewport={"width": 2200, "height": 1600}
        )
        for src in sources:
            page.goto(src.as_uri())
            page.wait_for_function("document.body.dataset.ready === '1'")
            page.evaluate("document.fonts.ready")
            out = src.with_suffix(".png")
            page.locator("#diagram").screenshot(path=str(out), omit_background=True)
            logger.info("{} -> {}", src.name, out.name)
        browser.close()


if __name__ == "__main__":
    names = sys.argv[1:]
    files = [HERE / n for n in names] if names else sorted(HERE.glob("[!_]*.html"))
    render(files)
