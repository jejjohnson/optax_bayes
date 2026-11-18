"""Draw optax_bayes's icon and logo: ``icon.svg`` and ``logo-{light,dark}.svg``.

The icon is the Bayesian learning rule: the posterior q(θ) sharpens as
training proceeds. Standard library plus loguru; the tile and wordmark
follow the other repos of the stack (radtrax, gaussx, kernellib).
Run from the repository root::

    uv run --no-project --with loguru python docs/assets/make_assets.py
"""

import math
from pathlib import Path

from loguru import logger


HERE = Path(__file__).resolve().parent

NAME = "optax_bayes"
WORDMARK = ("optax", "_bayes")
LOGO_WIDTH = 700
# Gold: the posterior read out of the optimizer state.
GRADIENT = ("#FCD34D", "#CA8A04")
THEMES = {
    "light": {"ink": "#1F2937", "accent": "#CA8A04"},
    "dark": {"ink": "#E5E7EB", "accent": "#FACC15"},
}
SANS = "ui-sans-serif, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
WHITE = "#FFFFFF"


def svg(width: int, height: int, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{NAME}">\n'
        f"<title>{NAME}</title>\n{body}\n</svg>\n"
    )


def tile(gradient_id: str) -> str:
    start, stop = GRADIENT
    return (
        f'<defs><linearGradient id="{gradient_id}" x1="0" y1="0" x2="1" y2="1">'
        f'<stop offset="0" stop-color="{start}"/>'
        f'<stop offset="1" stop-color="{stop}"/></linearGradient></defs>'
        '<rect x="4" y="4" width="120" height="120" rx="26" '
        f'fill="url(#{gradient_id})"/>'
    )


def polyline(
    points: list[tuple[float, float]], *, width: float, opacity: float = 1.0
) -> str:
    commands = [
        f"{'M' if i == 0 else 'L'}{x:.1f} {y:.1f}" for i, (x, y) in enumerate(points)
    ]
    return (
        f'<path d="{" ".join(commands)}" fill="none" stroke="{WHITE}" '
        f'stroke-width="{width}" stroke-opacity="{opacity}" '
        'stroke-linecap="round" stroke-linejoin="round"/>'
    )


def dot(x: float, y: float, radius: float) -> str:
    return f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{WHITE}"/>'


def glyph() -> str:
    """Three posteriors q(θ), each sharper than the last, as training proceeds."""

    def bell(mean: float, scale: float, height: float) -> list[tuple[float, float]]:
        points = []
        for step in range(101):
            x = 14 + 100 * step / 100
            density = math.exp(-0.5 * ((x - mean) / scale) ** 2)
            points.append((x, 92 - height * density))
        return points

    body = polyline([(14, 92), (114, 92)], width=2.4, opacity=0.4)
    body += polyline(bell(46, 20, 22), width=2.6, opacity=0.35)
    body += polyline(bell(62, 12, 38), width=3, opacity=0.6)
    body += polyline(bell(76, 6.5, 60), width=4.2)
    body += polyline([(40, 26), (60, 26)], width=3)
    body += f'<path d="M58 20 L68 26 L58 32 Z" fill="{WHITE}"/>'
    return body


def icon() -> str:
    return svg(128, 128, tile(f"{NAME}-icon") + glyph())


def logo(theme: str) -> str:
    colours = THEMES[theme]
    head, tail = WORDMARK
    mark = f'<g transform="translate(6 6)">{tile(f"{NAME}-{theme}")}{glyph()}</g>'
    wordmark = (
        f'<text x="154" y="92" font-family="{SANS}" font-size="72" font-weight="700" '
        f'letter-spacing="-1.5" fill="{colours["ink"]}">{head}'
        f'<tspan fill="{colours["accent"]}">{tail}</tspan></text>'
    )
    return svg(LOGO_WIDTH, 140, mark + wordmark)


def main() -> None:
    outputs = {"icon.svg": icon()}
    for theme in THEMES:
        outputs[f"logo-{theme}.svg"] = logo(theme)
    for filename, content in outputs.items():
        (HERE / filename).write_text(content)
        logger.info("wrote {}", HERE / filename)


if __name__ == "__main__":
    main()
