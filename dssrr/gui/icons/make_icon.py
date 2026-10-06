"""Generate the DSSRR application icon (SVG master + PNG/ICO raster exports).

The mark encodes what DSSRR does: a seismic time series whose middle span is
missing, reconstructed from the healthy reference segments on both sides.  The
two reference spans are drawn in white, the rebuilt span in accent blue, and a
pair of faint ticks brackets the gap that was filled.

Design notes (deliberately kept "clean and unfussy" per the project brief):

* flat solid fills only -- no gradients, so the mark stays crisp at 16 px;
* one geometry source (``_wave_points``) drives both the SVG and the raster
  exports, so the vector and bitmap versions can never drift apart;
* the accent blue ``#4CC2FF`` is a light tint of the GUI brand accent
  (``#2563EB``), which keeps the icon on-brand while staying legible on navy.

Usage::

    python -m dssrr.gui.icons.make_icon          # writes next to this file
    python dssrr/gui/icons/make_icon.py

Requires PyQt5 (QtSvg) for rasterising and Pillow for the .ico container; both
are already available in the GUI environment.
"""

from __future__ import annotations

import math
import os
import sys

# ---------------------------------------------------------------- design tokens

SIZE = 1024                 # master canvas, px
RADIUS = 224                # rounded-square corner radius

BG = "#0F2C4C"              # deep navy plate
BG_EDGE = "#1B4470"         # subtle rim so the plate reads on dark desktops
REF = "#FFFFFF"             # healthy reference spans
RECON = "#4CC2FF"           # reconstructed span (tint of the GUI brand accent)

STROKE = 46                 # waveform stroke width (master px)
TICK_ALPHA = 0.18           # opacity of the gap boundary ticks
TICK_WIDTH = 8.0            # tick bar width (master px)
TICK_EXTENT = 0.88          # tick half-height, as a fraction of AMP (kept inside
                            # the wave envelope so the ticks read as brackets
                            # rather than as poles sticking out of the trace)

X0, X1 = 118.0, 906.0       # waveform horizontal extent
Y_MID = 512.0               # waveform baseline
AMP = 198.0                 # peak amplitude

GAP_FROM, GAP_TO = 0.34, 0.66   # normalised span that was missing

CYCLES = 2.5                # dominant cycles across the plate
HARMONIC = 0.14             # amplitude of the second harmonic
ENV_DEPTH = 0.12            # amplitude-modulation depth

SAMPLES = 1800              # polyline resolution


def _envelope(t: float) -> float:
    """Gentle amplitude modulation so the trace reads as a real seismogram."""
    return (1.0 - ENV_DEPTH) + ENV_DEPTH * math.sin(2.0 * math.pi * 0.75 * t + 0.25) ** 2


def _wave(t: float) -> float:
    """Deterministic two-tone waveform, roughly -1..1."""
    return (
        math.sin(2.0 * math.pi * CYCLES * t + 0.15)
        + HARMONIC * math.sin(2.0 * math.pi * CYCLES * 2.2 * t + 1.5)
    )


def _wave_points() -> list[tuple[float, float]]:
    pts = []
    for i in range(SAMPLES + 1):
        t = i / SAMPLES
        x = X0 + (X1 - X0) * t
        y = Y_MID - AMP * _envelope(t) * _wave(t)
        pts.append((x, y))
    return pts


def _slice(pts, lo: float, hi: float):
    """Take the sub-polyline whose normalised x lies in [lo, hi]."""
    return [
        (x, y)
        for x, y in pts
        if lo - 1e-9 <= (x - X0) / (X1 - X0) <= hi + 1e-9
    ]


def _path(pts) -> str:
    return "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts)


# ------------------------------------------------------------------ SVG master

def build_svg() -> str:
    pts = _wave_points()
    left = _path(_slice(pts, 0.0, GAP_FROM))
    mid = _path(_slice(pts, GAP_FROM, GAP_TO))
    right = _path(_slice(pts, GAP_TO, 1.0))

    gap_x = X0 + (X1 - X0) * GAP_FROM
    gap_x2 = X0 + (X1 - X0) * GAP_TO
    tick_y = Y_MID - AMP * TICK_EXTENT
    tick_h = AMP * 2 * TICK_EXTENT

    ticks = "".join(
        f'\n    <rect x="{cx - TICK_WIDTH / 2:.1f}" y="{tick_y:.1f}" '
        f'width="{TICK_WIDTH:.1f}" height="{tick_h:.1f}" '
        f'rx="{TICK_WIDTH / 2:.1f}"/>'
        for cx in (gap_x, gap_x2)
    )

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{SIZE}" height="{SIZE}"
     viewBox="0 0 {SIZE} {SIZE}" role="img">
  <title>DSSRR</title>
  <desc>Dual-sided spectral reference reconstruction: a seismic trace whose
  missing middle span has been rebuilt from the healthy spans on both sides.</desc>

  <rect x="0" y="0" width="{SIZE}" height="{SIZE}" rx="{RADIUS}" ry="{RADIUS}"
        fill="{BG}"/>
  <rect x="3" y="3" width="{SIZE - 6}" height="{SIZE - 6}"
        rx="{RADIUS - 3}" ry="{RADIUS - 3}"
        fill="none" stroke="{BG_EDGE}" stroke-width="6"/>

  <g fill="{REF}" opacity="{TICK_ALPHA}">{ticks}
  </g>

  <g fill="none" stroke-linecap="round" stroke-linejoin="round">
    <path d="{left}" stroke="{REF}" stroke-width="{STROKE}"/>
    <path d="{right}" stroke="{REF}" stroke-width="{STROKE}"/>
    <path d="{mid}" stroke="{RECON}" stroke-width="{STROKE}"/>
  </g>
</svg>
"""


# -------------------------------------------------------------- raster exports

def render_png(svg_bytes: bytes, px: int):
    from PyQt5.QtCore import QByteArray
    from PyQt5.QtGui import QImage, QPainter
    from PyQt5.QtSvg import QSvgRenderer

    renderer = QSvgRenderer(QByteArray(svg_bytes))
    if not renderer.isValid():
        raise RuntimeError("QSvgRenderer rejected the generated SVG")
    img = QImage(px, px, QImage.Format_ARGB32_Premultiplied)
    img.fill(0)
    painter = QPainter(img)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
    renderer.render(painter)
    painter.end()
    return img


def _qimage_to_pil(qimg):
    from PyQt5.QtGui import QImage
    from PIL import Image

    qimg = qimg.convertToFormat(QImage.Format_RGBA8888)
    w, h = qimg.width(), qimg.height()
    ptr = qimg.constBits()
    ptr.setsize(qimg.byteCount())
    return Image.frombytes("RGBA", (w, h), bytes(ptr))


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))

    svg = build_svg()
    svg_path = os.path.join(here, "dssrr.svg")
    with open(svg_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(svg)
    print("wrote", svg_path)

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtGui import QGuiApplication

    app = QGuiApplication.instance() or QGuiApplication(sys.argv[:1])

    png_path = os.path.join(here, "dssrr.png")
    render_png(svg.encode("utf-8"), 512).save(png_path)
    print("wrote", png_path)

    # extra standalone sizes, handy for README / docs / packaging
    for px in (16, 24, 32, 48, 64, 128, 256):
        render_png(svg.encode("utf-8"), px).save(os.path.join(here, f"dssrr-{px}.png"))
    print("wrote size variants")

    ico_path = os.path.join(here, "dssrr.ico")
    master = _qimage_to_pil(render_png(svg.encode("utf-8"), 256))
    master.save(
        ico_path,
        format="ICO",
        sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )
    print("wrote", ico_path)
    del app
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
