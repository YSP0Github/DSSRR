#!/usr/bin/env sh
# ============================================================================
#  DSSRR GUI launcher (Linux / macOS)
#
#  Usage:
#      ./launch_gui.sh
#
#  Requirements: PyQt5 + matplotlib (+ obspy to open MiniSEED files):
#      pip install -e ".[gui]"
#
#  Set DSSRR_PYTHON to use a specific interpreter, e.g.
#      DSSRR_PYTHON=~/.venv/bin/python ./launch_gui.sh
# ============================================================================
set -e

# Run from the repo root so the package is importable without installation.
cd "$(dirname "$0")"

PY="${DSSRR_PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || PY=python

exec "$PY" -m dssrr.gui
