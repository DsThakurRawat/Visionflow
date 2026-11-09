#!/usr/bin/env python3
"""Run the traffic pipeline with an intuitive short syntax and auto popup.

Usage:
    uv run python run.py input
    uv run python run.py input2
    ./run.py input

Outputs are automatically saved into:
    - data/output_<name>.mp4
    - data/summary_<name>.csv

And the output video immediately pops up in a player window (mpv / ffplay).
Pass --no-play to disable the popup window.
"""

import sys
from pathlib import Path

# Ensure script directory is first on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from traffic_pipeline import main

if __name__ == "__main__":
    sys.exit(main())

