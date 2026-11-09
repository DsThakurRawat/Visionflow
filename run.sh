#!/usr/bin/env bash
# Quick runner for the traffic analytics pipeline.
#
# Usage:
#   ./run.sh input
#   ./run.sh input2
#   ./run.sh input.mp4
#
# Automatically reads from data/<name>.mp4, saves to:
#   - data/output_<name>.mp4
#   - data/summary_<name>.csv
# And automatically plays the output in a popup player window.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -z "$1" ]; then
    echo "Usage: ./run.sh <video_name> [optional flags]"
    echo ""
    echo "Examples:"
    echo "  ./run.sh input"
    echo "  ./run.sh input2"
    echo "  ./run.sh input2 --no-plates"
    exit 1
fi

INPUT_NAME="$1"
shift

uv run python traffic_pipeline.py "$INPUT_NAME" "$@"

