#!/usr/bin/env bash
# Isaac's Python runtime, without starting SimulationApp (training/evaluation).
set -euo pipefail
cd "$(dirname "$0")"
ISAAC_ROOT="${ISAAC_ROOT:-$HOME/dev/IsaacSim/_build/linux-aarch64/release}"
export LD_LIBRARY_PATH="$ISAAC_ROOT/exts/omni.isaac.ml_archive/pip_prebundle/torch/lib:${LD_LIBRARY_PATH:-}"
exec "${ISAAC_PYTHON:-$ISAAC_ROOT/python.sh}" "$@"
