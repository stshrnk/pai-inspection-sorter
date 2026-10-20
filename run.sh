#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
ISAAC_ROOT="${ISAAC_ROOT:-$HOME/dev/IsaacSim/_build/linux-aarch64/release}"
ISAAC_PYTHON="${ISAAC_PYTHON:-$ISAAC_ROOT/python.sh}"
if [[ ! -x "$ISAAC_PYTHON" ]]; then
    echo 'Set ISAAC_ROOT or ISAAC_PYTHON to an Isaac Sim installation.' >&2
    exit 2
fi
# GB10 requires the NVRTC libraries shipped with Isaac Sim's torch.
TORCH_LIB="$ISAAC_ROOT/exts/omni.isaac.ml_archive/pip_prebundle/torch/lib"
export LD_LIBRARY_PATH="$TORCH_LIB:${LD_LIBRARY_PATH:-}"
exec "$ISAAC_PYTHON" "${SORTER_ENTRYPOINT:-main.py}" "$@"
