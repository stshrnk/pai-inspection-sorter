#!/bin/bash
export DISPLAY=:0
export XAUTHORITY=/run/user/1000/gdm/Xauthority
ISAAC_PYTHON=~/dev/IsaacSim/_build/linux-aarch64/release/python.sh

# Ensure CUDA 12.9 NVRTC builtins are findable for GB10 (sm_121) support
TORCH_LIB=~/dev/IsaacSim/_build/linux-aarch64/release/exts/omni.isaac.ml_archive/pip_prebundle/torch/lib
export LD_LIBRARY_PATH="$TORCH_LIB:${LD_LIBRARY_PATH:-}"

cd "$(dirname "$0")"
exec "$ISAAC_PYTHON" main.py "$@"
