#!/bin/bash
# GPU environment setup for AMD Radeon 8060S (Strix Halo) on WSL2
#
# This configures PyTorch ROCm to use the GPU via librocdxg + DXG passthrough.
# Source this file before running training: source scripts/gpu_env.sh
#
# Required components:
#   - ROCm 7.2 system install (/opt/rocm)
#   - librocdxg v1.1.0 (installed as /usr/lib/x86_64-linux-gnu/libhsakmt.so.1)
#   - PyTorch nightly ROCm 7.2 (pip install torch --index-url https://download.pytorch.org/whl/nightly/rocm7.2)
#   - System HSA runtime replacing bundled one in _rocm_sdk_core
#   - HSA image shim library (~/.local/lib/libhsa_image_shim.so)

# Tell ROCm HSA runtime to detect GPU via DXG (loads librocdxg)
export HSA_ENABLE_DXG_DETECTION=1

# Disable SDMA (not supported via DXG passthrough)
export HSA_ENABLE_SDMA=0

# Override GFX version: gfx1151 (RDNA 3.5) → pretend gfx1100 (RDNA 3)
export HSA_OVERRIDE_GFX_VERSION=11.0.0

# Provide stub HSA image v2 APIs (system 7.2 runtime lacks them, HIP 7.2 expects them)
export LD_PRELOAD="${HOME}/.local/lib/libhsa_image_shim.so"

echo "GPU env configured for AMD Radeon 8060S (Strix Halo / WSL2)"
