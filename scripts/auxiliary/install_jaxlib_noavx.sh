#!/usr/bin/env bash
# ───────────────────────────────────────────────────────────────────────
# Install locally-built jaxlib (no-AVX) into the project venv.
#
# Usage: bash scripts/install_jaxlib_noavx.sh
# ───────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
WHEEL_DIR="/extra/wayne2/preserve/nntran5/jaxlib-wheels"

source "$PROJECT_DIR/venv/bin/activate"

# Find the built wheel
WHEEL=$(find "$WHEEL_DIR" -name 'jaxlib-*.whl' -type f | head -1)

if [[ -z "$WHEEL" ]]; then
    echo "ERROR: No jaxlib wheel found in $WHEEL_DIR"
    echo "       The build may still be running. Check with:"
    echo "       ps aux | grep bazel"
    exit 1
fi

echo "Installing: $WHEEL"
pip install --force-reinstall "$WHEEL"

echo ""
echo "Verifying JAX import..."
python3 -c "
import jax
import jaxlib
print(f'jax     {jax.__version__}')
print(f'jaxlib  {jaxlib.__version__}')
print(f'devices: {jax.devices()}')
print('SUCCESS: JAX imported without AVX errors!')
"

echo ""
echo "Run full smoke tests with: python3 scripts/smoke_test.py"
