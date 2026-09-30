#!/usr/bin/env bash
# Source this file; use the existing local setup when present, otherwise the
# caller's active Conda environment. All runtime paths can be overridden.
export ETPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export ETPO_SHARED_SETUP="${ETPO_SHARED_SETUP:-$(dirname "$ETPO_ROOT")/skillrise-setup}"
if [ -f "$ETPO_SHARED_SETUP/activate.sh" ]; then
    source "$ETPO_SHARED_SETUP/activate.sh"
fi
export SKILLRISE_ROOT="$ETPO_ROOT"
export SKILLRISE_SETUP="${SKILLRISE_SETUP:-$ETPO_ROOT/.runtime}"
export PYTHONPATH="$ETPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export SKILLRISE_DATA_ROOT="${SKILLRISE_DATA_ROOT:-$SKILLRISE_SETUP/data/verl-agent}"
export SKILLRISE_MODEL_PATH="${SKILLRISE_MODEL_PATH:-$SKILLRISE_SETUP/models/Qwen3-4B}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$SKILLRISE_SETUP/data/alfworld}"
export SCIWORLD_DATA="${SCIWORLD_DATA:-$SKILLRISE_SETUP/data/sciworld}"
export SKILLRISE_OUTPUT_ROOT="${ETPO_OUTPUT_ROOT:-$ETPO_ROOT/outputs}"
export WANDB_DIR="$SKILLRISE_OUTPUT_ROOT"
export WANDB_MODE="${WANDB_MODE:-offline}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
if [ -n "${JAVA_HOME:-}" ]; then
    export PATH="$JAVA_HOME/bin:$PATH"
    export SCIWORLD_JAVA_HOME="${SCIWORLD_JAVA_HOME:-$JAVA_HOME}"
fi
mkdir -p "$SKILLRISE_OUTPUT_ROOT"
cd "$ETPO_ROOT"
