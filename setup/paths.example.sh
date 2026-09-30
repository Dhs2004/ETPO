# Copy to config.local.sh in the repository root and adjust paths.
# Activate the skillrise Conda environment before sourcing this file.
export SKILLRISE_SETUP="$PWD/.runtime"
export SKILLRISE_MODEL_PATH="$SKILLRISE_SETUP/models/Qwen3-4B"
export SKILLRISE_DATA_ROOT="$SKILLRISE_SETUP/data/verl-agent"
export ALFWORLD_DATA="$SKILLRISE_SETUP/data/alfworld"
export SCIWORLD_DATA="$SKILLRISE_SETUP/data/sciworld"
# openjdk from conda-forge usually installs its JVM here; check your installation.
export JAVA_HOME="$CONDA_PREFIX/lib/jvm"
export SCIWORLD_JAVA_HOME="$JAVA_HOME"
export CLASSPATH=""
export WANDB_MODE=offline
