#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/activate.sh"
ENVIRONMENT="${1:-alfworld}"
if [ "$#" -gt 0 ]; then shift; fi
case "$ENVIRONMENT" in alfworld|webshop|sciworld|sokoban) ;; *) echo 'Usage: train.sh {alfworld|webshop|sciworld|sokoban} [Hydra overrides]' >&2; exit 2;; esac
export ETPO_PROJECT_NAME="etpo_${ENVIRONMENT}"
export ETPO_EXPERIMENT_NAME="${ETPO_EXPERIMENT_NAME:-etpo_${ENVIRONMENT}_qwen3-4b_$(date +%Y%m%d%H%M%S)}"
# Pass --cfg job to validate without starting GPU work. Otherwise fail early.
ETPO_CONFIG_ONLY=false
for arg in "$@"; do
    case "$arg" in --cfg|--cfg=*) ETPO_CONFIG_ONLY=true;; esac
done
if [ "$ETPO_CONFIG_ONLY" = false ]; then
    python -c 'import torch; assert torch.cuda.is_available(), "No CUDA GPU visible; use a GPU-enabled container"'
fi
ETPO_BACKEND_SCRIPT="examples/skillrise_${ENVIRONMENT}/skillrise_${ENVIRONMENT}_qwen3_4b.sh"
if [ "$ENVIRONMENT" = sokoban ]; then ETPO_BACKEND_SCRIPT="examples/etpo/train_sokoban.sh"; fi
bash "$ETPO_BACKEND_SCRIPT" vllm \
    etpo.enabled=true \
    "etpo.skill_root=$ETPO_ROOT/skills" \
    data.return_raw_chat=true data.truncation=error \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.use_torch_compile=false \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    trainer.n_gpus_per_node=8 trainer.project_name="etpo_${ENVIRONMENT}" \
    "$@"
