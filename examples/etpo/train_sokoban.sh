#!/usr/bin/env bash
set -euo pipefail
ENGINE="${1:-vllm}"
if [ "$#" -gt 0 ]; then shift; fi
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/activate.sh"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
ETPO_RUN_NAME="${ETPO_EXPERIMENT_NAME:-etpo_sokoban_qwen3-4b_$(date +%Y%m%d%H%M%S)}"
ETPO_RUN_DIR="$SKILLRISE_OUTPUT_ROOT/etpo_sokoban/$ETPO_RUN_NAME"
ETPO_INPUT_DIR="$ETPO_ROOT/.runtime/sokoban-input"
mkdir -p "$ETPO_RUN_DIR"
# Sokoban has no external task dataset or Java dependency.
python -m examples.etpo.prepare_sokoban --output-dir "$ETPO_INPUT_DIR" --train-rows 16 --val-rows 32
python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=skillrise \
  algorithm.gamma=0.95 +algorithm.step_gamma=0.95 +algorithm.traj_gamma=0.6 \
  +algorithm.curate_loss_weight=1.0 algorithm.gigpo.mode=mean_norm \
  algorithm.gigpo.step_advantage_w=1.0 algorithm.use_kl_in_reward=false \
  data.train_files="$ETPO_INPUT_DIR/train.parquet" data.val_files="$ETPO_INPUT_DIR/test.parquet" \
  data.train_batch_size=16 data.val_batch_size=32 \
  data.max_prompt_length=2048 data.max_response_length=256 \
  data.return_raw_chat=true data.truncation=error \
  actor_rollout_ref.model.path="$SKILLRISE_MODEL_PATH" \
  actor_rollout_ref.model.use_remove_padding=true \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=128 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
  actor_rollout_ref.actor.use_torch_compile=false actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.actor.use_kl_loss=false \
  actor_rollout_ref.actor.use_invalid_action_penalty=true \
  actor_rollout_ref.actor.invalid_action_penalty_coef=0.5 \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.rollout.name="$ENGINE" \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.enable_chunked_prefill=true \
  actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
  actor_rollout_ref.rollout.max_num_seqs=256 \
  reward_model.reward_manager=episode \
  env.env_name=skillrise_sokoban env.seed=0 env.rollout.n=8 env.num_attempts=3 \
  env.max_steps=40 env.max_turns=40 env.history_length=8 env.num_actions_per_turn=1 \
  env.sokoban.mode=text env.max_env_per_rollout=128 \
  etpo.enabled=true etpo.teacher_max_prompt_length=4096 \
  trainer.project_name=etpo_sokoban trainer.experiment_name="$ETPO_RUN_NAME" \
  trainer.default_local_dir="$ETPO_RUN_DIR" trainer.rollout_data_dir="$ETPO_RUN_DIR/rollouts" \
  trainer.n_gpus_per_node=8 trainer.nnodes=1 trainer.critic_warmup=0 \
  "trainer.logger=['console','wandb']" trainer.val_before_train=true \
  trainer.save_freq=10 trainer.save_start_step=10 trainer.test_freq=5 trainer.total_epochs=150 \
  trainer.ray_wait_register_center_timeout=3600 \
  "$@" 2>&1 | tee "$ETPO_RUN_DIR/train.log"
