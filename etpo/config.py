"""Fail early for settings whose ETPO semantics are not implemented."""
import math


def enabled(config):
    return bool(config.get('etpo', {}).get('enabled', False))


def validate(config):
    if not enabled(config):
        return
    e = config.etpo
    if config.env.num_attempts != 1 or config.env.get('val_num_attempts', 1) != 1:
        raise ValueError('ETPO release requires single-task K=1 training and validation')
    if config.algorithm.get('use_kl_in_reward', False):
        raise ValueError('ETPO does not use reward KL penalties')
    if e.teacher_mode != 'independent':
        raise ValueError('ETPO requires independent teacher and student weights')
    if config.actor_rollout_ref.actor.strategy != 'fsdp':
        raise ValueError('Independent ETPO currently requires FSDP1')
    if config.actor_rollout_ref.model.get('lora_rank', 0) > 0:
        raise ValueError('Independent ETPO currently requires full model training, not LoRA')
    if config.actor_rollout_ref.actor.ulysses_sequence_parallel_size != 1:
        raise ValueError('ETPO currently requires sequence parallel size 1')
    if config.algorithm.adv_estimator != 'grpo':
        raise ValueError('ETPO requires GRPO advantages, matching gap_DPO_GRPO')
    actor = config.actor_rollout_ref.actor
    if actor.get('clip_level', 'token') != 'token' or actor.tis_imp_ratio_cap > 0:
        raise ValueError('ETPO requires token clipping without TIS, matching gap_DPO_GRPO')
    if actor.use_kl_loss or actor.entropy_coeff != 0 or actor.loss_agg_mode != 'token-mean':
        raise ValueError('ETPO uses token-mean GRPO without auxiliary KL or entropy loss')
    if not isinstance(e.exclude_top20_gap_tokens, bool):
        raise ValueError('exclude_top20_gap_tokens must be boolean')
    if not isinstance(e.get('sft_success_rate_weighting', False), bool):
        raise ValueError('sft_success_rate_weighting must be boolean')
    if not config.env.env_name.startswith('skillrise_'):
        raise ValueError('ETPO requires a skillrise_ environment with task identity metadata')
    if not config.data.return_raw_chat or config.data.truncation != 'error':
        raise ValueError('ETPO requires return_raw_chat=True and truncation=error for exact context matching')
    if float(config.actor_rollout_ref.rollout.temperature) != 1.0:
        raise ValueError('ETPO currently requires rollout temperature=1 for matching SFT/distillation likelihoods')
    if config.actor_rollout_ref.rollout.multi_turn.enable:
        raise ValueError('ETPO stores one response per environment step; packed multi_turn is unsupported')
    obsolete = ('distill_coef', 'confidence_min', 'confidence_max', 'max_abs_log_ratio', 'max_success_trajectories', 'teacher_sft_lr_scale')
    if any(key in e for key in obsolete):
        raise ValueError('Obsolete ETPO settings: use credit_enabled, gate_beta and teacher_update_batch_size')
    for name in ('gate_beta', 'success_reward_min', 'teacher_lr'):
        if not math.isfinite(float(e[name])) or e[name] < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    if e.teacher_lr <= 0:
        raise ValueError('teacher_lr must be positive')
    for name in ('sft_epochs', 'sft_micro_batch_size', 'teacher_update_batch_size', 'teacher_max_prompt_length', 'max_skill_tokens'):
        if isinstance(e[name], bool) or int(e[name]) != e[name] or int(e[name]) < 1:
            raise ValueError(f'{name} must be a positive integer')
