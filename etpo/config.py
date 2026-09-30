"""Fail early for settings whose ETPO semantics are not implemented."""
import math


def enabled(config):
    return bool(config.get('etpo', {}).get('enabled', False))


def validate(config):
    if not enabled(config):
        return
    e = config.etpo
    if e.teacher_mode != 'shared':
        raise ValueError('ETPO implements shared weights: teacher = student + initial skill')
    if config.actor_rollout_ref.actor.strategy not in ('fsdp', 'fsdp2'):
        raise ValueError('ETPO requires the FSDP actor')
    if config.actor_rollout_ref.actor.ulysses_sequence_parallel_size != 1:
        raise ValueError('ETPO currently requires sequence parallel size 1')
    if config.algorithm.adv_estimator != 'skillrise':
        raise ValueError('ETPO uses SkillRise per-task/group reward advantages (adv_estimator=skillrise)')
    if not config.env.env_name.startswith('skillrise_'):
        raise ValueError('ETPO requires a skillrise_ environment with task identity metadata')
    if not config.data.return_raw_chat or config.data.truncation != 'error':
        raise ValueError('ETPO requires return_raw_chat=True and truncation=error for exact context matching')
    if float(config.actor_rollout_ref.rollout.temperature) != 1.0:
        raise ValueError('ETPO currently requires rollout temperature=1 for matching SFT/distillation likelihoods')
    if config.actor_rollout_ref.rollout.multi_turn.enable:
        raise ValueError('ETPO stores one response per environment step; packed multi_turn is unsupported')
    if not 0 < e.confidence_min < e.confidence_max <= 1:
        raise ValueError('Require 0 < confidence_min < confidence_max <= 1')
    for name in ('distill_coef', 'max_abs_log_ratio', 'success_reward_min', 'teacher_sft_lr_scale'):
        if not math.isfinite(float(e[name])) or e[name] < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    if e.teacher_sft_lr_scale <= 0 or e.max_abs_log_ratio <= 0:
        raise ValueError('SFT LR scale and log-ratio bound must be positive')
    for name in ('sft_epochs', 'sft_micro_batch_size', 'max_success_trajectories', 'teacher_max_prompt_length', 'max_skill_tokens'):
        if int(e[name]) < 1:
            raise ValueError(f'{name} must be positive')
    if not 0 <= config.etpo.distill_coef:
        raise ValueError('distill_coef must be nonnegative')
