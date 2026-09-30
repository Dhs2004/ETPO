"""Pure tensor ETPO objectives and strict success selection."""
import math
import random
from collections import defaultdict
import torch


def filtered_distillation(teacher_logp, old_student_logp, response_mask, *, confidence_min, confidence_max, max_abs_log_ratio):
    """Sampled reverse-KL policy-gradient credit, detached and held fixed per batch.

    Uses normalized sampled-token log probabilities, NOT raw logits. Rejected
    tokens contribute zero; the PPO loss remains normalized by all response tokens.
    """
    if teacher_logp.shape != old_student_logp.shape or response_mask.shape != teacher_logp.shape:
        raise ValueError('Teacher/student responses and masks must be token-aligned')
    t, s = teacher_logp.detach().float(), old_student_logp.detach().float()
    delta = t - s
    keep = (response_mask.bool() & torch.isfinite(t) & torch.isfinite(s)
            & (s <= 0) & (t >= math.log(confidence_min))
            & (t <= math.log(confidence_max)) & (delta.abs() <= max_abs_log_ratio))
    # where, rather than multiplication: rejected inf/nan must never enter the loss.
    credit = torch.where(keep, delta, torch.zeros_like(delta)).detach()
    return credit, keep


def masked_sft_sum(log_probs, mask):
    valid = mask.bool()
    if not torch.isfinite(log_probs[valid]).all():
        raise FloatingPointError('Nonfinite supervised token log probability')
    return -torch.where(valid, log_probs, torch.zeros_like(log_probs)).sum()


def select_successful_rows(batch, *, reward_min=10.0, require_valid_actions=True, max_trajectories=8, seed=0):
    """Select complete successful (traj_uid, task_pos) units using raw env labels.

    De-duplicate rows added by adjust_batch before summing rewards. Never select
    successes using shaped advantages, teacher likelihood, or another task's win.
    """
    n = batch.non_tensor_batch
    required = ('traj_uid', 'traj_idx', 'turn_idx', 'phase', 'etpo_env_won', 'rewards', 'is_action_valid')
    missing = [key for key in required if key not in n]
    if missing:
        raise ValueError(f'Missing ETPO outcome metadata: {missing}')
    units, seen = defaultdict(list), set()
    for i in range(len(batch)):
        if n['phase'][i] != 'play':
            raise ValueError('ETPO must collect student play only, without curator/reflection rows')
        key = (str(n['traj_uid'][i]), int(n['traj_idx'][i]))
        row_key = (*key, int(n['turn_idx'][i]))
        if row_key not in seen:
            units[key].append(i)
            seen.add(row_key)
    successful = []
    for key, rows in units.items():
        won = any(bool(n['etpo_env_won'][i]) for i in rows)
        reward = sum(float(n['rewards'][i]) for i in rows)
        valid = all(bool(n['is_action_valid'][i]) for i in rows)
        if won and math.isfinite(reward) and reward >= reward_min and (valid or not require_valid_actions):
            successful.append(key)
    successful.sort()
    random.Random(seed).shuffle(successful)
    selected = successful[:max_trajectories]
    rows = [i for key in selected for i in sorted(units[key], key=lambda j: int(n['turn_idx'][j]))]
    return rows, {'etpo/successful_tasks': len(successful), 'etpo/sft_tasks': len(selected),
                  'etpo/sft_rows': len(rows), 'etpo/unique_tasks': len(units)}
