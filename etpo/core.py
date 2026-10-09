"""Pure tensor ETPO objectives and strict success selection."""
import math
import random
from collections import defaultdict
import torch


def reward_aligned_credit(teacher_logp, old_student_logp, response_mask, advantages, *, gate_beta=5.0, exclude_top20_gap_tokens=False):
    """Same advantage scaling as gap_DPO_GRPO; no auxiliary loss or hard gate."""
    if not isinstance(exclude_top20_gap_tokens, bool):
        raise ValueError('exclude_top20_gap_tokens must be boolean')
    if not math.isfinite(gate_beta) or gate_beta < 0:
        raise ValueError('gate_beta must be finite and nonnegative')
    if any(x.shape != teacher_logp.shape for x in (old_student_logp, response_mask, advantages)):
        raise ValueError('Teacher/student responses and advantages must be token-aligned')
    valid = response_mask.bool()
    with torch.no_grad():
        if not all(torch.isfinite(x[valid]).all() for x in (teacher_logp, old_student_logp, advantages)):
            raise ValueError('Scores and advantages must be finite on response tokens')
        gap = torch.where(valid, teacher_logp.float() - old_student_logp.float(), 0.)
        reward = torch.where(valid, advantages, 0.)
        scale = 0.5 + torch.sigmoid(gate_beta * reward.sign() * gap)
        excluded = torch.zeros_like(valid)
        if exclude_top20_gap_tokens:
            # Global driver batch, valid response tokens only. Raw signed gap,
            # floor(N/5) tokens; deterministic row-major tie-breaking.
            flat_valid = valid.flatten().nonzero(as_tuple=True)[0]
            k = flat_valid.numel() // 5
            if k:
                order = torch.argsort(gap.flatten()[flat_valid], descending=True, stable=True)
                excluded.view(-1)[flat_valid[order[:k]]] = True
                scale = torch.where(excluded, 1., scale)
        adjusted = reward * scale.to(reward.dtype)
        count = valid.sum().clamp_min(1)
        metrics = {'etpo/credit_scale_mean': (scale.masked_fill(~valid, 0).sum()/count).item(),
                   'etpo/teacher_gap_mean': (gap.sum()/count).item(),
                   'etpo/excluded_gap_tokens': excluded.sum().item(),
                   'etpo/excluded_gap_fraction': (excluded.sum()/count).item()}
        def mean(values):
            return values.float().mean().item() if values.numel() else 0.0
        vg = gap[valid]
        metrics.update({'etpo/valid_tokens': valid.sum().item(),
                        'etpo/gap_positive_fraction': mean(vg > 0),
                        'etpo/gap_std': vg.std(unbiased=False).item() if vg.numel() else 0.,
                        'etpo/adv_abs_before': mean(reward[valid].abs()),
                        'etpo/adv_abs_after': mean(adjusted[valid].abs()),
                        'etpo/sign_flip_fraction': mean(adjusted[valid].sign() != reward[valid].sign()),
                        'etpo/excluded_gap_mean': mean(gap[excluded]),
                        'etpo/excluded_positive_adv_fraction': mean(reward[excluded] > 0),
                        'etpo/excluded_negative_adv_fraction': mean(reward[excluded] < 0)})
        for label, values in (('gap', vg), ('scale', scale[valid])):
            qs = torch.quantile(values.float(), values.new_tensor([.1,.5,.8,.9,.99])).tolist() if values.numel() else [0.]*5
            for percentile, value in zip((10,50,80,90,99), qs):
                metrics[f'etpo/{label}_p{percentile}'] = value
        for label, group in (('positive_adv', valid & (reward>0)), ('negative_adv', valid & (reward<0))):
            metrics[f'etpo/{label}_scale_mean'] = mean(scale[group])
            metrics[f'etpo/{label}_gap_mean'] = mean(gap[group])
    return adjusted.detach(), metrics


def masked_sft_sum(log_probs, mask, token_weights=None):
    valid = mask.bool()
    if not torch.isfinite(log_probs[valid]).all():
        raise FloatingPointError('Nonfinite supervised token log probability')
    values = torch.where(valid, log_probs, torch.zeros_like(log_probs))
    if token_weights is not None:
        if token_weights.shape != log_probs.shape or not torch.isfinite(token_weights).all() or (token_weights < 0).any():
            raise ValueError('SFT token weights must be aligned, finite and nonnegative')
        values = values * token_weights
    return -values.sum()


def success_trajectory_weights(batch, successful_indices):
    """Inverse eligible-success rate within each rollout task group.

    uid is the collector's shared sampling-group ID (not traj_uid); traj_idx
    keeps different task positions separate. Deduplicate turns/padding before
    counting trajectories. Called once per rollout, so groups never mix steps.
    """
    n = batch.non_tensor_batch
    if 'uid' not in n:
        raise ValueError('Weighted teacher SFT requires rollout task group uid')
    units = {}
    for i in range(len(batch)):
        key = (str(n['traj_uid'][i]), int(n['traj_idx'][i]))
        group = (str(n['uid'][i]), int(n['traj_idx'][i]))
        if key in units and units[key] != group:
            raise ValueError('Inconsistent task group within a trajectory')
        units[key] = group
    successes = {(str(n['traj_uid'][i]), int(n['traj_idx'][i])) for i in successful_indices}
    totals, counts = defaultdict(int), defaultdict(int)
    for key, group in units.items():
        totals[group] += 1
        counts[group] += int(key in successes)
    return {key: totals[units[key]] / counts[units[key]] for key in successes}


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
    selected = successful if max_trajectories is None else successful[:max_trajectories]
    rows = [i for key in selected for i in sorted(units[key], key=lambda j: int(n['turn_idx'][j]))]
    return rows, {'etpo/successful_tasks': len(successful), 'etpo/sft_tasks': len(selected),
                  'etpo/sft_rows': len(rows), 'etpo/unique_tasks': len(units)}
