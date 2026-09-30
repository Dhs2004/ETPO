"""Driver-side ordering: rollout -> success SFT -> teacher score -> student PPO."""
import torch
from etpo.core import filtered_distillation, select_successful_rows
from etpo.data import build_teacher_batch, supervised_subset
from verl.utils.metric import reduce_metrics


def evolve_and_score(batch, worker_group, tokenizer, skill_bank, config, global_step):
    e = config.etpo
    # old_log_probs were computed BEFORE SFT, under the rollout student policy.
    teacher = build_teacher_batch(batch, tokenizer, skill_bank, config)
    indices, metrics = select_successful_rows(
        batch, reward_min=e.success_reward_min,
        require_valid_actions=e.require_valid_actions,
        max_trajectories=e.max_success_trajectories, seed=e.seed + global_step)
    world_size = config.trainer.n_gpus_per_node * config.trainer.nnodes
    if indices and e.evolve_teacher:
        supervised = supervised_subset(teacher, indices, world_size)
        out = worker_group.update_etpo_teacher(supervised)
        metrics.update(reduce_metrics(out.meta_info['metrics']))
        metrics['etpo/sft_skipped'] = 0
    else:
        metrics['etpo/sft_skipped'] = 1
    if e.distill_coef > 0:
        # Snapshot the evolved shared teacher's sampled-token scores BEFORE PPO.
        scored = worker_group.compute_log_prob(teacher)
        teacher_logp = scored.batch['old_log_probs']
        credit, keep = filtered_distillation(
            teacher_logp, batch.batch['old_log_probs'], batch.batch['response_mask'],
            confidence_min=e.confidence_min, confidence_max=e.confidence_max,
            max_abs_log_ratio=e.max_abs_log_ratio)
        batch.batch['etpo_distill_advantages'] = credit
        total = batch.batch['response_mask'].sum().clamp_min(1)
        metrics['etpo/retained_fraction'] = (keep.sum() / total).item()
        metrics['etpo/credit_mean'] = (credit.sum() / total).item()
        metrics['etpo/retained_tokens'] = keep.sum().item()
        metrics['etpo/teacher_nonfinite_tokens'] = (~torch.isfinite(teacher_logp) & batch.batch['response_mask'].bool()).sum().item()
    else:
        batch.batch['etpo_distill_advantages'] = torch.zeros_like(batch.batch['old_log_probs'])
        metrics['etpo/retained_fraction'] = 0.0
    return batch, metrics
