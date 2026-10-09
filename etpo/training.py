"""Rollout -> accumulate successes -> batch-triggered SFT -> reward credit."""
import time
from etpo.deep_diagnostics import credit_diagnostics, teacher_objective_diagnostics
from etpo.diagnostics import rollout_metrics, weight_metrics, credit_outcome_metrics, write_audit
from etpo.core import reward_aligned_credit, select_successful_rows, success_trajectory_weights
from etpo.data import build_teacher_batch, supervised_subset
from verl.utils.metric import reduce_metrics


def evolve_and_score(batch, worker_group, tokenizer, skill_bank, config, global_step, success_buffer):
    e = config.etpo
    # The reward advantages and old policy probabilities precede any teacher SFT.
    teacher = build_teacher_batch(batch, tokenizer, skill_bank, config)
    indices, metrics = select_successful_rows(
        batch, reward_min=e.success_reward_min,
        require_valid_actions=e.require_valid_actions,
        max_trajectories=None, seed=e.seed + global_step)
    diagnostics, records = rollout_metrics(batch, indices)
    metrics.update(diagnostics)
    write_audit('trajectories', {'step':global_step,'trajectories':records})
    metrics['etpo/success_buffer_before'] = len(success_buffer)
    world_size = config.trainer.n_gpus_per_node * config.trainer.nnodes
    updates = 0
    sft_seconds = 0.
    if e.evolve_teacher:
        weighted = e.get('sft_success_rate_weighting', False)
        raw_weights = success_trajectory_weights(batch, indices)
        success_buffer.add(teacher, batch, indices, raw_weights=raw_weights, collected_step=global_step)
        # Consume full batches only, retaining the remainder for future steps.
        while len(success_buffer) >= e.teacher_update_batch_size:
            buffered = success_buffer.batch(e.teacher_update_batch_size, tokenizer.pad_token_id,
                                            success_rate_weighting=weighted)
            raw=success_buffer.raw_weights[:e.teacher_update_batch_size]
            lengths=[sum(row['response'].numel() for row in task) for task in success_buffer.tasks[:e.teacher_update_batch_size]]
            update_diagnostics=weight_metrics(raw,lengths)
            detailed,audit=teacher_objective_diagnostics(raw,lengths,success_buffer.audit_metadata[:e.teacher_update_batch_size],weighted,global_step,success_buffer.teacher_updates+1)
            update_diagnostics.update(detailed)
            metrics.update(update_diagnostics)
            write_audit('teacher_update_attempts', audit)
            if weighted:
                for key in ('sum', 'min', 'max'):
                    metrics[f'etpo/sft_trajectory_weight_{key}'] = buffered.meta_info[f'sft_trajectory_weight_{key}']
            supervised = supervised_subset(buffered, list(range(len(buffered))), world_size)
            started = time.monotonic()
            out = worker_group.update_etpo_teacher(supervised)
            sft_seconds += time.monotonic() - started
            metrics.update(reduce_metrics(out.meta_info['metrics']))
            audit['status']='completed'
            audit['training_metrics']=reduce_metrics(out.meta_info['metrics'])
            write_audit('teacher_updates',audit)
            success_buffer.consume(e.teacher_update_batch_size)
            updates += 1
    metrics.update({'etpo/sft_skipped': int(updates == 0),
                    'etpo/teacher_updates': updates,
                    'etpo/sft_tasks': updates * e.teacher_update_batch_size,
                    'etpo/buffered_success_tasks': len(success_buffer),
                    'etpo/teacher_updates_total': success_buffer.teacher_updates,
                    'etpo/sft_tasks_total': success_buffer.consumed_tasks,
                    'etpo/teacher_sft_seconds': sft_seconds})
    if e.credit_enabled:
        started = time.monotonic()
        scored = worker_group.compute_etpo_teacher_log_prob(teacher)
        metrics['etpo/teacher_score_seconds'] = time.monotonic() - started
        before = batch.batch['advantages']
        batch.batch['advantages'], credit_metrics = reward_aligned_credit(
            scored.batch['old_log_probs'], batch.batch['old_log_probs'],
            batch.batch['response_mask'], batch.batch['advantages'], gate_beta=e.gate_beta,
            exclude_top20_gap_tokens=e.exclude_top20_gap_tokens)
        metrics.update(credit_metrics)
        metrics.update(credit_outcome_metrics(batch,before,batch.batch['advantages'],scored.batch['old_log_probs']))
        metrics.update(credit_diagnostics(batch,before,batch.batch['advantages'],scored.batch['old_log_probs'],global_step,e.gate_beta))
    return batch, metrics
