"""Task grouping, global trajectory normalization, gradients and resume."""
import copy
import tempfile
import unittest
from pathlib import Path

import torch
from verl import DataProto
from etpo.buffer import SuccessBuffer
from etpo.core import select_successful_rows, success_trajectory_weights, masked_sft_sum
from etpo.data import supervised_subset
from etpo.config import validate
from test_etpo import make_batch, arr, configuration, LocalWorker, TinyBank
from etpo.training import evolve_and_score


def candidates():
    base, _ = make_batch()
    # Two groups of eight trajectories: easy=8 successes, hard=1 success.
    # Easy trajectories have two turns, hard ones one. Padding duplicates a row.
    rows, uid, traj, turns, wins, rewards = [], [], [], [], [], []
    for group, count, turn_count in [('easy', 8, 2), ('hard', 1, 1)]:
        for j in range(8):
            for turn in range(turn_count):
                rows.append(base.select_idxs([0 if group == 'easy' else 1]))
                uid.append(group); traj.append(f'{group}-{j}'); turns.append(turn)
                wins.append(j < count); rewards.append(10. / turn_count if j < count else 0.)
    b = DataProto.concat(rows)
    b.non_tensor_batch.update(uid=arr(uid), traj_uid=arr(traj), turn_idx=arr(turns),
                              etpo_env_won=arr(wins), rewards=arr(rewards))
    b = DataProto.concat([b, b.select_idxs([0])])
    return b


class WeightingTests(unittest.TestCase):
    def test_inverse_rate_counts_trajectories_not_turns_or_padding(self):
        b = candidates(); indices, _ = select_successful_rows(b, max_trajectories=None)
        weights = success_trajectory_weights(b, indices)
        self.assertEqual(len(weights), 9)
        self.assertEqual(sum(v for (uid, _), v in weights.items() if uid.startswith('easy')), 8.)
        self.assertEqual(weights[('hard-0', 0)], 8.)
        b.non_tensor_batch.pop('uid')
        with self.assertRaisesRegex(ValueError, 'group uid'):
            success_trajectory_weights(b, indices)

    def test_full_512_global_normalization_partial_groups_and_resume(self):
        b = candidates(); indices, _ = select_successful_rows(b, max_trajectories=None)
        weights = success_trajectory_weights(b, indices); buffer = SuccessBuffer()
        for _ in range(57): buffer.add(b, b, indices, weights)  # 513 successes, spans rollouts.
        data = buffer.batch(512, 0, success_rate_weighting=True)
        self.assertAlmostEqual(data.meta_info['sft_trajectory_weight_sum'], 512., places=9)
        self.assertAlmostEqual(data.batch['sft_token_weights'].sum().item(), 512., places=3)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'buffer.pt'; buffer.save(path)
            restored = SuccessBuffer(); restored.load(path)
            self.assertTrue(torch.equal(data.batch['sft_token_weights'],
                restored.batch(512, 0, True).batch['sft_token_weights']))
            restored.consume(512)
            self.assertEqual(len(restored), 1)
            self.assertAlmostEqual(restored.batch(1, 0, True).meta_info['sft_trajectory_weight_sum'], 1.)
            torch.save({'version':1,'tasks':buffer.tasks[:1]}, path)
            restored.load(path)
            restored.batch(1, 0)  # Old checkpoint remains usable with weighting off.
            with self.assertRaisesRegex(ValueError, 'legacy buffer'): restored.batch(1, 0, True)

    def test_trajectory_means_and_distributed_microbatch_gradients(self):
        b = candidates(); indices, _ = select_successful_rows(b, max_trajectories=None)
        indices.sort()
        weights = success_trajectory_weights(b, indices); buffer = SuccessBuffer()
        buffer.add(b, b, indices, weights)
        data = supervised_subset(buffer.batch(9, 0, True), list(range(17)), 4)
        self.assertEqual(len(data), 20)
        self.assertEqual(data.batch['sft_token_weights'][17:].sum().item(), 0.)
        raw = torch.arange(60, dtype=torch.float32).reshape(20,3).neg().requires_grad_()
        # Eight easy trajectories, each two turns x two response tokens; one hard with three.
        easy = [-raw[2*i:2*i+2,:2].mean() for i in range(8)]
        expected = (sum(easy) + 8 * (-raw[16,:3].mean())) / 16
        actual = masked_sft_sum(raw, data.batch['sft_mask'], data.batch['sft_token_weights']) / 9
        self.assertTrue(torch.allclose(actual, expected))
        expected_grad = torch.autograd.grad(expected, raw, retain_graph=True)[0]
        for micro_size in (1, 2, 5):
            raw.grad = None
            for start in range(0, 20, 5):  # Four ranks, five rows each.
                for offset in range(start, start+5, micro_size):
                    end = min(offset+micro_size, start+5)
                    loss = masked_sft_sum(raw[offset:end], data.batch['sft_mask'][offset:end],
                                          data.batch['sft_token_weights'][offset:end]) * 4 / 9
                    (loss / 4).backward()  # DDP averages four ranks.
            self.assertTrue(torch.allclose(raw.grad, expected_grad))

    def test_enabled_real_teacher_update_and_default_on(self):
        c=configuration();self.assertTrue(c.etpo.sft_success_rate_weighting)
        c.etpo.sft_success_rate_weighting='true'
        with self.assertRaisesRegex(ValueError,'boolean'): validate(c)
        c.etpo.sft_success_rate_weighting=True
        b,t=make_batch();b.non_tensor_batch['uid']=arr(['same-task']*2)
        w=LocalWorker(c);b.batch['old_log_probs']=w.compute_log_prob(b).batch['old_log_probs']
        student=[p.detach().clone() for p in w.model.parameters()]
        teacher=[p.detach().clone() for p in w.teacher_model.parameters()]
        _,metrics=evolve_and_score(b,w,t,TinyBank(),c,0,SuccessBuffer())
        self.assertEqual(metrics['etpo/sft_trajectory_weight_sum'],1.)
        self.assertTrue(all(torch.equal(x,p) for x,p in zip(student,w.model.parameters())))
        self.assertTrue(any(not torch.equal(x,p) for x,p in zip(teacher,w.teacher_model.parameters())))


if __name__ == '__main__': unittest.main()
