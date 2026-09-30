import json
import os
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from verl import DataProto
from verl.utils.model import compute_position_id_with_mask
from agent_system.environments.skillrise_sokoban.envs import SokobanWorker, parse_board, render_board, project_action
from agent_system.environments.skillrise_sokoban.env_manager import load_groups, make_envs, SokobanEnvironmentManager
from agent_system.multi_turn_rollout import TrajectoryCollector
from etpo.sokoban_levels import solve_board
from etpo.skills import SkillBank
from etpo.data import build_teacher_batch
from etpo.core import select_successful_rows
from test_etpo import ROOT, configuration, arr

TRAIN = ROOT / 'data/groups/etpo_sokoban_train_K3.jsonl'
VAL = ROOT / 'data/groups/etpo_sokoban_val.jsonl'


def config():
    c = configuration()
    c.env.env_name = 'skillrise_sokoban'
    c.env.num_attempts = 3
    c.env.rollout.n = 2
    c.env.max_steps = 40
    c.env.max_turns = 40
    c.env.sokoban.train_groups = str(TRAIN)
    c.env.sokoban.val_groups = str(VAL)
    c.data.train_batch_size = 2
    c.data.val_batch_size = 2
    c.data.max_prompt_length = 2048
    c.trainer.rollout_data_dir = None
    return c


class OracleGenerator:
    """Test-only BFS; never used by the training launcher or policy."""
    def __init__(self, tokenizer, manager):
        self.tokenizer, self.manager = tokenizer, manager

    def generate_sequences_agent(self, batch):
        direction = solve_board(self.manager.boards[0].splitlines())[0]
        ids = self.tokenizer.encode(f'<action>{direction}</action>', add_special_tokens=False) + [self.tokenizer.eos_token_id]
        response = torch.full((1, 32), self.tokenizer.pad_token_id, dtype=torch.long)
        response[0, :len(ids)] = torch.tensor(ids)
        rm = torch.zeros_like(response)
        rm[0, :len(ids)] = 1
        prompts = batch.batch['input_ids']
        mask = torch.cat([batch.batch['attention_mask'], rm], 1)
        return DataProto.from_dict(tensors={'prompts': prompts, 'responses': response,
            'input_ids': torch.cat([prompts, response], 1), 'attention_mask': mask,
            'position_ids': compute_position_id_with_mask(mask)},
            non_tensors={k:v for k,v in batch.non_tensor_batch.items() if k != 'raw_prompt_ids'})


class SokobanTests(unittest.TestCase):
    def test_board_symbols_and_invalid_levels(self):
        lines = ['#######', '# +*$ #', '#     #', '#######']
        fixed, state = parse_board(lines)
        self.assertEqual(render_board(fixed, state), lines)
        with self.assertRaises(ValueError):
            parse_board(['#####', '#@$. ', '#####'])
        with self.assertRaises(ValueError):
            parse_board(['#####', '#@..#', '#####'])

    def test_real_backend_sparse_success_and_terminal_latch(self):
        worker = SokobanWorker(max_steps=3)
        try:
            worker.load_game({'task_id':'one', 'board':['#####','#@$.#','#####']})
            obs, reward, done, info = worker.step('right')
            self.assertEqual(reward, 10)
            self.assertTrue(done and info['won'] and info['is_action_valid'])
            self.assertIn('*', obs)
            self.assertEqual(worker.step('left')[1], 0)
            worker.load_game({'task_id':'one', 'board':['#####','#@$.#','#####']})
            self.assertFalse(worker.won)
            self.assertEqual(worker.moves, 0)
            self.assertEqual(worker.step('up')[1], 0)
            self.assertFalse(worker.info(False)['is_action_valid'])
            self.assertFalse(worker.step(None)[2])
            self.assertTrue(worker.step(None)[2])
            self.assertFalse(worker.won)
        finally:
            worker.close()

    def test_projection_one_move_only(self):
        self.assertEqual(project_action('reason\n<action>RIGHT</action>'), 'right')
        for text in ['right', '<action>undo</action>', '<action>right,left</action>',
                     '<action>up</action><action>down</action>']:
            self.assertIsNone(project_action(text))

    def test_all_bundled_levels_solve_in_actual_gym(self):
        train, train_keys, train_ids = load_groups(TRAIN, 3)
        val, val_keys, val_ids = load_groups(VAL, 1)
        self.assertFalse(train_keys & val_keys)
        self.assertFalse(train_ids & val_ids)
        tasks = [task for group in train + val for task in group['tasks']]
        worker = SokobanWorker(max_steps=40)
        lengths = []
        try:
            for task in tasks:
                solution = solve_board(task['board'], max_steps=30)
                self.assertTrue(solution, task['task_id'])
                worker.load_game(task)
                total = 0
                for action in solution:
                    _, reward, done, info = worker.step(action)
                    self.assertTrue(info['is_action_valid'])
                    total += reward
                self.assertTrue(done and info['won'])
                self.assertEqual(total, 10)
                lengths.append(len(solution))
        finally:
            worker.close()
        report = {'train_levels': len(train_keys), 'validation_levels': len(val_keys),
                  'all_solved_in_gym': len(tasks), 'symmetric_split_overlap': 0,
                  'solution_length_min': min(lengths), 'solution_length_max': max(lengths),
                  'solver': 'test-only BFS, not an LLM evaluation', 'gpu_training': 'not tested'}
        output = ROOT / 'logs/sokoban-backend.json'
        output.parent.mkdir(exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + '\n')

    def test_group_repeat_restart_and_split_guard(self):
        c = config()
        train, val = make_envs(c)
        try:
            obs, _ = train.reset()
            self.assertEqual(obs['anchor'][0], obs['anchor'][1])
            old = train.envs.workers[0].task['task_id']
            train.advance()
            self.assertNotEqual(old, train.envs.workers[0].task['task_id'])
            before, _ = val.reset()
            val.step(['<action>right</action>'] * val.num_processes)
            after, _ = val.restart()
            self.assertEqual(before['anchor'], after['anchor'])
        finally:
            train.close()
            val.close()
        group = json.loads(TRAIN.read_text().splitlines()[0])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'val.jsonl'
            path.write_text(json.dumps({'K':1, 'tasks':[group['tasks'][0]]}) + '\n')
            c.env.sokoban.val_groups = str(path)
            with self.assertRaisesRegex(ValueError, 'overlap'):
                make_envs(c)

    def test_real_collector_success_selection_and_teacher_skill(self):
        c = config()
        c.env.rollout.n = 1
        groups, _, _ = load_groups(TRAIN, 3)
        manager = SokobanEnvironmentManager(c, groups[:1], 1, 1)
        tokenizer = AutoTokenizer.from_pretrained(os.environ['SKILLRISE_MODEL_PATH'], local_files_only=True)
        base = DataProto.from_dict(tensors={'input_ids':torch.ones(1,1,dtype=torch.long)},
            non_tensors={'raw_prompt':arr([[{'role':'user','content':''}]]), 'data_source':arr(['text'])})
        try:
            data = TrajectoryCollector(c, tokenizer).multi_turn_loop(base, OracleGenerator(tokenizer, manager), manager)
            self.assertEqual(set(data.non_tensor_batch['phase']), {'play'})
            self.assertEqual(set(data.non_tensor_batch['traj_idx']), {0,1,2})
            rows, metrics = select_successful_rows(data)
            self.assertEqual(metrics['etpo/sft_tasks'], 3)
            self.assertEqual(len(rows), len(data))
            for raw in data.non_tensor_batch['raw_prompt']:
                self.assertNotIn('Plan in pushes', str(raw))
            data.batch['response_mask'] = data.batch['attention_mask'][:, -32:]
            teacher = build_teacher_batch(data, tokenizer, SkillBank(ROOT/'skills','sokoban'), c)
            self.assertTrue(torch.equal(teacher.batch['responses'], data.batch['responses']))
            self.assertIn('Plan in pushes', tokenizer.decode(teacher.batch['prompts'][0]))
        finally:
            manager.close()


if __name__ == '__main__':
    unittest.main()
