"""Sokoban task groups with the metadata required by ETPO reward/SFT selection."""
import json
import random
from collections import defaultdict
import numpy as np
from ..base import EnvironmentManagerBase
from .envs import SokobanEnvs, parse_board, board_key, project_action

RULES = '''You are playing Sokoban. Put every box on a goal.
Board symbols: # wall, space floor, . goal, $ box, * box on goal,
@ player, + player on goal. Up/down change row; left/right change column.
You can walk on floor or goals, and push one adjacent box if the square beyond
is empty. You cannot pull boxes, walk through walls/boxes, undo, or reset.
Return exactly one move in <action>up</action>, <action>down</action>,
<action>left</action>, or <action>right</action>. You may reason before the tag.
'''


def load_groups(path, k):
    with open(path) as stream:
        groups = [json.loads(line) for line in stream if line.strip()]
    if not groups:
        raise ValueError(f'Empty Sokoban group file: {path}')
    identities, boards = set(), set()
    for group in groups:
        if group['K'] != k or len(group['tasks']) != k:
            raise ValueError('Sokoban group K differs from env.num_attempts')
        for task in group['tasks']:
            _, state = parse_board(task['board'])
            if not np.any(state == 4):
                raise ValueError('Initial Sokoban task cannot already be solved')
            key = board_key(task['board'])
            if task['task_id'] in identities or key in boards:
                raise ValueError('Duplicate Sokoban task or symmetric starting board')
            identities.add(task['task_id'])
            boards.add(key)
    return groups, boards, identities


class SokobanEnvironmentManager(EnvironmentManagerBase):
    meta_mode = 'skillrise'
    carry_mode = 'none'
    do_reflection = False

    def __init__(self, config, groups, group_count, group_n, task_mode='cross'):
        self.num_attempts = config.env.num_attempts
        self.max_turns = config.env.max_turns
        self.task_mode = task_mode
        self.group_n = group_n
        self.groups_per_chunk = group_count
        self.num_processes = group_count * group_n
        self.groups = list(groups)
        random.Random(config.env.seed).shuffle(self.groups)
        self.cursor = 0
        self.row_task_type = ['sokoban'] * self.num_processes
        self.history_length = config.env.history_length
        if self.history_length < 0:
            raise ValueError('history_length must be nonnegative')
        super().__init__(SokobanEnvs(self.num_processes, config.env.max_steps), project_action, config)

    def reset(self):
        chosen = [self.groups[(self.cursor + i) % len(self.groups)] for i in range(self.groups_per_chunk)]
        self.cursor += self.groups_per_chunk
        self.row_group = [g for g in chosen for _ in range(self.group_n)]
        self.curr_pos = 0
        return self._load()

    def _load(self):
        position = 0 if self.task_mode == 'repeat' else self.curr_pos
        tasks = [group['tasks'][position] for group in self.row_group]
        self.boards, infos = self.envs.load_games(tasks)
        self.history = [[] for _ in tasks]
        return self.build_text_obs(), infos

    def advance(self):
        self.curr_pos += 1
        if self.curr_pos >= self.num_attempts:
            raise ValueError('No more tasks in this group')
        return self._load()

    def restart(self):
        if self.task_mode != 'repeat':
            raise ValueError('Restart is only for held-out evaluation retries')
        return self.advance()

    def build_text_obs(self):
        texts = []
        for board, history in zip(self.boards, self.history):
            recent = history[-self.history_length:] if self.history_length else []
            trace = '\n'.join(recent)
            texts.append(RULES + f'\nRecent actions:\n{trace}\nCurrent board:\n```\n{board}\n```')
        return {'text': texts, 'image': None, 'anchor': list(self.boards)}

    def step(self, texts, phase='play'):
        if phase != 'play':
            raise ValueError('ETPO Sokoban only supports student play')
        actions = [project_action(text) for text in texts]
        self.boards, rewards, dones, infos = self.envs.step(actions)
        for history, action, info in zip(self.history, actions, infos):
            history.append(f'{action or "invalid format"}: {"moved" if info["is_action_valid"] else "no movement"}')
        return self.build_text_obs(), np.asarray(rewards), np.asarray(dones), infos

    def success_evaluator(self, **kwargs):
        result = defaultdict(list)
        for records, infos in zip(kwargs['total_batch_list'], kwargs['total_infos']):
            wons = [False] * self.num_attempts
            for record, info in zip(records, infos):
                if record['active_masks'] and record['phase'] == 'play':
                    wons[record['traj_idx']] |= bool(info['won'])
            cumulative = False
            for pos, won in enumerate(wons):
                cumulative |= won
                result[f'success_rate[{pos}]'].append(float(cumulative if self.task_mode == 'repeat' else won))
                if self.task_mode == 'cross':
                    result['success_rate'].append(float(won))
        return {key: np.asarray(value) for key, value in result.items()}


def make_envs(config):
    from agent_system.multi_turn_rollout.utils import compute_groups_per_chunk
    from pathlib import Path
    if not config.get('etpo', {}).get('enabled', False):
        raise ValueError('skillrise_sokoban is an ETPO play-only adapter; enable etpo.enabled')
    cfg = config.env
    if cfg.rollout.n < 1 or cfg.num_attempts < 1 or min(cfg.max_steps, cfg.max_turns) < 1:
        raise ValueError('Sokoban needs positive group_n, attempts, steps and turns')
    train_path = str(Path(cfg.sokoban.train_groups).expanduser())
    val_path = str(Path(cfg.sokoban.val_groups).expanduser())
    train, train_boards, train_ids = load_groups(train_path, cfg.num_attempts)
    val, val_boards, val_ids = load_groups(val_path, 1)
    if train_boards & val_boards or train_ids & val_ids:
        raise ValueError('Sokoban train and validation tasks overlap')
    groups = compute_groups_per_chunk(config.data.train_batch_size, cfg.rollout.n, cfg.max_env_per_rollout)
    envs = SokobanEnvironmentManager(config, train, groups, cfg.rollout.n)
    val_envs = SokobanEnvironmentManager(config, val, config.data.val_batch_size, 1, task_mode='repeat')
    return envs, val_envs
