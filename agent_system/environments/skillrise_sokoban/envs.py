"""Text-only adapter over gym-sokoban's actual transition engine."""
import hashlib
import re
import numpy as np
from gym_sokoban.envs.sokoban_env import SokobanEnv

ACTIONS = {'up': 1, 'down': 2, 'left': 3, 'right': 4}
DIRECTIONS = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}


def parse_board(lines):
    if not lines or len({len(row) for row in lines}) != 1:
        raise ValueError('Sokoban board must be a nonempty rectangle')
    fixed = np.ones((len(lines), len(lines[0])), dtype=np.int64)
    state = fixed.copy()
    for r, row in enumerate(lines):
        for c, char in enumerate(row):
            if char not in '# .$*@+':
                raise ValueError(f'Unknown Sokoban tile: {char!r}')
            fixed[r, c] = 0 if char == '#' else 2 if char in '.*+' else 1
            # gym-sokoban encodes a box ON a goal as 3, off-goal as 4.
            state[r, c] = {'#': 0, ' ': 1, '.': 2, '*': 3, '$': 4, '@': 5, '+': 5}[char]
    if not (np.all(fixed[0] == 0) and np.all(fixed[-1] == 0)
            and np.all(fixed[:, 0] == 0) and np.all(fixed[:, -1] == 0)):
        raise ValueError('Boards need a closed wall boundary')
    boxes = np.isin(state, [3, 4]).sum()
    if (state == 5).sum() != 1 or boxes < 1 or boxes != (fixed == 2).sum():
        raise ValueError('Require one player and an equal positive number of boxes/goals')
    return fixed, state


def render_board(fixed, state):
    rows = []
    for r in range(state.shape[0]):
        row = ''
        for c in range(state.shape[1]):
            value = state[r, c]
            row += ('+' if fixed[r, c] == 2 else '@') if value == 5 else {0: '#', 1: ' ', 2: '.', 3: '*', 4: '$'}[value]
        rows.append(row)
    return rows


def board_key(lines):
    """Canonical full starting state, including rotations and reflections."""
    board = np.array([list(row) for row in lines])
    variants = []
    for k in range(4):
        rotated = np.rot90(board, k)
        for x in (rotated, np.fliplr(rotated)):
            variants.append('\n'.join(''.join(row) for row in x))
    return hashlib.sha256(min(variants).encode()).hexdigest()


def project_action(text):
    # Exactly one environment action; no hidden fallback move for malformed output.
    matches = re.findall(r'<action>\s*(.*?)\s*</action>', str(text), flags=re.DOTALL)
    if len(matches) != 1:
        return None
    action = matches[0].strip().lower()
    return action if action in ACTIONS else None


class SokobanWorker:
    def __init__(self, max_steps=60):
        if max_steps < 1:
            raise ValueError('max_steps must be positive')
        self.max_steps = max_steps
        self.env = None
        self.done = False
        self.won = False
        self.moves = 0

    def load_game(self, task):
        fixed, state = parse_board(task['board'])
        if not np.any(state == 4):
            raise ValueError('Initial board is already solved')
        if self.env is not None:
            self.env.close()
        self.env = SokobanEnv(dim_room=state.shape, num_boxes=int(np.isin(state, [3, 4]).sum()),
                              max_steps=self.max_steps, reset=False)
        self.env.room_fixed = fixed
        self.env.room_state = state
        self.env.player_position = np.argwhere(state == 5)[0]
        self.env.num_env_steps = 0
        self.env.reward_last = 0
        self.env.boxes_on_target = int((state == 3).sum())
        self.env.box_mapping = {}
        self.moves = 0
        self.done = self.won = False
        self.task = task
        return self.observation(), self.info(False)

    def observation(self):
        return '\n'.join(render_board(self.env.room_fixed, self.env.room_state))

    def info(self, valid):
        return {'won': self.won, 'is_action_valid': bool(valid), 'task_type': 'sokoban',
                'task_id': self.task['task_id'], 'task_score': float(self.won),
                'steps': self.moves, 'truncated': self.done and not self.won}

    def step(self, action):
        if self.done:
            # Collector may still call completed rows while other rows continue.
            return self.observation(), 0.0, True, self.info(False)
        self.moves += 1
        valid = False
        if action in ACTIONS:
            _, _, native_done, native_info = self.env.step(ACTIONS[action], observation_mode='tiny_rgb_array')
            valid = bool(native_info['action.moved_player'])
            self.won = bool(self.env._check_if_all_boxes_on_target())
            self.done = bool(native_done)
        self.done = self.done or self.won or self.moves >= self.max_steps
        return self.observation(), 10.0 if self.won else 0.0, self.done, self.info(valid)

    def close(self):
        if self.env is not None:
            self.env.close()


class SokobanEnvs:
    """In-process vector pool: small NumPy boards need no per-board Ray actor."""
    def __init__(self, num_processes, max_steps):
        self.num_processes = num_processes
        self.workers = [SokobanWorker(max_steps) for _ in range(num_processes)]

    def load_games(self, tasks):
        if len(tasks) != self.num_processes:
            raise ValueError('Task batch must match the vector pool')
        result = [w.load_game(t) for w, t in zip(self.workers, tasks)]
        return [r[0] for r in result], [r[1] for r in result]

    def step(self, actions):
        if len(actions) != self.num_processes:
            raise ValueError('Action batch must match the vector pool')
        result = [w.step(a) for w, a in zip(self.workers, actions)]
        return tuple([r[i] for r in result] for i in range(4))

    def close(self):
        for worker in self.workers:
            worker.close()
