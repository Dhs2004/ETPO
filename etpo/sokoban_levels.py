"""Offline level construction/verification; never called by a training policy."""
from collections import deque
import random
import numpy as np
from agent_system.environments.skillrise_sokoban.envs import DIRECTIONS, parse_board, render_board, board_key


def solve_board(lines, max_steps=60, max_states=100000):
    """Bounded BFS oracle, used only to verify level solvability and backend tests."""
    fixed, state = parse_board(lines)
    start = (tuple(np.argwhere(state == 5)[0]), tuple(sorted(map(tuple, np.argwhere(np.isin(state, [3, 4]))))))
    goals = set(map(tuple, np.argwhere(fixed == 2)))
    queue = deque([(start, 0)])
    parents = {start: None}
    while queue:
        current, depth = queue.popleft()
        player, boxes_tuple = current
        boxes = set(boxes_tuple)
        if boxes == goals:
            path = []
            while parents[current] is not None:
                current, action = parents[current]
                path.append(action)
            return list(reversed(path))
        if depth >= max_steps:
            continue
        for action, (dr, dc) in DIRECTIONS.items():
            nxt = (player[0] + dr, player[1] + dc)
            if fixed[nxt] == 0:
                continue
            new_boxes = boxes
            if nxt in boxes:
                dest = (nxt[0] + dr, nxt[1] + dc)
                if fixed[dest] == 0 or dest in boxes:
                    continue
                new_boxes = (boxes - {nxt}) | {dest}
            successor = (nxt, tuple(sorted(new_boxes)))
            if successor not in parents:
                if len(parents) >= max_states:
                    return None
                parents[successor] = (current, action)
                queue.append((successor, depth + 1))
    return None


def generate_tasks(count, seed=0, dim_room=(6, 6), num_boxes=2, max_steps=30):
    """Reverse-generate puzzles; verify solvability and remove symmetric duplicates."""
    from gym_sokoban.envs.room_utils import generate_room
    old_py, old_np = random.getstate(), np.random.get_state()
    tasks, seen = [], set()
    try:
        for candidate in range(count * 100):
            candidate_seed = seed + candidate
            random.seed(candidate_seed)
            np.random.seed(candidate_seed)
            try:
                fixed, state, _ = generate_room(dim=dim_room, num_steps=25, num_boxes=num_boxes)
            except (RuntimeError, RuntimeWarning, ValueError):
                continue
            lines = render_board(fixed, state)
            key = board_key(lines)
            if key in seen:
                continue
            solution = solve_board(lines, max_steps=max_steps)
            if not solution or len(solution) < 4:
                continue
            seen.add(key)
            tasks.append({'task_id': key, 'board': lines, 'generation_seed': candidate_seed})
            if len(tasks) == count:
                return tasks
    finally:
        random.setstate(old_py)
        np.random.set_state(old_np)
    raise RuntimeError(f'Generated only {len(tasks)}/{count} distinct solvable boards')
