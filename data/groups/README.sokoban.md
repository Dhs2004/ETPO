# ETPO Sokoban level split

This is a small procedural starter dataset, not the official Boxoban benchmark.
Generated with gym-sokoban 0.0.6 using seed 20260930, 6×6 rooms, 2 boxes, 25 room
construction steps. Only initially unsolved boards with an offline BFS solution
of 4–30 moves are retained. Exact full starting boards are deduplicated across
both splits, including rotations/reflections; related layouts with different
starting states can still occur.

- `etpo_sokoban_train_K3.jsonl`: 64 groups × 3 distinct tasks = 192 levels.
- `etpo_sokoban_val.jsonl`: 32 held-out levels (K=1 in the file). The evaluator
  retries each same level up to the configured number of attempts.
- Fields: `task_id` (canonical board SHA-256), `board` (ASCII rows),
  `generation_seed`. No solution or optimal action is stored in the files.

To reproduce in a NEW directory (existing files are never overwritten):

```bash
source activate.sh
python -m examples.etpo.prepare_sokoban \
  --output-dir "$ETPO_ROOT/.runtime/sokoban-input" \
  --generate-levels --group-dir "$ETPO_ROOT/.runtime/new-sokoban-groups" \
  --train-groups 64 --val-levels 32 --k 3 --seed 20260930
```

The generator and test-only BFS live in `etpo/sokoban_levels.py`. Training loads
saved boards and calls the real gym-sokoban transition engine; it does not call
this solver or expose solutions in student/teacher prompts. The bundled boards
were all solved by the test oracle in the native environment, which verifies
backend correctness, not model skill.
