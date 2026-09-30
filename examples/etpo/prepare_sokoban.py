"""Build standalone input tables; optionally regenerate the bundled level split."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--train-rows', type=int, default=16)
    parser.add_argument('--val-rows', type=int, default=32)
    parser.add_argument('--generate-levels', action='store_true')
    parser.add_argument('--group-dir', default=str(Path(__file__).resolve().parents[2] / 'data/groups'))
    parser.add_argument('--train-groups', type=int, default=64)
    parser.add_argument('--val-levels', type=int, default=32)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--seed', type=int, default=20260930)
    args = parser.parse_args()
    if min(args.train_rows, args.val_rows, args.train_groups, args.val_levels, args.k) < 1:
        parser.error('Counts must be positive')
    if args.generate_levels:
        from etpo.sokoban_levels import generate_tasks
        group_dir = Path(args.group_dir)
        group_dir.mkdir(parents=True, exist_ok=True)
        paths = [group_dir / f'etpo_sokoban_train_K{args.k}.jsonl', group_dir / 'etpo_sokoban_val.jsonl']
        if any(p.exists() for p in paths):
            parser.error('Refusing to overwrite a level split; choose a fresh --group-dir')
        n_train = args.train_groups * args.k
        tasks = generate_tasks(n_train + args.val_levels, seed=args.seed)
        for path, split_tasks, k in [(paths[0], tasks[:n_train], args.k), (paths[1], tasks[n_train:], 1)]:
            groups = [{'group_id': i // k, 'task_type': 'sokoban', 'K': k, 'tasks': split_tasks[i:i+k]}
                      for i in range(0, len(split_tasks), k)]
            path.write_text(''.join(json.dumps(g) + '\n' for g in groups))
        print(f'Wrote {n_train} training and {args.val_levels} validation levels')
    import pyarrow as pa
    import pyarrow.parquet as pq
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for split, count in [('train', args.train_rows), ('test', args.val_rows)]:
        records = [{'data_source': 'text', 'prompt': [{'role': 'user', 'content': ''}],
                    'ability': 'agent', 'extra_info': {'split': split, 'index': i}}
                   for i in range(count)]
        pq.write_table(pa.Table.from_pylist(records), output / f'{split}.parquet')
    print('Input tables:', output)


if __name__ == '__main__':
    main()
