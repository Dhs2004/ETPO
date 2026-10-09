"""Download public task definitions and convert locally to single-task groups."""
import argparse
import json
from pathlib import Path
from urllib.request import urlopen


def main():
    p = argparse.ArgumentParser()
    p.add_argument('environment', choices=['alfworld', 'webshop', 'sciworld'])
    p.add_argument('--source', help='Local upstream group JSONL instead of downloading')
    p.add_argument('--output-dir', default='data/groups')
    args = p.parse_args()
    out = Path(args.output_dir) / f'etpo_{args.environment}_K1.jsonl'
    if out.exists():
        p.error('Output exists; choose a new directory to preserve the existing split')
    url = f'https://raw.githubusercontent.com/Within-yao/SkillRise/main/data/groups/skillrise_{args.environment}_K3.jsonl'
    text = Path(args.source).read_text() if args.source else urlopen(url, timeout=60).read().decode()
    records, seen = [], set()
    for line in text.splitlines():
        group = json.loads(line)
        for task in group['tasks']:
            if task.get('padded', False):
                continue
            key = json.dumps(task, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            records.append(dict(group_id=f'{args.environment}_K1_{len(records)}',
                                task_type=group['task_type'], K=1, tasks=[task]))
    if not records:
        raise ValueError('No tasks found')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(''.join(json.dumps(x)+'\n' for x in records))
    print(f'Prepared {len(records)} single-task groups at {out}')


if __name__ == '__main__':
    main()
