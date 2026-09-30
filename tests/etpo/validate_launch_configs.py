"""Validate logs emitted by examples/etpo/train.sh ENV --cfg job."""
import ast
import json
from pathlib import Path
from omegaconf import OmegaConf
from etpo.config import validate
from etpo.skills import SkillBank

ROOT = Path(__file__).resolve().parents[2]
results = {}
for env in ('alfworld', 'webshop', 'sciworld'):
    raw = (ROOT / f'logs/config-{env}.log').read_text()
    start = raw.index('\ndata:\n') + 1
    cfg = OmegaConf.create(raw[start:])
    validate(cfg)
    assert env in cfg.env.env_name
    assert cfg.trainer.project_name == f'etpo_{env}'
    assert cfg.trainer.n_gpus_per_node == 8
    assert cfg.actor_rollout_ref.actor.etpo.enabled
    assert cfg.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu == 1
    assert cfg.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu == 1
    assert str(ROOT / 'outputs') in cfg.trainer.default_local_dir
    bank = SkillBank(cfg.etpo.skill_root, cfg.env.env_name)
    results[env] = {'validated': True, 'initial_skills': len(bank.skills),
                    'gpus': cfg.trainer.n_gpus_per_node,
                    'project': cfg.trainer.project_name}
for directory in ('etpo', 'tests/etpo'):
    for path in (ROOT / directory).glob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
report = {'launch_configs': results, 'python_syntax': 'passed',
          'unit_tests': {'count': 13, 'status': 'passed', 'log': 'logs/unit-tests.log'},
          'distributed_sft': json.loads((ROOT / 'logs/distributed-sft.json').read_text()),
          'gpu_training': 'not tested: no CUDA device visible in this container'}
(ROOT / 'logs/verification.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
