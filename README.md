# ETPO

Anonymous method implementation for skill-conditioned teacher guidance in multi-turn reinforcement learning.

ETPO initializes an independent teacher from the student's initial model. The student interacts with the environment without teacher skills. The teacher scores the same sampled response under a skill-augmented context. Its token log-probability gap relative to the rollout policy rescales the signed GRPO advantage before the clipped policy update. There is no additional student KL distillation objective.

Successful complete trajectories enter a persistent buffer. Once a teacher update batch is available, the teacher performs supervised learning with normalized inverse-success-rate trajectory weights. Teacher parameters and optimizer are separate from the student. The initial skill text remains fixed.

- [Method and objective](docs/METHOD.md)
- [Installation and dataset download/configuration links](docs/DATA.md)
- [Runtime paths](setup/paths.example.sh)
- [Shared configuration](verl/trainer/config/ppo_trainer.yaml)
- [Training entry point](examples/etpo/train.sh)
- [Skill attribution](skills/SOURCES.md)

## Running

Follow `docs/DATA.md`, activate the dependencies, and configure local paths:

```bash
cp setup/paths.example.sh config.local.sh
# Edit the paths in config.local.sh.
source config.local.sh
source activate.sh
python scripts/prepare_groups.py alfworld
bash examples/etpo/train.sh alfworld --cfg job
bash examples/etpo/train.sh alfworld
```

Replace `alfworld` with `webshop` or `sciworld` after preparing their resources and task groups. For Sokoban, generate local levels as documented in `docs/DATA.md`.

Default launch settings use single-task groups (K=1), disable top-token exclusion and KL penalties, and save checkpoints every 50 steps. The rollout group size is distinct from task-group K. Numeric configuration values are algorithm/runtime settings, not reported experimental results.

Ablations use separate output directories:

```bash
bash examples/etpo/train.sh alfworld etpo.sft_success_rate_weighting=false
bash examples/etpo/train.sh alfworld etpo.evolve_teacher=false
```

The first disables trajectory weighting in teacher SFT; the second keeps teacher guidance but freezes teacher parameters. Neither adds a student distillation loss.

## Scope

This repository contains source code, configurations, and initial skill prompts. It contains no experimental results, training logs, checkpoints, evaluation trajectories, paper files, or benchmark records. External datasets are downloaded locally. A clean-machine GPU installation and full training run are required to validate a new hardware/software stack.

Upstream copyright and license notices are retained. ETPO-specific authorship and commit metadata are anonymous; the hosting account remains visible on GitHub.
