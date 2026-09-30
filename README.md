# ETPO

在 SkillRise 上实现 reward RL、过滤后的 token-level 自蒸馏，以及成功轨迹驱动的教师 SFT 进化。教师采用**学生共享权重 + 初始 skill 上下文**；学生 rollout 不带 skill。初始 skill 文本固定，SFT/PPO 共同更新模型参数。

基于 [SkillRise](https://github.com/Within-yao/SkillRise) 实现，可共用现有 `skillrise` Conda 环境、模型和环境资源。新机器请先阅读 [环境、模型与数据准备](setup/README.md)。

```bash
git clone https://github.com/Dhs2004/ETPO.git
cd ETPO
conda activate skillrise
# 新机器先按 setup/README.md 配置并 source config.local.sh
source activate.sh

# 8 GPU；环境可替换为 webshop 或 sciworld
WANDB_MODE=offline bash examples/etpo/train.sh alfworld

# 只检查配置，不需要 GPU
bash examples/etpo/train.sh alfworld --cfg job
```

ALFWorld 初始 skills 来自 SkillZero；WebShop、ScienceWorld 的通用 skills 为本项目编写。见 [来源](skills/SOURCES.md)。

完整设计、损失公式、更新顺序、成功筛选、参数、消融与运行说明见 [ETPO 实现文档](docs/etpo/IMPLEMENTATION.zh-CN.md)。

已通过 13 项 CPU 测试（含真实小型 Qwen3 参数更新）及双进程 CPU SFT 归一化验证。当前容器没有 GPU，尚未验证 8×L40 完整训练，也未复现论文指标。

```bash
python -m unittest discover -s tests/etpo -p 'test_*.py' -v
torchrun --standalone --nnodes=1 --nproc_per_node=2 tests/etpo/distributed_sft_smoke.py
```

上游代码保留原许可证与 NOTICE；SkillZero 内容许可证见 `skills/LICENSE.SkillZero`。

配置入口：`verl/trainer/config/ppo_trainer.yaml`；路径示例：`setup/paths.example.sh`；依赖约束及版本快照：`setup/`；可随仓库查看的 [验证记录](docs/etpo/verification/verification.json)。模型、数据、环境目录和运行日志不上传。
