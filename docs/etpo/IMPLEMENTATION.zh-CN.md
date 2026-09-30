# ETPO 实现说明

本项目从本机 `SDAR/SkillRise` 独立复制到 `SDAR/ETPO`（上游 HEAD 为 `2b38da3ad7b64414305d39609484803b07d3b9b0`，同时继承本机已有环境适配），在原有 verl、环境交互和分组优势估计上实现用户提供的 ETPO 方法。当前完成的是方法实现和 CPU 功能验证；尚未完成 GPU 训练、超参数搜索或论文指标复现。

## 1. 教师的定义与实现选择

本实现将“teacher model 就是学生模型 + skill”解释为**共享模型参数、不同输入上下文**：

- 学生策略：`πθ(y | h)`，输入任务和交互历史，不注入初始 skill。
- 教师策略：`qθ(y | h, S)`，在相同历史的 system 消息中加入初始 skill。
- 教师 SFT 与学生 PPO 更新同一份 actor 参数，复用同一 optimizer 和 checkpoint，不额外驻留独立教师模型。

因此教师 SFT 也会改变学生参数；这不是两份独立权重的 teacher/student 实现。初始 skill 文本在一次训练中固定，教师的“进化”发生在模型权重上。恢复到相同输出目录时会校验 skill 快照，防止悄悄换掉监督上下文。

保留 SkillRise 的每组多个任务、每个任务多次尝试及 reward 优势计算；关闭其 curator/reflection 和 skill 跨任务携带。ETPO 的 rollout 只包含学生 play 阶段。初始 skill 仅用于教师 SFT 和教师打分，不给学生 rollout 使用。

## 2. 每次迭代的更新顺序

1. 当前学生无 skill 与环境交互，保存响应 token、原始 chat、任务类型、任务位置、环境 reward、won 和动作有效性。
2. 在任何 SFT 更新之前计算并保存 `old_log_probs`，作为本轮 PPO 的旧策略分母。
3. 按 `(traj_uid, task_pos)` 聚合成功任务，选出完整任务轨迹。
4. 用相同历史加初始 skill 构造教师上下文，只对选中轨迹的响应 token 做 SFT。
5. 用更新后的共享模型，在所有本轮学生响应上计算教师 token log probability。教师无需生成另一条轨迹。
6. 筛选教师监督，生成 detached token credit；本批 PPO 的所有更新期间这些分数保持固定。
7. 用无 skill 的原学生上下文执行 reward PPO + 蒸馏辅助 PPO。下一轮使用更新后参数，经现有 FSDP/vLLM 同步路径开始 rollout。

没有成功轨迹时跳过 SFT，仍可使用现有 skill-conditioned teacher 打分。评分和 SFT 只使用训练 rollout，验证集不进入教师训练。

## 3. 损失函数

### 3.1 成功轨迹 SFT

设本批选中成功轨迹的有效响应 token 集合为 C：

`L_SFT = - (1 / |C|) Σ[t∈C] log qθ(y_t | h, S, y_<t)`。

不监督 prompt、环境 observation 或 padding。每个 SFT epoch 在完整成功子集上累积梯度后更新一次。默认一个 epoch，学习率为当前 actor 学习率的 0.1 倍；更新完成后恢复原学习率。共享 Adam 状态也会被 SFT 更新，学习率 scheduler 仅随原 PPO 更新路径推进。

### 3.2 置信度筛选和 token credit

记 `s_t = log π_old(y_t | h, y_<t)`，`u_t = log q_afterSFT(y_t | h,S,y_<t)`。

默认保留条件为：

- token 属于有效 response；
- 两个 log probability 都是有限数；
- `1e-4 ≤ exp(u_t) ≤ 0.99`；
- `|u_t - s_t| ≤ 5`。

保留时 `D_t = stop_gradient(u_t - s_t)`，否则为零。实现通过 `where` 去除 NaN/Inf，避免零乘无穷污染损失。正 credit 鼓励教师认为相对更好的 token，负 credit 抑制相对更差的 token。

这里过滤的是 **softmax 归一化后的采样 token 概率**，不是未归一化 logits。阈值是可调启发式，不能保证留下的指导都正确。仅计算学生实际生成 token 的概率，不存储完整词表分布，也不计算 full-vocabulary KL。

### 3.3 学生联合优化

记 `r_t(θ) = exp(log πθ(y_t|h,y_<t) - s_t)`，PPO surrogate 的基本形式为：

`L_clip(A) = mean_t[-min(r_t A_t, clip(r_t,1-ε_low,1+ε_high) A_t)]`。

最终使用：

`L_student = L_SkillRise_reward_PPO + α L_clip(D)`，默认 `α=0.1`。

实际 clipping（含负 advantage 的 dual clipping）调用原有 `compute_policy_loss`，复用配置中的边界。辅助项使用 token clipping；reward advantages 保持原值，不被教师分数覆盖。默认 loss aggregation 为有效 response token 均值，过滤掉的 token 仍计入归一化分母，因此保留率低时辅助项自然变弱。若手动改变上游 `loss_agg_mode`，归一化随之改变。

辅助项是基于采样 token 的 reverse-KL 策略梯度启发的 clipped surrogate，并非直接对两份完整分布求 KL。由于先进行共享参数 SFT，再进行 PPO，PPO 起点已经偏离 rollout 策略；仍保留 rollout 旧分母以反映这种变化。应监测 clipping 和 KL，必要时降低 SFT 学习率或 epoch 数。

RL 直接使用任务 reward，但不保证最终成功；SFT 和概率筛选同样不保证教师可靠性。方法效果需要实验验证。

## 4. 成功筛选与分布式正确性

成功条件默认同时满足：原始环境 `won=True`、该任务环境 reward 总和至少 10、所有动作有效。不会根据教师概率、shaped advantage 或其他任务的成功挑选样本。

先按任务位置和 turn 去掉 rollout 批次补齐产生的重复行，再计算 reward；避免重复 reward 或同一组内任务成功串用。默认最多选择 8 个完整成功任务，使用 `seed + global_step` 确定性打乱后选取，不在任务中间截断。

SFT 子集补齐到 worker world size 的倍数，补齐行监督 mask 为零。每个 rank 使用 `world_size / 全局有效token数` 缩放本地负 log probability 总和，以抵消 DDP/FSDP 的梯度平均。不同 rank 成功长度不同、或某个 rank 只有补齐行时仍对应全局 token 均值。

教师 prompt 重新套用同一 chat template，但 response IDs 和 mask 直接复制学生结果，绝不 decode 后重新 tokenize 响应。构造前检查原始 chat 重渲染与 rollout prompt token 完全一致；不一致或超预算直接报错，不静默截断。默认 skill 最多 2048 token，教师 prompt 最多 16384 token，响应另计。

## 5. 初始 skill

| 环境 | 初始内容 | 来源 |
| --- | --- | --- |
| ALFWorld | general + 按任务类型选择的 specific skill | SkillZero，提交 `1980cd589036ec9f04c64749bbd317a8dd0eef7a` |
| WebShop | 搜索、筛选、产品属性核对和购买动作通用指导 | 本次为 ETPO 编写 |
| ScienceWorld | 观察、导航、实验操作和目标核对通用指导 | 本次为 ETPO 编写 |

SkillZero 原始参考不提供此处的 WebShop/ScienceWorld skill，不将自编内容表述为其产物。ALFWorld 补充了 `pick_and_place_simple` 映射别名；未知类型回退到 general。来源与许可证见 `skills/SOURCES.md`、`skills/LICENSE.SkillZero` 和 `NOTICE`。

每次运行在 `trainer.default_local_dir/etpo_initial_skills.json` 保存文本、映射及 SHA-256。改 skill 做新实验时使用新输出目录。成功轨迹可能仍包含多余动作，此版本不会自动提炼或改写 skill 文本。

## 6. 代码位置

| 文件 | 职责 |
| --- | --- |
| `etpo/config.py` | 支持范围和参数校验 |
| `etpo/skills.py` | 初始 skill 加载、路由、快照 |
| `etpo/data.py` | 教师上下文、token 对齐、SFT mask 与补齐 |
| `etpo/core.py` | 成功筛选、概率 gate、蒸馏 credit、SFT token loss |
| `etpo/training.py` | SFT → 教师评分 → credit 的驱动逻辑 |
| `agent_system/multi_turn_rollout/skillrise_rollout_loop.py` | 学生无 skill rollout、环境结果元数据 |
| `verl/trainer/ppo/ray_trainer.py` | reward advantage 后、actor PPO 前接入 ETPO |
| `verl/workers/fsdp_workers.py` | SFT 分布式 RPC、模型与 optimizer offload 管理 |
| `verl/workers/actor/dp_actor.py` | 真正的 SFT backward/step 和蒸馏 PPO 辅助项 |
| `verl/trainer/config/ppo_trainer.yaml` | 默认关闭的 ETPO 配置 |
| `examples/etpo/train.sh` | 三个环境统一启动入口 |

同时修复继承代码中小批次补齐不足的问题，以及 CPU 验证时 CUDA 专用交叉熵与显存日志调用问题。原始 SkillRise 目录不依赖这些 ETPO 修改。

## 7. 运行与共用环境

以下绝对路径记录本机安装。GitHub 新 checkout 使用相对路径 `source activate.sh`，安装、数据来源和路径覆盖见 [setup/README.md](../../setup/README.md)。

项目：`/mnt/bn/vai3d-hl-lwb/user/dhs/SDAR/ETPO`。

共用环境：`/mnt/bn/vai3d-hl-lwb/user/dhs/miniconda3/envs/skillrise`。不需要再安装一份 torch/vLLM。项目 activation 会设置本项目优先的 PYTHONPATH、Java、数据和输出路径：

```bash
source /mnt/bn/vai3d-hl-lwb/user/dhs/bashrc
conda activate skillrise
source /mnt/bn/vai3d-hl-lwb/user/dhs/SDAR/ETPO/activate.sh
```

默认模型为共享的本地 Qwen3-4B。ALFWorld 数据、ScienceWorld jar、WebShop 数据继续使用已有 setup；WebShop Lucene 索引链接到原 SkillRise 的 1000 商品索引，因此不能删除该索引目录。此数据规模不等同于完整 WebShop benchmark。

在有 8 张 L40 的容器上，三个任务分别启动：

```bash
WANDB_MODE=offline bash examples/etpo/train.sh alfworld
WANDB_MODE=offline bash examples/etpo/train.sh webshop
WANDB_MODE=offline bash examples/etpo/train.sh sciworld
```

脚本已设置 8 GPU、actor/logprob microbatch=1、梯度检查点和 sequence parallel=1。参数与 optimizer offload 继承各环境脚本设置（ALFWorld 开启，ScienceWorld 默认关闭），可用 `actor_rollout_ref.actor.fsdp_config.param_offload=true actor_rollout_ref.actor.fsdp_config.optimizer_offload=true` 显式开启以节约 GPU 显存。共享权重避免额外一份教师常驻，但教师上下文更长，且每轮增加 SFT 与教师评分计算。8×L40 是本配置的目标硬件；显存峰值、吞吐和完整训练稳定性尚未实测，不作一定可跑的承诺。

建议先做一步 GPU 冒烟测试（会包含末步验证，可能耗时）：

```bash
WANDB_MODE=offline bash examples/etpo/train.sh alfworld \
  trainer.total_training_steps=1 trainer.val_before_train=false \
  trainer.save_freq=1 trainer.save_start_step=0
```

该命令仍使用原有 rollout 批量和验证集规模，以保持分组/整除关系。如需缩小批量，应一起检查 `data.train_batch_size`、`env.rollout.n`、`ppo_mini_batch_size` 与 GPU 数的关系。发生 OOM 时先缩小教师 prompt 预算或环境历史长度、降低 vLLM 显存占比，并检查成功子集规模；避免仅增大上下文上限。

无 GPU 时检查最终 Hydra 配置：

```bash
bash examples/etpo/train.sh alfworld --cfg job
bash examples/etpo/train.sh webshop --cfg job
bash examples/etpo/train.sh sciworld --cfg job
```

`--cfg` 不执行模型训练；上游脚本仍会检查依赖、准备输入表和输出目录。训练输出位于本项目 `outputs/etpo_<environment>/...`。默认通过现有 verl checkpoint 保存共享模型和 optimizer；恢复时需复用输出目录和相同 skill 快照。分布式 checkpoint 恢复还需 GPU 验证。

## 8. 参数、消融与监测

| 参数 | 默认 | 作用 |
| --- | --- | --- |
| `etpo.enabled` | false（入口设 true） | 启用 ETPO |
| `etpo.teacher_mode` | shared | 只支持共享参数 |
| `etpo.evolve_teacher` | true | 成功轨迹 SFT |
| `etpo.distill_coef` | 0.1 | 蒸馏辅助权重 |
| `etpo.confidence_min/max` | 0.0001 / 0.99 | 采样 token 教师概率区间 |
| `etpo.max_abs_log_ratio` | 5 | 教师/旧学生 log ratio 绝对值上限 |
| `etpo.success_reward_min` | 10 | 任务累计成功 reward 阈值 |
| `etpo.require_valid_actions` | true | 排除含无效动作的成功任务 |
| `etpo.max_success_trajectories` | 8 | 每轮成功任务数量上限 |
| `etpo.sft_epochs` | 1 | 成功子集更新次数 |
| `etpo.sft_micro_batch_size` | 1 | 每 rank SFT microbatch |
| `etpo.teacher_sft_lr_scale` | 0.1 | 相对当前 actor LR |
| `etpo.teacher_max_prompt_length` | 16384 | 教师 prompt 上限 |
| `etpo.max_skill_tokens` | 2048 | skill 文本 token 上限 |

消融以启动参数覆盖：

- `etpo.evolve_teacher=false`：移除成功 SFT，保留 skill-conditioned 蒸馏。由于共享 actor 仍被 PPO 更新，这不等于冻结独立教师。
- `etpo.distill_coef=0`：保留成功 SFT 和 reward RL，不计算教师蒸馏评分。
- 两者同时关闭：ETPO rollout 结构下的 reward RL 基线；仍加载/校验 skill 上下文。
- `etpo.enabled=false`：恢复上游 SkillRise 路径，包含其自身机制，不等于上述纯 RL 消融。

关注 `etpo/successful_tasks`、`etpo/sft_tasks`、`etpo/sft_skipped`、`etpo/sft_loss`、`etpo/sft_grad_norm`、`etpo/retained_fraction`、`etpo/credit_mean`、`etpo/teacher_nonfinite_tokens`、`etpo/distill_loss`，并与 reward、成功率、`actor/pg_clipfrac` 和 `actor/ppo_kl` 联合判断。低保留率先检查教师概率分布，不应直接视为实现出错。

目前只支持三个文本环境、FSDP/FSDP2 actor、sequence parallel=1、temperature=1、raw chat、禁止 prompt 截断；不支持独立教师、packed multi-turn、视觉输入。FSDP2 和 GPU FSDP 虽接入同一代码路径，尚未实测。

## 9. 验证记录与边界

```bash
python -m unittest discover -s tests/etpo -p 'test_*.py' -v
torchrun --standalone --nnodes=1 --nproc_per_node=2 tests/etpo/distributed_sft_smoke.py
```

本机已通过 13 项测试：skill 来源/映射、成功筛选去重和任务隔离、过滤符号/NaN/Inf/全拒绝、精确 token 对齐、零权重补齐、无成功跳过、skill 快照一致性、配置拒绝路径、真实 collector 与 Qwen tokenizer、可训练小模型完整 SFT→PPO 更新与模型/optimizer 重载，以及小型真实 Qwen3 模型 forward/backward。

双进程 CPU DDP/Gloo 测试中，一个 rank 只有零监督补齐行；两个 rank 参数一致，与单进程全局 token 均值参考的最大参数差为 **0.0**。详见 `logs/unit-tests.log`、`logs/distributed-sft.log`、`logs/distributed-sft.json`。

三个环境的最终 Hydra 配置均成功解析，并通过 ETPO 参数、初始 skill、8 GPU、microbatch 和输出目录检查。汇总见 `logs/verification.json`；可在生成配置日志后运行 `python tests/etpo/validate_launch_configs.py` 重新校验。

这些结果证明被测试的更新顺序、张量逻辑和 CPU 分布式归一化工作正常，不能替代 CUDA、FlashAttention、GPU FSDP、vLLM 权重同步、真实环境完整训练和分布式 checkpoint 的验证。当前容器无可用 GPU，因此没有训练收益、最终成功率或论文结果可报告。
