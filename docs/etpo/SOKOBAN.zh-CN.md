# Sokoban 环境与初始 skill

已作为第四个环境接入 ETPO。复用现有 `skillrise` 环境中的 `gym-sokoban==0.0.6`，不需要 Java、外部商品索引或新 Conda 环境。这里使用该库真实的走动、推箱、目标和终止判定，没有用脚本奖励替代推箱过程。

## 来源与技能

- 环境：[mpSchrader/gym-sokoban](https://github.com/mpSchrader/gym-sokoban)，MIT。
- 初始技能：`skills/sokoban/general_skills.md`，为 ETPO 独立编写的 10 条通用解题规则，包括不可逆推箱检查、非目标角落死锁、墙边陷阱、玩家站位、通道推箱顺序、多箱目标分配。
- 检索到了公开的 `sokoban-benchmark-player` skill，但其流程依赖 undo/reset 和结果文件输出，且未发现仓库许可证，因此未复制其文本。当前 skill 不是 SkillZero/SDAR/AgentOPSD 发布的原始技能。
- WebShop、ScienceWorld 的其他现成来源调查见 [来源核查](SKILL-SOURCE-AUDIT.zh-CN.md)。它们的 ETPO 默认 skill 本次未替换。

## 交互与奖励

文本棋盘：`#` 墙、空格地面、`.` 目标、`$` 箱子、`*` 目标上的箱子、`@` 玩家、`+` 目标上的玩家。每轮只接受一个 `<action>up|down|left|right</action>` 中的方向（实际写一个单词，例如 `<action>right</action>`），不支持拉箱、撤销、重开或一次提交多个方向。

所有箱子到达目标时 `won=True`，只在成功首次发生时给 reward=10。其他步骤为 0；不使用 gym-sokoban 的逐步惩罚和箱子上目标的中间奖励。格式错误、撞墙和推不动的动作记为无效，消耗一个步数；默认 ETPO SFT 排除含无效动作的轨迹。结束后的重复 step 返回零奖励，防止向量环境中已结束的行重复累计成功奖励。没有自动死锁检测终止，未解开的死局运行到步数上限。

学生只看到游戏规则、当前棋盘和近期动作反馈；策略性 skill 只注入教师上下文。成功轨迹 SFT、teacher token scoring 和学生联合 PPO 与其他 ETPO 环境共用。

## 任务组织

随仓库附带 6×6、2 箱的 192 个训练关卡和 32 个验证关卡。训练按 64 组×3 个不同关卡组织，每组默认 8 次并行尝试；验证对同一未见初始棋盘重试最多 3 次，报告累计 pass@k。初始棋盘去重包含旋转和镜像，加载时再次检查 train/val 的 task ID 和规范棋盘哈希没有交集。

这是小规模可复现起始集，不是完整 Boxoban 基准或严格的布局 OOD 划分。关卡生成和离线 BFS 最短解验证在预处理/测试阶段完成，动作答案不保存到训练关卡文件，也不提供给模型。详见 [关卡说明](../../data/groups/README.sokoban.md)。

向量环境在 CPU 进程内运行，不为每个轻量棋盘启动 Ray actor；GPU actor 和 vLLM 仍走原有分布式训练架构。加载保存的棋盘，无训练时随机生成成本。此版不保存环境采样 cursor 到 checkpoint，与上游环境相同，不保证恢复后的关卡顺序逐步完全一致。

## 启动

```bash
source activate.sh
bash examples/etpo/train.sh sokoban --cfg job
WANDB_MODE=offline bash examples/etpo/train.sh sokoban
```

默认 Qwen3-4B、8 GPU、训练 batch=16 组、每组 N=8、K=3、每关最多 40 步/40 轮、每次响应最多 256 token、学生 prompt 上限 2048、教师 prompt 上限 4096。训练输入 parquet 由本地空任务占位行构造，不下载 geometry3k；实际关卡由环境 group 文件选取。

一步 GPU 检查：

```bash
WANDB_MODE=offline bash examples/etpo/train.sh sokoban \
  trainer.total_training_steps=1 trainer.val_before_train=false \
  trainer.save_freq=1 trainer.save_start_step=0
```

更换关卡文件：

```bash
bash examples/etpo/train.sh sokoban \
  env.sokoban.train_groups=/path/to/train_K3.jsonl \
  env.sokoban.val_groups=/path/to/val.jsonl
```

固定棋盘决定尺寸和箱子数，旧配置中的 `env.sokoban.dim_room/num_boxes/search_depth` 不会重生成这些已保存关卡。需要新难度时使用离线生成器或提供新的合法棋盘文件，并保持每组 K 与 `env.num_attempts` 一致。BFS 解长不进入 reward 或教师监督。

## 验证

```bash
python -m unittest discover -s tests/etpo -p 'test_*.py' -v
```

Sokoban 测试覆盖真实 backend 的成功奖励、终止锁定、无效动作与步数耗尽、隐藏目标字符、全部 224 个关卡的实际动作执行、分组重复/跨任务切换/验证重试、训练验证交集拒绝，以及真实 TrajectoryCollector + Qwen tokenizer 的轨迹采集、成功筛选与 teacher token 对齐。

结果快照见 [sokoban-backend.json](verification/sokoban-backend.json)。测试中的 BFS 是正确性 oracle，其成功率不是 LLM 成功率。GPU FSDP/vLLM 完整训练与 8×L40 性能仍未实测。
