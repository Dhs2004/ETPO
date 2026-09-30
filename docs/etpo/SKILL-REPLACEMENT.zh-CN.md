# WebShop / ScienceWorld 默认 skill 替换

本次按用户要求替换原自编默认文件。运行入口和配置无需修改，新训练自动使用替换后的内容。

| 环境 | 当前默认来源 | 处理方式 |
| --- | --- | --- |
| WebShop | SDAR `skills/webshop/general_skills.md` | 原文复制，15 条通用策略 |
| ScienceWorld | SkillNet 的 8 个 ScienceWorld skill | 筛选、压缩并适配为一个通用文件 |

两者继续通过 `skill_files.general_skills` 加载。当前需求是替换通用 skills，因此未启用 WebShop 的 7 类专门技能或把 ScienceWorld 的 61 个文件全部注入上下文。学生 rollout 仍不带 skill，教师 SFT 和教师打分使用新内容。

## 固定来源

- SDAR：`308153bb5f7d63cd57f301f432d616a072220e1b`，Apache-2.0。
- SkillNet：`358ec824577e8a5aa6857c0eff268e492703606b`，MIT，Copyright (c) 2026 ZJUNLP。

完整 URL、来源文件 SHA-256、运行文件 SHA-256 和 ScienceWorld 的逐节映射见 [skills/provenance.json](../../skills/provenance.json)。许可证分别保存在 `skills/LICENSE.SDAR` 和 `skills/LICENSE.SkillNet`。

## ScienceWorld 适配

| 默认文件章节 | SkillNet 原始技能 |
| --- | --- |
| 1. 任务解析 | scienceworld-task-parser |
| 2. 房间观察 | scienceworld-room-scanner；导航条件参考 task-parser |
| 3. 获取和放置物品 | scienceworld-inventory-manager |
| 4. 容器与设备检查 | scienceworld-container-inspector |
| 5. 工具操作 | scienceworld-tool-user |
| 6. 温度测量 | scienceworld-temperature-measurer |
| 7. 动作消歧 | scienceworld-ambiguous-action-resolution |
| 8. 条件等待 | scienceworld-controlled-waiting |

原始文件保存在 `skills/sources/skillnet/scienceworld/`，用于追溯，不在运行映射中，教师不会直接加载这些原文件。

适配保留源文件的操作流程，去掉具体任务的示例答案、固定阈值/箱子颜色、对未附带脚本的调用和未验证的库存容量假设。将“所有容器已打开”“必须传送”“所有任务先 focus”等表述改为遵循实际任务与可用动作。动作以 ETPO 所需的 `<action>...</action>` 输出；消歧数字也放在动作标签内。不会对每道任务自动附加加热、聚焦或等待步骤。

WebShop 文件保持 SDAR 原文。其筛选、排序、缩略图等建议只在界面确实提供这些信息或控件时适用；教师上下文已声明当前任务与观测优先。复用上游技能不代表其每条启发式都最优，也没有因此产生已验证的训练收益。

## 使用与复查

```bash
source activate.sh
bash examples/etpo/train.sh webshop
bash examples/etpo/train.sh sciworld
```

启动默认生成新的输出目录。旧运行目录中的 `etpo_initial_skills.json` 保留原技能快照；更换 skill 后复用旧目录会触发已有一致性检查，不能覆盖快照以伪装成同一实验。需要延续旧实验时，从对应 Git 提交或已有快照恢复旧 skill。

本次检查使用真实 Qwen3 tokenizer，验证新文件低于 `max_skill_tokens=2048`，按未知任务类型调用也会加载对应 general skill，并验证来源哈希及已有快照保护测试。结果见 [skill-replacement.json](verification/skill-replacement.json)。未运行 GPU 训练，也未测量替换后的成功率。
