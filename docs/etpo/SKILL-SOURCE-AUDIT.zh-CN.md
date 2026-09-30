# WebShop / ScienceWorld 初始 skill 来源核查

核查日期：2026-09-30。仅调查公开源码和技能资产，本次未替换 ETPO 的训练 skills。

| 项目 | 核查提交 | WebShop | ScienceWorld |
| --- | --- | --- | --- |
| ZJU-REAL/SkillZero（Skill0） | 1980cd589036ec9f04c64749bbd317a8dd0eef7a | skills 目录没有，仅 ALFWorld/Search | 未发现 |
| JasonZhujp/Skill0_5 | 703e635619901c9c84f76caff0907a37d1a262a8 | memory_data/webshop_ood/ 下有 ID/OOD skill JSON | 未发现 |
| ZJU-REAL/SDAR | 308153bb5f7d63cd57f301f432d616a072220e1b | skills/webshop/general_skills.md + 7 类特定技能 | 未发现对应技能资产 |
| ZethWang/AgentOPSD | 0c478b2d7cdc201d9b1f076ec5b3dec7e88a161b | skills/webshop/，通用 Markdown 与 SDAR 字节一致 | 未发现 |
| zjunlp/SkillNet | 358ec824577e8a5aa6857c0eff268e492703606b | experiments/src/skills/webshop/：23 个 SKILL.md | experiments/src/skills/scienceworld/：61 个 SKILL.md |

“未发现”限于本次核查的公开版本，不表示所有分支、历史版本或未公开数据都没有。SkillZero 与 SDAR 当前远端 HEAD 已核对；另外三个项目读取的是当次最新公开 HEAD。

## 直接来源

- https://github.com/ZJU-REAL/SkillZero/tree/1980cd589036ec9f04c64749bbd317a8dd0eef7a/skills
- https://github.com/ZJU-REAL/SDAR/tree/308153bb5f7d63cd57f301f432d616a072220e1b/skills/webshop
- https://github.com/ZethWang/AgentOPSD/tree/0c478b2d7cdc201d9b1f076ec5b3dec7e88a161b/skills/webshop
- https://github.com/JasonZhujp/Skill0_5/tree/703e635619901c9c84f76caff0907a37d1a262a8/memory_data/webshop_ood
- https://github.com/zjunlp/SkillNet/tree/358ec824577e8a5aa6857c0eff268e492703606b/experiments/src/skills/scienceworld
- https://github.com/zjunlp/SkillNet/tree/358ec824577e8a5aa6857c0eff268e492703606b/experiments/src/skills/webshop

## WebShop 复用选择

优先 SDAR：其 general_skills.md 有 15 条通用策略，另有 apparel、footwear、home_decor、electronics、accessories、beauty_health、other 共 7 类特定技能与 skill_mapping.json。格式最接近 ETPO 当前 general + task-specific 加载结构。SDAR 根许可证为 Apache-2.0，复用时保留许可和归属。

AgentOPSD 的 general_skills.md 与 SDAR 对应文件逐字节一致。Skill0.5 两份 WebShop JSON 均包含 15 条 general_skills、12 条 common_mistakes；ID 有 4 类 task_specific_skills，OOD 有 3 类。已验证其 ID 的 general_skills 数组与 SDAR claude_style_skills.json 完全相同。因此不是三个独立构建的通用技能库。当前 Skill0.5 根目录未发现 LICENSE 文件，直接复用时优先采用许可明确的 SDAR 来源。

SDAR 中“筛选/排序/缩略图”等建议仍需与 ETPO 的文本界面实际可用动作匹配；有现成文件并不代表每条操作均可无修改执行。接入分类技能时还需核对 ETPO 环境输出的任务类型与分类映射。

## ScienceWorld 复用选择

SkillNet 是本次找到的明确公开资产来源。通过未截断的递归 Git tree 计数：ScienceWorld 61 个 SKILL.md、WebShop 23 个 SKILL.md。其 experiments/README.md 提供 ScienceWorld --use_skill 运行方式，因此不只是提到环境名称。

实际抽查六个 ScienceWorld 文件：room-scanner、task-interpreter、object-selector、circuit-builder、room-teleporter、task-focuser。每个文件包括技能描述与操作步骤，部分包括完整任务示例。根 LICENSE 为 MIT。

建议抽取经过核对的通用观察、任务理解、对象消歧、测量规则，再根据任务类型选择专门操作技能。不要把 61 个文件全部注入每个教师 prompt：ETPO 有 skill token 预算，且不相关步骤会干扰监督。

需要适配的实际例子：circuit-builder 写死蓝箱代表导体、橙箱代表非导体；room-teleporter 假定允许 teleport；task-focuser 把 focus on 广泛描述为测量等操作的前置要求。应服从实际任务指令、环境可用动作和成功规则，不能把这些例子提升为所有任务的通用规则。技能构建轨迹与 ETPO 训练/测试变体的重合关系尚未核实；正式报告实验时应记录其外部知识来源及使用范围。

附加排查了 AlphaLab-USTC/Skill1 和 jinyangwu/OPID 的公开文件树；OPID 包含 ScienceWorld 环境源码，但环境实现不能作为已发布初始 skill 的证据，因此未列为可直接复用的 ScienceWorld skill 来源。
