# 环境与数据准备

现有安装可以直接复用 `skillrise` 环境。根目录 `activate.sh` 会优先加载相邻的 `skillrise-setup/activate.sh`；若不存在，使用当前已激活的环境和可配置路径。下述新机器安装步骤针对 Linux x86_64、Python 3.10、CUDA 12 系列；尚未在全新机器上完整重装验证。

## 依赖

```bash
conda env create -f setup/environment.yml
conda activate skillrise
python -m pip install 'setuptools<81' wheel
python -m pip install -r setup/requirements-core.txt -c setup/constraints.txt
python -m pip install -r setup/requirements-web-science.txt -c setup/constraints.txt
python -m pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

FlashAttention 使用与本机 Python、PyTorch、CUDA 和 C++ ABI 匹配的 wheel。已验证的组合：Python 3.10、torch 2.6.0、vLLM 0.8.5.post1、transformers 4.51.3、flash-attn 2.7.4.post1；对应 wheel 为：

```bash
python -m pip install --no-deps 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'
```

其他硬件/ABI 不应直接使用此 wheel。`constraints.txt` 是关键兼容约束；`requirements-resolved.txt` 是本机已安装版本快照，用于核对，并非跨平台 lockfile。它将本机 wheel 路径归一化为版本号，不包含本机路径或私有下载地址。根目录的 `requirements.txt` 保留上游清单。

本机曾修复 decord 0.6.0、TextWorld 1.7.0 上游 wheel 的平台元数据；若安装后 `pip check` 报对应错误：

```bash
mkdir -p setup/wheels/upstream
python -m pip download --no-deps --only-binary=:all: -d setup/wheels/upstream decord==0.6.0 textworld==1.7.0
python setup/repair_wheel_metadata.py
python -m pip install --no-deps --force-reinstall setup/wheels/metadata-fixed/*.whl
python -m pip check
```

修复只调整 WHEEL Tag 并更新 RECORD，不修改二进制。下载与模型文件不进入 Git。

## 模型和路径

在仓库根目录执行：

```bash
cp setup/paths.example.sh config.local.sh
# 编辑 config.local.sh；已有模型可直接设置 SKILLRISE_MODEL_PATH
source config.local.sh
source activate.sh
```

可通过 Hugging Face 下载默认模型：

```bash
huggingface-cli download Qwen/Qwen3-4B --local-dir "$SKILLRISE_MODEL_PATH"
```

`config.local.sh` 不会被自动加载，需显式 source。共享安装存在时，其 activation 会提供已有数据/Java 默认路径；新机器使用上述配置。不要安装其他仓库的 verl 覆盖此项目，activation 用 PYTHONPATH 选择本仓库源码。

## 环境资源

模型、ALFWorld 游戏、WebShop 商品与索引、ScienceWorld jar 不随仓库上传。`data/groups/*.jsonl` 是保留的训练任务分组定义，不是模型 rollout。

- **ALFWorld**：使用官方 [ALFWorld releases](https://github.com/alfworld/alfworld/releases) 的文字环境游戏资源；可用已安装的 `alfworld-download` 工具准备资源（该工具也可能下载视觉资源）。本机使用的 JSON/PDDL/TextWorld 包名及 SHA-256 见 `download-checksums.json`。在 `ALFWORLD_DATA` 下应能找到 `json_2.1.1/train/...` 等游戏和 `logic/alfred.pddl`、`logic/alfred.twl2`；logic 模板在本仓库 `agent_system/environments/alfworld/alfworld/data/`。训练分组中的相对游戏路径必须实际存在。
- **WebShop**：从 [YWZBrandon/webshop-data](https://huggingface.co/datasets/YWZBrandon/webshop-data) 获取 `items_shuffle_1000.json`、`items_ins_v2_1000.json`、`items_human_ins.json`，放到 `$SKILLRISE_SETUP/data/webshop/`。这是 1000 商品配置。原始来源与哈希见 `web-science-data-sources.json`。新 checkout 中创建数据链接并构建 Lucene 索引：

  ```bash
  ln -s "$SKILLRISE_SETUP/data/webshop" agent_system/environments/webshop/webshop/data
  bash setup/build_webshop_index.sh
  ```

  该脚本要求 Java 11 和 spaCy 小模型。已有 `data` 或 `indexes` 时直接复用，不覆盖。本机现有索引位于原 SkillRise，链接不会上传到 GitHub。
- **ScienceWorld**：`scienceworld==1.2.3` 提供环境 jar；需要 Java 11。变体划分保存为 `$SCIWORLD_DATA/variations_idx/L0_idx.json`，来源为 [BEACON L0_idx.json](https://raw.githubusercontent.com/ZJU-REAL/BEACON/main/agent_system/environments/env_package/sciworld/variations_idx/L0_idx.json)，哈希见 `web-science-data-sources.json`。设置 `SCIWORLD_JAVA_HOME`，保持训练/验证划分不混用。
- **训练输入表**：启动脚本会调用 `examples.data_preprocess.prepare`，首次使用需要联网下载 `hiyouga/geometry3k`，或事先缓存。环境任务选择仍由本仓库 group 文件控制。

资源需要遵守各自的使用条款。上述来源和哈希用于复核本机安装，不把 CPU 后端验证等同于 GPU 训练效果。

## 配置、验证与训练

完整可调参数位于 `verl/trainer/config/ppo_trainer.yaml`，默认 ETPO 关闭；统一入口自动开启。查看三种环境最终配置：

```bash
bash examples/etpo/train.sh alfworld --cfg job
bash examples/etpo/train.sh webshop --cfg job
bash examples/etpo/train.sh sciworld --cfg job
```

8 GPU 启动、一步训练、消融和限制详见 [实现文档](../docs/etpo/IMPLEMENTATION.zh-CN.md)。验证快照位于 [docs/etpo/verification](../docs/etpo/verification/)；它记录本次 CPU 测试结果，不代表在新机器上执行过。真实 collector 测试需要 `SKILLRISE_MODEL_PATH` 下的 Qwen3 tokenizer，其余算法测试使用小模型。

## Sokoban

现有依赖已包含 `gym_sokoban==0.0.6`。Sokoban 不使用 Java 或上述外部数据，关卡随仓库提供。运行 `bash examples/etpo/train.sh sokoban`；细节见 [Sokoban 文档](../docs/etpo/SOKOBAN.zh-CN.md)。
