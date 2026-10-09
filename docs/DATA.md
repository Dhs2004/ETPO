# Installation and external resources

## Dependencies and paths

Use Linux with a CUDA-capable GPU. Start from [environment.yml](../setup/environment.yml), [core dependencies](../setup/requirements-core.txt), [environment dependencies](../setup/requirements-web-science.txt), and [compatibility constraints](../setup/constraints.txt). Install a FlashAttention build matching your Python, PyTorch, CUDA and C++ ABI.

```bash
conda env create -f setup/environment.yml
conda activate skillrise
python -m pip install -r setup/requirements-core.txt -c setup/constraints.txt
python -m pip install -r setup/requirements-web-science.txt -c setup/constraints.txt
python -m spacy download en_core_web_sm
cp setup/paths.example.sh config.local.sh
# Edit paths and then:
source config.local.sh
source activate.sh
```

Use this repository's `verl` through `PYTHONPATH`; do not overwrite it with another installed checkout. Java 11 is needed for WebShop indexing and ScienceWorld. Dependency compatibility depends on the host; the supplied files are setup references rather than a portable GPU lockfile.

The default model can be obtained from [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B). Download it to `SKILLRISE_MODEL_PATH`. No weights are included here.

## Dataset and configuration links

| Environment | Public download / setup | ETPO configuration |
|---|---|---|
| ALFWorld | [Official repository and instructions](https://github.com/alfworld/alfworld), [releases](https://github.com/alfworld/alfworld/releases) | [Environment paths](../examples/skillrise_alfworld/env.sh), [launch configuration](../examples/skillrise_alfworld/skillrise_alfworld_qwen3_4b.sh) |
| WebShop | [Official WebShop setup](https://github.com/princeton-nlp/WebShop), [downloadable product data](https://huggingface.co/datasets/YWZBrandon/webshop-data) | [Environment paths](../examples/skillrise_webshop/env.sh), [launch configuration](../examples/skillrise_webshop/skillrise_webshop_qwen3_4b.sh), [index builder](../setup/build_webshop_index.sh) |
| ScienceWorld | [Official repository](https://github.com/allenai/ScienceWorld), [Python package](https://pypi.org/project/scienceworld/), [variation configuration](https://github.com/ZJU-REAL/BEACON/blob/main/agent_system/environments/env_package/sciworld/variations_idx/L0_idx.json) | [Environment paths](../examples/skillrise_sciworld/env.sh), [launch configuration](../examples/skillrise_sciworld/skillrise_sciworld_qwen3_4b.sh) |
| Sokoban | [gym-sokoban generator](https://github.com/mpSchrader/gym-sokoban); generate levels locally | [Input preparation](../examples/etpo/prepare_sokoban.py), [launch configuration](../examples/etpo/train_sokoban.sh) |

### ALFWorld

Install the package, run `alfworld-download`, and set `ALFWORLD_DATA` to the downloaded resource root. TextWorld game definitions must match the upstream task definitions. Preserve the official training/validation split. The environment's PDDL/TextWorld logic templates are part of the backend code; game records are external.

### WebShop

Download `items_shuffle_1000.json`, `items_ins_v2_1000.json`, and `items_human_ins.json` from the linked data repository into `$SKILLRISE_SETUP/data/webshop`. Follow the official data license and setup instructions. Link the data into the backend and build its Lucene index:

```bash
ln -s "$SKILLRISE_SETUP/data/webshop" agent_system/environments/webshop/webshop/data
bash setup/build_webshop_index.sh
```

The default launcher selects the small catalog. Product data, instruction definitions and indexes must use a consistent version.

### ScienceWorld

The `scienceworld` package supplies the environment jar. Set `SCIWORLD_JAVA_HOME` to Java 11. Download the linked variation file to `$SCIWORLD_DATA/variations_idx/L0_idx.json`. Keep training and validation variations separate.

### Task groups

[Upstream task definitions](https://github.com/Within-yao/SkillRise/tree/main/data/groups) are downloaded separately. The preparation script converts upstream records into single-task groups; upstream multi-task groups are never used directly for training:

```bash
python scripts/prepare_groups.py alfworld
python scripts/prepare_groups.py webshop
python scripts/prepare_groups.py sciworld
```

For a version-pinned or custom upstream task file, pass `--source /path/to/groups.jsonl`. Generated groups remain local and are ignored by Git. Record the downloaded source revision in your own experiment manifest. Do not combine incompatible resource versions or change a split within an existing run.

The text input-table preprocessing also downloads [geometry3k](https://huggingface.co/datasets/hiyouga/geometry3k) as an upstream table scaffold; environment task selection is driven by the task groups. Cache it before offline execution.

### Sokoban

```bash
python -m examples.etpo.prepare_sokoban --output-dir .runtime/sokoban-input --generate-levels --k 1
bash examples/etpo/train.sh sokoban --cfg job
```

Levels are generated and checked locally, not included as benchmark records. Teacher skills are fixed guidance, not solver solutions.

## Inspect before training

Use `bash examples/etpo/train.sh ENVIRONMENT --cfg job` after preparing dependencies and data. Verify K=1, training/validation attempts of one, top-token exclusion disabled, KL penalties disabled, checkpoint interval, model path and output directory. Configuration inspection can invoke input preparation; it does not launch GPU training.
