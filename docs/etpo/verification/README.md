# 验证快照

记录本次本机 CPU 验证和配置解析结果；不是训练指标。`verification.json` 中 `logs/` 路径指向原机器日志，运行日志未上传。`distributed-sft.json` 是两进程 CPU DDP 与单进程 SFT 的参数一致性结果。GPU FSDP、vLLM rollout 和 8×L40 完整训练尚未验证。

复查命令（先完成环境和 tokenizer 准备）：

```bash
source activate.sh
python -m unittest discover -s tests/etpo -p 'test_*.py' -v
mkdir -p logs
torchrun --standalone --nnodes=1 --nproc_per_node=2 tests/etpo/distributed_sft_smoke.py
for task in alfworld webshop sciworld; do
  bash examples/etpo/train.sh "$task" --cfg job > "logs/config-$task.log" 2>&1
done
python tests/etpo/validate_launch_configs.py
```
