# Handoff

## 当前目标

项目已重构为论文主线：未知模态发现、簇覆盖调度、未配对 Split Learning、共享语义学习、prototype alignment。

## 已完成

- 删除 `trainers/split_multimodal_trainer.py`，移除 `pseudo_paired_concat` 旧训练路径。
- 新增 `data/client.py`，统一 `Client` 对象。
- 新增 `clustering/fingerprint.py`，支持 encoder/signal/hybrid fingerprint，不包含 input dimension hint。
- `clustering/cluster.py` 支持 KMeans、HDBSCAN、ISODATA；KMeans 可在 `known_k: null` 时用 silhouette 估计 K。
- 新增 `scheduling/schedulers.py`，包含 Random、RoundRobin、Oracle、ProposedClusterCoverage。
- 新增 `trainers/prototypes.py`，按 `(cluster_id, class_id)` 维护 EMA prototype。
- 重写 `trainers/pretrain_cluster.py`，真实模态只用于 discovery metrics。
- 重写 `trainers/unpaired_split_multimodal_trainer.py`，每个客户端独立采样 batch，不做 concat/alignment-by-sample。
- 更新 `configs/uci_har.yaml`、`configs/mhealth.yaml`、`configs/pamap2.yaml` 为新流程。
- 重写 `README.md` 与 `docs/extension_guide.md`。

## 验证

已通过：

```bash
python -m py_compile $(rg --files -g '*.py')
```

未完成端到端训练 smoke test：当前 Python 环境缺少 `torch`，执行合成数据测试时在 `import torch` 失败。

## 下一步建议

1. 安装依赖后运行三阶段最小实验，建议临时设置 `pretrain.epochs: 1`、`training.global_rounds: 2`。
2. 检查 ISODATA 自动估计簇数是否稳定，必要时调 `split_std_threshold` 和 `merge_distance_threshold`。
3. 为 D2D 添加真实 latency profile 后，再把 `d2d_metrics` 从占位计算替换为协作前后时延统计。
