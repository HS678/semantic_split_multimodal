# Handoff

## 当前主线

当前工作树只保留 `mmbind_fusion_split_learning` 作为活动方法。旧 unpaired baseline 不再作为当前主线源码、入口、配置、测试或主文档中的活动路线。

## 代码阅读顺序

1. `configs/uci_har.yaml`：先理解一份完整配置。
2. `scripts/stage1_partition.py`：理解 Stage 1 partition 如何保存到 `local/results/partition/`。
3. `src/semantic_split_multimodal/data/datasets.py`：理解三个数据集的统一 loader contract。
4. `src/semantic_split_multimodal/data/partitioner.py`：理解单模态 client 和 `test_multimodal.pt`。
5. `scripts/stage2_discovery.py` 与 `learning/pretrain.py`：理解 encoder pretraining、fingerprint 和 clustering。
6. `learning/scheduling.py`：理解 balanced per-cluster random round-robin scheduler。
7. `learning/binding.py`：理解 same-label pseudo binding。
8. `learning/models.py`：理解 encoder、`ClusterAdapter` 和 `ConcatMLPFusionServer`。
9. `scripts/stage3_train.py` 与 `learning/fusion_sl.py`：理解 Stage 3 训练、backward、checkpoint 和 evaluation 调用。
10. `evaluation/fusion_eval.py` 与 `evaluation/oracle_mapping.py`：理解 naturally paired evaluation。

## 协议红线

- 不做 FedAvg。
- 不运行旧 unpaired 方法。
- `hidden_modality_id` 不得用于训练、调度、binding、fusion slot 构造或模型输入。
- `hidden_modality_id` 只允许用于 discovery audit 和 evaluation-only oracle mapping。
- 训练期 same-label binding 不表示实例级 naturally paired。
- Stage 2 只保留 `kmeans` 和 `adaptive_isodata`。
- D2D 尚未实现。

## 当前验证重点

维护者修改代码后应至少运行：

```bash
python -m compileall src scripts tests
python -c "import semantic_split_multimodal"
python -m pytest tests
python scripts/stage1_partition.py --help
python scripts/stage2_discovery.py --help
python scripts/stage3_train.py --help
```

正式实验前，再按 `docs/experiment_walkthrough.md` 顺序检查 Stage 1、Stage 2、Stage 3 的输入输出目录。
