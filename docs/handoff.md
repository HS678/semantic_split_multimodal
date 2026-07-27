# Handoff

## 当前主线

当前工作树只保留 `mmbind_fusion_split_learning` 作为活动方法。旧方法已由安全标签 `pre-remove-unpaired-4e52965` 和旧分支 `feature/unpaired-shared-semantic` 保留在 Git 历史中，不再作为当前分支源码、入口、配置、测试或主文档中的活动路线。

## 代码阅读顺序

1. `configs/uci_har.yaml`：先理解一份完整配置。
2. `scripts/stage1_partition.py`：理解数据如何进入 run 目录。
3. `src/semantic_split_multimodal/data/datasets.py`：理解三个数据集的统一 loader contract。
4. `src/semantic_split_multimodal/data/partitioner.py`：理解单模态 client 和 `test_multimodal.pt`。
5. `scripts/stage2_discovery.py` 与 `learning/pretrain.py`：理解 encoder pretraining、fingerprint 和 clustering。
6. `learning/scheduling.py`：理解 predicted-cluster coverage scheduler。
7. `learning/binding.py`：理解 same-label pseudo binding。
8. `learning/models.py`：理解 encoder、`ClusterAdapter` 和 `ConcatMLPFusionServer`。
9. `learning/fusion_sl.py`：理解 Stage 3 训练、backward、checkpoint 和 evaluation 调用。
10. `evaluation/fusion_eval.py` 与 `evaluation/oracle_mapping.py`：理解 naturally paired evaluation。

## 协议红线

- 不做 FedAvg。
- 不运行旧 unpaired 方法。
- `hidden_modality_id` 不得用于训练、调度、binding、fusion slot 构造或模型输入。
- `hidden_modality_id` 只允许用于 discovery metrics 和 evaluation-only oracle mapping。
- 训练期 same-label binding 不表示实例级 naturally paired。
- D2D 尚未实现。
- unknown-Q discovery 当前仍需实验验证。

## 当前验证重点

维护者修改代码后应至少运行：

```bash
/home/shuang/miniconda3/envs/mpsl/bin/python -m compileall src scripts tests
/home/shuang/miniconda3/envs/mpsl/bin/python -c "import semantic_split_multimodal"
/home/shuang/miniconda3/envs/mpsl/bin/python -m pytest tests
/home/shuang/miniconda3/envs/mpsl/bin/python scripts/stage1_partition.py --help
/home/shuang/miniconda3/envs/mpsl/bin/python scripts/stage2_discovery.py --help
/home/shuang/miniconda3/envs/mpsl/bin/python scripts/stage3_train.py --help
```

正式实验前，再按 `docs/experiment_walkthrough.md` 顺序检查 Stage 1、Stage 2、Stage 3 的输入输出目录。
