# Architecture

## 主线

当前活动方法唯一为 `mmbind_fusion_split_learning`。它由三个 stage 组成：

1. Stage 1：raw naturally paired multimodal dataset -> single-modality client partition -> `test_multimodal.pt`
2. Stage 2：client encoder autoencoder pretraining -> fingerprint extraction -> clustering -> `pred_cluster`
3. Stage 3：predicted-cluster full-coverage scheduling -> same-label pseudo binding -> ClusterAdapter -> concat MLP fusion -> classifier -> CE loss -> Split Learning backward -> naturally paired evaluation

项目不是 Federated Learning，不做 FedAvg。每个客户端只训练自己的 encoder；服务器训练 fusion server，并把 activation gradient 返回给产生该 activation 的客户端。

## 关键目录树

```text
src/semantic_split_multimodal/
  data/
    client.py
    datasets.py
    partitioner.py
    registry.py
  discovery/
    clustering.py
    fingerprint.py
  learning/
    binding.py
    fusion_sl.py
    models.py
    pretrain.py
    scheduling.py
  evaluation/
    fusion_eval.py
    metrics.py
    oracle_mapping.py
  utils/
    config.py
    device.py
    results.py
    seed.py
scripts/
  stage1_partition.py
  stage2_discovery.py
  stage2_discovery_only.py
  stage3_train.py
  stage3_train_only.py
```

## 文件职责

- `data/client.py`：定义单模态 client payload 的内存对象和序列化字段。
- `data/datasets.py`：加载 UCI-HAR、MHEALTH、PAMAP2，并输出统一 loader contract。
- `data/partitioner.py`：把 naturally paired train split 划成单模态客户端，并保存 naturally paired test payload。
- `data/registry.py`：按 config 中的数据集类型分发 loader。
- `discovery/fingerprint.py`：从预训练 encoder 提取客户端 fingerprint。
- `discovery/clustering.py`：执行 KMeans、HDBSCAN 或 ISODATA 聚类。
- `learning/pretrain.py`：Stage 2 主流程，负责 encoder autoencoder pretraining、fingerprint、clustering 和 discovery metrics。
- `learning/scheduling.py`：基于 `pred_cluster` 做 proposed cluster coverage scheduling。
- `learning/binding.py`：根据训练 label 和 `pred_cluster` 构造 exact same-label pseudo multimodal batch。
- `learning/models.py`：定义 client encoder、autoencoder、`ClusterAdapter` 和 `ConcatMLPFusionServer`。
- `learning/fusion_sl.py`：Stage 3 正式 fusion Split Learning trainer、checkpoint 保存和 naturally paired final evaluation 调用。
- `evaluation/fusion_eval.py`：读取 `test_multimodal.pt` 做 naturally paired fusion evaluation。
- `evaluation/oracle_mapping.py`：仅在 evaluation 阶段用 `hidden_modality_id` 建立 oracle mapping。
- `evaluation/metrics.py`：计算 discovery 和 learning metrics。
- `utils/config.py`：读取 YAML config。
- `utils/device.py`：选择 CPU、CUDA 或 MPS。
- `utils/results.py`：把 config 中的输出目录解析为当前 run 的四阶段目录。
- `utils/seed.py`：设置 Python、NumPy 和 PyTorch 随机种子。
- `scripts/stage1_partition.py`：Stage 1 CLI。
- `scripts/stage2_discovery_only.py`：Stage 2-only CLI，从冻结的 Stage 1 partition 读取输入并写入独立输出目录。
- `scripts/stage3_train_only.py`：Stage 3-only CLI，从冻结的 Stage 1/Stage 2 输入训练正式 fusion Split Learning 模型并写入独立输出目录。
- `scripts/stage2_discovery.py`：Stage 2 兼容 CLI，用于单一共享 run 目录配置。
- `scripts/stage3_train.py`：Stage 3 兼容 CLI，用于单一共享 run 目录配置。

## 边界

`hidden_modality_id` 只允许出现在 Stage 2 discovery metrics 和 evaluation-only oracle mapping。训练、调度、binding、fusion slot 和模型输入都基于 `pred_cluster` 与 label，不读取真实模态归属。

D2D 尚未实现。unknown-Q discovery 当前仍需实验验证。
