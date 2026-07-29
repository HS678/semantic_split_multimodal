# Architecture

## 主线

当前活动方法为 `mmbind_fusion_split_learning`，由三个可分开运行的 stage 组成：

1. Stage 1：raw naturally paired multimodal dataset -> subject-disjoint train/validation/test -> train-only single-modality client partition -> `validation_multimodal.pt` / `test_multimodal.pt`
2. Stage 2：client encoder autoencoder pretraining -> fingerprint extraction -> clustering -> `pred_cluster`
3. Stage 3：balanced per-cluster random round-robin scheduling -> same-label pseudo binding -> ClusterAdapter -> concat MLP fusion -> classifier -> CE loss -> Split Learning backward -> naturally paired evaluation

项目不是 Federated Learning，不做 FedAvg。每个客户端只训练自己的 encoder；服务器训练 fusion server，并把 activation gradient 返回给产生该 activation 的客户端。

## 关键目录树

```text
src/semantic_split_multimodal/
  data/
  discovery/
  learning/
  evaluation/
  utils/
scripts/
  stage1_partition.py
  stage2_discovery.py
  stage3_train.py
```

## 文件职责

- `data/client.py`：定义单模态 client payload 的内存对象和序列化字段。
- `data/datasets.py`：加载 UCI-HAR、MHEALTH、PAMAP2，并输出统一 loader contract。
- `data/partitioner.py`：把 naturally paired train split 划成单模态客户端，并保存 naturally paired validation/test payload。
- `data/registry.py`：按 config 中的数据集类型分发 loader。
- `discovery/fingerprint.py`：从预训练 encoder 提取客户端 fingerprint。
- `discovery/clustering.py`：执行 `kmeans` 或 `adaptive_isodata` 聚类。
- `learning/pretrain.py`：Stage 2 主流程，负责 encoder pretraining、fingerprint、clustering 和 discovery metrics。
- `learning/scheduling.py`：基于 `pred_cluster` 做每簇均衡的随机轮询调度。
- `learning/binding.py`：根据训练 label 和 `pred_cluster` 构造 same-label pseudo multimodal batch。
- `learning/models.py`：定义 client encoder、autoencoder、`ClusterAdapter` 和 `ConcatMLPFusionServer`。
- `learning/fusion_sl.py`：Stage 3 fusion Split Learning trainer、validation checkpoint selection、early stopping、best checkpoint 恢复和一次性 test 调用。
- `evaluation/fusion_eval.py`：读取 `validation_multimodal.pt` 或 `test_multimodal.pt` 做 naturally paired fusion evaluation。
- `evaluation/plot_training_curves.py`：从训练和验证 CSV 生成 `training_curves.png`，也提供独立重绘 CLI。
- `evaluation/oracle_mapping.py`：仅在 evaluation 阶段用 `hidden_modality_id` 建立 oracle mapping。
- `evaluation/metrics.py`：计算 discovery 和 learning metrics。
- `utils/results.py`：解析 Stage 1 partition、Stage 2 cluster 和 Stage 3 experiment 输出路径。
- `scripts/stage1_partition.py`：Stage 1 CLI。
- `scripts/stage2_discovery.py`：Stage 2 CLI，从冻结 Stage 1 partition 读取输入并写入独立 cluster 目录。
- `scripts/stage3_train.py`：Stage 3 CLI，从冻结 Stage 1/Stage 2 输入训练 fusion Split Learning 模型并写入 experiment 目录。

## 结果边界

- Stage 1：`local/results/partition/<dataset>/<partition_signature>/`
- Stage 2：`local/results/cluster/<dataset>/<partition_signature>/<cluster_method>/`
- Stage 3：`local/results/experiments/<dataset>/<run_id>/`

Stage 2 不回写 Stage 1。Stage 3 不回写 Stage 1 或 Stage 2。

## 协议边界

`hidden_modality_id` 只允许出现在 Stage 2 discovery metrics 和无梯度的 validation/test evaluation-only oracle mapping。训练 forward/backward、调度、binding、fusion slot 和模型输入都基于 `pred_cluster` 与 label，不读取真实模态归属。validation mapping 只产生指标和 checkpoint 选择信号，不进入 optimizer 或训练数据流。

D2D 尚未实现。
