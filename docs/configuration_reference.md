# Configuration Reference

三个正式配置位于 `configs/uci_har.yaml`、`configs/mhealth.yaml` 和 `configs/pamap2.yaml`。

## 顶层字段

- `seed`：基础随机种子，三个正式 YAML 均固定为 `42`。Stage 3 可通过 CLI `--seed` 只覆盖本次运行的内存配置；不会修改 YAML，也不会影响 Stage 1/Stage 2。
- `device`：`auto`、`cpu`、`cuda` 或 `mps`。
- `experiment_name`：实验名，主要用于记录。
- `results.base_dir`：结果根目录基准，默认 `./local/results`。
- `num_classes`：分类类别数。
- `encoder_hidden_dim`：client encoder 输出维度。

## dataset

- `dataset.type`：`uci_har`、`mhealth` 或 `pamap2`。
- `dataset.root`：原始数据目录。
- `dataset.modality_scheme`：传感器到模态的划分方案。
- `dataset.train_subjects` / `dataset.test_subjects`：subject split。
- `dataset.window_size` / `dataset.stride`：时间序列窗口设置。

## partition

- `partition.clients_per_modality`：每个真实模态拆分出的单模态客户端数量。

`partition_signature` 由模态名和客户端数量组成，例如：

```text
acc_10clients_gyro_10clients
```

## pretrain

- `pretrain.epochs`：每个 client autoencoder 预训练 epoch 数。
- `pretrain.batch_size`：预训练 batch size。
- `pretrain.lr`：预训练学习率。
- `pretrain.weight_decay`：预训练 weight decay。
- `pretrain.max_samples`：可选采样上限。

## fingerprint

- `fingerprint.type`：fingerprint 类型。
- `fingerprint.batch_size`：提取 fingerprint 时的 batch size。
- `fingerprint.max_batches`：每个 client 最多使用的 batch 数。

## cluster

- `cluster.method`：只支持 `kmeans` 或 `adaptive_isodata`。
- `cluster.known_k`：known-Q kmeans 实验的聚类数量；unknown-Q adaptive ISODATA 应为 `null`。
- `cluster.adaptive.*`：adaptive ISODATA 参数，三个公开数据集应保持统一。

## training

- `training.scheduler`：正式主线使用 `balanced_cluster_round_robin`。
- `training.global_rounds`：Stage 3 global round 数。
- `training.local_steps`：每个 global round 内复用 selected clients 的 local step 数。
- `training.batch_size`：client local batch size。
- `training.eval_batch_size`：naturally paired evaluation batch size。
- `training.eval_every`：naturally paired evaluation 频率。正式配置与 `training.global_rounds` 相等，只在最终轮评估：UCI-HAR `50/50`、MHEALTH `50/50`、PAMAP2 `100/100`。
- `training.clients_per_cluster_per_round`：每个预测簇每轮选中的客户端数量 `r`。每轮总客户端数为 `r * estimated_Q`，其中 `estimated_Q` 来自 Stage 2 的 `pred_cluster`，不是训练阶段读取的真实 Q。
- `training.client_lr`：client encoder 学习率。
- `training.server_lr`：fusion server 学习率。

## binding

- `binding.type`：正式主线为 `label_random`。
- `binding.batch_size`：pseudo multimodal batch size。

## fusion

- `fusion.type`：正式主线为 `concat_mlp`。
- `fusion.adapter_dim`：每个 cluster slot 的 `ClusterAdapter` 输出维度。
- `fusion.hidden_dim`：concat MLP hidden dim。
- `fusion.num_layers`：concat MLP hidden layer 数。
- `fusion.dropout`：fusion dropout。

## CLI 注入的输出目录

Stage 3 CLI 在内存配置中注入输出路径，正式 YAML 不保存 `result` 或 `result_model` 字段。所有输出写入：

```text
local/results/experiments/<dataset>/<run_id>/
```

训练日志、评估日志、最终指标、兼容性最佳指标、最终模型和兼容性最佳模型都直接保存在该目录下。正式结果使用 `final_metrics.json` 和 `final_model.pt`。

## d2d

`d2d.enabled` 当前必须保持 `false`。D2D 尚未实现，配置项只作为显式禁用记录。
