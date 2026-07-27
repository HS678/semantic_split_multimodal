# Configuration Reference

三个正式配置位于 `configs/uci_har.yaml`、`configs/mhealth.yaml` 和 `configs/pamap2.yaml`。Stage 3 不再需要方法选择字段，入口固定执行正式 fusion Split Learning。

## 顶层字段

- `seed`：随机种子。
- `device`：`auto`、`cpu`、`cuda` 或 `mps`。
- `experiment_name`：实验名，主要用于记录。
- `results.base_dir`：当前 run 的根目录基准。
- `results.run_id`：当前 run 名称。
- `num_modalities`：数据集真实模态数，用于配置和 sanity check；训练期不作为真实模态 oracle 使用。
- `num_classes`：分类类别数。
- `encoder_hidden_dim`：client encoder 输出维度。
- `learning_rate`、`batch_size`：兼容默认值；Stage 3 优先读取 `training.*`。

## dataset

- `dataset.type`：`uci_har`、`mhealth` 或 `pamap2`。
- `dataset.root`：原始数据目录。
- `dataset.modality_scheme`：传感器到模态的划分方案。
- `dataset.train_subjects` / `dataset.test_subjects`：subject split。
- `dataset.window_size` / `dataset.stride`：时间序列窗口设置。
- 其他数据集字段：过滤、归一化和标签纯度设置，按 loader 实际支持字段读取。

## model

- `model.encoder.type`：当前三数据集使用 `time_series`。
- `model.encoder.conv_channels`、`kernel_sizes`、`dropout`：1D encoder 参数。
- `model.server.*`：server/fusion 的默认维度来源；`fusion.*` 会覆盖 fusion server 的主要参数。

## partition

- `partition.output_dir`：Stage 1 输出目录，经过 `utils.results.configure_result_run` 后会落在当前 run 的 `01_dataset_partition/`。
- `partition.clients_per_modality`：每个真实模态拆分出的单模态客户端数量。

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

- `cluster.output_dir`：Stage 2 输出目录，当前 run 中为 `02_cluster_results/`。
- `cluster.method`：`kmeans`、`hdbscan` 或 `isodata`。
- `cluster.known_k`：known-Q 实验的聚类数量。
- `cluster.isodata.*`：ISODATA 参数。

## training

- `training.scheduler`：正式主线使用 `proposed_cluster_coverage`。
- `training.global_rounds`：Stage 3 global round 数。
- `training.local_steps`：每个 global round 内复用 selected clients 的 local step 数。
- `training.batch_size`：client local batch size。
- `training.eval_batch_size`：naturally paired evaluation batch size。
- `training.eval_every`：evaluation 频率。
- `training.clients_per_round`：每轮选中客户端数量。
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

## evaluation

- `evaluation.metrics`：记录期望输出的 learning metrics。正式 evaluation 实际输出 loss、accuracy、macro-F1 和样本数。

## result 和 result_model

- `result.output_dir`：Stage 3 训练/eval 日志目录。
- `result_model.output_dir`：checkpoint 和 mapping 产物目录。

## d2d

`d2d.enabled` 当前必须保持 `false`。D2D 尚未实现，配置项只作为显式禁用记录。
