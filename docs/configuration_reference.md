# Configuration Reference

五个数据集配置位于 `configs/uci_har.config`、`configs/mhealth.config`、`configs/pamap2.config`、`configs/cmu_mosei.config` 和 `configs/iemocap.config`。完整字段、类型和取值说明位于 `configs/config.config`。当前五份开发配置统一使用 `true_cluster`，属于 Oracle/debug 实验。

## 顶层字段

- `seed`：基础随机种子，四个正式 YAML 均固定为 `42`。Stage 3 可通过 CLI `--seed` 只覆盖本次运行的内存配置；不会修改 YAML，也不会影响 Stage 1/Stage 2。
- `device`：`auto`、`cpu`、`cuda` 或 `mps`。
- `experiment_name`：实验名，主要用于记录。
- `results.base_dir`：结果根目录基准，默认 `./local/results`。
- `num_classes`：分类类别数。
- `encoder_hidden_dim`：client encoder 输出维度。

## dataset

- `dataset.type`：`uci_har`、`mhealth`、`pamap2`、`cmu_mosei` 或 `iemocap`。
- `dataset.root`：原始数据目录。
- `dataset.split_protocol`：传感器数据集固定为 `subject_disjoint_tvt_v1`；CMU-MOSEI 固定为 `official_video_disjoint_tvt_v1`；IEMOCAP 固定为 `session_disjoint_123_4_5_v1`。该字段写入 partition signature。
- `dataset.modality_scheme`：传感器到模态的划分方案。
- `dataset.train_subjects` / `dataset.validation_subjects` / `dataset.test_subjects`：互斥的 subject split。
- `dataset.window_size` / `dataset.stride`：时间序列窗口设置。
- `dataset.task` / `dataset.label_protocol`：CMU-MOSEI 固定为 `binary_sentiment` / `negative_vs_non_negative`，即 `< 0` 为负类、`>= 0` 为非负类。
- `dataset.temporal_pooling`：CMU-MOSEI audio/visual 固定为 `mean`。
- `dataset.normalize`：CMU-MOSEI 为 `true` 时，mean pooling 后三个模态都只使用 train 统计量标准化。
- `dataset.processed_root` / `dataset.feature_recipe`：IEMOCAP 三模态冻结序列缓存目录及固定特征配方。
- `dataset.train_sessions` / `dataset.validation_sessions` / `dataset.test_sessions`：IEMOCAP 固定为 `[1,2,3]` / `[4]` / `[5]`。

正式 subject 划分固定为：

```text
UCI-HAR train:      1,3,5,6,7,8,11,15,16,17,21,22,26,27,28,29,30
UCI-HAR validation: 14,19,23,25
UCI-HAR test:       2,4,9,10,12,13,18,20,24

MHEALTH train:      2,3,4,6,7,8
MHEALTH validation: 1,5
MHEALTH test:       9,10

PAMAP2 train:       101,102,103,105,107
PAMAP2 validation:  104,106
PAMAP2 test:        108,109

CMU-MOSEI 使用来源仓库官方 split TSV，不重新按标签或样本随机划分。
IEMOCAP train: Session 1,2,3；validation: Session 4；test: Session 5。
```

按当前窗口与过滤配置实际加载后的样本数为：

```text
UCI-HAR: train=5888, validation=1464, test=2947
MHEALTH: train=3152, validation=1059, test=1043
PAMAP2:  train=9298, validation=3757, test=2096
CMU-MOSEI: train=16327, validation=1871, test=4662
IEMOCAP: train=3259, validation=1031, test=1241
```

三个 split 均覆盖各数据集配置的全部类别。

## partition

- `partition.clients_per_modality`：每个真实模态拆分出的单模态客户端数量。

`partition_signature` 由模态名、客户端数量和 split protocol 组成，例如：

```text
acc_10clients_gyro_10clients__subject_disjoint_tvt_v1
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

## fingerprint_visualization

- `fingerprint_visualization.enabled`：是否在 Stage 2 完成后保存 fingerprint 并生成 PCA 审计图。
- `fingerprint_visualization.method`：当前论文主图固定为 `pca`。
- `fingerprint_visualization.standardize`：PCA 前是否逐 fingerprint 维度标准化。
- `fingerprint_visualization.show_client_ids`：是否在点旁标注 client ID。
- `fingerprint_visualization.show_ellipses`：是否显示分组协方差椭圆。
- `fingerprint_visualization.ellipse_confidence`：协方差椭圆置信水平。
- `fingerprint_visualization.png_dpi`：PNG 预览分辨率；PDF 始终为矢量格式。

## cluster

- `cluster.method`：只支持 `kmeans` 或 `adaptive_isodata`。
- `cluster.known_k`：known-Q kmeans 实验的聚类数量；unknown-Q adaptive ISODATA 应为 `null`。
- `cluster.adaptive.*`：adaptive ISODATA 参数，四个正式数据集保持当前统一配置。不得根据 CMU-MOSEI 聚类结果反向修改聚类算法或参数设计。

## training

- `training.scheduler`：正式主线使用 `balanced_cluster_round_robin`。
- `training.global_rounds`：Stage 3 最大 global round 数，正式配置为 `200`。
- `training.local_steps`：每个 global round 内复用 selected clients 的 local step 数。
- `training.batch_size`：client local batch size。
- `training.eval_batch_size`：naturally paired validation/test batch size。
- `training.validation_every`：naturally paired validation 间隔，正式配置为 `10`。
- `training.early_stopping.patience`：validation macro-F1 连续未改善次数，正式配置为 `3`。
- `training.early_stopping.min_rounds`：允许 early stop 前至少完成的 rounds，正式配置为 `50`。
- `training.early_stopping.min_delta`：macro-F1 被视为改善所需的最小增量，正式配置为 `0.001`。
- `training.clients_per_cluster_per_round`：每个预测簇每轮选中的客户端数量 `r`。每轮总客户端数为 `r * estimated_Q`，其中 `estimated_Q` 来自 Stage 2 的 `pred_cluster`，不是训练阶段读取的真实 Q。
- `training.client_lr`：client encoder 学习率。
- `training.server_lr`：fusion server 学习率。

## binding

- `binding.type`：正式主线为 `label_random`。
- `binding.batch_size`：pseudo multimodal batch size。

## fusion

- `fusion.type`：正式主线为 `concat_mlp`。
- `fusion.training_objective`：融合训练机制。`label_random_ce` 保留原有完整伪元组分类 CE，并作为默认值；`mmbind_weighted_contrastive` 在相同融合结构上联合计算完整伪元组 CE、跨所选簇同标签 group contrastive loss 和单簇异构输入 CE。也可通过 Stage 3 CLI 的 `--fusion-training-objective` 临时覆盖，最终值会写入 `resolved_config.config`。
- `fusion.adapter_dim`：每个 cluster slot 的 `ClusterAdapter` 输出维度。
- `fusion.hidden_dim`：concat MLP hidden dim。
- `fusion.num_layers`：concat MLP hidden layer 数。
- `fusion.dropout`：fusion dropout。
- `fusion.mmbind.temperature`：跨预测簇余弦对比的 temperature，默认 `0.1`。
- `fusion.mmbind.contrastive_weight`：group contrastive loss 系数，默认 `0.1`。
- `fusion.mmbind.heterogeneous_ce_weight`：只保留一个预测簇、其他 adapted slots 置零时的分类 loss 系数，默认 `0.5`。

MMBind 式分支不修改 `ClusterAdapter + Concat Fusion + Classifier` 的推理结构，不读取真实模态名或 `hidden_modality_id`。完全相同标签的 binding confidence 当前为 `1.0`；没有软标签或共享传感器相似度时，它属于 MMBind label-shared 场景的适配实现，而不是原论文所有步骤的完整复现。

## CLI 注入的输出目录

Stage 3 CLI 在内存配置中注入输出路径，正式 YAML 不保存 `result` 或 `result_model` 字段。所有输出写入：

```text
local/results/experiments/<oracle_true_cluster|predicted_cluster>/<dataset>/<config_signature>/seed-<seed>/attempt-<nn>/
```

解析后的配置、训练日志、验证日志、最佳验证指标、最终测试指标、`best_model.pt`、诊断用 `last_model.pt` 和训练曲线都直接保存在该目录下。正式模型使用 `best_model.pt`，正式 test 指标使用 `final_metrics.json`。

## d2d

`d2d.enabled` 当前必须保持 `false`。D2D 尚未实现，配置项只作为显式禁用记录。
