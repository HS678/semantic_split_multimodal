# Experiment Walkthrough

本文按一次真实运行顺序说明当前主线。示例命令见根目录 README；这里重点解释数据流。

## 1. 配置加载

- 输入：`configs/<dataset>.yaml`
- 对应文件：`scripts/stage1_partition.py`、`scripts/stage2_discovery.py`、`scripts/stage3_train.py`
- 关键函数：`semantic_split_multimodal.utils.config.load_config`
- 协议限制：D2D 尚未实现，保持 `d2d.enabled: false`

## 2. Stage 1 Partition

- 输入：raw naturally paired dataset、`dataset.*`、`partition.*`
- 输出目录：`local/results/partition/<dataset>/<partition_signature>/`
- 输出文件：`train_clients/client_*.pt`、`client_meta.csv`、`validation_multimodal.pt`、`test_multimodal.pt`、`partition_config.json`
- 对应文件：`data/registry.py`、`data/datasets.py`、`data/partitioner.py`
- 关键函数：`load_dataset`、`run_stage1_partition`

`partition_signature` 由每个模态名、`clients_per_modality` 和 split protocol 组成，例如 `acc_10clients_gyro_10clients__subject_disjoint_tvt_v1`。新签名不会与旧 train/test 结果目录冲突。

## 3. Dataset Loader

- 输入：原始数据目录
- 输出：统一 loader contract
- 对应文件：`data/datasets.py`
- 关键函数：`load_uci_har_dataset`、`load_mhealth_dataset`、`load_pamap2_dataset`、`load_cmu_mosei_dataset`

loader 必须返回 `train`、`validation`、`test`、`modality_names` 和 `modality_input_shapes`。传感器数据集的三个 split subject 必须互斥；CMU-MOSEI 必须严格使用官方 split ID。每个 modality tensor 的第 0 维必须与 labels 对齐，归一化只能用 train 统计量。

## 4. Client Payload

- 输入：loader 返回的 train split
- 输出：每个 client 一个单模态 payload
- 对应文件：`data/partitioner.py`、`data/client.py`
- 协议限制：`hidden_modality_id` 只作为 metadata 保存，不给训练调度、binding 或 fusion 使用

## 5. validation_multimodal.pt / test_multimodal.pt

- 输入：loader 返回的 naturally paired validation/test split
- 输出：包含所有模态同 index 样本和 label 的 payload
- 协议限制：validation/test label 不参与输入构造，只用于 loss、accuracy、macro-F1 和 weighted-F1

## 6. Stage 2 Discovery

- 输入：Stage 1 `train_clients/`、`pretrain.*`、`fingerprint.*`、`cluster.*`
- 输出目录：`local/results/cluster/<dataset>/<partition_signature>/<cluster_method>/`
- 输出文件：`true_cluster.csv`、`pred_cluster.csv`、`pretrained_encoders/`、`stage2_metadata.json`
- 对应文件：`scripts/stage2_discovery.py`、`learning/pretrain.py`
- 关键函数：`run_stage2_discovery`

Stage 2 先为每个单模态 client 预训练 autoencoder encoder，再提取 fingerprint，最后执行 `kmeans` 或 `adaptive_isodata`。`hidden_modality_id` 只能在聚类完成后用于 post-hoc audit，不能反馈给 PCA、split/merge、Q 选择、seed 选择或参数调整。

## 7. Clustering Output

- `true_cluster.csv`：真实模态簇，仅用于 audit。
- `pred_cluster.csv`：预测簇，是 Stage 3 的正式输入。
- `stage2_metadata.json`：保存 Git SHA、runtime、Stage 1 输入路径、配置快照和 discovery metrics。

Stage 3 只信任 `pred_cluster`，不读取真实模态 id 做训练。其技术门槛是合法 `pred_cluster.csv`、与 Stage 1 一致的 client IDs 和逐客户端 pretrained encoder；`true_cluster.csv` 与 `stage2_metadata.json` 是可选 audit，内容缺失或 `discovery_status` 非成功都不阻止训练。

## 8. Stage 3 Training

- 输入：Stage 1 partition、Stage 2 cluster、`training.*`、`binding.*`、`fusion.*`
- 输出目录：`local/results/experiments/<dataset>/<run_id>/`
- 输出文件：`resolved_config.yaml`、`train_log.csv`、`validation_log.csv`、`best_metrics.json`、`final_metrics.json`、`best_model.pt`、`last_model.pt`、`training_curves.png`、`stage3_metadata.json`
- 对应文件：`scripts/stage3_train.py`、`learning/fusion_sl.py`
- 关键函数：`run_mmbind_fusion_stage3_split_training`

正式 YAML 的基础 `seed` 为 `42`。`scripts/stage3_train.py --seed <N>` 只覆盖本次 Stage 3 的内存配置，适合在复用冻结 Stage 1/Stage 2 输入时运行 `101/202/303/404/505` 五个正式种子。

## 9. Scheduler

- 输入：所有 train clients 的 `pred_cluster`
- 输出：每轮 selected clients
- 对应文件：`learning/scheduling.py`
- 关键函数：`build_scheduler`、`BalancedClusterRoundRobinScheduler.sample_round`
- 协议限制：scheduler 只读取 `pred_cluster`；每个预测簇每轮固定选择 `training.clients_per_cluster_per_round` 个客户端；簇内随机轮询池无放回抽样，耗尽后排除本轮已选客户端再重置

## 10. Same-Label Binding

- 输入：selected client activation batches
- 输出：pseudo multimodal batch
- 对应文件：`learning/binding.py`
- 关键函数：`build_label_random_pseudo_batch`
- 协议限制：只保证 label 相同，不表示自然配对或真实跨模态实例身份

## 11. Fusion And Backward

- 输入：每个 cluster slot 的 activation
- 输出：class logits、server 参数更新、activation gradients、client encoder 参数更新
- 对应文件：`learning/fusion_sl.py`、`learning/models.py`
- 关键函数：`ConcatMLPFusionServer.forward`、`_train_local_step`、`_backward_to_clients`

fusion slot 由 `pred_cluster -> cluster_to_slot` 固定映射决定。训练 loss 是 `CrossEntropy(logits, labels)`。server 更新不做 FedAvg，gradient 只返回产生该 activation 的 client。

## 12. Naturally Paired Validation And Test

- 输入：`validation_multimodal.pt` 或 `test_multimodal.pt`、representative client encoders、fusion server、evaluation-only oracle mapping
- 输出：loss、accuracy、macro-F1、weighted-F1
- 对应文件：`evaluation/fusion_eval.py`、`evaluation/oracle_mapping.py`
- 关键函数：`build_oracle_eval_mapping`、`evaluate_naturally_paired_fusion`
- 协议限制：validation/test label 只用于 metrics；oracle mapping 只用于无梯度 evaluation，不进入训练数据流

## 13. Official Final Result

- 论文主结果：`final_metrics.json`
- 论文 checkpoint：`best_model.pt`
- 最佳验证指标：`best_metrics.json`
- 停止时诊断 checkpoint：`last_model.pt`

正式配置每 10 rounds 执行 naturally paired validation，由 validation macro-F1 选择 `best_model.pt` 并控制 early stopping。训练结束后恢复该 checkpoint，test 只执行一次。`last_model.pt` 不参与正式 test。

`stage3_metadata.json` 用于审计运行状态、输入路径、Git SHA、runtime、scheduler、estimated_Q 和完整配置快照。
