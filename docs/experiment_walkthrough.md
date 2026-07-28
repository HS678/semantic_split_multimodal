# Experiment Walkthrough

本文按一次真实运行顺序说明当前主线。示例命令使用 UCI-HAR；MHEALTH 和 PAMAP2 只需替换 config。

## 1. 配置加载

- 输入：`configs/uci_har.yaml`
- 输出：Python dict config
- 对应文件：`scripts/stage1_partition.py`、`scripts/stage2_discovery_only.py`、`scripts/stage3_train_only.py`
- 关键函数：`semantic_split_multimodal.utils.config.load_config`
- 读取字段：全部 YAML 字段，后续 stage 只读取自己需要的部分
- 结果文件：无
- 协议限制：配置不得启用旧方法；D2D 尚未实现，保持 `d2d.enabled: false`

## 2. Stage 1

- 输入：raw naturally paired dataset、`dataset.*`、`partition.*`
- 输出：单模态 train clients 和 naturally paired test payload
- 对应文件：`scripts/stage1_partition.py`、`data/registry.py`、`data/datasets.py`、`data/partitioner.py`
- 关键函数：`load_dataset`、`run_stage1_partition`
- 读取字段：`dataset.type`、`dataset.root`、数据集窗口参数、`partition.clients_per_modality`、`partition.output_dir`
- 生成结果目录：`local/results/partition/<dataset>/<modality_names>_<clients_per_modality>clients/`
- 生成结果文件：`train_clients/client_*.pt`、`client_meta.csv`、`test_multimodal.pt`、`partition_config.json`
- 协议限制：train clients 是单模态；`test_multimodal.pt` 保留自然配对样本索引，仅供 evaluation

## 3. Loader

- 输入：原始数据目录
- 输出：统一 loader contract
- 对应文件：`data/datasets.py`
- 关键函数：`load_uci_har`、`load_mhealth`、`load_pamap2`
- 读取字段：`dataset.modality_scheme`、`window_size`、`stride`、subject split、normalization 设置
- 生成结果文件：无，返回内存对象
- 协议限制：三个数据集都必须返回 `modality_names` 和 `modality_input_shapes`

## 4. Partition

- 输入：loader 返回的 train split
- 输出：每个 client 一个单模态 payload
- 对应文件：`data/partitioner.py`、`data/client.py`
- 关键函数：`run_stage1_partition`、`Client.to_payload`
- 读取字段：`partition.clients_per_modality`
- 生成结果文件：`train_clients/client_*.pt`、`client_meta.csv`
- 协议限制：`hidden_modality_id` 只作为 metadata 保存，不给训练调度、binding 或 fusion 使用

## 5. test_multimodal.pt

- 输入：loader 返回的 test split
- 输出：包含所有模态同 index 样本和 label 的 payload
- 对应文件：`data/partitioner.py`
- 关键函数：`run_stage1_partition`
- 读取字段：`modality_names`、`modality_input_shapes`
- 生成结果文件：`test_multimodal.pt`
- 协议限制：test label 不参与训练或测试输入构造

## 6. Stage 2

- 输入：Stage 1 train clients、`pretrain.*`、`fingerprint.*`、`cluster.*`
- 输出：pretrained encoders、fingerprints、`pred_cluster`
- 对应文件：`scripts/stage2_discovery_only.py`、`learning/pretrain.py`
- 关键函数：`run_stage2_discovery`
- 读取字段：`pretrain.epochs`、`pretrain.batch_size`、`fingerprint.type`、`cluster.method`、`cluster.known_k`
- 生成结果目录：`local/results/experiment/<dataset>/<run_id>/02_cluster_results/`
- 生成结果文件：`pretrained_encoders/*_encoder.pt`、`fingerprints.npy`、`cluster_assignments.csv`、`cluster_metrics.json`
- 协议限制：`hidden_modality_id` 只用于 discovery metrics；unknown-Q discovery 当前仍需实验验证

## 7. Encoder Pretraining

- 输入：每个单模态 client payload
- 输出：client encoder state dict
- 对应文件：`learning/pretrain.py`、`learning/models.py`
- 关键函数：`create_client_encoder`、autoencoder pretraining loop
- 读取字段：`model.encoder.*`、`encoder_hidden_dim`、`pretrain.*`
- 生成结果文件：`pretrained_encoders/client_XXX_encoder.pt`
- 协议限制：每个 client 独立预训练，不做跨客户端 FedAvg

## 8. Fingerprint

- 输入：pretrained encoder 和 client samples
- 输出：fingerprint matrix
- 对应文件：`discovery/fingerprint.py`
- 关键函数：fingerprint extraction helpers
- 读取字段：`fingerprint.type`、`fingerprint.batch_size`、`fingerprint.max_batches`
- 生成结果文件：`fingerprints.npy`
- 协议限制：fingerprint 用于 clustering，不直接作为 Stage 3 模型输入

## 9. Clustering

- 输入：fingerprints
- 输出：`pred_cluster`
- 对应文件：`discovery/clustering.py`、`learning/pretrain.py`
- 关键函数：`cluster_fingerprints`
- 读取字段：`cluster.method`、`cluster.known_k`、`cluster.isodata.*`
- 生成结果文件：`cluster_assignments.csv`、`cluster_metrics.json`
- 协议限制：Stage 3 只信任 `pred_cluster`，不使用真实模态 id

## 10. Stage 3

- 输入：Stage 1 clients、Stage 2 assignments 和 pretrained encoders
- 输出：训练日志、evaluation metrics、checkpoint
- 对应文件：`scripts/stage3_train_only.py`、`learning/fusion_sl.py`
- 关键函数：`run_mmbind_fusion_stage3_split_training`
- 读取字段：`training.*`、`binding.*`、`fusion.*`、`result.*`、`result_model.*`
- 生成结果目录：`local/results/experiment/<dataset>/<run_id>/`
- 生成结果文件：`03_training_evaluation/train_log.csv`、`eval_log.csv`、`final_metrics.json`、`04_model_artifacts/*`
- 协议限制：Stage 3 入口只有正式 fusion Split Learning

## 11. Scheduler

- 输入：所有 train clients 的 `pred_cluster`
- 输出：每轮 selected clients
- 对应文件：`learning/scheduling.py`
- 关键函数：`build_scheduler`、`ProposedClusterCoverageScheduler.sample_round`
- 读取字段：`training.scheduler`、`training.clients_per_round`
- 生成结果文件：训练日志中的 `selected_client_ids`、`selected_cluster_ids`
- 协议限制：主线 scheduler 不读取 `hidden_modality_id`

## 12. Local Batch

- 输入：selected clients 的本地单模态 samples 和 labels
- 输出：每个 client 的 mini-batch
- 对应文件：`learning/fusion_sl.py`
- 关键函数：`FusionSplitClient.sample_batch`
- 读取字段：`training.batch_size`
- 生成结果文件：无，统计写入 train log
- 协议限制：每个 client 独立采样，不要求实例级配对

## 13. Client Forward

- 输入：local batch samples
- 输出：client encoder activation
- 对应文件：`learning/fusion_sl.py`、`learning/models.py`
- 关键函数：`FusionSplitClient.forward_detached`
- 读取字段：`model.encoder.*`、`encoder_hidden_dim`
- 生成结果文件：无
- 协议限制：上传的是 detached activation，不上传原始样本

## 14. Upload Activation

- 输入：detached activation 和 labels
- 输出：`ClientActivationBatch`
- 对应文件：`learning/fusion_sl.py`、`learning/binding.py`
- 关键函数：`_collect_selected_activations`
- 读取字段：无
- 生成结果文件：无
- 协议限制：activation metadata 使用 `pred_cluster`，不使用真实模态身份

## 15. Same-Label Binding

- 输入：selected client activation batches
- 输出：pseudo multimodal batch
- 对应文件：`learning/binding.py`
- 关键函数：`build_label_random_pseudo_batch`
- 读取字段：`binding.batch_size`
- 生成结果文件：训练日志中的 binding 成功率、common labels、pseudo batch size
- 协议限制：只保证 label 相同，不表示自然配对

## 16. Cluster Slot

- 输入：pseudo batch 中的 `slot_activations`
- 输出：按 `cluster_to_slot` 排列的 fusion slots
- 对应文件：`learning/fusion_sl.py`、`learning/models.py`
- 关键函数：`ConcatMLPFusionServer.forward`
- 读取字段：`fusion.*`
- 生成结果文件：`04_model_artifacts/cluster_to_slot.json`
- 协议限制：slot 由 `pred_cluster` 决定

## 17. Fusion

- 输入：所有 cluster slot activation
- 输出：fused representation
- 对应文件：`learning/models.py`
- 关键函数：`ClusterAdapter.forward`、`ConcatMLPFusionServer.forward`
- 读取字段：`fusion.adapter_dim`、`fusion.hidden_dim`、`fusion.num_layers`、`fusion.dropout`
- 生成结果文件：无
- 协议限制：每个 pseudo tuple 必须覆盖全部 cluster slot

## 18. Classifier

- 输入：concat fused representation
- 输出：class logits
- 对应文件：`learning/models.py`
- 关键函数：`ConcatMLPFusionServer.forward`
- 读取字段：`num_classes`
- 生成结果文件：无
- 协议限制：不按 label 或真实模态选择预测路径

## 19. CE Loss

- 输入：logits 和 pseudo labels
- 输出：cross-entropy loss
- 对应文件：`learning/fusion_sl.py`
- 关键函数：`_train_local_step`
- 读取字段：无
- 生成结果文件：`train_log.csv` 中的 `loss`
- 协议限制：训练 loss 是 CE，不加入 prototype alignment

## 20. Server Backward

- 输入：CE loss
- 输出：server 参数更新和 activation gradients
- 对应文件：`learning/fusion_sl.py`
- 关键函数：`_train_local_step`
- 读取字段：`training.server_lr`
- 生成结果文件：`train_log.csv` 中的 `server_update_l1`
- 协议限制：server 更新不做 FedAvg

## 21. Activation Gradient Routing

- 输入：每个 participating activation 的 gradient
- 输出：对应 client 的 backward 调用
- 对应文件：`learning/fusion_sl.py`
- 关键函数：`_backward_to_clients`
- 读取字段：无
- 生成结果文件：无
- 协议限制：gradient 只返回产生该 activation 的 client

## 22. Client Backward

- 输入：activation gradient
- 输出：client encoder 参数更新
- 对应文件：`learning/fusion_sl.py`
- 关键函数：`FusionSplitClient.backward_from_server`
- 读取字段：`training.client_lr`
- 生成结果文件：`train_log.csv` 中的 `client_update_l1`
- 协议限制：客户端只更新本地 encoder

## 23. Naturally Paired Evaluation

- 输入：`test_multimodal.pt`、representative client encoders、fusion server、oracle eval mapping
- 输出：loss、accuracy、macro-F1
- 对应文件：`evaluation/fusion_eval.py`、`evaluation/oracle_mapping.py`
- 关键函数：`build_oracle_eval_mapping`、`evaluate_naturally_paired_fusion`
- 读取字段：`training.eval_batch_size`
- 生成结果文件：`eval_log.csv`、`final_metrics.json`、`oracle_eval_modality_to_cluster.json`
- 协议限制：test label 只用于 metrics；oracle mapping 只用于 evaluation

## 24. Checkpoint

- 输入：server state、client encoder state、cluster mapping、metrics
- 输出：best 和 last checkpoint
- 对应文件：`learning/fusion_sl.py`
- 关键函数：`_save_checkpoint`
- 读取字段：`result_model.output_dir`
- 生成结果文件：`best_mmbind_fusion_checkpoint.pt`、`last_mmbind_fusion_checkpoint.pt`、`best_server_model.pt`、`best_client_encoders/*.pt`、`best_model_info.json`
- 协议限制：checkpoint 保存主线 fusion server 和 client encoders，不保存旧方法状态

## 25. final_metrics.json

- 输入：final evaluation、round 统计、oracle mapping
- 输出：最终可审计指标
- 对应文件：`learning/fusion_sl.py`
- 关键函数：`run_mmbind_fusion_stage3_split_training`
- 读取字段：无
- 生成结果文件：`03_training_evaluation/final_metrics.json`
- 协议限制：论文主结果应使用 `final_eval` 的 naturally paired metrics
