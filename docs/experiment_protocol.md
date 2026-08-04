# Experiment Protocol

## 假设

训练阶段服务器不知道真实模态数量、真实模态名称和客户端真实模态归属。Stage 2 可以为了 discovery metrics 读取 `hidden_modality_id`；Stage 3 的无梯度 naturally paired validation/test 可以为了 evaluation-only oracle mapping 读取 `hidden_modality_id`。除此之外，训练协议不得使用真实模态身份。

## Stage 1

输入是 naturally paired 多模态数据集。loader 输出统一 contract：`train`、`validation`、`test` 三个 split，每个 split 均包含 `modalities` 和 `labels`，另含 `modality_names` 与 `modality_input_shapes`。传感器数据集按固定 subject 划分且互斥。partitioner 只将 train 按模态拆成多个单模态 client，同时保存自然配对的 `validation_multimodal.pt` 和 `test_multimodal.pt`。归一化统计量只从 train 拟合。

## Stage 2

每个单模态 client encoder 先做 autoencoder pretraining。随后从 encoder 和数据统计中提取 fingerprint，并对 fingerprint 聚类得到 `pred_cluster`。known-Q 实验使用 `cluster.known_k`；unknown-Q 仍需实验验证，不能默认视为稳定结论。

## Stage 3

Stage 3 启动时只把 `pred_cluster.csv`、Stage 1 client IDs 和逐客户端 pretrained encoder 视为技术门槛。`true_cluster.csv`、`stage2_metadata.json` 及其中的 `discovery_status` 只记录为可选 audit，不得决定训练是否启动。正式 YAML 保持基础 `seed: 42`；CLI `--seed` 只覆盖本次 Stage 3 的内存配置。

每个 global round 使用 `balanced_cluster_round_robin` scheduler。设 Stage 2 得到的预测簇集合为 `C`，配置 `training.clients_per_cluster_per_round = r`，则每轮从每个 `pred_cluster` 独立选择 `r` 个客户端，总客户端数为 `r * |C|`。每个 cluster 内维护独立随机轮询池：池内无放回随机抽样；当剩余客户端不足以补满本轮 `r` 个时，先取完剩余客户端，再将该 cluster 的池重置为排除本轮已选客户端后的其余客户端，并继续随机补足。

每个 selected client 独立采样 labeled batch，forward 后上传 detached activation。server 用 exact same-label random binding 构造 pseudo multimodal tuple：每个 tuple 在所有 cluster slot 上 label 相同，但不表示这些样本在实例级 naturally paired。

fusion slot 由 `pred_cluster` 和 `cluster_to_slot` 固定映射决定。`ConcatMLPFusionServer` 对每个 slot 使用 `ClusterAdapter`，拼接所有 adapted activation，经 MLP classifier 输出 logits。训练 loss 是 `CrossEntropyLoss(logits, pseudo_labels)`。server backward 后，activation gradient 只路由回参与该 pseudo batch 的客户端 encoder。

## Validation And Test

每 10 rounds 读取 Stage 1 保存的 `validation_multimodal.pt`，按相同 sample index 同时取所有模态，并在 `torch.no_grad()` 下计算 loss、accuracy、macro-F1 和 weighted-F1。validation label 不参与输入构造、binding、筛选、排序或模态选择。validation weighted-F1 只用于选择 `best_model.pt` 和 early stopping；macro-F1 仅用于结果报告。

evaluation-only oracle mapping 只用于把 naturally paired validation/test 的真实模态 id 映射到 Stage 2 的 `pred_cluster`，并选择 representative client encoder。mapping 固定且不读取 validation/test label，不进入 scheduler、binding、optimizer、训练 forward/backward 或梯度路径。若一个真实模态被拆成多个 cluster，或一个 cluster 合并多个真实模态，则 evaluation 返回 failed，`loss`、`accuracy`、`macro_f1` 和 `weighted_f1` 为 `null`。聚类失败或质量不理想不会触发任何聚类算法、调度、binding、fusion 或 Split Learning 设计变更，Stage 3 仍按已有 `pred_cluster` 进入后续步骤。

正式配置最多训练 200 rounds，每 10 rounds validation，最少训练 50 rounds。validation weighted-F1 改善量必须超过 `0.001`；连续 3 次未改善则 early stop。训练结束先保存 `last_model.pt`，再恢复 `best_model.pt`，最后对 `test_multimodal.pt` 完整评估一次。`best_model.pt` 是正式 checkpoint，`final_metrics.json` 是正式 test 结果。

## 限制

- 不做 FedAvg。
- 不运行旧 unpaired 方法。
- 不实现或启用 D2D；`d2d.enabled` 当前应保持 `false`。
- 不把 same-label binding 解释成实例级自然配对。
- 不把 unknown-Q discovery 结果当作已稳定主结论。
