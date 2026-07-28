# Experiment Protocol

## 假设

训练阶段服务器不知道真实模态数量、真实模态名称和客户端真实模态归属。Stage 2 可以为了 discovery metrics 读取 `hidden_modality_id`；Stage 3 final evaluation 可以为了 evaluation-only oracle mapping 读取 `hidden_modality_id`。除此之外，训练协议不得使用真实模态身份。

## Stage 1

输入是 naturally paired 多模态数据集。loader 输出统一 contract：`train.modalities`、`train.labels`、`test.modalities`、`test.labels`、`modality_names` 和 `modality_input_shapes`。partitioner 将 train split 按模态拆成多个单模态 client，同时保存自然配对的 `test_multimodal.pt`。

## Stage 2

每个单模态 client encoder 先做 autoencoder pretraining。随后从 encoder 和数据统计中提取 fingerprint，并对 fingerprint 聚类得到 `pred_cluster`。known-Q 实验使用 `cluster.known_k`；unknown-Q 仍需实验验证，不能默认视为稳定结论。

## Stage 3

每个 global round 使用 `balanced_cluster_round_robin` scheduler。设 Stage 2 得到的预测簇集合为 `C`，配置 `training.clients_per_cluster_per_round = r`，则每轮从每个 `pred_cluster` 独立选择 `r` 个客户端，总客户端数为 `r * |C|`。每个 cluster 内维护独立随机轮询池：池内无放回随机抽样；当剩余客户端不足以补满本轮 `r` 个时，先取完剩余客户端，再将该 cluster 的池重置为排除本轮已选客户端后的其余客户端，并继续随机补足。

每个 selected client 独立采样 labeled batch，forward 后上传 detached activation。server 用 exact same-label random binding 构造 pseudo multimodal tuple：每个 tuple 在所有 cluster slot 上 label 相同，但不表示这些样本在实例级 naturally paired。

fusion slot 由 `pred_cluster` 和 `cluster_to_slot` 固定映射决定。`ConcatMLPFusionServer` 对每个 slot 使用 `ClusterAdapter`，拼接所有 adapted activation，经 MLP classifier 输出 logits。训练 loss 是 `CrossEntropyLoss(logits, pseudo_labels)`。server backward 后，activation gradient 只路由回参与该 pseudo batch 的客户端 encoder。

## Evaluation

正式 evaluation 读取 Stage 1 保存的 `test_multimodal.pt`，按相同 sample index 同时取所有模态。test label 只用于 CE loss、accuracy 和 macro-F1，不参与输入构造、binding、筛选、排序或模态选择。

evaluation-only oracle mapping 只用于把真实测试模态 id 映射到 Stage 2 的 `pred_cluster`，并选择 representative client encoder。若一个真实模态被拆成多个 cluster，或一个 cluster 合并多个真实模态，则 evaluation 返回 failed，`loss`、`accuracy` 和 `macro_f1` 为 `null`。

## 限制

- 不做 FedAvg。
- 不运行旧 unpaired 方法。
- 不实现或启用 D2D；`d2d.enabled` 当前应保持 `false`。
- 不把 same-label binding 解释成实例级自然配对。
- 不把 unknown-Q discovery 结果当作已稳定主结论。
