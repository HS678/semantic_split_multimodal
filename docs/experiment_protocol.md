# Experiment Protocol

## Unknown Modality 假设

训练阶段服务器不知道真实模态数量、真实模态名称和客户端真实模态归属。`hidden_modality_id` 只允许用于 Stage 2 discovery metrics 和 evaluation-only oracle mapping。

## Training Pseudo Binding

训练允许使用 training label 进行 exact same-label random pseudo binding。该 binding 不使用 instance-level correspondence，也不读取 test label。fusion slot 只由 `pred_cluster` 和 `cluster_to_slot` 决定。

## Naturally Paired Testing

正式 evaluation 使用 Stage 1 保存的 `test_multimodal.pt`。所有模态按同一个 sample index 构造输入；test label 只用于 loss、accuracy、macro-F1，不参与 binding、筛选或排序。

## Global Round / Local Step

每个 global round 调用一次 scheduler。被选客户端集合在该 round 的所有 `local_steps` 内固定；每个 local step 独立采样 batch。empty binding 只跳过当前 local step，所有 local step 均失败时记录 `empty_binding_round`。

## Metrics

Stage 2 discovery 输出 `true_Q`、`estimated_Q`、`abs_Q_error`、`ACC`、`NMI`、`ARI`。Stage 3 输出训练日志、learning metrics 和 naturally paired evaluation metrics。

## Mapping Failure

oracle mapping 仅用于 evaluation。真实模态被拆分或预测簇合并真实模态时，`accuracy`、`macro_f1`、`loss` 为 `null`，不做 majority 静默修复。

## Baseline 和消融

`unpaired_split_learning` 保留 shared semantic baseline、`PrototypeBank` 和 `SharedSemanticBackbone`。baseline 不能改写为主线方法，主线也不引入 baseline 的 prototype alignment 作为默认训练目标。
