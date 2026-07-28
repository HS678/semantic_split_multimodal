# Output Reference

## Stage 1 Partition

默认目录：

```text
local/results/partition/<dataset>/<partition_signature>/
```

文件：

- `train_clients/client_*.pt`：单模态 client payload，包含 samples、labels、`hidden_modality_id`、encoder type 和 input shape。
- `client_meta.csv`：client metadata。`hidden_modality_id` 只允许用于 Stage 2 audit 和 evaluation-only oracle mapping。
- `test_multimodal.pt`：naturally paired test payload，包含 `modalities`、`modality_names`、`modality_input_shapes` 和 `label`。
- `partition_config.json`：Stage 1 运行配置摘要。

## Stage 2 Cluster

默认目录：

```text
local/results/cluster/<dataset>/<partition_signature>/<cluster_method>/
```

文件：

- `true_cluster.csv`：`client_id -> true_cluster`，仅用于 post-hoc audit。
- `pred_cluster.csv`：`client_id -> pred_cluster`，Stage 3 的训练输入。
- `pretrained_encoders/client_XXX_encoder.pt`：Stage 2 预训练 encoder state dict。
- `stage2_metadata.json`：完整配置快照、Git SHA、runtime、Stage 1 输入路径、聚类方法和 discovery metrics。

Stage 2 不保留单独的 fingerprint 文件或独立 diagnostics 文件。相关审计信息写入 `stage2_metadata.json`。

## Stage 3 Experiment

默认目录：

```text
local/results/experiments/<dataset>/<run_id>/
```

文件：

- `train_log.csv`：每轮训练指标，包括 selected clients、selected clusters、binding 成功率、pseudo batch size、coverage 和参数更新量。
- `eval_log.csv`：周期性 naturally paired evaluation 指标。
- `final_metrics.json`：最终轮 evaluation、round 汇总、oracle eval mapping、cluster ids 和协议字段。
- `best_metrics.json`：macro-F1 最优轮的 evaluation metrics，作为论文主结果。
- `best_model.pt`：最优主线 checkpoint。
- `final_model.pt`：最后一轮 checkpoint。
- `stage3_metadata.json`：完整配置快照、Git SHA、runtime、Stage 1 输入、Stage 2 输入、scheduler、run type 和完成状态。

## final_metrics.json 关键字段

- `final_eval.eval_status`：`success` 或 `failed`。
- `final_eval.eval_failure_reason`：mapping failure 或 `null`。
- `final_eval.loss`、`accuracy`、`macro_f1`：naturally paired evaluation 指标。
- `oracle_eval_mapping.mapping_type`：固定为 `oracle_evaluation_only`。
- `total_global_rounds`、`effective_global_rounds`、`empty_binding_rounds`：global round 统计。
- `binding_success_rate`、`local_step_binding_success_rate`：binding 有效性统计。
- `scheduler`、`binding`、`fusion`：记录当前主线协议组件。

论文主结果读取 `best_metrics.json`；`final_metrics.json` 保留为最终轮诊断。
