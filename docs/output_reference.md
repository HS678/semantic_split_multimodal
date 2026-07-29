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

- `true_cluster.csv`：`client_id -> true_cluster`，仅用于可选 post-hoc audit；Stage 3 不要求该文件存在，也不以其一致性或可解析性决定是否训练，读取异常只记录到 input audit。
- `pred_cluster.csv`：`client_id -> pred_cluster`，Stage 3 的训练输入。
- `pretrained_encoders/client_XXX_encoder.pt`：Stage 2 预训练 encoder state dict。
- `stage2_metadata.json`：完整配置快照、Git SHA、runtime、Stage 1 输入路径、聚类方法和 discovery metrics；Stage 3 将其作为可选审计输入，文件读取异常、`discovery_status`、ACC、NMI、ARI 和其中报告的 Q 均不构成训练门槛。

Stage 2 不保留单独的 fingerprint 文件或独立 diagnostics 文件。相关审计信息写入 `stage2_metadata.json`。

Stage 3 的技术输入门槛仅包括完整且合法的 `pred_cluster.csv`、与 Stage 1 一致的 client IDs，以及逐客户端可加载的 pretrained encoder 文件。

## Stage 3 Experiment

默认目录：

```text
local/results/experiments/<dataset>/<run_id>/
```

文件：

- `train_log.csv`：每轮训练指标，包括 selected clients、selected clusters、binding 成功率、pseudo batch size、coverage 和参数更新量。
- `eval_log.csv`：naturally paired evaluation 指标；正式 final-only 配置仅包含最终轮记录。
- `final_metrics.json`：正式论文指标，包含最终轮 evaluation、round 汇总、oracle eval mapping、cluster ids 和协议字段。
- `best_metrics.json`：兼容性输出；final-only 模式下对应同一个最终轮。
- `best_model.pt`：兼容性 checkpoint；final-only 模式下与最终轮模型状态一致。
- `final_model.pt`：正式论文 checkpoint，即最后一轮 checkpoint。
- `stage3_metadata.json`：完整配置快照、Git SHA、runtime、Stage 1 输入、Stage 2 输入、scheduler、run type 和完成状态。

## final_metrics.json 关键字段

- `final_eval.eval_status`：`success` 或 `failed`。
- `final_eval.eval_failure_reason`：mapping failure 或 `null`。
- `final_eval.loss`、`accuracy`、`macro_f1`：naturally paired evaluation 指标。
- `oracle_eval_mapping.mapping_type`：固定为 `oracle_evaluation_only`。
- `total_global_rounds`、`effective_global_rounds`、`empty_binding_rounds`：global round 统计。
- `binding_success_rate`、`local_step_binding_success_rate`：binding 有效性统计。
- `scheduler`、`binding`、`fusion`：记录当前主线协议组件。
- `evaluation_mode`：正式配置为 `final_only`。
- `official_result`：明确记录正式指标文件为 `final_metrics.json`、正式 checkpoint 为 `final_model.pt`，选择规则为 `final_round`。

论文主结果读取 `final_metrics.json`，模型读取 `final_model.pt`。`best_*` 文件只保留接口兼容性。
