# Output Reference

## Stage 1 Partition

默认目录：

```text
local/results/partition/<dataset>/<partition_signature>/
```

文件：

- `train_clients/client_*.pt`：单模态 client payload，包含 samples、labels、`hidden_modality_id`、encoder type 和 input shape。
- `client_meta.csv`：client metadata。`hidden_modality_id` 只允许用于 Stage 2 audit 和无梯度 validation/test evaluation-only oracle mapping。
- `validation_multimodal.pt`：naturally paired validation payload；用于 checkpoint 选择和 early stopping。
- `test_multimodal.pt`：naturally paired test payload，包含 `modalities`、`modality_names`、`modality_input_shapes` 和 `label`。
- `partition_config.json`：Stage 1 运行配置摘要，包含完整 `dataset_config`、label mapping、split protocol、可用的 subject 列表和三个 split 样本数。

## Stage 2 Cluster

默认目录：

```text
local/results/cluster/<dataset>/<partition_signature>/<cluster_method>/
```

文件：

- `true_cluster.csv`：`client_id -> true_cluster`。当前开发配置的 Stage 3 固定读取它；该结果必须标记为 Oracle/debug。
- `pred_cluster.csv`：`client_id -> pred_cluster`。Stage 2 始终生成并审计，后续论文正式配置切换为该输入。
- `pretrained_encoders/client_XXX_encoder.pt`：Stage 2 预训练 encoder state dict。
- `fingerprints.npz`：逐客户端原始 fingerprint、client ID、审计标签及二维 PCA 坐标，用于结果复现和绘图审计。
- `fingerprint_pca.pdf`：基于聚类前 fingerprint 的双面板矢量 PCA 图，可直接用于论文。
- `fingerprint_pca.png`：同图的 600 DPI PNG 预览版本。
- `fingerprint_pca_metadata.json`：PCA explained variance、fingerprint 维度、绘图参数及无泄漏声明。
- `stage2_metadata.json`：完整配置快照、Git SHA、runtime、Stage 1 输入路径、聚类方法和 discovery metrics；Stage 3 将其作为可选审计输入，文件读取异常、`discovery_status`、ACC、NMI、ARI 和其中报告的 Q 均不构成训练门槛。

PCA 坐标仅由 fingerprint 计算。`true_cluster` 与 `pred_cluster` 只用于 post-hoc audit 着色，不得参与 PCA 拟合、聚类或参数选择。

Stage 3 按 `training.cluster_assignment_source` 检查所选 cluster CSV、与 Stage 1 一致的 client IDs，以及逐客户端可加载的 pretrained encoder 文件。

## Stage 3 Experiment

默认目录：

```text
local/results/experiments/<oracle_true_cluster|predicted_cluster>/<dataset>/<config_signature>/seed-<seed>/attempt-<nn>/
```

文件：

- `source_config.config`：本次 run 原样复制的输入配置文件。
- `resolved_config.config`：本次 run 的完整解析配置，包括最终 seed、attempt 和输入输出路径。
- `train_log.csv`：每轮训练指标，包括 selected clients、selected clusters、binding 成功率、pseudo batch size、coverage 和参数更新量。
- `validation_log.csv`：每 10 rounds 的 naturally paired validation loss、accuracy、macro-F1、weighted-F1、是否更新 best 和 patience 计数。
- `best_metrics.json`：最佳 validation 指标、`best_round` 和选择规则。
- `final_metrics.json`：恢复 `best_model.pt` 后一次性 naturally paired test 的正式指标。
- `best_model.pt`：validation weighted-F1 选择的正式 checkpoint。
- `last_model.pt`：训练停止时 checkpoint，仅用于诊断。
- `training_curves.png`：由 `train_log.csv` 和 `validation_log.csv` 生成的训练/验证曲线，不包含 test 曲线。
- `stage3_metadata.json`：完整配置快照、Git SHA、runtime、Stage 1 输入、Stage 2 输入、scheduler、run type 和完成状态。

Stage3 会自动生成曲线。手动重绘入口位于：

```bash
python -m semantic_split_multimodal.evaluation.plot_training_curves --run-dir <run_dir>
```

## final_metrics.json 关键字段

- `test_eval_status`：`success` 或 `failed`。
- `test_eval_failure_reason`：mapping failure 或 `null`。
- `test_loss`、`test_accuracy`、`test_macro_f1`、`test_weighted_f1`：naturally paired test 指标。
- `checkpoint` / `selected_by`：固定为 `best_model.pt` / `validation_weighted_f1`。
- `oracle_eval_mapping.mapping_type`：固定为 `oracle_evaluation_only`。
- `configured_global_rounds`、`executed_global_rounds`、`effective_global_rounds`、`empty_binding_rounds`：global round 统计。
- `binding_success_rate`、`local_step_binding_success_rate`：binding 有效性统计。
- `scheduler`、`binding`、`fusion`：记录当前主线协议组件。
- `validation_protocol`：固定为 naturally paired evaluation-only oracle mapping。
- `best_round`、`stop_round`、`stop_reason`：checkpoint 选择和停止状态。
- `test_evaluation_count`：成功正式 run 必须为 `1`。
- `official_result`：正式指标文件为 `final_metrics.json`、正式 checkpoint 为 `best_model.pt`，选择规则为 `best_validation_weighted_f1`。

论文主结果读取 `final_metrics.json`，模型读取 `best_model.pt`。`last_model.pt` 不参与论文测试结果。
