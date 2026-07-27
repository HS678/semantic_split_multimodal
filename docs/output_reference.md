# Output Reference

当前 run 目录由 `utils.results.configure_result_run` 解析。正式配置默认写入 `local/results/<dataset>/<run_id>/`；formal 实验配置可写入 `local/results/formal/<dataset>/known_q/`。

## 01_dataset_partition

- `train_clients/client_*.pt`：单模态 client payload，包含 samples、labels、`hidden_modality_id`、encoder type 和 input shape。
- `client_meta.csv`：client metadata。`hidden_modality_id` 只允许用于 Stage 2 metrics 和 evaluation-only oracle mapping。
- `test_multimodal.pt`：naturally paired test payload，包含 `modalities`、`modality_names`、`modality_input_shapes` 和 `label`。
- `partition_config.json`：Stage 1 运行配置摘要。

## 02_cluster_results

- `pretrained_encoders/client_XXX_encoder.pt`：Stage 2 预训练 encoder state dict。
- `fingerprints.npy`：client fingerprint matrix。
- `cluster_assignments.csv`：`client_id -> pred_cluster`。
- `cluster_metrics.json`：discovery metrics，包括 `true_Q`、`estimated_Q`、`ACC`、`NMI`、`ARI`。
- `cluster_config.json`：clustering 配置摘要。

## 03_training_evaluation

- `train_log.csv`：每轮训练指标，包括 selected clients、selected clusters、binding 成功率、pseudo batch size、coverage 和参数更新量。
- `eval_log.csv`：周期性 naturally paired evaluation 指标。
- `final_metrics.json`：最终 evaluation、round 汇总、oracle eval mapping、cluster ids 和协议字段。
- `best_metrics.json`：macro-F1 最优轮的 evaluation metrics。
- `config_used.yaml`：解析后的运行配置。
- `cluster_result.txt`、`cluster_metrics.json`：复制或记录 discovery 结果，便于 Stage 3 目录自包含审计。

## 04_model_artifacts

- `best_mmbind_fusion_checkpoint.pt`：最优主线 checkpoint，包含 server state、client encoder states、`pred_cluster_assignments`、`cluster_ids`、`cluster_to_slot`、resolved config 和 metrics。
- `last_mmbind_fusion_checkpoint.pt`：最后一轮主线 checkpoint。
- `best_server_model.pt`：最优 fusion server state dict。
- `best_client_encoders/client_XXX_encoder.pt`：最优轮 client encoder state dict。
- `cluster_to_slot.json`：`pred_cluster -> fusion slot` 映射。
- `oracle_eval_modality_to_cluster.json`：evaluation-only oracle mapping 结果。
- `best_model_info.json`：最优指标和 cluster slot 摘要。

## final_metrics.json 关键字段

- `final_eval.eval_status`：`success` 或 `failed`。
- `final_eval.eval_failure_reason`：mapping failure 或 `null`。
- `final_eval.loss`、`accuracy`、`macro_f1`：正式 naturally paired evaluation 指标；mapping failure 时为 `null`。
- `oracle_eval_mapping.mapping_type`：固定为 `oracle_evaluation_only`。
- `total_global_rounds`、`effective_global_rounds`、`empty_binding_rounds`：global round 统计。
- `total_attempted_local_steps`、`total_effective_local_steps`、`total_empty_binding_local_steps`：local step 统计。
- `binding_success_rate`、`local_step_binding_success_rate`：binding 有效性统计。
- `scheduler`、`binding`、`fusion`：记录当前主线协议组件。

论文主结果应读取 `final_eval` 的 naturally paired metrics，不应使用诊断性 client-only 指标。
