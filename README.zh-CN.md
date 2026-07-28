# semantic_split_multimodal

面向未知模态环境的语义对齐分布式多模态 Split Learning 实验框架。

英文说明见 [README.md](README.md)。

## 项目概览

本仓库提供一套可复现的三阶段多模态 Split Learning 实验流程。当前唯一主线方法是 `mmbind_fusion_split_learning`：

1. Stage 1：把 naturally paired 多模态训练数据划分为单模态客户端。
2. Stage 2：使用 adaptive ISODATA 在 unknown-Q 设置下发现客户端模态簇。
3. Stage 3：基于 `pred_cluster` 训练 MMBind 风格的多模态 fusion Split Learning 模型。

本项目不是 Federated Learning，不使用 FedAvg。客户端上传 detached activation；服务器计算 cross-entropy loss，完成 fusion model 的 backward，然后把 activation gradient 路由回对应客户端 encoder。

## 协议约束

训练阶段不使用真实模态名称、真实模态 ID、真实 Q，也不使用 oracle modality scheduler。Stage 1 保存的 `hidden_modality_id` 只允许用于 discovery 完成后的 post-hoc audit 和 evaluation-only oracle mapping。

`hidden_modality_id` 不得参与 PCA、split/merge 决策、Q 选择、seed 选择、scheduler、binding、fusion slot、模型输入或训练 loss。

最终评估从冻结的 Stage 1 partition 目录读取 `test_multimodal.pt`。测试标签只用于计算指标，不用于构造测试输入。

## 安装

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

`hdbscan` 只有在选择 HDBSCAN 聚类模式时才需要。默认配置使用 adaptive ISODATA。

## 数据目录

仓库不包含数据集。请把原始数据放到：

```text
local/datasets/
```

默认期望目录：

```text
local/datasets/uci_har/
local/datasets/mhealth/
local/datasets/pamap2/
```

本地参考资料可以保留在：

```text
local/references/
```

整个 `local/` 目录都会被 Git 忽略。不要提交数据集、reference、checkpoint、日志、fingerprint、生成配置或正式实验结果。

## 配置文件

公开配置文件：

```text
configs/uci_har.yaml
configs/mhealth.yaml
configs/pamap2.yaml
```

当 `clients_per_modality: 10` 时，默认 Stage 1 partition 名称为：

```text
UCI-HAR: local/results/partition/uci_har/acc-gyro_10clients
MHEALTH: local/results/partition/mhealth/accelerometer-gyroscope-magnetometer-ecg_10clients
PAMAP2:  local/results/partition/pamap2/accelerometer-gyroscope-magnetometer_10clients
```

## 结果目录

Stage 1 的数据划分是可复用资产，保存到：

```text
local/results/partition/<dataset>/<modality_names>_<clients_per_modality>clients/
```

Stage 2、Stage 3 和后续 D2D 属于同一套实验运行，保存到：

```text
local/results/experiment/<dataset>/<run_id>/
  02_cluster_results/
  02_discovery_logs/
  03_training_evaluation/
  04_model_artifacts/
```

建议使用明确的 `run_id`，例如 `run_1`、`run_2`、`run_3`。

防覆盖规则：

- Stage 1 如果发现 partition 目录已存在且非空，会拒绝覆盖。
- Stage 2 如果发现已有 Stage 2 输出，会拒绝覆盖。
- Stage 3 可以复用 Stage 2 创建的同一个 experiment run 目录，但如果已有 Stage 3 输出，会拒绝覆盖。

## Stage 1：生成可复用数据划分

运行单个数据集：

```bash
python scripts/stage1_partition.py --config configs/uci_har.yaml
```

也可以使用批量脚本：

```bash
scripts/run_stage1_partitions.sh uci_har
scripts/run_stage1_partitions.sh mhealth
scripts/run_stage1_partitions.sh pamap2
scripts/run_stage1_partitions.sh all
```

Stage 1 输出：

```text
train_clients/client_*.pt
client_meta.csv
test_multimodal.pt
partition_config.json
```

如果配置不变，这个 Stage 1 partition 可以被多次 Stage 2/Stage 3 实验复用。

## Stage 2：Unknown-Q 模态发现

UCI-HAR：

```bash
python scripts/stage2_discovery_only.py \
  --config configs/uci_har.yaml \
  --stage1-dir local/results/partition/uci_har/acc-gyro_10clients \
  --output-root local/results/experiment \
  --tag run_1 \
  --run-type user_formal
```

MHEALTH：

```bash
python scripts/stage2_discovery_only.py \
  --config configs/mhealth.yaml \
  --stage1-dir local/results/partition/mhealth/accelerometer-gyroscope-magnetometer-ecg_10clients \
  --output-root local/results/experiment \
  --tag run_1 \
  --run-type user_formal
```

PAMAP2：

```bash
python scripts/stage2_discovery_only.py \
  --config configs/pamap2.yaml \
  --stage1-dir local/results/partition/pamap2/accelerometer-gyroscope-magnetometer_10clients \
  --output-root local/results/experiment \
  --tag run_1 \
  --run-type user_formal
```

Stage 2 输出：

```text
02_cluster_results/pretrained_encoders/*_encoder.pt
02_cluster_results/fingerprints.npy
02_cluster_results/cluster_assignments.csv
02_cluster_results/cluster_metrics.json
02_cluster_results/adaptive_diagnostics.json
02_discovery_logs/stage2_only_metadata.json
02_discovery_logs/stage2_only_config_used.yaml
```

运行 Stage 3 前，请先检查：

```text
local/results/experiment/<dataset>/<run_id>/02_cluster_results/cluster_metrics.json
```

理想情况下应看到 `discovery_status: discovery_success`。

## Stage 3：Fusion Split Learning

Stage 3 使用和 Stage 2 相同的 `run_id`。

UCI-HAR：

```bash
python scripts/stage3_train_only.py \
  --config configs/uci_har.yaml \
  --stage1-dir local/results/partition/uci_har/acc-gyro_10clients \
  --stage2-dir local/results/experiment/uci_har/run_1/02_cluster_results \
  --output-root local/results/experiment \
  --tag run_1 \
  --run-type user_formal
```

MHEALTH：

```bash
python scripts/stage3_train_only.py \
  --config configs/mhealth.yaml \
  --stage1-dir local/results/partition/mhealth/accelerometer-gyroscope-magnetometer-ecg_10clients \
  --stage2-dir local/results/experiment/mhealth/run_1/02_cluster_results \
  --output-root local/results/experiment \
  --tag run_1 \
  --run-type user_formal
```

PAMAP2：

```bash
python scripts/stage3_train_only.py \
  --config configs/pamap2.yaml \
  --stage1-dir local/results/partition/pamap2/accelerometer-gyroscope-magnetometer_10clients \
  --stage2-dir local/results/experiment/pamap2/run_1/02_cluster_results \
  --output-root local/results/experiment \
  --tag run_1 \
  --run-type user_formal
```

Stage 3 输出：

```text
03_training_evaluation/train_log.csv
03_training_evaluation/eval_log.csv
03_training_evaluation/final_metrics.json
04_model_artifacts/best_mmbind_fusion_checkpoint.pt
04_model_artifacts/last_mmbind_fusion_checkpoint.pt
04_model_artifacts/cluster_to_slot.json
```

论文主结果应读取：

```text
03_training_evaluation/final_metrics.json
```

其中 `final_eval` 字段是 naturally paired final evaluation 指标。

## 重新运行实验

如果 Stage 1 数据划分配置没有变化，复用已有 partition，直接开启新的 experiment run：

```bash
python scripts/stage2_discovery_only.py \
  --config configs/uci_har.yaml \
  --stage1-dir local/results/partition/uci_har/acc-gyro_10clients \
  --output-root local/results/experiment \
  --tag run_2 \
  --run-type user_formal
```

然后 Stage 3 也使用 `--tag run_2`。

如果修改了数据预处理、模态划分方式、客户端数量或其他 Stage 1 partition 配置，请重新运行 Stage 1。如果旧 partition 目录已经存在，请在确认不再需要后手动移动或删除。

## 测试

语法检查：

```bash
python -m compileall src scripts tests
```

完整测试：

```bash
PYTHONPATH=src python -m pytest tests -q
```

## 项目结构

```text
configs/    # UCI-HAR、MHEALTH、PAMAP2 配置
docs/       # 协议、架构、输出和交接文档
scripts/    # Stage 1、Stage 2-only、Stage 3-only 入口
src/        # semantic_split_multimodal package
tests/      # 当前主线的单元测试和回归测试
local/      # 被 Git 忽略的本地数据、参考资料、输出和 checkpoint
```

本仓库不发布数据集、checkpoint、正式实验结果或尚未完成的消融结论。
