# semantic_split_multimodal

面向未知模态环境的语义对齐分布式多模态 Split Learning 实验框架。

English README: [README.md](README.md)。

## 项目概览

本仓库提供一套可复现的三阶段实验流程：

1. Stage 1：把 naturally paired 多模态数据划分为单模态客户端。
2. Stage 2：从 client encoder fingerprint 中发现未知客户端模态簇。
3. Stage 3：基于 `pred_cluster` 训练 MMBind 风格的 fusion Split Learning 模型。

本项目不是 Federated Learning，不使用 FedAvg。客户端上传 detached activation；服务器计算 cross-entropy loss，执行 fusion model 的 backward，再把 activation gradient 回传给对应客户端 encoder。

## 协议约束

训练阶段不使用真实模态名称、真实模态 ID、真实 Q，也不使用 oracle modality scheduler。Stage 1 保存的 `hidden_modality_id` 只允许用于 discovery 完成后的 post-hoc audit 和 evaluation-only oracle mapping。

Stage 3 使用按预测簇均衡的随机轮询调度、label-guided semantic pseudo binding、`ClusterAdapter`、concat fusion、既有 classifier 和 Split Learning 梯度回传。最终评估读取冻结 Stage 1 的 `test_multimodal.pt`；测试标签只用于计算指标，不用于构造测试输入。

当前只保留两种聚类方法：

- `kmeans`
- `adaptive_isodata`

## 安装

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

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

本地参考资料可以保留在 `local/references/`。整个 `local/` 目录都会被 Git 忽略。

## 结果目录

Stage 1 的数据划分是可复用资产：

```text
local/results/partition/<dataset>/<模态1>_<客户端数>clients_<模态2>_<客户端数>clients_.../
```

Stage 2 的聚类结果按数据集、partition signature 和聚类方法分开保存：

```text
local/results/cluster/<dataset>/<partition_signature>/<cluster_method>/
```

Stage 3 的训练和测试结果是正式实验 run：

```text
local/results/experiments/<dataset>/<run_id>/
```

当 `clients_per_modality: 10` 时，默认 partition signature 为：

```text
UCI-HAR: local/results/partition/uci_har/acc_10clients_gyro_10clients
MHEALTH: local/results/partition/mhealth/accelerometer_10clients_gyroscope_10clients_magnetometer_10clients_ecg_10clients
PAMAP2:  local/results/partition/pamap2/accelerometer_10clients_gyroscope_10clients_magnetometer_10clients
```

每个阶段都会拒绝覆盖已有的非空输出目录。Stage 3 请使用新的 `run_id`，例如 `run_1`、`run_2` 或 `adaptive_seed7`。

## Stage 1：数据划分

运行单个数据集：

```bash
python scripts/stage1_partition.py --config configs/uci_har.yaml
```

三个数据集依次运行：

```bash
python scripts/stage1_partition.py --config configs/uci_har.yaml
python scripts/stage1_partition.py --config configs/mhealth.yaml
python scripts/stage1_partition.py --config configs/pamap2.yaml
```

Stage 1 直接在 partition 目录下写入：

```text
train_clients/client_*.pt
client_meta.csv
test_multimodal.pt
partition_config.json
```

如果数据预处理、模态划分方式和客户端数量不变，这个 partition 可以被多次 Stage 2/Stage 3 实验复用。

## Stage 2：未知 Q 模态发现

UCI-HAR：

```bash
python scripts/stage2_discovery.py \
  --config configs/uci_har.yaml \
  --stage1-dir local/results/partition/uci_har/acc_10clients_gyro_10clients \
  --output-root local/results/cluster \
  --run-type user_formal
```

MHEALTH：

```bash
python scripts/stage2_discovery.py \
  --config configs/mhealth.yaml \
  --stage1-dir local/results/partition/mhealth/accelerometer_10clients_gyroscope_10clients_magnetometer_10clients_ecg_10clients \
  --output-root local/results/cluster \
  --run-type user_formal
```

PAMAP2：

```bash
python scripts/stage2_discovery.py \
  --config configs/pamap2.yaml \
  --stage1-dir local/results/partition/pamap2/accelerometer_10clients_gyroscope_10clients_magnetometer_10clients \
  --output-root local/results/cluster \
  --run-type user_formal
```

Stage 2 只保留：

```text
true_cluster.csv
pred_cluster.csv
pretrained_encoders/
stage2_metadata.json
```

运行 Stage 3 前，请检查 `stage2_metadata.json`。成功结果应包含 `metrics.discovery_status: discovery_success`。

## Stage 3：Fusion Split Learning

Stage 3 从冻结的 Stage 1 partition 和冻结的 Stage 2 cluster 目录读取输入。建议先只运行 UCI-HAR，再运行较大的数据集。

UCI-HAR：

```bash
python scripts/stage3_train.py \
  --config configs/uci_har.yaml \
  --stage1-dir local/results/partition/uci_har/acc_10clients_gyro_10clients \
  --stage2-dir local/results/cluster/uci_har/acc_10clients_gyro_10clients/adaptive_isodata \
  --output-root local/results/experiments \
  --run-id run_1 \
  --run-type user_formal
```

MHEALTH：

```bash
python scripts/stage3_train.py \
  --config configs/mhealth.yaml \
  --stage1-dir local/results/partition/mhealth/accelerometer_10clients_gyroscope_10clients_magnetometer_10clients_ecg_10clients \
  --stage2-dir local/results/cluster/mhealth/accelerometer_10clients_gyroscope_10clients_magnetometer_10clients_ecg_10clients/adaptive_isodata \
  --output-root local/results/experiments \
  --run-id run_1 \
  --run-type user_formal
```

PAMAP2：

```bash
python scripts/stage3_train.py \
  --config configs/pamap2.yaml \
  --stage1-dir local/results/partition/pamap2/accelerometer_10clients_gyroscope_10clients_magnetometer_10clients \
  --stage2-dir local/results/cluster/pamap2/accelerometer_10clients_gyroscope_10clients_magnetometer_10clients/adaptive_isodata \
  --output-root local/results/experiments \
  --run-id run_1 \
  --run-type user_formal
```

Stage 3 直接在 `local/results/experiments/<dataset>/<run_id>/` 下写入：

```text
train_log.csv
eval_log.csv
final_metrics.json
best_metrics.json
best_model.pt
final_model.pt
stage3_metadata.json
```

论文主结果读取 `best_metrics.json`。`final_metrics.json` 保留为最终轮诊断结果。

## 测试

```bash
PYTHONPATH=src python -m pytest tests -q
```
