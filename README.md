# semantic_split_multimodal

这是一个轻量级、单进程的 split multimodal learning 实验框架，用于搭建论文中的非 D2D 基础实验流程。

当前项目目标不是实现完整 D2D 机制，而是先跑通：

1. 数据划分 `dataPartition`
2. 本地预训练与模态感知聚类 `pretrain + clustering`
3. 基于预测模态簇的 split multimodal training

D2D 暂时只在配置文件中预留：

```yaml
d2d:
  enabled: false
```

## 目录结构

```text
semantic_split_multimodal/
  clustering/      # KMeans、ISODATA 和聚类指标
  configs/         # 各数据集 YAML 实验配置
  data/            # 数据集适配器、数据注册表、Stage 1 数据划分逻辑
  docs/            # 数据集扩展和后续 D2D 接入说明
  experiments/     # 三个阶段的独立入口脚本
  models/          # 客户端 encoder 等 PyTorch 模型
  trainers/        # Stage 2 预训练和 Stage 3 split learning 训练逻辑
  utils/           # 配置、随机种子、设备、结果目录等工具函数
  results/         # 运行时生成结果，已被 git 忽略
  requirements.txt
  README.md
```

只跟踪源码、配置、文档和依赖文件。原始数据和实验结果不提交到仓库。

## 环境准备

建议使用 Python 3.10+。

```bash
pip install -r requirements.txt
```

主要依赖：

```text
torch
numpy
scikit-learn
pyyaml
```

## 结果保存规则

所有生成内容统一放在：

```text
results/<dataset_name>/<yy_mm_dd_HH_MM_SS_mmm>/
```

一次完整实验的目录形态如下：

```text
results/
  pamap2/
    latest_run.txt
    26_07_13_12_30_20_338/
      run_meta.yaml
      01_dataset_partition/
      02_cluster_results/
      03_training_evaluation/
      04_model_artifacts/
```

四个子目录含义：

```text
01_dataset_partition/       # Stage 1：客户端单模态训练数据和多模态配对测试数据
02_cluster_results/         # Stage 2：fingerprints、聚类结果、聚类指标、预训练 encoder
03_training_evaluation/     # Stage 2 同步摘要、Stage 3 训练日志、评估日志、最终指标
04_model_artifacts/         # Stage 3 最优 server 模型、最优 client encoders、模型信息
```

Stage 1 会创建新的时间戳 run 目录。Stage 2 和 Stage 3 默认读取同一数据集的 `latest_run.txt`，因此三条命令会写入同一个实验目录。

## 支持的数据集

### UCI-HAR

默认路径：

```text
data/uci-har/
```

期望结构：

```text
data/uci-har/
  train/
    Inertial Signals/
      body_acc_x_train.txt
      body_acc_y_train.txt
      body_acc_z_train.txt
      total_acc_x_train.txt
      total_acc_y_train.txt
      total_acc_z_train.txt
      body_gyro_x_train.txt
      body_gyro_y_train.txt
      body_gyro_z_train.txt
    y_train.txt
  test/
    Inertial Signals/
      body_acc_x_test.txt
      body_acc_y_test.txt
      body_acc_z_test.txt
      total_acc_x_test.txt
      total_acc_y_test.txt
      total_acc_z_test.txt
      body_gyro_x_test.txt
      body_gyro_y_test.txt
      body_gyro_z_test.txt
    y_test.txt
```

模态划分：

```text
acc  = body_acc_x/y/z + total_acc_x/y/z
gyro = body_gyro_x/y/z
```

运行命令：

```bash
python experiments/stage1_partition.py --config configs/uci_har.yaml
python experiments/stage2_pretrain_cluster.py --config configs/uci_har.yaml
python experiments/stage3_train_sl.py --config configs/uci_har.yaml
```

### MHEALTH

默认路径：

```text
data/MHEALTHDATASET/
```

期望文件：

```text
data/MHEALTHDATASET/
  mHealth_subject1.log
  ...
  mHealth_subject10.log
  README.txt
```

模态划分：

```text
chest            = chest acceleration + ECG
left_ankle       = left-ankle acceleration + gyroscope + magnetometer
right_lower_arm  = right-lower-arm acceleration + gyroscope + magnetometer
```

默认使用 subjects `1-8` 训练，subjects `9-10` 测试。标签 `0` 作为空标签丢弃。

运行命令：

```bash
python experiments/stage1_partition.py --config configs/mhealth.yaml
python experiments/stage2_pretrain_cluster.py --config configs/mhealth.yaml
python experiments/stage3_train_sl.py --config configs/mhealth.yaml
```

### PAMAP2

默认路径：

```text
data/pamap2+physical+activity+monitoring/
```

期望结构：

```text
data/pamap2+physical+activity+monitoring/
  PAMAP2_Dataset/
    PAMAP2_Dataset/
      Protocol/
        subject101.dat
        ...
        subject109.dat
```

当前正式实验只使用三个传感器类型模态：

```text
accelerometer = hand/chest/ankle 的 16g acceleration
gyroscope     = hand/chest/ankle 的 gyroscope
magnetometer  = hand/chest/ankle 的 magnetometer
```

当前配置不使用 `heart_rate`：

```yaml
dataset:
  type: pamap2
  modality_scheme: sensor_type
  include_heart_rate: false

num_modalities: 3
```

PAMAP2 的 orientation 列不使用，因为数据集说明中将其标记为 invalid。活动标签 `0` 被丢弃，其余 Protocol 活动标签重映射为 `0-11`。

运行命令：

```bash
python experiments/stage1_partition.py --config configs/pamap2.yaml
python experiments/stage2_pretrain_cluster.py --config configs/pamap2.yaml
python experiments/stage3_train_sl.py --config configs/pamap2.yaml
```

当前 PAMAP2 三模态 baseline 使用：

```yaml
model:
  encoder:
    type: cnn_gru
  server:
    hidden_dim: 256
    num_layers: 2
    dropout: 0.1

training:
  global_rounds: 100
  clients_per_cluster_per_round: 4
  client_lr: 0.0005
  server_lr: 0.0005

alignment:
  enabled: true
  type: supervised_contrastive
  lambda_align: 0.01
```

## 三阶段流水线说明

### Stage 1：数据划分

入口：

```bash
python experiments/stage1_partition.py --config configs/<dataset>.yaml
```

输入：

```text
dataset.root 指向的原始真实数据集
```

输出：

```text
results/<dataset>/<run_id>/01_dataset_partition/
  train_clients/
    client_000.pt
    client_001.pt
    ...
  test_multimodal.pt
  client_meta.csv
  partition_config.json
```

行为：

- 读取原始数据。
- 按配置划分模态。
- 构造客户端单模态训练数据。
- 构造多模态配对测试数据。
- 每个客户端只持有一种模态。

`client_meta.csv` 包含：

```text
client_id, modality_id, modality_name, num_samples
```

注意：真实 `modality_name` 只用于聚类评估，不用于 Stage 3 正式调度。

### Stage 2：预训练与聚类

入口：

```bash
python experiments/stage2_pretrain_cluster.py --config configs/<dataset>.yaml
```

输入：

```text
results/<dataset>/<run_id>/01_dataset_partition/train_clients/client_*.pt
```

输出：

```text
results/<dataset>/<run_id>/02_cluster_results/
  fingerprints.npy
  cluster_assignments.csv
  cluster_metrics.json
  pretrained_encoders/
  cluster_config.json
```

同步摘要：

```text
results/<dataset>/<run_id>/03_training_evaluation/
  cluster_result.txt
  cluster_metrics.json
```

行为：

- 为每个客户端初始化对应 encoder。
- 进行本地 representation learning。
- 提取 fingerprint。
- 使用 KMeans 或 ISODATA 聚类。
- 输出聚类 accuracy、NMI、ARI。

`cluster_assignments.csv` 包含：

```text
client_id, true_modality, pred_cluster
```

### Stage 3：Split Multimodal Training

入口：

```bash
python experiments/stage3_train_sl.py --config configs/<dataset>.yaml
```

输入：

```text
01_dataset_partition/train_clients/client_*.pt
01_dataset_partition/test_multimodal.pt
02_cluster_results/cluster_assignments.csv
02_cluster_results/pretrained_encoders/*.pt
```

输出：

```text
results/<dataset>/<run_id>/03_training_evaluation/
  train_log.csv
  eval_log.csv
  final_metrics.json
  best_metrics.json
  config_used.yaml

results/<dataset>/<run_id>/04_model_artifacts/
  best_server_model.pt
  best_client_encoders/
  best_model_info.json
```

行为：

- 只根据 `pred_cluster` 构造 `cluster_to_clients`。
- 正式训练调度不使用真实 `modality_name`。
- 每轮从每个预测簇中采样 `r = training.clients_per_cluster_per_round` 个客户端。
- 如果预测簇数量为 `Q_star`，则每轮参与客户端数为 `K_t = Q_star * r`。
- 调度结构为二维结构：

```python
selected[cluster_id][group_id] = client
```

- 构造 `r` 个 modality-complete groups。
- 按排序后的 `cluster_id` 拼接 feature，避免无序 concat。
- 保留：

```python
feature_map[(cluster_id, group_id)] -> client
```

- server backward 后，将每个 feature 的梯度返回到对应 client encoder。
- 测试使用 `test_multimodal.pt` 中的多模态配对测试集。

## 如何查看结果

查看某数据集最新 run：

```powershell
Get-Content results\pamap2\latest_run.txt
```

查看聚类指标：

```text
results/<dataset>/<run_id>/02_cluster_results/cluster_metrics.json
```

查看训练日志：

```text
results/<dataset>/<run_id>/03_training_evaluation/train_log.csv
```

查看评估日志：

```text
results/<dataset>/<run_id>/03_training_evaluation/eval_log.csv
```

查看最终指标：

```text
results/<dataset>/<run_id>/03_training_evaluation/final_metrics.json
```

查看最优指标：

```text
results/<dataset>/<run_id>/03_training_evaluation/best_metrics.json
```

论文实验通常优先报告 `best_metrics.json`，并同时说明最优轮次；`final_metrics.json` 表示最后一轮结果，可能受后期过拟合或训练漂移影响。

## 当前已实现内容

- UCI-HAR 双模态数据划分。
- MHEALTH 三模态真实数据适配。
- PAMAP2 三传感器模态真实数据适配。
- 三阶段独立入口。
- run-local 结果目录管理。
- 本地 encoder 预训练。
- fingerprint 提取。
- KMeans 聚类。
- 简化 ISODATA 聚类。
- 聚类 accuracy、NMI、ARI。
- 基于 `pred_cluster` 的 balanced cluster scheduling。
- 二维调度结构 `selected[cluster_id][group_id]`。
- 有序 feature concat。
- `feature_map` 梯度回传。
- 多模态配对测试。
- accuracy、macro-F1、weighted-F1、confusion matrix、per-class accuracy。
- best model 保存。
- `supervised_contrastive` 跨模态同标签表示对齐。

## 后续 TODO

- 增加 early stopping，减少后期训练漂移。
- 对正式论文实验做多 seed 统计，报告 mean/std。
- 对 PAMAP2、MHEALTH 继续做 window size 和 stride 对照。
- 为后续 4、5、6 模态数据集增加更细粒度 encoder。
- 将 D2D 时延测试作为独立模块接入，不混入当前三阶段主流程。
- 有小型 fixture 后再补充轻量 smoke tests。

## 扩展新数据集

详细说明见：

```text
docs/extension_guide.md
```

最小接入步骤：

1. 在 `data/` 下实现新数据集 adapter。
2. 返回统一数据结构：

```python
{
    "train": {
        "modalities": [torch.Tensor, torch.Tensor, ...],
        "labels": torch.LongTensor,
    },
    "test": {
        "modalities": [torch.Tensor, torch.Tensor, ...],
        "labels": torch.LongTensor,
    },
    "root": "resolved dataset root",
    "modality_names": ["modality_a", "modality_b", ...],
    "modality_input_dims": [768, 384, ...],
    "modality_input_shapes": [[6, 128], [3, 128], ...],
}
```

3. 在 `data/dataset_registry.py` 注册：

```python
register_dataset_loader("my_dataset", load_my_dataset)
```

4. 新增 `configs/my_dataset.yaml`。
5. 继续使用同样三条命令运行 Stage 1、Stage 2、Stage 3。

这样可以在不改实验入口的情况下扩展到 3 模态以上数据集。
