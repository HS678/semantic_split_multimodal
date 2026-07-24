# semantic_split_multimodal

这是论文实验代码，用于研究未知模态数量、未知客户端模态归属、未配对单模态客户端环境下的分布式多模态 Split Learning。

当前主流程：

```text
Raw multimodal dataset
-> Independent modality client partition
-> Each client owns one modality
-> Encoder pretraining
-> Fingerprint extraction
-> Unknown modality discovery
-> Cluster-based client scheduling
-> Unpaired Split Learning
-> Shared semantic learning
-> Prototype-based cross-modal alignment
-> Evaluation
```

已删除旧设计：`pseudo_paired_concat`、样本级 multimodal fusion、按标签构造伪多模态样本、训练阶段依赖真实 modality label、input dimension hint、固定真实模态数的 ISODATA 设置。

## 目录结构

```text
clustering/     # KMeans/HDBSCAN/ISODATA 与 fingerprint
configs/        # 数据集、encoder、cluster、scheduler、alignment 配置
data/           # 数据集适配器、Client 对象、独立单模态 client partition
evaluation/     # discovery/scheduling/learning/D2D 指标
experiments/    # 三阶段入口
models/         # encoder registry 与 server 模块
scheduling/     # Random/RoundRobin/Oracle/Proposed schedulers
trainers/       # 预训练聚类与未配对 Split Learning
utils/          # 配置、设备、随机种子、结果目录
```

## 安装

建议 Python 3.10+。

```bash
pip install -r requirements.txt
```

`hdbscan` 是可选聚类方法的依赖；如果只运行 `kmeans` 或 `isodata`，不会在导入阶段强制使用它。

## 运行

```bash
python experiments/stage1_partition.py --config configs/uci_har.yaml
python experiments/stage2_pretrain_cluster.py --config configs/uci_har.yaml
python experiments/stage3_train_sl.py --config configs/uci_har.yaml
```

同样支持：

```bash
python experiments/stage1_partition.py --config configs/mhealth.yaml
python experiments/stage2_pretrain_cluster.py --config configs/mhealth.yaml
python experiments/stage3_train_sl.py --config configs/mhealth.yaml

python experiments/stage1_partition.py --config configs/pamap2.yaml
python experiments/stage2_pretrain_cluster.py --config configs/pamap2.yaml
python experiments/stage3_train_sl.py --config configs/pamap2.yaml
```

## 数据与 Client

Stage 1 将原始多模态数据独立切分为单模态客户端。每个 `Client` 包含：

```text
client_id
hidden_modality_id
samples
labels
encoder_type
pred_cluster
```

训练阶段只使用 `samples`、`labels`、`encoder_type`、`pred_cluster`。`hidden_modality_id` 只允许用于 discovery/evaluation 指标和 `OracleModalityScheduler` 消融。

## 模型

客户端 encoder 通过 `models/encoders.py` 的 registry 创建：

- `TimeSeriesEncoder`：1D CNN，用于 UCI-HAR、PAMAP2、MHEALTH。
- `ImageEncoder`：ResNet18 接口。
- `VideoEncoder`：预留 3D CNN 接口。
- `AudioEncoder`：预留 CNN 接口。

服务器端：

```text
ClusterAdapter
-> SharedSemanticBackbone
-> Classifier
```

不同预测簇的 client activation 先经过对应 `ClusterAdapter`，再进入统一 semantic space。

## Fingerprint 与聚类

支持三类 fingerprint：

- `encoder`：预训练 encoder 表示统计。
- `signal`：信号统计。
- `hybrid`：encoder + signal。

支持聚类：

- `kmeans`
- `hdbscan`
- `isodata`

Stage 2 输出：

```text
cluster_assignments.csv
cluster_metrics.json
fingerprints.npy
pretrained_encoders/
```

`cluster_metrics.json` 包含 ARI、NMI、ACC、estimated modality number。真实模态只用于这些 discovery 指标。

## Scheduler

统一接口位于 `scheduling/schedulers.py`：

- `RandomScheduler`
- `RoundRobinScheduler`
- `OracleModalityScheduler`
- `ProposedClusterCoverageScheduler`

默认使用：

```yaml
training:
  scheduler: proposed_cluster_coverage
  clients_per_round: 8
```

每轮客户端数量由 `clients_per_round` 固定。Proposed scheduler 优先覆盖所有预测模态簇。

## Unpaired Split Learning

Stage 3 不做 feature concat，不做 sample alignment，不构造伪多模态样本。每个被选客户端独立采样 batch：

```text
client encoder
-> activation upload
-> cluster adapter
-> shared semantic backbone
-> classifier
```

损失函数：

```text
L = L_cls + lambda * L_align
```

`L_align` 使用 `PrototypeBank`，按 `(cluster_id, class_id)` 维护 EMA prototype。

## Evaluation

当前指标：

- Discovery：ARI、NMI、ACC、estimated modality number。
- Scheduling：coverage、participation fairness。
- Learning：accuracy、macro F1、per modality accuracy。
- D2D：latency、speedup 占位指标；真实 D2D 协作逻辑后续接入。

## 配置要点

```yaml
fingerprint:
  type: hybrid

cluster:
  method: isodata
  known_k: null

training:
  multimodal_mode: unpaired_split_learning
  scheduler: proposed_cluster_coverage
  clients_per_round: 8

alignment:
  enabled: true
  type: prototype
  lambda_align: 0.05
  ema_momentum: 0.9
```

## 当前验证状态

已通过：

```bash
python -m py_compile $(rg --files -g '*.py')
```

未能在当前环境跑完整训练 smoke test，因为 Python 环境缺少 `torch`。安装依赖后建议先用较小 `global_rounds` 验证三阶段端到端运行。
