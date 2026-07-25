# Architecture

## 三阶段架构

Stage 1 位于 `scripts/stage1_partition.py`，调用 `semantic_split_multimodal.data.partitioner`。它把原始多模态数据拆成单模态客户端，同时保存 naturally paired `test_multimodal.pt`。

Stage 2 位于 `scripts/stage2_discovery.py`，调用 `semantic_split_multimodal.learning.pretrain` 和 `semantic_split_multimodal.discovery`。它执行 autoencoder pretraining、fingerprint extraction 和 KMeans/HDBSCAN/ISODATA clustering。

Stage 3 位于 `scripts/stage3_train.py`。主线调用 `semantic_split_multimodal.learning.fusion_sl`，baseline 调用 `semantic_split_multimodal.learning.baseline_unpaired`。

## 模块职责

- `data/`：数据集 loader、Client payload、单模态客户端划分。
- `discovery/`：fingerprint 和聚类。
- `learning/models.py`：client encoders、autoencoder 组件、`ClusterAdapter`、`ConcatMLPFusionServer`、`SharedSemanticBackbone`、`ClassifierHead`。
- `learning/binding.py`：same-label random pseudo binding。
- `learning/scheduling.py`：客户端选择策略。
- `evaluation/`：discovery metrics、naturally paired fusion eval、oracle evaluation mapping。
- `utils/`：配置、设备、结果目录、随机种子。

## 数据流

原始数据进入 Stage 1 后产生 `train_clients/client_*.pt` 和 `test_multimodal.pt`。Stage 2 读取训练客户端，保存 `cluster_assignments.csv`、`cluster_metrics.json`、`fingerprints.npy` 和 pretrained encoders。Stage 3 读取 Stage 1/2 产物，基于 `pred_cluster` 调度、binding 和 fusion，并输出训练日志、评估指标和 checkpoint。

## Client / Server 边界

client 只持有本地单模态样本、label 和 encoder。server 接收 detached activation，完成 pseudo binding、fusion、classification、loss 和 backward，再把 activation gradient 返回对应 client 更新 encoder。

## 主线与 baseline

主线是 MMBind-style label-random fusion Split Learning。baseline 是 unpaired shared semantic Split Learning，保留 `PrototypeBank` 和 `SharedSemanticBackbone`，但不替代主方法。
