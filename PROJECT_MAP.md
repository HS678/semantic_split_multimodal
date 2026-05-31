# PROJECT_MAP

## 1) 项目整体目标
`semantic_split_multimodal` 的主目标是：在单进程内模拟“分布式 split multimodal learning”的训练协议。  
该项目**不是 Federated Learning**，不实现 FedAvg，不做客户端参数聚合。  
客户端只负责本地 encoder 前向与接收服务器梯度后的本地反向更新；服务器负责语义投影、对齐、融合、分类与损失反传。

## 2) 目录说明
- `data/`：数据构造与数据集适配（synthetic、real npz、UCI-HAR）。
- `clients/`：客户端节点实现（本地 encoder、采样、split-learning 接口）。
- `clustering/`：Stage 1 聚类与聚类评估（KMeans、指标，ISODATA stub）。
- `server/`：服务器训练核心与测试评估逻辑。
- `models/`：模型基础模块（encoder/projector/fusion/classifier；attention stub）。
- `losses/`：语义对齐相关损失（SupCon、prototype stub loss）。
- `trainers/`：Stage 2 训练编排、调度器、D2D stub。
- `experiments/`：训练入口脚本与多 seed 脚本。
- `tests/`：协议一致性、梯度路由、调度、公平性、UCI-HAR 无泄漏等测试。
- `utils/`：配置读取、随机种子、设备选择、采样器。
- `configs/`：默认配置、UCI-HAR 配置、消融配置。
- `experiments/results/`：训练日志与 summary 的 JSON 结果产物目录。
- `log/`：运行日志文件目录。

## 3) 文件职责表（主要 .py）
| 文件路径 | 所属阶段 | 主要职责 | 核心类/函数 | 输入 | 输出 | 是否参与主训练流程 | 是否可以暂时忽略 | 备注 |
|---|---|---|---|---|---|---|---|---|
| `experiments/run_stage2_training.py` | 入口编排 | 读取配置、CLI 覆盖、准备数据、启动 Stage2Trainer | `main`, `_prepare_dataset` | yaml 配置、CLI 参数 | 完整训练过程与控制台日志 | 是 | 否 | 主实验入口 |
| `trainers/stage2_trainer.py` | Stage 1+2 主控 | 聚类、调度、round/step 训练循环、评估、日志落盘 | `Stage2Trainer.run`, `cluster_clients` | `cfg`、客户端池、test_set | 指标、JSON 日志 | 是 | 否 | 主流程核心 |
| `trainers/schedulers.py` | Stage 2 调度 | 全模态公平随机调度；保留其他调度器 stub | `FairRandomFullModalityScheduler.select` | `cluster_to_clients` | 每轮选中 client map | 是 | 否 | 两个 stub：`PairedFullModalityScheduler`、`GlobalRandomScheduler` |
| `clients/client_node.py` | Stage 2 客户端侧 | client 采样、encoder 前向、接收梯度反向更新、聚类表示提取 | `SplitClient` | 客户端分片数据、batch | `z_client`、`z_server`、更新后 encoder | 是 | 否 | split 协议关键点在此 |
| `server/server_core.py` | Stage 2 服务器侧 | 语义 batch 构造、投影/对齐/融合/分类、反传、梯度回填 | `SemanticBatchBuilder.build`, `SplitServer.train_step` | 多 client payload | `grad_to_clients`、loss 统计 | 是 | 否 | 禁止二次 detach 已落实 |
| `server/evaluation.py` | 评估 | 使用 paired test set 做多模态评估 | `evaluate_paired_test` | `clients_by_modality`、`server_model`、`test_set` | `acc/macro_f1/top1_acc` | 是 | 否 | 目前使用每模态第一个 client encoder 作为代理 |
| `clustering/kmeans_cluster.py` | Stage 1 聚类 | KMeans 聚类与聚类指标评估 | `run_kmeans`, `evaluate_clustering` | client representations、GT cluster | pred cluster、mapping、CM、acc/NMI/ARI | 是 | 否 | `ISODATAClusterer` 为 stub |
| `models/modules.py` | 模型模块 | encoder/projector/fusion/classifier 定义 | `ClientEncoder`, `SemanticProjector`, `ConcatFusion`, `ClassifierHead` | 特征向量 | 中间特征/分类 logit | 是 | 否 | `AttentionFusion` 为 stub |
| `losses/semantic_losses.py` | 损失 | SupCon 对齐损失、prototype 占位损失 | `SupervisedContrastiveLoss`, `PrototypeLoss` | `features`、`labels` | loss tensor | 是 | 否 | `PrototypeLoss` 目前恒 0 |
| `data/synthetic_dataset.py` | 数据准备 | 合成配对数据、训练测试切分、单模态 client 分片 | `make_synthetic_paired_dataset`, `build_client_pool` | cfg 参数 | `train/test`、client dict 列表 | 是 | 否 | 含 label-skew 与最小标签保障 |
| `data/uci_har_adapter.py` | 数据准备 | UCI-HAR 读取与双模态构造（acc/gyro） | `load_uci_har_dataset` | UCI-HAR 根目录 | train/test paired dict | 是 | 否 | 当前口径：`body_acc(3x128)` 与 `body_gyro(3x128)` |
| `data/real_dataset_adapter.py` | 数据准备 | 通用 npz 配对数据读取 | `load_real_paired_dataset` | root/train/test/full npz | train/test paired dict | 是 | 视场景 | 用于后续真实数据接入 |
| `utils/samplers.py` | 训练辅助 | 客户端类均衡采样 | `ClassBalancedBatchSampler` | client labels、batch_size | batch indices | 是 | 否 | 每个 client 本地采样 |
| `utils/config.py` | 工具 | yaml 配置读取 | `load_config` | 配置路径 | dict | 是 | 否 | 简单配置入口 |
| `utils/device.py` | 工具 | 设备选择 | `select_device` | `device` 字符串 | `torch.device` | 是 | 否 | 支持 auto/cpu/cuda/cuda:1 |
| `utils/seed.py` | 工具 | 随机种子设置 | `set_seed` | seed | 无 | 是 | 否 | 设置 random/numpy/torch |
| `experiments/run_multi_seed.py` | 实验辅助 | 多 seed 重复实验并聚合均值方差 | `main` | 配置、seed 列表 | multi-seed summary JSON | 否（主流程外） | 是 | 不影响单次主训练逻辑 |
| `trainers/d2d_offloading.py` | Stub | D2D offloading 预留接口 | `D2DOffloading.route` | 任意 | 抛出 `NotImplementedError` | 否 | 是 | v1 stub |

## 4) 当前最核心文件
- `clients/client_node.py`
- `clustering/kmeans_cluster.py`
- `server/server_core.py`
- `trainers/stage2_trainer.py`
- `trainers/schedulers.py`
- `server/evaluation.py`
- `experiments/run_stage2_training.py`

## 5) Stub 文件标记
- 已发现并确认：
  - `clustering/kmeans_cluster.py` 中 `ISODATAClusterer.fit_predict`（stub）
  - `trainers/schedulers.py` 中 `PairedFullModalityScheduler`（stub）
  - `trainers/schedulers.py` 中 `GlobalRandomScheduler`（stub）
  - `models/modules.py` 中 `AttentionFusion`（stub）
  - `trainers/d2d_offloading.py` 中 `D2DOffloading`（stub）
