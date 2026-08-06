# MSL

未知模态环境下语义对齐的分布式分割多模态学习（Semantic-aligned Distributed Split Multimodal Learning）。

英文说明见 [README.md](README.md)。完整设计决策见 [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md)。

## 总览

三阶段可复现实验框架：

1. **Stage 1（数据划分）** — `scripts/stage1_partition.py` 把自然配对的多模态数据拆成单模态客户端。只有 train 被划分；test 保持自然配对（`test_multimodal.pt`）。
2. **Stage 2（模态发现）** — `scripts/stage2_discovery.py` 为每个客户端预训练 encoder、提取 fingerprint、用 adaptive ISODATA 聚类，输出 `pred_cluster.csv`（另存 `true_cluster.csv` 供审计）。
3. **Stage 3（训练）** — `scripts/stage3_train.py` 执行聚类感知调度、label-guided semantic pseudo binding、`ClusterAdapter` + concat fusion、Split Learning，固定轮数训练。无验证集：`global_rounds` 结束后直接用 `last_model.pt` 在 `test_multimodal.pt` 上评估一次。

本项目不是联邦学习，不使用 FedAvg。客户端上传 detached 激活；服务器计算 loss、反向传播融合模型，并把激活梯度路由回客户端 encoder。

## 环境安装

需要 Python 3.10+ 与 PyTorch。创建环境并安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

使用 CUDA 时先安装与显卡匹配的 `torch` / `torchvision` / `torchaudio`，再安装其余依赖。

## 协议

- 训练、调度、binding、fusion slot 构造只使用 `pred_cluster` 与 label。`hidden_modality_id` / 真实模态名只用于 Stage 2 审计和 evaluation-only oracle mapping。
- 无验证集：固定 `global_rounds=200`，无 early stopping、无 best checkpoint 选择。
- 正式指标：`acc` / `macro_f1` / `weighted_f1`。
- 配置项 `training.cluster_assignment_source=true_cluster` 用于 oracle 上限对比；正式无泄漏主线为 `pred_cluster`。
- D2D 尚未实现，`d2d.enabled=false` 仅保留扩展口。

## 数据集

支持四个数据集，划分内置在 `src/MSL/data/datasets.py` / `src/MSL/data/iemocap.py`：

| 数据集 | 划分协议 | 说明 |
| --- | --- | --- |
| UCI-HAR | `subject_disjoint_70_30` | 官方 70/30 固定划分；5 个 seed |
| MHEALTH | `subject_5fold_foldN` | subject 级 5 折；1 个 seed |
| PAMAP2 | `subject_9fold_loso_foldN` | 9 折 LOSO，12 类活动，不含心率；1 个 seed |
| IEMOCAP | `session_5fold_loso_foldN` | 5 折 session-LOSO，audio/video/text；1 个 seed |

每个数据集的固定参数（num_classes、root、encoder、预训练/训练 lr、mmbind 权重、聚类默认参数）内置在 `src/MSL/data/dataset_defaults.py`，不重复写入配置文件。

下载公开数据集并放到 `local/datasets/`：

- **UCI-HAR**：官方 `UCI HAR Dataset`（把 `train/`、`test/` 目录放到 `local/datasets/uci_har/`）。
- **MHEALTH**：UCI 的 `MHEALTHDATASET.zip`，解压为 `local/datasets/mhealth/`（内含 `MHEALTHDATASET/`）。
- **PAMAP2**：UCI 的 `PAMAP2_Dataset.zip`，解压为 `local/datasets/pamap2/`（`PAMAP2_Dataset/Protocol/subject10*.dat`）。
- **IEMOCAP**：完整版放在 `local/datasets/IEMOCAP/IEMOCAP_full/IEMOCAP_full_release/`（需 CMU IEMOCAP 许可）。

IEMOCAP 冻结特征需要在 Stage 1 前准备一次：

```bash
PYTHONPATH=src python -m MSL.data.prepare_iemocap --device cuda
```

## 配置

`configs/` 下每个数据集/折一份独立完整配置（约 14 行），统一分段结构：

```ini
[config]      # experiment_name、base_dir（seed/device 已内置；--seed 覆盖）
[partition]   # type、split_protocol、clients_per_modality
[train]       # cluster_assignment_source、scheduler、fusion_training_objective、global_rounds
# cluster / d2d / other 段均有内置默认，仅在需要覆盖时写出。
```

所有输出路径由 `base_dir` + 数据集 + 划分协议自动生成。`configs/config.config` 是字段参考模板。

## 运行

单阶段运行：

```bash
python scripts/stage1_partition.py --config configs/uci_har.config
python scripts/stage2_discovery.py --config configs/uci_har.config
python scripts/stage3_train.py --config configs/uci_har.config --seed 101
```

一键启动脚本（每个数据集完整执行 Stage1 → Stage2 → Stage3 → 汇总）：

```bash
nohup bash tools/single/launch_msl_uci_har.sh > "tools/single/uci_har_msl_$(date '+%Y%m%d_%H%M%S').log" 2>&1 &
nohup bash tools/serial/launch_msl_all.sh > "tools/serial/msl_all_$(date '+%Y%m%d_%H%M%S').log" 2>&1 &
nohup bash tools/parallel/launch_msl_parallel.sh > "tools/parallel/main_$(date '+%Y%m%d_%H%M%S').log" 2>&1 &
```

启动脚本对已存在的输出目录自动跳过（可断点续跑），日志写入 `local/results_msl/logs/`。

## 结果目录

```text
local/results_msl/
├── partition/<dataset>/<partition_signature>/   # Stage 1（train_clients/、test_multimodal.pt 等）
├── cluster/<dataset>/<partition_signature>/adaptive_isodata/   # Stage 2（含 visualization/）
├── experiments/<scope>/<dataset>/<loss>/attempt-<nn>/seed-<ss>/   # Stage 3 运行
│   └── summary.json
└── summary/<dataset>.json                      # 数据集级汇总
```

Stage 输出不覆盖已有非空目录；重复同一配置/seed 需要递增 `stage3.attempt`。

## 汇总格式

```bash
python scripts/summarize_results.py --results-root local/results_msl
```

```json
{
  "fold1": {"acc": 0.85, "macro_f1": 0.81, "weighted_f1": 0.84},
  "fold2": {"acc": 0.87, "macro_f1": 0.82, "weighted_f1": 0.85},
  "average": {"acc": 0.86, "macro_f1": 0.815, "weighted_f1": 0.845}
}
```

## 测试

```bash
PYTHONPATH=src python -m pytest tests -q
```
