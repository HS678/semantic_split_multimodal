# MSL

未知模态环境下语义对齐的分布式分割多模态学习（Semantic-aligned Distributed Split Multimodal Learning）。

英文说明见 [README.md](README.md)。

## 总览

三阶段可复现实验框架：

1. **Stage 1（数据划分）** — `scripts/MSL/stage1_partition.py` 把自然配对的多模态数据拆成单模态客户端。只有 train 被划分；test 保持自然配对（`test_multimodal.pt`）。
2. **Stage 2（模态发现）** — `scripts/MSL/stage2_discovery.py` 为每个客户端预训练 encoder、提取 fingerprint、用 adaptive ISODATA 聚类，输出 `pred_cluster.csv`（另存 `true_cluster.csv` 供审计）。
3. **RQ2 / Stage 3（训练）** — `experiments/run_rq2_training.py` 使用共享 Split Learning trainer，通过 policy 切换 Ours、RandomSL、KMeans-2/3/5-SL 和 Oracle-SL。无验证集：`global_rounds` 结束后直接用 `last_model.pt` 在 `test_multimodal.pt` 上评估一次。

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
- RandomSL 每轮从所有客户端中完全随机选择 `r * Q_hat` 个不同客户端；`pred_cluster` 只在选择后用于 slot organization 和 coverage 统计，不用于补齐或重抽。
- 测试使用 tolerant evaluation routing：仅在 evaluation layer 读取 `hidden_modality_id` 和 `pred_cluster`，构造 `N_mq/P_mq`，对每个 `(true modality, pred cluster)` 的训练后 client encoders 做参数平均，并支持 correct、split、merge、split+merge。

## 数据集

支持四个数据集，划分内置在 `src/MSL/data/datasets.py` / `src/MSL/data/iemocap.py`：

| 数据集 | 划分协议 | 说明 |
| --- | --- | --- |
| UCI-HAR | `subject_disjoint_70_30` | 官方 train/test；正式 seeds 为 `42,123,2025,3407,7777` |
| MHEALTH | `subject_5fold_foldN` | subject 级 5 折；正式 seeds 为 `42,123,2025,3407,7777` |
| PAMAP2 | `subject_9fold_loso_foldN` | subject 级 9 折 LOSO；12 类活动，不含心率；正式 seeds 为 `42,123,2025,3407,7777` |
| IEMOCAP | `session_5fold_loso_foldN` | 5 折 session-LOSO，audio/video/text；正式 seeds 为 `42,123,2025,3407,7777` |

每个数据集的固定参数（num_classes、root、encoder、预训练/训练 lr、mmbind 权重、聚类默认参数）内置在 `src/MSL/data/dataset_defaults.py`，不重复写入配置文件。

下载公开数据集并放到 `local/datasets/`：

- **UCI-HAR**：官方 `UCI HAR Dataset`（把 `train/`、`test/` 目录放到 `local/datasets/UCI-HAR/`）。
- **MHEALTH**：UCI 的 `MHEALTHDATASET.zip`，解压为 `local/datasets/MHEALTH/`（内含 `MHEALTHDATASET/`）。
- **PAMAP2**：UCI 的 `PAMAP2_Dataset.zip`，解压为 `local/datasets/PAMAP2/`（`PAMAP2_Dataset/Protocol/subject10*.dat`）。
- **IEMOCAP**：完整版放在 `local/datasets/IEMOCAP/IEMOCAP_full/IEMOCAP_full_release/`（需 CMU IEMOCAP 许可）。

IEMOCAP 冻结特征需要在 Stage 1 前准备一次：

```bash
PYTHONPATH=src python -m MSL.data.prepare_iemocap --device cuda
```

## 配置

每个脚本都可以直接通过 `--dataset` 加载数据集默认参数。用 `--print-config` 可以查看某个数据集的完整 resolved 参数：

```bash
python scripts/MSL/stage3_train.py --dataset mhealth --fold 1 --print-config
python scripts/baseline/randomSL/stage3_train.py --dataset pamap2 --fold 2 --print-config
```

命令行参数会覆盖数据集默认参数：

```bash
python scripts/MSL/stage3_train.py --dataset uci_har --seed 101 --global-rounds 50 --client-lr 0.0001
```

数据集默认参数集中在 `src/MSL/data/dataset_defaults.py`，正式命令行 parser 在 `src/MSL/utils/experiment_args.py`。多折数据集通过 `--fold N` 从数据集默认模板生成划分协议，命令行参数覆盖默认值。

运行前可以用 `--print-config` 查看完整解析后的参数：

```bash
python scripts/MSL/stage3_train.py --dataset mhealth --fold 1 --print-config
```

所有输出路径由 `base_dir` + 数据集 + 划分协议自动生成。主线默认写入 `results/MSL/`，baseline 默认写入 `results/baseline/randomSL/`。每次运行会在结果目录保存 `resolved_config.json`。

## 运行

先生成可复用的 Stage1 和 Stage2 产物。Stage2 必须在对应 Stage1 目录存在后才能运行。

```bash
bash tools/dataset/uci_har/stage1.sh
bash tools/dataset/uci_har/stage2.sh
```

多折数据集的脚本里写了显式 fold 循环：

```bash
bash tools/dataset/mhealth/stage1.sh
bash tools/dataset/mhealth/stage2.sh
bash tools/dataset/pamap2/stage1.sh
bash tools/dataset/pamap2/stage2.sh
bash tools/dataset/iemocap/stage1.sh
bash tools/dataset/iemocap/stage2.sh
```

Stage1 和 Stage2 都存在后，可以运行 RQ1 discovery：

```bash
python experiments/run_rq1_discovery.py \
  --dataset pamap2 \
  --fold 1 \
  --seed 42 \
  --method adaptive_isodata

python experiments/run_rq1_discovery.py --dataset pamap2 --fold 1 --seed 42 --method kmeans2
python experiments/run_all_rq1.py
```

RQ1 支持 `adaptive_isodata`、`kmeans2`、`kmeans3`、`kmeans5`。KMeans 方法复用同一套 Stage2 pretrained encoders 和 fingerprints，不重新预训练。

RQ2 单方法运行：

```bash
python experiments/run_rq2_training.py \
  --dataset pamap2 \
  --fold 1 \
  --seed 42 \
  --method ours
```

RQ2 支持 `ours`、`randomsl`、`kmeans2`、`kmeans3`、`kmeans5`、`oracle`。全部方法共享 trainer、binding、server、loss 和 evaluator；只切换 discovery topology、scheduler policy 和 slot number。

完整 RQ2：

```bash
python experiments/run_all_rq2.py
```

旧 Stage3 入口仍可用于兼容运行：

```bash
python scripts/MSL/stage3_train.py --dataset uci_har --seed 101
```

Stage3 启动脚本只复用已有 Stage1/Stage2 产物并写入 Stage3 结果：

```bash
bash tools/launch_msl.sh
bash tools/launch_random_sl.sh
```

Stage3 启动脚本默认 `MAX_JOBS=2` 并行运行。资源不够时用 `MAX_JOBS=1 bash tools/launch_msl.sh` 串行；资源允许时可以调大 `MAX_JOBS`。如果缺少所需 Stage1/Stage2 产物，脚本会直接失败。

## 结果目录

```text
results/
├── rq1/
│   ├── raw/
│   ├── aggregated/
│   └── artifacts/
└── rq2/
    ├── raw/
    ├── aggregated/
    ├── checkpoints/
    └── topologies/
```

Stage1/Stage2 公共产物仍保留在 `results/MSL/` 或既有的 `local/results_msl/` 下。新 `experiments/` 入口会优先查找真实 Stage1/Stage2 产物，找不到会明确报错，不生成 mock 数据。

Stage 输出不覆盖已有非空目录；重复同一 fold/seed 会自动生成下一个 `attempt-<nn>`。

## 汇总格式

```bash
python scripts/MSL/summarize_results.py --results-root results/MSL
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
conda run -n mpsl python -m pytest tests -q
```

最小真实 smoke 示例：

```bash
conda run -n mpsl python experiments/run_rq1_discovery.py --dataset mhealth --fold 1 --seed 42 --method kmeans2 --results-root results_smoke
conda run -n mpsl python experiments/run_rq2_training.py --dataset mhealth --fold 1 --seed 42 --method randomsl --global-rounds 1 --device cpu --results-root results_smoke
```
