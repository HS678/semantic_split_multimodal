# MSL

未知模态环境下语义对齐的分布式分割多模态学习。

本项目按从原始数据重新生成 pipeline artifacts 和正式实验结果的方式组织。`local/datasets/` 保存四个原始数据集；IEMOCAP 的 `model_cache/` 可作为预训练模型下载缓存保留。IEMOCAP frozen features、PAMAP2 window cache、client partitions、modality discovery artifacts 和正式实验结果均由当前代码生成。本项目文件夹按“可直接复现实验的完整工作环境”组织：源码、正式协议、数据目录、IEMOCAP 预处理特征缓存和运行入口在同一目录树下。复现时只使用本文档列出的当前路径。

## 快速复现

进入项目根目录并激活环境：

```bash
cd /path/to/semantic_split_multimodal
conda activate mpsl
```

确认 Python、CUDA 和依赖：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
python -m pytest tests -q
```

默认正式复现面向单张 GPU，例如单张 RTX 4090。不要同时启动多个 discovery 或 training process；所有正式 launcher 都按固定顺序串行运行，避免显存竞争、OOM 和 runtime 干扰。

完整复现按当前路径从头生成 IEMOCAP frozen feature cache、数据划分、模态聚类和实验结果：

```bash
conda activate mpsl

python -m compileall src pipeline experiments tools tests -q
python -m pytest tests -q

python pipeline/prepare_iemocap_features.py --device cuda

bash tools/data/launch_prepare_clients_all.sh

bash tools/data/launch_discover_modalities_all.sh

python experiments/run_all.py \
  --results-root results \
  --device cuda \
  --require-cuda
```

后台完整复现命令：

```bash
mkdir -p logs && nohup bash -lc 'python -m compileall src pipeline experiments tools tests -q && python -m pytest tests -q && python pipeline/prepare_iemocap_features.py --device cuda && bash tools/data/launch_prepare_clients_all.sh && bash tools/data/launch_discover_modalities_all.sh && python experiments/run_all.py --results-root results --device cuda --require-cuda' > logs/full_reproduce.log 2>&1 &
```

查看进度：

```bash
tail -f logs/full_reproduce.log
```

## 目录结构

```text
.
├── README.zh-CN.md
├── requirements.txt
├── pyproject.toml
├── src/MSL/
├── pipeline/
├── experiments/
├── tools/
├── tests/
├── local/datasets/        # 本地数据和预处理缓存，不提交到 Git
└── results/               # 运行生成结果，不提交到 Git
```

核心代码在 `src/MSL/`：

```text
src/MSL/
├── protocol.py
├── data.py
├── datasets/
│   ├── uci_har.py
│   ├── mhealth.py
│   ├── pamap2.py
│   └── iemocap.py
├── discovery.py
├── models.py
├── pretrain.py
├── scheduling.py
├── training.py
├── binding.py
├── evaluation.py
└── utils.py
```

## 环境

推荐配置好 `mpsl` conda 环境后，开始实验。从源码安装依赖：

```bash
pip install -r requirements.txt
```

CUDA 复现实验建议显式使用：

```bash
--device cuda --require-cuda
```

如果 `torch.cuda.is_available()` 为 `False`，先不要跑正式实验，检查当前 shell 是否能访问 GPU 驱动和 CUDA 版 PyTorch。

单 GPU 正式复现约束：

- `pipeline/prepare_iemocap_features.py --device cuda` 必须在 client preparation 前单独执行。
- `tools/data/launch_prepare_clients_all.sh` 串行执行 UCI-HAR、MHEALTH、PAMAP2、IEMOCAP。
- `tools/data/launch_discover_modalities_all.sh` 串行执行 UCI-HAR、MHEALTH、PAMAP2、IEMOCAP。
- `experiments/run_all.py`、`experiments/run_all_discovery.py`、`experiments/msl/run_all.py` 按 formal grid 串行执行。
- 不建议同时启动多个 discovery/training 进程。

## 数据

默认数据位置：

```text
local/datasets/
├── UCI-HAR/
├── MHEALTH/
├── PAMAP2/
└── IEMOCAP/
    ├── IEMOCAP_full/IEMOCAP_full_release/
    ├── processed/mfcc_mobilevit_xs_distilbert_v1/
    └── model_cache/
```

IEMOCAP 的正式 loader 不直接从视频/音频/文本原始文件训练，而是读取 frozen processed cache：

```text
local/datasets/IEMOCAP/processed/mfcc_mobilevit_xs_distilbert_v1/
├── manifest.json
├── audio.pt
├── video.pt
├── text.pt
└── metadata.json
```

生成该 cache 的入口是：

```bash
python pipeline/prepare_iemocap_features.py --device cuda
```

## 正式协议

正式实验参数唯一来源是 `src/MSL/protocol.py`。CLI 参数只用于临时 override；`protocol_manifest.json` 是运行后写出的协议快照，不作为输入配置。

冻结协议为 `fixed_total_clients_per_round&modality_coverage`：

- UCI-HAR / MHEALTH: `global_rounds=200`
- IEMOCAP / PAMAP2: `global_rounds=300`
- discovery 方法：`adaptive_isodata`, `kmeans2`, `kmeans3`, `kmeans4`, `kmeans5`
- training 方法：`ours`, `randomsl`, `kmeans2`, `kmeans3`, `kmeans4`, `kmeans5`, `oracle`
- RQ2 client budget 使用固定每轮总客户端数 `clients_per_round`，不使用每个 cluster 固定客户端数：
  - UCI-HAR: `clients_per_round=4`
  - MHEALTH: `clients_per_round=8`
  - PAMAP2: `clients_per_round=6`
  - IEMOCAP: `clients_per_round=6`
- `ours`, `kmeans2`, `kmeans3`, `kmeans4`, `kmeans5`, `oracle` 将 `clients_per_round` 均匀轮转分配到当前方法的 clusters；当 `clients_per_round < cluster_num` 时，允许部分 cluster 本轮调度 0 个客户端。
- `randomsl` 使用相同 `clients_per_round` 总预算，但保持全局随机选择客户端。

正式 run grid：

- UCI-HAR: 5 个 seed，fold 为 `fold_00`
- MHEALTH: 5 folds，seed 固定 `42`
- PAMAP2: 8 folds，seed 固定 `42`
- IEMOCAP: 5 folds，seed 固定 `42`

训练、调度、binding、fusion slot 构造只使用 `pred_cluster` 与 label。`hidden_modality_id` / 真实模态名不参与训练决策，仅用于 discovery 审计、RQ2 每轮真实模态覆盖率统计和 evaluation-only tolerant routing。

RQ2 每轮 `train_log.csv` 会记录：

```text
selected_client_ids
selected_client_ids_by_cluster_json
per_cluster_budget_json
per_cluster_selected_json
selected_hidden_modality_ids_json
per_modality_selected_json
modality_coverage
full_modality_coverage
```

`modality_coverage` 按真实 `hidden_modality_id` 计算；`full_modality_coverage=1` 表示该轮被调度客户端覆盖了全部真实模态。最终 `final_metrics.json` 和 `result.json` 会汇总 `modality_full_coverage_rate`。

## 复现步骤

1. 检查环境：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m compileall src pipeline experiments tools tests -q
python -m pytest tests -q
```

2. 准备 IEMOCAP processed cache：

```bash
python pipeline/prepare_iemocap_features.py --device cuda
```

该步骤必须在 client preparation 前单独执行，不要和其它 GPU-heavy 任务并发运行。

3. 生成单模态 client partition：

```bash
bash tools/data/launch_prepare_clients_all.sh
```

该 launcher 严格串行执行四个数据集，顺序固定为 UCI-HAR、MHEALTH、PAMAP2、IEMOCAP；每个数据集写入独立日志，任一失败立即停止。

单数据集入口：

```bash
bash tools/dataset/uci_har/prepare_clients.sh
bash tools/dataset/mhealth/prepare_clients.sh
bash tools/dataset/pamap2/prepare_clients.sh
bash tools/dataset/iemocap/prepare_clients.sh
```

输出位置：

```text
results/pipeline/clients/<dataset>/<partition_signature>/
```

每个有效目录至少包含：

```text
train_clients/
test_multimodal.pt
```

4. 发现模态簇并保存 pretrained encoders / fingerprints：

```bash
bash tools/data/launch_discover_modalities_all.sh
```

该 launcher 严格串行执行四个数据集，顺序固定为 UCI-HAR、MHEALTH、PAMAP2、IEMOCAP；每个数据集写入独立日志，任一失败立即停止。不要同时启动多个 discovery 进程。

单数据集入口：

```bash
bash tools/dataset/uci_har/discover_modalities.sh
bash tools/dataset/mhealth/discover_modalities.sh
bash tools/dataset/pamap2/discover_modalities.sh
bash tools/dataset/iemocap/discover_modalities.sh
```

输出位置：

```text
results/pipeline/discovery/<dataset>/<partition_signature>/adaptive_isodata/
```

每个有效目录至少包含：

```text
pred_cluster.csv
pretrained_encoders/
visualization/fingerprints.npz
```

5. 运行 RQ1 discovery comparison：

```bash
python experiments/run_all_discovery.py --results-root results
```

6. 运行 RQ2 全部训练方法：

```bash
python experiments/msl/run_all.py --results-root results --device cuda --require-cuda
```

7. 或者一键运行 RQ1 + RQ2：

```bash
python experiments/run_all.py --results-root results --device cuda --require-cuda
```

training formal grid 由 Python runner 串行执行。不要同时启动多个 training 进程。

## 单次调试

单次 RQ1：

```bash
python experiments/discovery_comparison.py --dataset pamap2 --fold 1 --seed 42 --method adaptive_isodata
```

单次 RQ2 Ours：

```bash
python experiments/msl/train.py --dataset pamap2 --fold 1 --seed 42 --device cuda --require-cuda
```

单次 baseline：

```bash
python experiments/training.py --dataset pamap2 --fold 1 --seed 42 --method randomsl --device cuda --require-cuda
python experiments/training.py --dataset pamap2 --fold 1 --seed 42 --method kmeans4 --device cuda --require-cuda
python experiments/training.py --dataset pamap2 --fold 1 --seed 42 --method oracle --device cuda --require-cuda
```

## 结果

正式结果写入：

```text
results/
├── protocol_manifest.json
├── pipeline/
│   ├── clients/
│   └── discovery/
├── discovery/
│   └── aggregated/
├── msl/
│   └── aggregated/
├── baselines/
│   └── aggregated/
└── aggregated/
```

其中：

- `results/pipeline/clients/`：单模态 client partition；
- `results/pipeline/discovery/`：预训练 encoder、fingerprint 和模态发现结果；
- `results/discovery/`：RQ1 discovery comparison 原始结果和聚合结果；
- `results/msl/`：Ours 训练原始结果、曲线、checkpoint 和聚合结果；
- `results/baselines/`：RandomSL、KMeans-SL、Oracle-SL 训练原始结果、曲线、checkpoint 和聚合结果；
- `results/aggregated/`：跨方法聚合表。

实验入口只读取当前 `results/pipeline/` 生成的 pipeline artifacts。

## 职责边界

- `src/MSL/protocol.py`：正式实验协议唯一参数来源，只含协议常量和纯查询函数。
- `src/MSL/datasets/`：各数据集专属读取、预处理、split、windowing、normalization。
- `pipeline/prepare_iemocap_features.py`：IEMOCAP 原始音频/视频/文本到 frozen feature cache 的离线准备。
- `pipeline/prepare_clients.py`：从 dataset loader 输出生成单模态 client partition。
- `pipeline/discover_modalities.py`：从 prepared clients 生成 pretrained encoders、fingerprints 和 adaptive discovery artifacts。
- `experiments/discovery_comparison.py`：RQ1 单次 discovery comparison。
- `experiments/run_all_discovery.py`：RQ1 全部正式实验。
- `experiments/training.py`：RQ2 共享 runner，负责 method policy、topology artifact、结果目录、resume hash 和 CLI。
- `experiments/msl/train.py`：Ours 单次训练入口。
- `experiments/msl/run_all.py`：RQ2 全部训练方法入口。
- `src/MSL/training.py`：实际 split learning 训练循环、loss、binding 调用、server/client 更新和评估输出。

## Baseline

内部 baseline 放在 `experiments/baselines/`，共享 `experiments/training.py` 的实验层 runner；算法 trainer / binding / server / evaluator 仍在 `src/MSL/training.py` 等核心模块中。

KMeans-SL 统一由 `experiments/baselines/kmeans_sl.py --k 2/3/4/5` 和 `experiments/training.py --method kmeansK` 支持。

外部论文 baseline 预留在 `experiments/baselines/external/`，不强行塞入共享 trainer。
