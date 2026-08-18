# MSL

未知模态环境下语义对齐的分布式分割多模态学习。

## 目录

```text
protocol / pipeline / experiments / src / tools / tests
```

核心代码在 `src/MSL/`：

```text
src/MSL/
├── __init__.py
├── protocol.py
├── data.py
├── iemocap.py
├── discovery.py
├── models.py
├── pretrain.py
├── scheduling.py
├── training.py
├── binding.py
├── evaluation.py
└── utils.py
```

## 协议

正式实验参数唯一来源是 `src/MSL/protocol.py`。CLI 参数只用于临时 override；`protocol_manifest.json` 是运行后写出的协议快照，不作为输入配置。

冻结协议为 `iemocap300&kmeans4`：

- UCI-HAR / MHEALTH: `global_rounds=200`
- IEMOCAP / PAMAP2: `global_rounds=300`
- discovery 方法：`adaptive_isodata`, `kmeans2`, `kmeans3`, `kmeans4`, `kmeans5`
- training 方法：`ours`, `randomsl`, `kmeans2`, `kmeans3`, `kmeans4`, `kmeans5`, `oracle`
- 不做 membership canonicalization，保留原始 cluster-ID 行为
- feasibility repair 保留，仅在 cluster size `< r` 时保证 cluster-aware scheduler 可执行

训练、调度、binding、fusion slot 构造只使用 `pred_cluster` 与 label。`hidden_modality_id` / 真实模态名只用于 discovery 审计和 evaluation-only tolerant routing。

## 数据准备

数据放在 `local/datasets/`：

- UCI-HAR: `local/datasets/UCI-HAR/`
- MHEALTH: `local/datasets/MHEALTH/`
- PAMAP2: `local/datasets/PAMAP2/`
- IEMOCAP: `local/datasets/IEMOCAP/IEMOCAP_full/IEMOCAP_full_release/`

IEMOCAP 冻结特征准备：

```bash
PYTHONPATH=src python tools/prepare_iemocap.py --device cuda
```

生成单模态客户端：

```bash
bash tools/dataset/uci_har/prepare_clients.sh
bash tools/dataset/mhealth/prepare_clients.sh
bash tools/dataset/pamap2/prepare_clients.sh
bash tools/dataset/iemocap/prepare_clients.sh
```

发现模态簇：

```bash
bash tools/dataset/uci_har/discover_modalities.sh
bash tools/dataset/mhealth/discover_modalities.sh
bash tools/dataset/pamap2/discover_modalities.sh
bash tools/dataset/iemocap/discover_modalities.sh
```

## 实验

Discovery comparison：

```bash
python experiments/discovery_comparison.py --dataset pamap2 --fold 1 --seed 42 --method adaptive_isodata
python experiments/run_all_discovery.py --results-root results
```

MSL 主方法：

```bash
python experiments/msl/train.py --dataset pamap2 --fold 1 --seed 42 --method ours
python experiments/msl/run_all.py --methods ours --results-root results --device auto --require-cuda
```

Baselines：

```bash
python experiments/msl/run_all.py --methods randomsl kmeans2 kmeans3 kmeans4 kmeans5 oracle --results-root results --device auto --require-cuda
```

一键运行 discovery comparison + 全部 training 方法：

```bash
python experiments/run_all.py --results-root results --device auto --require-cuda
```

## 结果

```text
results/
├── protocol_manifest.json
├── discovery/
├── msl/
└── baselines/
```

公共 client partition 和 modality discovery 产物仍写入 `results/MSL/partition` 与 `results/MSL/cluster`，作为 pipeline 产物供 experiments 复用。

## Baseline 接入

内部 baseline 放在 `experiments/baselines/`，优先共享 `experiments/msl/train.py` 中的 trainer / binding / server / evaluator，仅切换 method policy。

KMeans-SL 统一由 `experiments/baselines/kmeans_sl.py` 和 `experiments/msl/train.py --method kmeansK` 支持，`K=2/3/4/5`。

外部论文 baseline 放在 `experiments/baselines/external/`，不强行塞入共享 trainer。

## 验证

```bash
python -m compileall src pipeline experiments tools tests -q
python -m pytest tests -q
```
