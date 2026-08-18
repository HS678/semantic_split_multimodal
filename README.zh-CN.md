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

## 协议

正式实验参数唯一来源是 `src/MSL/protocol.py`。CLI 参数只用于临时 override；`protocol_manifest.json` 是运行后写出的协议快照，不作为输入配置。

冻结协议为 `iemocap300&kmeans4`：

- UCI-HAR / MHEALTH: `global_rounds=200`
- IEMOCAP / PAMAP2: `global_rounds=300`
- discovery 方法：`adaptive_isodata`, `kmeans2`, `kmeans3`, `kmeans4`, `kmeans5`
- training 方法：`ours`, `randomsl`, `kmeans2`, `kmeans3`, `kmeans4`, `kmeans5`, `oracle`

训练、调度、binding、fusion slot 构造只使用 `pred_cluster` 与 label。`hidden_modality_id` / 真实模态名只用于 discovery 审计和 evaluation-only tolerant routing。

## 职责边界

- `src/MSL/protocol.py`：正式实验协议唯一参数来源，只含协议常量和纯查询函数。
- `src/MSL/datasets/`：各数据集专属读取、预处理、split、windowing、normalization。
- `src/MSL/data.py`：统一 dataset dispatcher、client partition、artifact save/load 与公共校验。
- `pipeline/`：生成 client/discovery artifacts。
- `experiments/training.py`：实验层共享 runner，负责 method policy、topology artifact、结果目录、resume hash 和 CLI。
- `src/MSL/training.py`：算法 trainer，负责实际 split learning 训练循环、loss、binding 调用、server/client 更新和评估输出。

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
python experiments/msl/train.py --dataset pamap2 --fold 1 --seed 42
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


Pipeline artifact 和正式实验结果统一使用当前 `results/` 目录结构：

```text
results/
├── protocol_manifest.json
├── pipeline/
│   ├── clients/
│   └── discovery/
├── discovery/
├── msl/
└── baselines/
```

其中：

* `results/pipeline/clients/`：保存单模态 client partition；
* `results/pipeline/discovery/`：保存预训练 encoder、fingerprint 和模态发现结果；
* `results/discovery/`：保存 Adaptive ISODATA 与 KMeans 的 discovery comparison 结果；
* `results/msl/`：保存本文方法 Ours 的训练结果；
* `results/baselines/`：保存 RandomSL、KMeans-SL 和 Oracle-SL 的训练结果。

实验只读取当前 `results/pipeline/` 生成的 pipeline artifacts，不读取其他历史目录。

## Baseline 接入

内部 baseline 放在 `experiments/baselines/`，共享 `experiments/training.py` 的实验层 runner；算法 trainer / binding / server / evaluator 仍在 `src/MSL/training.py` 等核心模块中。

KMeans-SL 统一由 `experiments/baselines/kmeans_sl.py --k 2/3/4/5` 和 `experiments/training.py --method kmeansK` 支持。

外部论文 baseline 放在 `experiments/baselines/external/`，不强行塞入共享 trainer。

## 验证

```bash
python -m compileall src pipeline experiments tools tests -q
python -m pytest tests -q
```
