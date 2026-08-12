# Tools

这些脚本只负责启动实验，不保存参数默认值。正式参数默认值在：

```bash
src/MSL/data/dataset_defaults.py
src/MSL/utils/experiment_args.py
```

## 1. 先生成 Stage1 和 Stage2

Stage1/Stage2 是可复用产物。Stage2 必须在对应 Stage1 已存在后运行。

UCI-HAR：

```bash
bash tools/dataset/uci_har/stage1.sh
bash tools/dataset/uci_har/stage2.sh
```

MHEALTH：

```bash
bash tools/dataset/mhealth/stage1.sh
bash tools/dataset/mhealth/stage2.sh
```

PAMAP2：

```bash
bash tools/dataset/pamap2/stage1.sh
bash tools/dataset/pamap2/stage2.sh
```

IEMOCAP：

```bash
bash tools/dataset/iemocap/stage1.sh
bash tools/dataset/iemocap/stage2.sh
```

## 2. 再运行 Stage3

主线 MMBind-style Fusion Split Learning：

```bash
bash tools/launch_msl.sh
```

randomSL baseline：

```bash
bash tools/launch_random_sl.sh
```

默认并行数是 2。可以用 `MAX_JOBS` 调整：

```bash
MAX_JOBS=4 bash tools/launch_msl.sh
MAX_JOBS=1 bash tools/launch_random_sl.sh
```

`MAX_JOBS=1` 就是串行运行，适合显存紧张时使用。

## 3. 单独跑一个任务

也可以直接调用 Python 脚本：

```bash
python3 scripts/MSL/stage3_train.py --dataset uci_har --seed 101
python3 scripts/MSL/stage3_train.py --dataset mhealth --fold 1 --seed 42
python3 scripts/baseline/randomSL/stage3_train.py --dataset pamap2 --fold 1 --seed 42
```

运行前查看完整解析参数：

```bash
python3 scripts/MSL/stage3_train.py --dataset mhealth --fold 1 --print-config
```

## 4. 输出位置

主线输出：

```bash
results/MSL/
```

baseline 输出：

```bash
results/baseline/randomSL/
```

每次运行会保存 `resolved_config.json`，用于记录本次实际使用的完整参数。
