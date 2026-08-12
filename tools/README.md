# Tools

这些脚本只负责启动实验，不保存参数默认值。正式参数默认值在：

```bash
src/MSL/data/dataset_defaults.py
src/MSL/utils/experiment_args.py
```

日志统一放在：

```bash
tools/logs/<dataset>/stage1.log
tools/logs/<dataset>/stage2.log
tools/logs/<dataset>/stage3.log
```

`--dataset uci_har/mhealth/pamap2/iemocap` 是代码里的数据集 key，不是磁盘数据目录名。

## UCI-HAR

Stage1：

```bash
bash tools/dataset/uci_har/stage1.sh
```

```bash
mkdir -p tools/logs/uci_har
nohup bash tools/dataset/uci_har/stage1.sh > tools/logs/uci_har/stage1.log 2>&1 &
```

Stage2：

```bash
bash tools/dataset/uci_har/stage2.sh
```

```bash
mkdir -p tools/logs/uci_har
nohup bash tools/dataset/uci_har/stage2.sh > tools/logs/uci_har/stage2.log 2>&1 &
```

Stage3：

```bash
for seed in 101 202 303 404 505; do
  python3 scripts/MSL/stage3_train.py --dataset uci_har --seed $seed
done
```

```bash
mkdir -p tools/logs/uci_har
nohup bash -c 'for seed in 101 202 303 404 505; do python3 scripts/MSL/stage3_train.py --dataset uci_har --seed $seed; done' > tools/logs/uci_har/stage3.log 2>&1 &
```

## MHEALTH

Stage1：

```bash
bash tools/dataset/mhealth/stage1.sh
```

```bash
mkdir -p tools/logs/mhealth
nohup bash tools/dataset/mhealth/stage1.sh > tools/logs/mhealth/stage1.log 2>&1 &
```

Stage2：

```bash
bash tools/dataset/mhealth/stage2.sh
```

```bash
mkdir -p tools/logs/mhealth
nohup bash tools/dataset/mhealth/stage2.sh > tools/logs/mhealth/stage2.log 2>&1 &
```

Stage3：

```bash
for fold in 1 2 3 4 5; do
  python3 scripts/MSL/stage3_train.py --dataset mhealth --fold $fold --seed 42
done
```

```bash
mkdir -p tools/logs/mhealth
nohup bash -c 'for fold in 1 2 3 4 5; do python3 scripts/MSL/stage3_train.py --dataset mhealth --fold $fold --seed 42; done' > tools/logs/mhealth/stage3.log 2>&1 &
```

## PAMAP2

Stage1：

```bash
bash tools/dataset/pamap2/stage1.sh
```

```bash
mkdir -p tools/logs/pamap2
nohup bash tools/dataset/pamap2/stage1.sh > tools/logs/pamap2/stage1.log 2>&1 &
```

Stage2：

```bash
bash tools/dataset/pamap2/stage2.sh
```

```bash
mkdir -p tools/logs/pamap2
nohup bash tools/dataset/pamap2/stage2.sh > tools/logs/pamap2/stage2.log 2>&1 &
```

Stage3：

```bash
for fold in 1 2 3 4 5 6 7 8 9; do
  python3 scripts/MSL/stage3_train.py --dataset pamap2 --fold $fold --seed 42
done
```

```bash
mkdir -p tools/logs/pamap2
nohup bash -c 'for fold in 1 2 3 4 5 6 7 8 9; do python3 scripts/MSL/stage3_train.py --dataset pamap2 --fold $fold --seed 42; done' > tools/logs/pamap2/stage3.log 2>&1 &
```

## IEMOCAP

Stage1：

```bash
bash tools/dataset/iemocap/stage1.sh
```

```bash
mkdir -p tools/logs/iemocap
nohup bash tools/dataset/iemocap/stage1.sh > tools/logs/iemocap/stage1.log 2>&1 &
```

Stage2：

```bash
bash tools/dataset/iemocap/stage2.sh
```

```bash
mkdir -p tools/logs/iemocap
nohup bash tools/dataset/iemocap/stage2.sh > tools/logs/iemocap/stage2.log 2>&1 &
```

Stage3：

```bash
for fold in 1 2 3 4 5; do
  python3 scripts/MSL/stage3_train.py --dataset iemocap --fold $fold --seed 42
done
```

```bash
mkdir -p tools/logs/iemocap
nohup bash -c 'for fold in 1 2 3 4 5; do python3 scripts/MSL/stage3_train.py --dataset iemocap --fold $fold --seed 42; done' > tools/logs/iemocap/stage3.log 2>&1 &
```

## 全部 Stage3

如果 Stage1/Stage2 都已经生成，可以直接跑所有数据集的 Stage3。默认并行数是 2：

```bash
bash tools/launch_msl.sh
```

```bash
mkdir -p tools/logs/all
nohup bash tools/launch_msl.sh > tools/logs/all/stage3_msl.log 2>&1 &
```

指定并行数：

```bash
MAX_JOBS=4 bash tools/launch_msl.sh
```

```bash
mkdir -p tools/logs/all
nohup env MAX_JOBS=4 bash tools/launch_msl.sh > tools/logs/all/stage3_msl_max4.log 2>&1 &
```

randomSL baseline：

```bash
bash tools/launch_random_sl.sh
```

```bash
mkdir -p tools/logs/all
nohup bash tools/launch_random_sl.sh > tools/logs/all/stage3_random_sl.log 2>&1 &
```

## 查看日志

```bash
tail -f tools/logs/uci_har/stage1.log
tail -f tools/logs/uci_har/stage2.log
tail -f tools/logs/uci_har/stage3.log
```

## 查看参数

运行前可以查看完整解析参数：

```bash
python3 scripts/MSL/stage3_train.py --dataset mhealth --fold 1 --print-config
```

## 输出位置

主线结果：

```bash
results/MSL/
```

baseline 结果：

```bash
results/baseline/randomSL/
```

每次运行会保存 `resolved_config.json`，用于记录本次实际使用的完整参数。
