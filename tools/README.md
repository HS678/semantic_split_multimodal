# Tools

这些脚本只负责启动实验，默认参数在代码里：

```bash
src/MSL/data/dataset_defaults.py
src/MSL/utils/experiment_args.py
```

运行顺序必须是：

```bash
Stage1 partition -> Stage2 discovery -> Stage3 training
```

日志统一放在：

```bash
tools/logs/<dataset>/stage1.log
tools/logs/<dataset>/stage2.log
tools/logs/<dataset>/stage3.log
tools/logs/all/*.log
```

## 单数据集启动

### UCI-HAR

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

### MHEALTH

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

### PAMAP2

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
for fold in 1 2 3 4 5 6 7 8; do
  python3 scripts/MSL/stage3_train.py --dataset pamap2 --fold $fold --seed 42
done
```

```bash
mkdir -p tools/logs/pamap2
nohup bash -c 'for fold in 1 2 3 4 5 6 7 8; do python3 scripts/MSL/stage3_train.py --dataset pamap2 --fold $fold --seed 42; done' > tools/logs/pamap2/stage3.log 2>&1 &
```

### IEMOCAP

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

## 一键启动

Stage1 一键并行启动四个数据集：

```bash
bash tools/data/launch_stage1_all.sh
```

```bash
mkdir -p tools/logs/all
nohup bash tools/data/launch_stage1_all.sh > tools/logs/all/stage1_all.log 2>&1 &
```

Stage2 一键并行启动四个数据集：

```bash
bash tools/data/launch_stage2_all.sh
```

```bash
mkdir -p tools/logs/all
nohup bash tools/data/launch_stage2_all.sh > tools/logs/all/stage2_all.log 2>&1 &
```

Stage3 一键启动全部 MSL 主线实验，默认并行数是 2：

```bash
bash tools/launch_msl.sh
```

```bash
mkdir -p tools/logs/all
nohup bash tools/launch_msl.sh > tools/logs/all/stage3_msl.log 2>&1 &
```

Stage3 指定并行数：

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
tail -f tools/logs/all/stage1_all.log
tail -f tools/logs/all/stage2_all.log
tail -f tools/logs/all/stage3_msl.log
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

Stage1 和 Stage2 的 `resolved_config.json` 只保存本阶段相关参数；Stage3 保存完整训练参数。
