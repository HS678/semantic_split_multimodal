# semantic_split_multimodal

面向未知模态环境的语义对齐分布式多模态 Split Learning 实验框架。

English README: [README.md](README.md)。

## 项目概览

本仓库提供一套可复现的三阶段实验流程：

1. Stage 1：把 naturally paired 多模态数据划分为单模态客户端。
2. Stage 2：从 client encoder fingerprint 中发现未知客户端模态簇。
3. Stage 3：基于 `pred_cluster` 训练 MMBind 风格的 fusion Split Learning 模型。

本项目不是 Federated Learning，不使用 FedAvg。客户端上传 detached activation；服务器计算 cross-entropy loss，执行 fusion model 的 backward，再把 activation gradient 回传给对应客户端 encoder。

## 协议约束

训练 forward/backward 不使用真实模态名称、真实模态 ID、真实 Q，也不使用 oracle modality scheduler。Stage 1 保存的 `hidden_modality_id` 只允许用于 discovery 完成后的 post-hoc audit，以及无梯度的 naturally paired validation/test evaluation-only oracle mapping。

Stage 3 使用按预测簇均衡的随机轮询调度、label-guided semantic pseudo binding、`ClusterAdapter`、concat fusion、既有 classifier 和 Split Learning 梯度回传。训练期间每 10 rounds 读取 `validation_multimodal.pt` 做 naturally paired validation；训练结束后加载 validation macro-F1 选出的 `best_model.pt`，再读取 `test_multimodal.pt` 做一次最终测试。validation/test label 只用于计算 loss、accuracy、macro-F1 和 weighted-F1，不用于构造输入。

当前只保留两种聚类方法：

- `kmeans`
- `adaptive_isodata`

## 安装

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

仓库不包含数据集。请把原始数据放到：

```text
local/datasets/
```

默认期望目录：

```text
local/datasets/uci_har/
local/datasets/mhealth/
local/datasets/pamap2/
local/datasets/cmu_mosei/
```

CMU-MOSEI 目录需要包含：

```text
features/BERT_MOSEI.pkl
features/COAVAREP_aligned_MOSEI.pkl
features/FACET_aligned_MOSEI.pkl
splits/df_MOSEI.tsv
splits/df_valid_MOSEI.tsv
splits/df_test_MOSEI.tsv
```

三个 split TSV 来自特征来源仓库 [Ighina/MultiModalSA](https://github.com/Ighina/MultiModalSA/tree/master/data)，必须保持原始行顺序以便与 BERT 特征逐项核验。

本地参考资料可以保留在 `local/references/`。整个 `local/` 目录都会被 Git 忽略。

## 结果目录

Stage 1 的数据划分是可复用资产：

```text
local/results/partition/<dataset>/<partition_signature>/
```

Stage 2 的聚类结果按数据集、partition signature 和聚类方法分开保存：

```text
local/results/cluster/<dataset>/<partition_signature>/<cluster_method>/
```

Stage 3 的训练和测试结果是正式实验 run：

```text
local/results/experiments/<dataset>/<run_id>/
```

当 `clients_per_modality: 10` 时，默认 partition signature 为：

```text
UCI-HAR: local/results/partition/uci_har/acc_10clients_gyro_10clients__subject_disjoint_tvt_v1
MHEALTH: local/results/partition/mhealth/accelerometer_10clients_gyroscope_10clients_magnetometer_10clients_ecg_10clients__subject_disjoint_tvt_v1
PAMAP2:  local/results/partition/pamap2/accelerometer_10clients_gyroscope_10clients_magnetometer_10clients__subject_disjoint_tvt_v1
CMU-MOSEI: local/results/partition/cmu_mosei/text_10clients_audio_10clients_visual_10clients__official_video_disjoint_tvt_v1
```

每个阶段都会拒绝覆盖已有的非空输出目录。三划分 Stage 3 请使用新的 `run_id`，例如 `adaptive_tvt_seed101`。

## Stage 1：数据划分

运行单个数据集：

```bash
python scripts/stage1_partition.py --config configs/uci_har.yaml
```

四个数据集依次运行：

```bash
python scripts/stage1_partition.py --config configs/uci_har.yaml
python scripts/stage1_partition.py --config configs/mhealth.yaml
python scripts/stage1_partition.py --config configs/pamap2.yaml
python scripts/stage1_partition.py --config configs/cmu_mosei.yaml
```

Stage 1 直接在 partition 目录下写入：

```text
train_clients/client_*.pt
client_meta.csv
validation_multimodal.pt
test_multimodal.pt
partition_config.json
```

UCI-HAR、MHEALTH、PAMAP2 使用固定、互斥的 subject-level train/validation/test 划分；CMU-MOSEI 使用来源仓库的官方 video-disjoint train/validation/test，样本数分别为 16,327、1,871、4,662。CMU-MOSEI 任务为 `polarity < 0` 对 `polarity >= 0` 的二分类；audio/visual 按时间 mean pooling，三个模态都只用 train 统计量标准化。只有 train 会进入单模态 client partition 和 Stage 2；validation/test 保持 naturally paired。

## Stage 2：未知 Q 模态发现

UCI-HAR：

```bash
python scripts/stage2_discovery.py \
  --config configs/uci_har.yaml \
  --stage1-dir local/results/partition/uci_har/acc_10clients_gyro_10clients__subject_disjoint_tvt_v1 \
  --output-root local/results/cluster \
  --run-type user_formal
```

MHEALTH：

```bash
python scripts/stage2_discovery.py \
  --config configs/mhealth.yaml \
  --stage1-dir local/results/partition/mhealth/accelerometer_10clients_gyroscope_10clients_magnetometer_10clients_ecg_10clients__subject_disjoint_tvt_v1 \
  --output-root local/results/cluster \
  --run-type user_formal
```

PAMAP2：

```bash
python scripts/stage2_discovery.py \
  --config configs/pamap2.yaml \
  --stage1-dir local/results/partition/pamap2/accelerometer_10clients_gyroscope_10clients_magnetometer_10clients__subject_disjoint_tvt_v1 \
  --output-root local/results/cluster \
  --run-type user_formal
```

CMU-MOSEI：

```bash
python scripts/stage2_discovery.py \
  --config configs/cmu_mosei.yaml \
  --stage1-dir local/results/partition/cmu_mosei/text_10clients_audio_10clients_visual_10clients__official_video_disjoint_tvt_v1 \
  --output-root local/results/cluster \
  --run-type user_formal
```

Stage 2 只保留：

```text
true_cluster.csv
pred_cluster.csv
pretrained_encoders/
stage2_metadata.json
```

Stage 3 的技术输入要求是完整的 `pred_cluster.csv` 和逐客户端 `pretrained_encoders/`。`true_cluster.csv` 与 `stage2_metadata.json` 仅用于可选审计；文件缺失、真实簇不一致或 `discovery_status` 非成功都不会作为 Stage 3 启动门槛。

## Stage 3：Fusion Split Learning

Stage 3 从冻结的 Stage 1 partition 和冻结的 Stage 2 cluster 目录读取输入。正式 YAML 的基础 `seed` 固定为 `42`，供 Stage 1/Stage 2 和默认 Stage 3 使用；`--seed` 只在内存中覆盖本次 Stage 3 的实验种子，不修改 YAML，也不影响冻结的 Stage 1/Stage 2。建议先只运行 UCI-HAR，再运行较大的数据集。

UCI-HAR：

```bash
python scripts/stage3_train.py \
  --config configs/uci_har.yaml \
  --stage1-dir local/results/partition/uci_har/acc_10clients_gyro_10clients__subject_disjoint_tvt_v1 \
  --stage2-dir local/results/cluster/uci_har/acc_10clients_gyro_10clients__subject_disjoint_tvt_v1/adaptive_isodata \
  --output-root local/results/experiments \
  --run-id adaptive_tvt_seed101 \
  --seed 101 \
  --run-type user_formal
```

MHEALTH：

```bash
python scripts/stage3_train.py \
  --config configs/mhealth.yaml \
  --stage1-dir local/results/partition/mhealth/accelerometer_10clients_gyroscope_10clients_magnetometer_10clients_ecg_10clients__subject_disjoint_tvt_v1 \
  --stage2-dir local/results/cluster/mhealth/accelerometer_10clients_gyroscope_10clients_magnetometer_10clients_ecg_10clients__subject_disjoint_tvt_v1/adaptive_isodata \
  --output-root local/results/experiments \
  --run-id adaptive_tvt_seed101 \
  --seed 101 \
  --run-type user_formal
```

PAMAP2：

```bash
python scripts/stage3_train.py \
  --config configs/pamap2.yaml \
  --stage1-dir local/results/partition/pamap2/accelerometer_10clients_gyroscope_10clients_magnetometer_10clients__subject_disjoint_tvt_v1 \
  --stage2-dir local/results/cluster/pamap2/accelerometer_10clients_gyroscope_10clients_magnetometer_10clients__subject_disjoint_tvt_v1/adaptive_isodata \
  --output-root local/results/experiments \
  --run-id adaptive_tvt_seed101 \
  --seed 101 \
  --run-type user_formal
```

CMU-MOSEI：

```bash
python scripts/stage3_train.py \
  --config configs/cmu_mosei.yaml \
  --stage1-dir local/results/partition/cmu_mosei/text_10clients_audio_10clients_visual_10clients__official_video_disjoint_tvt_v1 \
  --stage2-dir local/results/cluster/cmu_mosei/text_10clients_audio_10clients_visual_10clients__official_video_disjoint_tvt_v1/adaptive_isodata \
  --output-root local/results/experiments \
  --run-id adaptive_tvt_seed101 \
  --seed 101 \
  --run-type user_formal
```

Stage 3 直接在 `local/results/experiments/<dataset>/<run_id>/` 下写入：

```text
resolved_config.yaml
train_log.csv
validation_log.csv
final_metrics.json
best_metrics.json
best_model.pt
last_model.pt
training_curves.png
stage3_metadata.json
```

正式配置最多训练 200 rounds，每 10 rounds 做 naturally paired validation，最少训练 50 rounds；validation macro-F1 连续 3 次未改善且改善量不足 `0.001` 时 early stop。`best_model.pt` 是 validation 选择的正式 checkpoint；`last_model.pt` 只用于诊断。训练结束后加载 `best_model.pt`，test 只完整评估一次并写入 `final_metrics.json`。

`training_curves.png` 会由 Stage3 自动生成。需要根据已有 CSV 手动重绘时运行：

```bash
PYTHONPATH=src /home/shuang/miniconda3/envs/mpsl/bin/python \
  -m semantic_split_multimodal.evaluation.plot_training_curves \
  --run-dir local/results/experiments/<dataset>/<run_id>
```

正式五种 Stage 3 随机种子为 `101`、`202`、`303`、`404`、`505`。每次运行使用独立 `run_id`，例如：

```bash
python scripts/stage3_train.py \
  --config configs/uci_har.yaml \
  --stage1-dir local/results/partition/uci_har/acc_10clients_gyro_10clients__subject_disjoint_tvt_v1 \
  --stage2-dir local/results/cluster/uci_har/acc_10clients_gyro_10clients__subject_disjoint_tvt_v1/adaptive_isodata \
  --output-root local/results/experiments \
  --run-id adaptive_tvt_seed202 \
  --seed 202 \
  --run-type user_formal
```

每个数据集都有独立正式实验脚本，分别完整执行 Stage1、Stage2 和五个 Stage3 seeds：

```bash
nohup bash local/tools/launch_uci_har_formal.sh \
  > "local/tools/uci_har_formal_$(date '+%Y%m%d_%H%M%S').log" 2>&1 &

nohup bash local/tools/launch_mhealth_formal.sh \
  > "local/tools/mhealth_formal_$(date '+%Y%m%d_%H%M%S').log" 2>&1 &

nohup bash local/tools/launch_pamap2_formal.sh \
  > "local/tools/pamap2_formal_$(date '+%Y%m%d_%H%M%S').log" 2>&1 &

nohup bash local/tools/launch_cmu_mosei_formal.sh \
  > "local/tools/cmu_mosei_formal_$(date '+%Y%m%d_%H%M%S').log" 2>&1 &
```

`launch_stage3_formal.sh` 保留为一次顺序启动全部四个数据集正式实验的聚合脚本：

```bash
nohup bash local/tools/launch_stage3_formal.sh \
  > "local/tools/formal_all_$(date '+%Y%m%d_%H%M%S').log" 2>&1 &
```

脚本使用 `adaptive_tvt_seed<N>`，不会覆盖旧 train/test 正式 run。`local/` 被 Git 忽略，因此该启动脚本和运行日志只保留在本地。

## 测试

```bash
PYTHONPATH=src python -m pytest tests -q
```
