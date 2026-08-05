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

Stage 3 使用按预测簇均衡的随机轮询调度、label-guided semantic pseudo binding、`ClusterAdapter`、concat fusion、既有 classifier 和 Split Learning 梯度回传。训练期间每 10 rounds 读取 `validation_multimodal.pt` 做 naturally paired validation；训练结束后加载 validation weighted-F1 选出的 `best_model.pt`，再读取 `test_multimodal.pt` 做一次最终测试。validation/test label 只用于计算 loss、accuracy、macro-F1 和 weighted-F1，不用于构造输入。

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
local/datasets/IEMOCAP/IEMOCAP_full/IEMOCAP_full_release/
```

IEMOCAP 使用 Full 版本和 `angry / happy-or-excited / sad / neutral` 四分类协议。Stage 1 前先生成冻结的 MFCC、MobileViT-XS 和 DistilBERT 序列特征：

```bash
PYTHONPATH=src python -m semantic_split_multimodal.data.prepare_iemocap --device cuda
```

固定划分为 Session 1-3 train、Session 4 validation、Session 5 test。音频采用三层 1D Conv 后接 GRU；视频和文本分别对冻结的 MobileViT-XS 帧特征、DistilBERT token 特征使用 GRU。

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
local/results/experiments/<oracle_true_cluster|predicted_cluster>/<dataset>/<config_signature>/seed-<seed>/attempt-<nn>/
```

当 `clients_per_modality: 10` 时，默认 partition signature 为：

```text
UCI-HAR: local/results/partition/uci_har/acc_10clients_gyro_10clients__subject_disjoint_tvt_v1
MHEALTH: local/results/partition/mhealth/accelerometer_10clients_gyroscope_10clients_magnetometer_10clients_ecg_10clients__subject_disjoint_tvt_v1
PAMAP2:  local/results/partition/pamap2/accelerometer_10clients_gyroscope_10clients_magnetometer_10clients__subject_disjoint_tvt_v1
IEMOCAP: local/results/partition/iemocap/audio_10clients_video_10clients_text_10clients__session_disjoint_123_4_5_v1
```

每个阶段都会拒绝覆盖已有的非空输出目录。同配置、同 seed 重跑 Stage 3 时，在 `.config` 中递增 `stage3.attempt`。

## Stage 1：数据划分

运行单个数据集：

```bash
python scripts/stage1_partition.py --config configs/uci_har.config
```

四个数据集依次运行：

```bash
python scripts/stage1_partition.py --config configs/uci_har.config
python scripts/stage1_partition.py --config configs/mhealth.config
python scripts/stage1_partition.py --config configs/pamap2.config
python scripts/stage1_partition.py --config configs/iemocap.config
```

Stage 1 直接在 partition 目录下写入：

```text
train_clients/client_*.pt
client_meta.csv
validation_multimodal.pt
test_multimodal.pt
partition_config.json
```

UCI-HAR、MHEALTH、PAMAP2 使用固定、互斥的 subject-level train/validation/test 划分。只有 train 会进入单模态 client partition 和 Stage 2；validation/test 保持 naturally paired。

IEMOCAP 采用固定且互斥的 Session 1-3/4/5 划分，共 5,531 条四分类语句。补齐后的序列长度会贯穿 Stage 1、encoder 预训练、fingerprint、Split Learning 和 naturally paired evaluation。`configs/iemocap.config` 按本次对照实验要求显式使用 `true_cluster`，属于 Oracle/debug 结果，不是无泄漏主结果。

## Stage 2：未知 Q 模态发现

UCI-HAR：

```bash
python scripts/stage2_discovery.py --config configs/uci_har.config
```

MHEALTH：

```bash
python scripts/stage2_discovery.py --config configs/mhealth.config
```

PAMAP2：

```bash
python scripts/stage2_discovery.py --config configs/pamap2.config
```

IEMOCAP：

```bash
python scripts/stage2_discovery.py --config configs/iemocap.config
```

Stage 2 只保留：

```text
true_cluster.csv
pred_cluster.csv
pretrained_encoders/
fingerprints.npz
fingerprint_pca.pdf
fingerprint_pca.png
fingerprint_pca_metadata.json
stage2_metadata.json
```

`fingerprint_pca.pdf` 是可直接用于论文的矢量图；`fingerprint_pca.png` 为 600 DPI 预览图。PCA 坐标只由聚类前 client fingerprint 计算，真实模态与预测簇仅用于聚类完成后的两个审计着色面板，不参与 PCA 拟合或聚类。

Stage 2 始终生成 `pred_cluster.csv`、PCA 图和聚类审计。本轮开发实验的 Stage 3 配置固定选择 `true_cluster.csv`；论文正式实验后再切换为 `pred_cluster` 并调节现有聚类参数。

## Stage 3：Fusion Split Learning

Stage 3 从 `.config` 中读取冻结的 Stage 1/Stage 2 输入、seed、输出根目录和 attempt。命令行路径参数仅保留为可选调试覆盖项。

Stage 3 的簇分配来源由下列配置控制：

```ini
[training]
cluster_assignment_source=true_cluster
```

调试 Stage 3、需要绕过预测聚类时，可改为 `true_cluster`。此时训练、调度、binding、fusion slot 和 evaluation mapping 统一读取 `true_cluster.csv`；该模式使用真实模态簇，属于 oracle/debug 实验，不得作为无模态泄漏的正式主结果，且应使用可明确区分的 `attempt`。

UCI-HAR：

```bash
python scripts/stage3_train.py --config configs/uci_har.config
```

MHEALTH：

```bash
python scripts/stage3_train.py --config configs/mhealth.config
```

PAMAP2：

```bash
python scripts/stage3_train.py --config configs/pamap2.config
```

IEMOCAP true-cluster Oracle/debug 对照：

```bash
python scripts/stage3_train.py --config configs/iemocap.config
```

Stage 3 在上述 config-signature/seed/attempt 目录下写入：

```text
source_config.config
resolved_config.config
train_log.csv
validation_log.csv
final_metrics.json
best_metrics.json
best_model.pt
last_model.pt
training_curves.png
stage3_metadata.json
```

正式配置最多训练 200 rounds，每 10 rounds 做 naturally paired validation，最少训练 50 rounds；validation weighted-F1 连续 3 次未改善且改善量不足 `0.001` 时 early stop。`best_model.pt` 是 validation 选择的正式 checkpoint；`last_model.pt` 只用于诊断。训练结束后加载 `best_model.pt`，test 只完整评估一次并写入 `final_metrics.json`。

`training_curves.png` 会由 Stage3 自动生成。需要根据已有 CSV 手动重绘时运行：

```bash
PYTHONPATH=src /home/shuang/miniconda3/envs/mpsl/bin/python \
  -m semantic_split_multimodal.evaluation.plot_training_curves \
  --run-dir local/results/experiments/<cluster_scope>/<dataset>/<config_signature>/seed-<seed>/attempt-<nn>
```

正式五种 Stage 3 随机种子为 `101`、`202`、`303`、`404`、`505`。修改 `.config` 的 `seed`；同一 seed 重跑时递增 `stage3.attempt`。

```bash
python scripts/stage3_train.py --config configs/uci_har.config
```

四个数据集各有独立实验脚本，分别完整执行 Stage1、Stage2 和五个 Stage3 seeds。当前配置固定 `true_cluster`，因此 Stage3 输出属于 Oracle/debug：

```bash
nohup bash local/tools/launch_uci_har_formal.sh \
  > "local/tools/uci_har_formal_$(date '+%Y%m%d_%H%M%S').log" 2>&1 &

nohup bash local/tools/launch_mhealth_formal.sh \
  > "local/tools/mhealth_formal_$(date '+%Y%m%d_%H%M%S').log" 2>&1 &

nohup bash local/tools/launch_pamap2_formal.sh \
  > "local/tools/pamap2_formal_$(date '+%Y%m%d_%H%M%S').log" 2>&1 &

nohup bash local/tools/launch_iemocap_formal.sh \
  > "local/tools/iemocap_formal_$(date '+%Y%m%d_%H%M%S').log" 2>&1 &
```

`launch_stage3_formal.sh` 是顺序启动上述四个数据集的聚合脚本：

```bash
nohup bash local/tools/launch_stage3_formal.sh \
  > "local/tools/formal_all_$(date '+%Y%m%d_%H%M%S').log" 2>&1 &
```

脚本通过 seed 覆盖生成独立 `seed-<seed>/attempt-01` 目录，不会覆盖已有 run。`local/` 被 Git 忽略，因此启动脚本和运行日志只保留在本地。

## 测试

```bash
PYTHONPATH=src python -m pytest tests -q
```
