# semantic_split_multimodal

这是一个用于论文实验的分布式多模态 Split Learning 项目。当前活动主线唯一为 `mmbind_fusion_split_learning`：先把 naturally paired 多模态数据拆成单模态客户端，再发现未知客户端模态簇，最后基于 `pred_cluster` 做 full-coverage scheduling、same-label pseudo binding 和 concat MLP fusion Split Learning。

项目不是 Federated Learning，不做 FedAvg；客户端上传 detached activation，服务器计算 CE loss 和 backward，再把 activation gradient 路由回对应客户端 encoder。

## 环境

建议使用固定解释器：

```bash
/home/shuang/miniconda3/envs/mpsl/bin/python -m pip install -r requirements.txt
/home/shuang/miniconda3/envs/mpsl/bin/python -m pip install -e .
```

`hdbscan` 只在选择 HDBSCAN 聚类时需要。

## 目录

```text
configs/    # 三个数据集的正式配置
scripts/    # Stage 1 / Stage 2 / Stage 3 入口
src/        # semantic_split_multimodal package
tests/      # 主线单元与回归测试
docs/       # 架构、协议、配置、输出和交接文档
local/      # 本地数据、运行结果和 checkpoint；不提交
```

## 三阶段运行

```bash
/home/shuang/miniconda3/envs/mpsl/bin/python scripts/stage1_partition.py --config configs/uci_har.yaml
/home/shuang/miniconda3/envs/mpsl/bin/python scripts/stage2_discovery.py --config configs/uci_har.yaml
/home/shuang/miniconda3/envs/mpsl/bin/python scripts/stage3_train.py --config configs/uci_har.yaml
```

把 config 替换为 `configs/mhealth.yaml` 或 `configs/pamap2.yaml` 可运行其他数据集。Stage 3 入口只有正式 fusion Split Learning 路径，不再暴露旧方法选择。

只检查第一阶段数据划分时，可以使用脚本入口：

```bash
# 单个数据集
scripts/run_stage1_partitions.sh uci_har

# 三个数据集依次划分
scripts/run_stage1_partitions.sh all

# 如需指定解释器
PYTHON_BIN=/home/shuang/miniconda3/envs/mpsl/bin/python scripts/run_stage1_partitions.sh all
```

该脚本只运行 Stage 1，不运行 discovery 或训练。每次运行会通过 `scripts/stage1_partition.py` 创建新的 timestamp run，输出到 `local/results/<dataset>/<run_id>/01_dataset_partition/`。

## 本地防覆盖运行

`scripts/stage1_partition.py` 在 config 没有 `results.run_id` 时会创建 timestamp run；但如果 config 写死了 `results.run_id`，重复运行同一套配置会覆盖同一个结果目录。为了做 smoke、复现实验或临时检查，推荐使用本地 runner 自动生成不重复的 `run_id`：

```bash
/home/shuang/miniconda3/envs/mpsl/bin/python local/tools/run_pipeline.py \
  --config local/configs/formal/uci_har_known_q.yaml \
  --base-dir local/results/smoke \
  --tag quick_check
```

runner 会：

- 读取模板 config，但不修改原文件；
- 生成类似 `quick_check_20260727_173012` 的 `run_id`；
- 把运行副本写到 `local/configs/smoke/`；
- 依次运行 Stage 1、Stage 2、Stage 3；
- 如果目标 `run_dir` 已存在，直接拒绝运行，避免覆盖。

输出目录示例：

```text
local/results/smoke/uci_har/quick_check_20260727_173012/
  01_dataset_partition/
  02_cluster_results/
  03_training_evaluation/
  04_model_artifacts/
```

常用选项：

```bash
# 只预览，不写文件、不运行 stage
/home/shuang/miniconda3/envs/mpsl/bin/python local/tools/run_pipeline.py \
  --config local/configs/formal/uci_har_known_q.yaml \
  --base-dir local/results/smoke \
  --tag quick_check \
  --dry-run

# 明确指定 run_id
/home/shuang/miniconda3/envs/mpsl/bin/python local/tools/run_pipeline.py \
  --config local/configs/formal/uci_har_known_q.yaml \
  --base-dir local/results/formal \
  --run-id known_q_rerun_20260727_01
```

## 关键输出

默认运行目录由 `results.base_dir` 和 `results.run_id` 决定，典型结构为：

```text
local/results/<dataset>/<run_id>/
  01_dataset_partition/
  02_cluster_results/
  03_training_evaluation/
  04_model_artifacts/
```

正式评估读取 `01_dataset_partition/test_multimodal.pt`，输出写入 `03_training_evaluation/final_metrics.json`。best checkpoint 写入 `04_model_artifacts/best_mmbind_fusion_checkpoint.pt`。

## 文档阅读顺序

1. `docs/architecture.md`
2. `docs/experiment_protocol.md`
3. `docs/experiment_walkthrough.md`
4. `docs/configuration_reference.md`
5. `docs/output_reference.md`
6. `docs/handoff.md`
