# semantic_split_multimodal

用于研究未知模态数量、未知客户端模态归属、未配对单模态客户端环境下的分布式多模态 Split Learning 实验项目。

主线方法保持为三阶段流程：单模态客户端划分、未知模态发现、基于 `pred_cluster` 的 MMBind-style label-random fusion Split Learning。`unpaired_split_learning` 仍作为 shared semantic baseline/ablation 保留。

## 环境安装

建议使用固定环境：

```bash
/home/shuang/miniconda3/envs/mpsl/bin/python -m pip install -r requirements.txt
/home/shuang/miniconda3/envs/mpsl/bin/python -m pip install -e .
```

`hdbscan` 仅在选择 HDBSCAN 聚类时需要。

## 目录结构

```text
configs/    # uci_har、mhealth、pamap2 配置
scripts/    # Stage 1 / Stage 2 / Stage 3 可执行入口
src/        # semantic_split_multimodal Python package
tests/      # 直接调用或 pytest 运行的测试
docs/       # 架构、实验协议、扩展与交接文档
local/      # 本地数据、结果、日志、checkpoint、论文；不提交
```

`local/` 不进入 Git。原始数据放在 `local/datasets/`，结果放在 `local/results/`，日志放在 `local/logs/`，checkpoint 放在 `local/checkpoints/`，本地参考论文放在 `local/references/`。

## 三阶段运行命令

```bash
/home/shuang/miniconda3/envs/mpsl/bin/python scripts/stage1_partition.py --config configs/uci_har.yaml
/home/shuang/miniconda3/envs/mpsl/bin/python scripts/stage2_discovery.py --config configs/uci_har.yaml
/home/shuang/miniconda3/envs/mpsl/bin/python scripts/stage3_train.py --config configs/uci_har.yaml
```

将配置替换为 `configs/mhealth.yaml` 或 `configs/pamap2.yaml` 可运行其他数据集。

## 方法入口

主方法：

```yaml
training:
  multimodal_mode: mmbind_fusion_split_learning
```

baseline：

```yaml
training:
  multimodal_mode: unpaired_split_learning
```

Stage 3 根据 `training.multimodal_mode` 自动选择主线 fusion SL 或 unpaired shared semantic baseline。

## 输出位置

默认运行目录位于：

```text
local/results/<dataset-or-experiment>/<run_id>/
```

每个 run 内保留 Stage 1 数据划分、Stage 2 discovery 结果、Stage 3 训练评估日志和模型产物。

## 文档

- `docs/architecture.md`
- `docs/experiment_protocol.md`
- `docs/extension_guide.md`
- `docs/handoff.md`
