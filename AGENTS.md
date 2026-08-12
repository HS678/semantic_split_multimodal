# Codex Instructions

## Communication

- 所有解释、计划、问题、总结使用中文。
- 命令、代码、路径、配置字段、错误信息保持原文。
- 修改任何文件前必须说明：修改文件、修改原因、影响范围。
- 不确定时先问；不自行改变实验设计；不做无关大规模重构。

## Project

项目：

```text
Semantic-aligned Distributed Split Multimodal Learning in Unknown Modality Environments
```

正式主线：

```text
MMBind-style Fusion Split Learning
```

核心流程：

```text
Stage1 data construction
-> Stage2 Modality Discovery
-> pred_cluster
-> Cluster-aware Scheduling
-> Label-guided Semantic Pseudo Binding
-> ClusterAdapter + Concat Fusion
-> Split Learning
-> Naturally Paired Evaluation
```

当前实现目录：

```text
src/MSL/
configs/MSL/
scripts/MSL/
```

baseline 只放对照方法：

```text
src/baseline/
configs/baseline/
scripts/baseline/
```

## No Leakage

训练阶段禁止使用：

```text
hidden_modality_id
true modality name
true modality count
true_cluster
```

这些只能用于：

```text
Stage2 discovery audit
evaluation-only oracle mapping
```

正式主线必须使用：

```text
training.cluster_assignment_source=pred_cluster
```

`true_cluster` 只能作为显式 oracle upper bound / sanity check，不能作为 unknown-modality 主结果。

## Algorithm Constraints

未经明确要求，禁止修改：

- Stage2 discovery 协议：client encoder pretraining -> fingerprint extraction -> clustering -> `pred_cluster`
- predicted-cluster coverage scheduling
- label-guided semantic pseudo binding
- `ClusterAdapter + Concat Fusion + Classifier`
- Split Learning feature upload / server backward / gradient return / client update 流程
- naturally paired evaluation：使用 `test_multimodal.pt`，test label 只用于 loss 和 metrics

训练调度、binding、fusion slot 构造只能依据：

```text
pred_cluster
label
```

禁止用 PCA 图替代 discovery 证明；必须记录：

```text
estimated_Q
hungarian_ACC
NMI
ARI
discovery_status
split/mix audit
```

## Experiment Rules

- 正式结果保存到 `local/results_msl/`。
- baseline 结果保存到 `local/results_baseline/`。
- 不覆盖已有正式实验结果。
- 无验证集；`test_multimodal.pt` 只在训练结束后评估一次。
- 多 seed / 多 fold 汇总 `mean ± std`。
- 修改数据划分、模型、loss、optimizer、scheduler、binding、fusion、evaluation 前必须说明影响。

当前数据划分：

```text
UCI-HAR: official train/test split, multi-seed Stage3
MHEALTH: subject-level 5-fold
PAMAP2: subject-level 9-fold LOSO
IEMOCAP: session-level 5-fold LOSO
```

客户端划分是 label-stratified IID，用于隔离 unknown modality heterogeneity；不要把 non-IID / Dirichlet 作为主线。

## Coding And Validation

- 优先最小修改，保持现有接口和结构。
- 不引入未经验证的新依赖。
- 新增模块必须说明位置、职责、与现有代码关系。
- 修改代码后至少执行 compile/import check；涉及流程时执行相关 pytest 或 smoke run。
- 删除文件前确认用途、引用和替代版本。

## Git

禁止自动执行：

```text
git commit
git push
git reset
git checkout
```

允许查看：

```text
git status
git diff
git log
```

## Completion Report

完成后用中文总结：

- 新增 / 修改 / 删除文件
- 修改内容与原因
- 是否影响算法协议和实验结果
- Validation 命令与结果
- Risks
- 如有 tracked 修改，给出建议 commit message，不自动 commit
