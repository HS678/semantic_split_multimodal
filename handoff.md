# Handoff

## 当前主线

本仓库当前主线是未知模态环境下的分布式多模态 Split Learning：客户端只持有单一模态数据，客户端之间没有 instance correspondence，训练数据有标签；服务器不知道真实模态数量 `Q` 和客户端真实模态身份。

这不是 Federated Learning / FedAvg 路线。当前实现采用 Split Learning：客户端前端 encoder 产生 activation，服务器端完成 binding、fusion、classification、loss 和 backward，再把 activation gradient 路由回对应客户端更新。

重要边界：

- `hidden_modality_id` 只允许用于 clustering evaluation 和 evaluation-only oracle mapping。
- `hidden_modality_id` 禁止用于训练、调度、binding、fusion slot 构造或任何模型输入。
- MMBind-style label-random pseudo binding 是借鉴机制，不是本项目原创贡献。
- 本项目原创主线应聚焦：未知模态数量/归属发现、基于 `pred_cluster` 的簇覆盖调度、后续用于缓解 slow client 等待的 D2D 协同机制。

## 三阶段流程

### Stage 1: Partition

`experiments/stage1_partition.py` 将原始多模态样本拆成单模态 client dataset。每个 client 只看到一个 hidden modality。训练/验证客户端样本不要求跨客户端配对。

Stage 1 同时保留 naturally paired test samples 到 `test_multimodal.pt`，用于最终测试时按原始样本构造多模态输入；测试阶段不使用 label 做 binding、筛选或排序。

### Stage 2: Discovery

`experiments/stage2_pretrain_cluster.py` / `trainers/pretrain_cluster.py` 对每个客户端 encoder 做 autoencoder pretrain，提取 fingerprint，然后用 KMeans、HDBSCAN 或 ISODATA 聚类。

Stage 2 输出 `estimated_Q` 和每个客户端的 `pred_cluster`。真实模态只用于 discovery metrics：`true_Q`、`estimated_Q`、`abs_Q_error`、`ACC`、`NMI`、`ARI`。

### Stage 3: MMBind Fusion Split Learning

主训练入口是 `experiments/stage3_train_sl.py` 中的 `run_mmbind_fusion_stage3_split_training`，主 trainer 是 `trainers/mmbind_fusion_split_trainer.py`。

每个 global round 先由 scheduler 选出一组固定客户端；随后在 `training.local_steps` 个 local step 内重复独立 batch-level SL update：

1. selected clients 各自独立采样 labeled batch。
2. 客户端 encoder 前向得到 detached activations。
3. 服务器按 exact same-label random pseudo binding 组装伪多模态样本。
4. binding 必须覆盖所有被调度的 `pred_cluster`；否则只跳过当前 local step。
5. `pred_cluster` 映射到稳定的 cluster slot，经 `ClusterAdapter` 对齐维度。
6. server 端 concat MLP fusion 和 classifier 计算 CE loss。
7. server backward 后，把每个 activation 的 gradient 路由回对应客户端执行 optimizer step。

如果某个 local step 没有共同 label，只跳过该 local step；如果整个 round 的所有 local step 都无法形成 binding，则记录 `round_status=empty_binding_round`。

## 测试协议

训练阶段可以使用 training label 做 label-random pseudo binding。测试阶段 label 只能用于 loss、accuracy、macro-F1 计算，不能参与 binding、筛选、排序或输入构造。

当前评估使用 naturally paired test samples。`evaluation/fusion_eval.py` 根据 Stage 1 保留的 `test_multimodal.pt` 构造测试 batch，并使用 Stage 2 discovery result 与 evaluation-only oracle mapping 将真实测试模态路由到 `pred_cluster` slot。

oracle mapping 只在 evaluation helper 内使用；mapping 失败时对应指标为 `null`，不做 majority-vote 修复，不填 `0`，也不回退到真实模态参与训练。

## 已完成

- Stage 1 单模态 client partition，并保留 naturally paired test set。
- Stage 2 autoencoder pretrain、fingerprint、KMeans/HDBSCAN/ISODATA discovery。
- Discovery metrics：`true_Q`、`estimated_Q`、`abs_Q_error`、`ACC`、`NMI`、`ARI`。
- ProposedClusterCoverage scheduler 基于 `pred_cluster` 做覆盖调度。
- Stage 3 MMBind fusion SL 主线：label-random pseudo binding、cluster-to-slot、`ClusterAdapter`、concat MLP fusion、classifier、CE loss、split backward、client optimizer update。
- `training.local_steps` 支持在同一 selected-client set 内执行多个 local step。
- Empty binding local step 只跳过当前 step；整轮失败时记录 `empty_binding_round`。
- Checkpoint 保存和 naturally paired evaluation。
- Evaluation-only oracle mapping 失败时返回 `null` 指标。
- `README.md` 和 `docs/extension_guide.md` 已描述当前三阶段主线。
- `results_smoke_*/`、`presude-results/` 等本地 smoke/参考产物已隔离到 ignore 规则。
- 本轮已将根目录下历史 tracked `results_unpaired/` 和 `results_unpaired_tuning/` 作为自动生成运行结果从 Git 跟踪中移除。

## Baseline / Ablation

`trainers/unpaired_split_multimodal_trainer.py` 仍保留旧的 SharedSemanticBackbone、PrototypeBank、prototype alignment 逻辑，仅作为 baseline/ablation 入口，不是当前主线。

后续写论文或 README 时应避免把 shared semantic learning / prototype alignment 描述成主方法贡献。

## 尚未完成

- D2D 协同机制尚未实现；目前只有 D2D 相关指标/接口占位。
- 还没有真实 slow-client latency profile，也没有协作前后等待时间对比实验。
- 还没有大规模端到端实验结果和消融表。
- HDBSCAN/ISODATA 的稳定性仍需在真实数据上复核。
- 论文贡献表述还需要把 MMBind-style binding 与本项目原创点清晰拆开。

## 验证状态

当前验证环境：

```bash
/home/shuang/miniconda3/envs/mpsl/bin/python
torch 2.4.1+cu121
```

`pytest` 当前未安装，因此测试通过直接调用 test functions 验证。已直接调用 14 个测试函数，覆盖：

- MMBind fusion split trainer：binding、覆盖失败、empty binding、local steps。
- label-random binding。
- concat MLP fusion。
- fusion evaluation protocol。
- oracle evaluation mapping。

已通过：

```bash
/home/shuang/miniconda3/envs/mpsl/bin/python -m py_compile $(rg --files -g '*.py')
```

## Git 状态基线

本轮开始分支：

```bash
feature/mmbind-fusion-sl
```

本轮开始 HEAD：

```bash
d3a1aff chore: ignore local smoke artifacts
```

稳定功能标签：

```bash
v2-mmbind-fusion-sl -> ecd28b0 feat: add MMBind fusion split learning pipeline
```

当前远端为：

```bash
origin git@github.com:HS678/semantic_split_multimodal.git
```

用户另行授权推送目标远端：

```bash
git@github.com:HS678/my-project.git
```

## 下一步建议顺序

1. 在当前分支完成本轮 `handoff.md` 与历史 tracked 运行结果治理提交。
2. 推送到用户授权的 `git@github.com:HS678/my-project.git`，标签名使用 `v2-mmbind-fusion-sl`。
3. 基于 `v2-mmbind-fusion-sl` 跑最小真实数据 smoke：Stage 1、Stage 2、Stage 3。
4. 检查 `estimated_Q`、`abs_Q_error`、ACC/NMI/ARI 是否符合预期。
5. 检查 Stage 3 每轮 `selected_clients`、binding success rate、`empty_binding_round` 比例。
6. 对比 Random / RoundRobin / Oracle / ProposedClusterCoverage。
7. 增加 slow-client latency profile。
8. 实现 D2D 协同前后等待时间统计。
9. 再补 D2D 消融实验与论文表格。
