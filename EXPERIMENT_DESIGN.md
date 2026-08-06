# 正式实验框架设计文档

> 本文档记录 `msl` 分支正式实验框架的全部设计决策，按实验运行步骤（Stage 1 → Stage 2 → Stage 3 → Evaluation）组织。
> 状态标记：✅ 已确定；⏳ 待讨论/待定；🔄 后续实现时细化。
> 本文件是设计记录，不是可执行配置；正式配置见 `configs/`（当前唯一正式方案）。

---

## 1. 总体设计决策

### 1.1 数据集范围

✅ 正式实验只保留四个数据集，CMU-MOSEI 已从代码、配置、测试与文档中移除：

- UCI-HAR（`uci_har`）
- MHEALTH（`mhealth`）
- PAMAP2（`pamap2`）
- IEMOCAP（`iemocap`，四分类 `angry / happy-or-excited / sad / neutral`）

### 1.2 评估协议：各数据集采用领域主流协议

✅ 每个数据集采用其领域最常用的评估协议，便于与主流文献直接对比；不同数据集协议不同是论文常见做法：

| 数据集 | 协议 | 报告口径 |
| --- | --- | --- |
| UCI-HAR | 官方 70/30 固定划分（train 21 / test 9 subject） | 单点精度（test 2,947 样本，统计可靠） |
| MHEALTH | subject-level 5 折（每折 2 个 subject） | 均值 ± 标准差 |
| PAMAP2 | subject-level 9 折 LOSO（每折 1 个 subject） | 均值 ± 标准差 |
| IEMOCAP | 5 折 session-LOSO（每折 1 个 Session test） | 均值 ± 标准差 |

理由：MHEALTH / PAMAP2 采用 subject 级交叉验证（subject 数少，单次划分偏差风险高）；UCI-HAR 官方划分样本量充足（test 9 个 subject / 2,947 样本），可直接与官方划分文献对比；IEMOCAP 采用固定 Session 划分（文献常见），论文说明即可。

### 1.3 无验证集：固定轮数 + last_model（已确定）

✅ 当前正式方案不使用验证集：

- 数据划分只有 train / test 两段，代码中已无 `validation_subjects/sessions`、`validation_multimodal.pt`、`best_model.pt` 等验证集产物；
- Stage3 固定 `global_rounds` 训练，无 early stopping、无 best checkpoint 选择；
- 训练结束后直接使用 `last_model.pt` 对 naturally paired `test_multimodal.pt` 评估一次；
- 与主流 LOSO/固定划分论文一致，论文无需说明验证集口径。

### 1.4 D2D 模块与正式实验的关系

✅ D2D（device-to-device）尚未实现（`d2d.enabled=false` 仅保留配置入口）。

- 正式精度主实验：四个数据集 × 5 折完整运行；
- D2D 效率实验（未来实现后）：作为独立章节，可在 1~2 个代表性数据集或单折上做通信量/时延/收敛轮数对比，不必 5 折全跑；
- 统一训练协议下，D2D 与其他方法的对比更公平（相同固定轮数训练规则）。

### 1.5 无泄漏红线（沿用并强化）

- `hidden_modality_id` / 真实模态名 / 真实模态数只允许用于 Stage 2 审计与 evaluation-only oracle mapping；
- 训练、调度、binding、fusion slot 一律只使用 `pred_cluster` 与 label；
- test 只用于最终报告，任何训练期决策不得接触 test；
- 5 折分组在 Stage 1 固定，不允许根据 test 结果调整分组。

---

## 2. 通用 Stage 1 参数

✅ 已确定：

| 参数 | 值 | 说明 |
| --- | --- | --- |
| `config.seed` | 42（基础）；多 seed 见下 | 基础随机种子 |
| `partition.clients_per_modality` | 10 | 每个真实模态拆成 10 个单模态客户端 |
| `dataset.normalize` | true | 只用 train split 统计量标准化 |
| `results.base_dir` | `./local/results_msl` | 正式结果根目录 |

### 2.1 seed 策略

✅ 已确定：

- MHEALTH（5 折）/ PAMAP2（9 折 LOSO）：1 个 seed（42）——多折已提供方差；
- UCI-HAR / IEMOCAP（单次固定划分）：5 个 seed（101, 202, 303, 404, 505），报告均值 ± 标准差；
- Stage 3 通过 CLI `--seed` 覆盖内存配置，不修改源 `.config`。

### 2.2 split_protocol 命名规范

✅ 已确定：

| 数据集 | split_protocol | 说明 |
| --- | --- | --- |
| uci_har | `subject_disjoint_70_30` | 官方固定 70/30 划分 |
| mhealth | `subject_5fold_foldN`（N=1..5） | fold 编号进签名，目录天然隔离 |
| pamap2 | `subject_9fold_loso_foldN`（N=1..9） | 明确 9 折 LOSO |
| iemocap | `session_5fold_loso_foldN`（N=1..5） | 5 折 session-LOSO |

原则：划分方式一变，签名必变；fold 编号进入签名，保证 5 折 / 9 折产物互不覆盖、全程可追溯。

---

## 3. Stage 1 数据划分设计（各数据集协议与分组）

> 每折的 train = 该数据集其余全部 subject/session；无独立验证集。
> 分组固定后写入正式配置，禁止根据结果调整。

### 3.1 UCI-HAR：官方 70/30 固定划分

✅ 已确定（与官方协议一致，单次固定划分，无验证集）：

- train（21 subject）：官方 70% 训练集（含 14, 19, 23, 25）
- test（9 subject）：2, 4, 9, 10, 12, 13, 18, 20, 24
- 样本数：train 7,352 / test 2,947

说明：train/test 与官方 70/30 划分完全一致，可与官方划分文献直接对比数字。

### 3.2 MHEALTH（10 subject，5 折 × 2 subject）

✅ 已确定：

| 折 | test subjects | 原始行数 |
| --- | --- | ---: |
| fold1 | 1, 10 | 259,584 |
| fold2 | 9, 6 | 233,472 |
| fold3 | 2, 7 | 235,009 |
| fold4 | 8, 4 | 245,760 |
| fold5 | 3, 5 | 241,920 |

✅ 无验证集：每折 train 8 个 subject / test 2 个 subject，划分内置在数据集代码中。

### 3.3 PAMAP2：9 折 subject-LOSO

✅ 已确定（与原论文及多数论文一致）：

| 折 | test subject | train subjects |
| --- | --- | --- |
| fold1 | 101 | 102, 103, 104, 105, 106, 107, 108, 109 |
| fold2 | 102 | 101, 103, 104, 105, 106, 107, 108, 109 |
| fold3 | 103 | 101, 102, 104, 105, 106, 107, 108, 109 |
| fold4 | 104 | 101, 102, 103, 105, 106, 107, 108, 109 |
| fold5 | 105 | 101, 102, 103, 104, 106, 107, 108, 109 |
| fold6 | 106 | 101, 102, 103, 104, 105, 107, 108, 109 |
| fold7 | 107 | 101, 102, 103, 104, 105, 106, 108, 109 |
| fold8 | 108 | 101, 102, 103, 104, 105, 106, 107, 109 |
| fold9 | 109 | 101, 102, 103, 104, 105, 106, 107, 108 |

设计约束与论文说明：

- subject 109 原始记录很少（约 85 秒 @100Hz），该折 test 窗口数远少于其他折（此前运行约 64 个窗口）——这是数据集固有特性，主流 9 折 LOSO 均包含该折，报告 9 折均值 ± 标准差；
- 不包含心率通道（代码内置，模态固定 acc/gyro/mag）；
- 无验证集：每折 train 8 个 subject / test 1 个 subject，划分内置在数据集代码中；
- 论文需注明：采用 9 折 LOSO（原论文推荐协议），subject 109 折的波动属于数据集特性。

### 3.4 IEMOCAP：5 折 session-LOSO

✅ 已确定（5 折 session-LOSO，无验证集）：

- 每折 test 1 个 Session，train 其余 4 个 Session，划分内置在数据集代码中；
- 报告 5 折均值 ± 标准差。

### 3.5 多折与"单模态客户端 / 多模态配对"结构的兼容性

✅ 已确认无结构问题：

- 交叉验证的"折"发生在 subject 级数据划分层，每个 fold 独立完整运行 Stage1 → Stage2 → Stage3；
- 每折内部结构与非交叉验证一致：该折 train subjects 拆成单模态客户端，test subjects 保留自然配对多模态；
- 客户端拆分只在 train 侧进行，test subject 与 train 不相交，无泄漏；
- 每折 Stage 2 独立聚类，`pred_cluster` 可能不同；若某折聚类质量差导致 evaluation mapping 失败，该折无指标，需记录原因而非静默跳过（实验阶段验证）。

### 3.6 窗口与模态参数（mhealth / pamap2）

✅ 已确定：

| 参数 | mhealth | pamap2 |
| --- | --- | --- |
| `dataset.modality_scheme` | sensor_type | sensor_type |
| `dataset.window_size` | 128（1.28s @50Hz） | 200（2s @100Hz） |
| `dataset.stride` | 64（50% overlap） | 100（50% overlap） |
| `dataset.min_label_purity` | 0.6 | 0.6 |
| `dataset.drop_null` / `drop_other` | true | true |
| `dataset.include_heart_rate` | — | false |
| `dataset.normalize` | true | true |

说明：

- mhealth 128/64 与主流一致（2.56s @50Hz，50% overlap）；
- pamap2 采用论文常见的 2s 窗口 + 50% overlap（200/100），替代原先 128/128 无重叠设置；
- `min_label_purity=0.6` 用于过滤跨动作边界的混合标签窗口，仅使用 train 侧 label，无泄漏；
- MMBind 参考源码（`local/references/mmbind/.../PAMAP2/preprocess/process_data.py`）采用 1000 帧无重叠块 + 7 类动作，属于其简化处理，不作为本项目参考。

---

## 4. Stage 2（模态发现）设计

✅ 已确定：

### 4.1 聚类输入源

- 当前固定 `cluster_assignment_source=true_cluster`（oracle 调试），先保证全流程跑通；
- 后续调优聚类参数、确认 `pred_cluster` 完全正确后，再切换为 `pred_cluster` 作为正式无泄漏主线；
- 切换时机由聚类质量验证（discovery 指标）决定，不以 test 结果回调。

### 4.2 fingerprint 类型

| 数据集 | fingerprint.type | 说明 |
| --- | --- | --- |
| uci_har | hybrid | 保持现状（已 discovery_success） |
| mhealth | signal | 保持现状（signal 指纹已成功区分 4 模态） |
| pamap2 | signal | 保持现状（signal 指纹已成功区分 3 模态） |
| iemocap | hybrid | 采用 hybrid |

### 4.3 预训练目标与参数

- 预训练目标：`classification`（任务语义初始化 encoder，有利于 Stage 3 精度）；
- fingerprint 的模态区分由 `signal` / hybrid 中的 signal 部分承担（实证：mhealth/pamap2 signal 成功、encoder 指纹聚成 1 簇失败）；
- 参数沿用当前运行成功的配置（uci_har 25 epochs / 0.0005；mhealth/pamap2 25~30 epochs / 0.0002~0.0003；iemocap 25 epochs / 0.0002；class_weighting 按数据集 inverse_sqrt 或 none）；
- 未来切换 `pred_cluster` 时若 encoder 指纹区分不足，再比较 `reconstruction` vs `classification` 的影响。

### 4.4 聚类参数

- 保持当前 `adaptive_isodata` 参数（`q_max=8`、`min_cluster_size=2`、`min_split_silhouette=0.10~0.20`、`pca_variance=0.95`、`seeds=[11,23,37,53,71]`）；
- 暂不调优（当前使用 `true_cluster`，聚类质量不影响训练）；后续统一调优后切换 `pred_cluster`。

### 4.5 每折独立性

- MHEALTH 5 折 / PAMAP2 9 折：每折独立预训练、提取 fingerprint、聚类，`pred_cluster` 每折可能不同；
- 最终实验报告给出多折聚合结果（均值 ± 标准差）。

### 4.6 沿用现状的细节

- fingerprint 提取参数：`batch_size=64`、`max_batches=4`；
- 指纹可视化：保留 `fingerprint_visualization.enabled=true`（PCA 双面板审计图继续输出到 `adaptive_isodata/visualization/`）；
- 每折目录隔离：Stage 2 输出目录自动带 fold（`<dataset>/<partition_signature>/adaptive_isodata/`），无 run_name 层。

---

## 5. Stage 3（训练）与 Evaluation 设计

✅ 已确定（沿用当前运行成功的参数）：

### 5.1 训练机制（方法核心协议，不改）

- 调度：`balanced_cluster_round_robin`（pred/true_cluster 均衡轮询）；
- 绑定：`label_random`（exact same-label pseudo binding）；
- 融合结构：`ClusterAdapter + Concat Fusion + Classifier`；
- 训练目标：`mmbind_weighted_contrastive`（CE + 跨簇 contrastive + 异构 CE，论文主线）；
- Split Learning：server backward → activation gradient 回传客户端。

> 定位说明：`mmbind_weighted_contrastive` 为借用的 MMBind 模块，**不是本论文的贡献点**；保持现状、不调优、不做消融。论文贡献点集中在分布式 Split Learning 框架与未知模态环境下的模态发现/调度/伪绑定流程。

### 5.2 训练规模（无验证集，固定轮数）

- `global_rounds=200`（无验证集，直接固定轮数）；
- 无 early stopping、无 best checkpoint 选择；
- `local_steps=1`；batch_size / lr / weight_decay / max_grad_norm / class_weighting 按数据集内置在 `dataset_defaults.py`（均为当前运行成功的参数）。

### 5.3 评估指标

- 正式指标：**acc / macro_f1 / weighted_f1**（论文只报告这三个）；
- 训练结束直接使用 `last_model.pt` 对 naturally paired `test_multimodal.pt` 评估一次；
- 汇总格式：`{foldN/seedN: {acc, macro_f1, weighted_f1}, average: {...}}`。

### 5.4 seed 与聚合报告

- MHEALTH（5 折）/ PAMAP2（9 折）：1 seed（42），报告 5/9 折 test 均值 ± 标准差；
- UCI-HAR / IEMOCAP：5 seed（101, 202, 303, 404, 505），报告均值 ± 标准差；
- 汇总脚本 `scripts/summarize_results.py` 已支持多折 / 多 seed 聚合。

---

## 6. 待定事项清单

| # | 事项 | 状态 | 影响 |
| --- | --- | --- | --- |
| 1 | mhealth / pamap2 窗口参数（window_size、stride、min_label_purity） | ✅ 已定（128/64；200/100；0.6） | Stage 1 样本量与 encoder 输入长度 |
| 2 | `split_protocol` 命名规范（含 fold 签名） | ✅ 已定 | 目录签名与可追溯性 |
| 3 | 多 seed 数量 | ✅ 已定（交叉验证 1 seed；单次划分 5 seed） | 算力成本与方差报告 |
| 4 | Stage 3 训练规模（无验证集固定轮数） | ✅ 已定（200 轮，无早停、无 best checkpoint） | 训练成本与收敛 |
| 5 | `pred_cluster` 切换验证（聚类调优后） | ⏳ | 无泄漏主线能否跑通（当前先用 true_cluster） |
| 6 | 汇总脚本多折聚合 | ✅ 已实现 | 正式结果报告 |

---

## 7. 决策记录

| 日期 | 决策 |
| --- | --- |
| 2026-08-04 | 移除 CMU-MOSEI，正式实验只保留四个数据集 |
| 2026-08-04 | `seed=42`，`clients_per_modality=10`，pamap2 不包含心率 |
| 2026-08-04 | 采用各数据集领域主流协议：UCI-HAR 官方 70/30、MHEALTH 5 折、PAMAP2 9 折 LOSO、IEMOCAP 5 折 session-LOSO |
| 2026-08-04 | 确定 MHEALTH 5 折分组（fold1: 1,10；fold2: 9,6；fold3: 2,7；fold4: 8,4；fold5: 3,5） |
| 2026-08-04 | 窗口参数：MHEALTH 128/64（50% overlap）、PAMAP2 200/100（2s、50% overlap）、min_label_purity 均为 0.6 |
| 2026-08-04 | seed 策略：MHEALTH/PAMAP2 用 1 seed（42）；UCI-HAR/IEMOCAP 用 5 seed（101,202,303,404,505） |
| 2026-08-06 | 最终 split_protocol 命名：uci_har `subject_disjoint_70_30`；mhealth `subject_5fold_foldN`；pamap2 `subject_9fold_loso_foldN`；iemocap `session_5fold_loso_foldN` |
| 2026-08-06 | 确定无验证集：删除 validation 相关代码与产物，固定 `global_rounds=200`，训练结束用 `last_model.pt` 测试一次 |
| 2026-08-06 | 论文指标只保留 acc / macro_f1 / weighted_f1；汇总格式 `{foldN/seedN, average}` |
| 2026-08-04 | Stage 2：固定 true_cluster（后续调聚类后切 pred_cluster）；fingerprint 保持现状（uci_har hybrid、mhealth/pamap2 signal、iemocap hybrid）；预训练用 classification；聚类参数暂不调 |
| 2026-08-04 | Stage 3：沿用当前配置（mmbind_weighted_contrastive、200 轮、无验证集） |

---

## 8. 正式实验运行说明（MSL）

### 8.1 配置位置

- `configs/uci_har.config`（官方 70/30，单次划分 × 5 seed）
- `configs/iemocap/fold1~5.config`（5 折 session-LOSO）
- `configs/mhealth/fold1~5.config`（5 折）
- `configs/pamap2/fold1~9.config`（9 折 LOSO）
- 全部为独立完整配置（无 extends）；无验证集，`training.validation_enabled=false`、固定 `global_rounds=200`；结果写入 `local/results_msl/`

### 8.2 运行前提

- Stage 1 已全部完成：20 个 partition 已生成于 `local/results_msl/partition/`（已验证样本数与设计一致）；
- Stage 2 / Stage 3 需要 GPU 环境。

### 8.3 执行命令（GPU 环境）

单数据集启动（每个脚本含该数据集全部 seed/折的 Stage1→Stage2→Stage3 + 一键汇总）：

```bash
nohup bash tools/single/launch_msl_uci_har.sh > "tools/single/uci_har_msl_$(date '+%Y%m%d_%H%M%S').log" 2>&1 &
nohup bash tools/single/launch_msl_iemocap.sh > "tools/single/iemocap_msl_$(date '+%Y%m%d_%H%M%S').log" 2>&1 &
nohup bash tools/single/launch_msl_mhealth.sh > "tools/single/mhealth_msl_$(date '+%Y%m%d_%H%M%S').log" 2>&1 &
nohup bash tools/single/launch_msl_pamap2.sh > "tools/single/pamap2_msl_$(date '+%Y%m%d_%H%M%S').log" 2>&1 &
```

四数据集串行（顺序执行各数据集脚本）：

```bash
nohup bash tools/serial/launch_msl_all.sh > "tools/serial/msl_all_$(date '+%Y%m%d_%H%M%S').log" 2>&1 &
```

四数据集并行（推荐，各数据集独立 Stage1→Stage2→Stage3）：

```bash
nohup bash tools/parallel/launch_msl_parallel.sh > "tools/parallel/main_$(date '+%Y%m%d_%H%M%S').log" 2>&1 &
```

脚本各阶段输出已存在时自动跳过（断点续跑）；日志写入 `local/results_msl/logs/`。

### 8.4 结果聚合

```bash
python scripts/summarize_results.py --results-root local/results_msl
```

输出：

- `local/results_msl/summary/<dataset>.json`：每个数据集每折/每 seed 的 test 指标（acc/macro_f1/weighted_f1）+ average；
- `local/results_msl/summary/summary.json`：四个数据集聚合总览。

### 8.5 注意事项

- 无验证集：无早停、无 best checkpoint 选择，固定 200 轮后使用 `last_model.pt` 在测试集评估一次；
- 当前阶段 `cluster_assignment_source=true_cluster`（oracle 上限）；正式无泄漏主线（`pred_cluster`）待聚类参数调优后切换，届时 pred_cluster 结果为主表、true_cluster 作为 oracle 上限对比；
- 每个 Stage 2 / Stage 3 输出目录防覆盖，重跑需清理对应目录或递增 `stage3.attempt`；
- 本仓库修改不自动提交，commit/push 由维护者确认后执行。
