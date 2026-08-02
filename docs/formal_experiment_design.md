# 五数据集正式实验设计

## 不变的主线协议

五个数据集均使用 `MMBind-style Fusion Split Learning`：Stage2 模态发现、簇覆盖调度、
label-guided semantic pseudo binding、`ClusterAdapter + Concat Fusion`、Split Learning，
最终使用自然配对的 `test_multimodal.pt`。当前开发实验按已确认方案固定
`training.cluster_assignment_source=true_cluster`；Stage2 仍输出 PCA 图和预测簇审计结果。
D2D 仅保留配置入口，不进入本轮精度实验。

模型优化只发生在单模态 encoder、预训练目标和训练稳定性参数，不读取
`hidden_modality_id`、真实模态名或实例级跨模态配对。Stage2 的 classification 预训练只使用
每个单模态客户端自身的训练标签。

## 数据集设计

| 数据集 | 单模态 encoder | 数据划分 | 正式主指标 | 补充指标 |
| --- | --- | --- | --- | --- |
| UCI-HAR | 3 层 Temporal CNN + 2 层 BiGRU + masked attention | 既有 subject-disjoint train/validation/test | weighted-F1 | accuracy、macro-F1、UA |
| MHEALTH | 3 层 Temporal CNN + 2 层 BiGRU + masked attention | 既有 subject-disjoint train/validation/test | weighted-F1 | accuracy、macro-F1、UA、per-class F1 |
| PAMAP2 | 3 层较宽 Temporal CNN + 2 层 BiGRU + masked attention | 既有 subject-disjoint train/validation/test | weighted-F1 | accuracy、macro-F1、UA、per-class F1 |
| CMU-MOSEI | normalized MLP + GELU + dropout | 官方 video-disjoint train/validation/test | binary-F1 与 accuracy | weighted-F1、macro-F1、混淆矩阵 |
| IEMOCAP | audio: Conv-GRU；video/text: 2 层 BiGRU；均使用 masked attention；视频输入为冻结 MobileViT-XS 帧特征 | 5-fold Session LOSO；每折一个完整 Session test；validation 从其余四个 Session 内按 dialog_id 分组产生 | 5 折 WA/accuracy、UA/macro-recall、macro-F1 的均值和标准差 | weighted-F1、per-class F1、聚合混淆矩阵 |

## 正式配置

- `configs/formal/uci_har.config`
- `configs/formal/mhealth.config`
- `configs/formal/pamap2.config`
- `configs/formal/cmu_mosei.config`
- `configs/formal/iemocap_fold1.config` 至 `iemocap_fold5.config`

所有配置使用 `fusion.training_objective=mmbind_weighted_contrastive`，并保存源配置、解析后配置、
Stage2 指纹/PCA/聚类审计、checkpoint、validation 曲线和一次正式 test 结果。

## 测试隔离和执行顺序

1. compile、pytest 和短 smoke 只验证代码路径；smoke 配置必须设置 `evaluation.run_test=false`。
2. 参数选择只读取 train 与 naturally paired validation 指标，禁止读取 test 指标后回调参数。
3. 参数冻结后将 `evaluation.run_test=true`，每个正式 checkpoint 只执行一次 test。
4. IEMOCAP 不挑选“最好的一折”，必须汇报全部五折的均值、标准差和聚合混淆矩阵。
5. 结果低于预期时如实保留，不打印、补写或手工构造任何指标。

## 当前运行状态（2026-08-02）

| 数据集 | Stage2 | Stage3/test | Test accuracy | Test weighted-F1 | 说明 |
| --- | --- | --- | ---: | ---: | --- |
| UCI-HAR | `Q=2, ACC/NMI/ARI=1.0` | complete | 0.8751 | 0.8711 | 第 120 轮早停，best round 60 |
| MHEALTH | `Q=4, ACC/NMI/ARI=1.0` | complete | 0.9492 | 0.9426 | signal fingerprint，best round 240 |
| PAMAP2 | `Q=3, ACC/NMI/ARI=1.0` | complete | 0.4156 | 0.3737 | validation-test subject shift 明显，结果保留且不按 test 回调参数 |
| CMU-MOSEI | `Q=2, ACC=0.6667`，discovery failure | complete | 0.8509 | 0.8478 | Stage3 按当前方案使用 true cluster；binary-F1=0.8977 |
| IEMOCAP | fold1 Stage1 complete | incomplete | — | — | GPU 执行额度阻塞；fold1 Stage2 与全部五折 Stage3 尚未运行 |

机器可读和 Markdown 汇总由以下命令从真实 `final_metrics.json` 生成：

```bash
python scripts/summarize_formal_results.py
```

输出为 `local/results/formal_summary.json` 与 `local/results/formal_summary.md`。IEMOCAP 未完成时，
脚本只写 `missing_test_sessions`，不会生成占位精度。
