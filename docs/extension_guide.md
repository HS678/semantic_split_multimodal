# Extension Guide

本项目采用 `src` layout，内部 import 统一使用 `semantic_split_multimodal...`。

## 新数据集适配器

数据集 loader 位于 `src/semantic_split_multimodal/data/datasets.py`，注册入口位于 `src/semantic_split_multimodal/data/registry.py`。loader 返回结构保持：

```python
{
    "train": {"modalities": [x0, x1, x2], "modality_lengths": [l0, l1, l2], "labels": y_train},
    "validation": {"modalities": [x0_val, x1_val, x2_val], "labels": y_val},
    "test": {"modalities": [x0_test, x1_test, x2_test], "labels": y_test},
    "root": str(root),
    "modality_names": ["mod0", "mod1", "mod2"],
    "modality_input_shapes": [[c0, t0], [c1, t1], [c2, t2]],
    "modality_encoder_types": ["conv_gru", "gru", "gru"],
}
```

现有 loader 包括 `uci_har`、`mhealth`、`pamap2`、`cmu_mosei` 和 `iemocap`。IEMOCAP 的 loader 独立位于 `data/iemocap.py`；可变长序列通过可选 `modality_lengths` 传递。CMU-MOSEI 与 IEMOCAP 的数据适配逻辑均不改变 Stage2/Stage3 核心算法。

规则：

- 每个 modality tensor 的第 0 维必须与 labels 对齐。
- train/validation/test 应优先采用互斥的 subject-level 划分。
- Stage 1 只会切分 train 的每个模态生成单模态 clients；validation/test 保持 naturally paired。
- 预处理统计量只能从 train 拟合并应用到 validation/test。
- 真实 modality name/id 只用于 discovery audit 和无梯度 validation/test evaluation-only oracle mapping。
- 不要把真实模态数或真实模态名传入训练、binding、fusion slot 构造。

## Encoder

encoder 与 server 组件集中在 `src/semantic_split_multimodal/learning/models.py`。通过 registry 增加 encoder：

```yaml
model:
  encoder:
    type: time_series
```

已有类型：`time_series`、`image`、`video`、`audio`、`mlp`、`gru`、`conv_gru`。

## 聚类

fingerprint 与 clustering 位于 `src/semantic_split_multimodal/discovery/`。当前只保留：

- `kmeans`
- `adaptive_isodata`

新增聚类算法前，需要同步：

- `discovery/clustering.py`
- `learning/pretrain.py`
- `scripts/stage2_discovery.py`
- Stage 2 输出审计测试
- README 和 `docs/output_reference.md`

## Stage 3 约束

必须保持：

- scheduler 使用 `pred_cluster`。
- selected clients 在一个 global round 的所有 `local_steps` 内固定。
- 每个 local step 独立采样 batch。
- binding 使用 exact same-label random pseudo batch。
- fusion slot 严格由 `pred_cluster` 和固定 `cluster_to_slot` 决定。
- 主线只使用 CE loss、server backward、activation gradient routing、client optimizer update。
- empty binding 只跳过当前 local step；整轮失败时记录 `empty_binding_round`。

## 本地文件

数据、日志、结果、checkpoint 和本地论文全部放在 `local/`，不提交到 Git。
