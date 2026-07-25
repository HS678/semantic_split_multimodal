# Extension Guide

本项目采用 `src` layout，内部 import 统一使用 `semantic_split_multimodal...`。

## 新数据集适配器

数据集 loader 位于 `src/semantic_split_multimodal/data/datasets.py`，注册入口位于 `src/semantic_split_multimodal/data/registry.py`。loader 返回结构保持：

```python
{
    "train": {"modalities": [x0, x1, x2], "labels": y_train},
    "test": {"modalities": [x0_test, x1_test, x2_test], "labels": y_test},
    "root": str(root),
    "modality_names": ["mod0", "mod1", "mod2"],
    "modality_input_shapes": [[c0, t0], [c1, t1], [c2, t2]],
}
```

规则：

- 每个 modality tensor 的第 0 维必须与 labels 对齐。
- Stage 1 会独立切分每个模态，生成单模态 clients。
- 真实 modality name/id 只用于 discovery metrics 和 evaluation-only oracle mapping。
- 不要把真实模态数或真实模态名传入训练、binding、fusion slot 构造。

## Encoder

encoder 与 server 组件集中在 `src/semantic_split_multimodal/learning/models.py`。通过 registry 增加 encoder：

```yaml
model:
  encoder:
    type: time_series
```

已有类型：`time_series`、`image`、`video`、`audio`、`mlp`。

## 聚类

fingerprint 与 clustering 位于 `src/semantic_split_multimodal/discovery/`。支持 `kmeans`、`hdbscan`、`isodata`，`known_k: null` 时保持现有估计逻辑。

## Stage 3 约束

必须保持：

- scheduler 使用 `pred_cluster`。
- selected clients 在一个 global round 的所有 `local_steps` 内固定。
- 每个 local step 独立采样 batch。
- binding 使用 exact same-label random pseudo batch。
- fusion slot 严格由 `pred_cluster` 和 `cluster_to_slot` 决定。
- 主线只使用 CE loss、server backward、activation gradient routing、client optimizer update。
- empty binding 只跳过当前 local step；整轮失败时记录 `empty_binding_round`。

## 本地文件

数据、日志、结果、checkpoint 和本地论文全部放在 `local/`，不提交到 Git。
