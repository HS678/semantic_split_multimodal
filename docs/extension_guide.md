# Extension Guide

本项目现在只维护未知模态发现 + 未配对 Split Learning 主线。

## 新数据集适配器

在 `data/` 下新增 adapter，并在 `data/dataset_registry.py` 注册。loader 返回：

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
- 真实 modality name/id 只用于评估，不用于训练调度。
- 不要添加 input dimension hint 或按真实模态数固定聚类。

## Encoder

只通过 `models/encoders.py` 的 registry 增加 encoder：

```yaml
model:
  encoder:
    type: time_series
```

已有类型：`time_series`、`image`、`video`、`audio`、`mlp`。

## 聚类

主实验建议：

```yaml
fingerprint:
  type: hybrid

cluster:
  method: isodata
  known_k: null
  isodata:
    initial_k: 3
    min_clusters: 1
    max_clusters: null
```

可替换为 `kmeans` 或 `hdbscan`。`kmeans` 在 `known_k: null` 时会用 silhouette 在小范围内估计 K。

## Stage 3 约束

必须保持：

- 每个 client 独立采样 batch。
- scheduler 使用 `pred_cluster`。
- 不做 feature concat。
- 不做 sample alignment。
- 不按标签构造伪多模态样本。
- 分类损失加 prototype alignment 损失。

## D2D

当前只记录 latency/speedup 指标占位。真实 D2D 协作应作为独立模块接入，消费 client/server 元数据和通信/计算 latency profile，不要重新引入样本级融合。
