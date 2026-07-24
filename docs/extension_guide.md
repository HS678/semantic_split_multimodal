# Extension Guide

本项目主线为未知模态发现 + 簇覆盖调度 + MMBind-style label binding + fusion Split Learning。旧的未配对 shared semantic trainer 保留为 baseline/ablation。

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
- 训练阶段禁止访问 `hidden_modality_id` 和真实 modality name。
- fusion slot 严格使用 `pred_cluster`。
- 允许 label-level semantic binding，但不存在 instance-level correspondence。
- Phase 1 使用 anchor-based same-label random pairing。
- pseudo sample 必须覆盖所有 `pred_cluster` slot，否则丢弃该 binding。
- Phase 1 只使用 concat MLP fusion 和分类损失。
- Phase 1 不使用 weighted contrastive loss、prototype alignment、missing modality mask。

## D2D

当前 Phase 1 暂不实现 D2D。后续真实 D2D 协作应作为独立模块接入，消费 client/server 元数据和通信/计算 latency profile。
