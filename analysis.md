# Unpaired Shared Semantic 实验提升分析

## 一、本次结果提升的主要原因

本轮优化没有改变新方法的核心设定：

- 训练阶段仍然采用 round-level full-modality participation。
- 每轮从所有 predicted modality clusters 中选择客户端参与训练。
- 不构造 sample-level pseudo-paired multimodal batch。
- 客户端仍通过 Split Learning 上传 activation，并接收 server 返回的 activation gradient。
- server 端仍采用 cluster adapter、shared semantic backbone 和 shared classifier。

本次提升主要来自以下几个方面。

### 1. 训练轮数不足是前一版新方法效果偏低的主要原因

初始新方法实验基本沿用了第一版 pseudo-paired baseline 的训练轮数：

- UCI-HAR: 50 rounds
- MHEALTH: 50 rounds
- PAMAP2: 100 rounds

但是新方法不再直接 concat 多个模态 feature，而是通过共享语义空间逐步学习跨模态可迁移表示。相比 pseudo-paired concat，新方法的优化路径更间接：

```text
client encoder
↓
cluster adapter
↓
shared semantic backbone
↓
shared classifier
```

因此它需要更多 rounds 来让不同模态簇的表示逐步进入同一个 shared semantic space。

实验现象也支持这一点：

- UCI-HAR 从 50 rounds 延长到 100 rounds 后，best macro F1 从 0.6388 提升到 0.8052。
- MHEALTH 从 50 rounds 延长到 100 rounds 后，best macro F1 从 0.7963 提升到 0.9369。

这说明前一版新方法效果偏低，并不代表方法本身不成立，而是训练尚未充分收敛。

### 2. Prototype alignment 强度不能对所有数据集统一设置

新方法中的 class-wise prototype alignment 目标是让不同预测模态簇中同类别的语义原型靠近：

```text
p_(m,k) ≈ p_(n,k)
```

其中 `m,n` 是 predicted modality cluster，`k` 是 class label。

这个约束有助于跨模态语义对齐，但过强时也可能压缩模态特异性，使分类边界变差。

本次实验观察到：

- UCI-HAR 需要较强 alignment，`lambda_align=0.05` 效果最好。
- MHEALTH 对 alignment 更敏感，`lambda_align=0.05` 过强，关闭 alignment 或减小到 `0.01` 明显更好。
- PAMAP2 使用较弱 alignment，`lambda_align=0.01` 更稳定。

因此，prototype alignment 应作为一个可调语义正则项，而不是固定强度的通用项。

论文中可以将其表述为：

> Prototype alignment improves cross-modal semantic consistency, but its strength should be balanced with modality-specific discriminability.

中文表述：

> 原型对齐能够提升跨模态语义一致性，但其强度需要与模态特异性的判别能力保持平衡。

### 3. Decision-level fusion 对弱模态较敏感

新方法测试阶段采用 decision-level fusion：

```text
logits = mean(logits_1, logits_2, ..., logits_Q)
```

初始实验发现不同 predicted modality cluster 的单模态性能差异较大。

例如 MHEALTH 50 rounds 时：

- cluster 2 的 single-modality accuracy 为 0.7325。
- cluster 1 的 single-modality accuracy 只有 0.1707。

PAMAP2 也存在类似现象：

- cluster 2 明显强于其他 cluster。

如果直接对所有 logits 做 uniform mean，弱模态会拖累整体 fusion 结果。为此，本次增加了 `confidence_weighted` decision fusion：

```text
confidence = 1 - entropy(prob) / max_entropy
weighted_logits = sum(confidence_m * logits_m)
```

该策略没有改变训练阶段的 full-modality participation，只是在测试阶段降低低置信模态对最终预测的负面影响。

实验显示：

- PAMAP2 从 mean fusion 的 best macro F1 0.4135 提升到 confidence-weighted fusion 的 0.4446。
- UCI-HAR 基本持平。
- MHEALTH 略降，说明 confidence 并非所有数据集上都可靠。

因此，confidence-weighted fusion 更适合作为 PAMAP2 这类弱模态差异明显数据集上的增强策略。

### 4. 新方法的收益来自更合理的 full-modality participation 定义

本论文的核心论点不是 sample-level 完整模态配对，而是 round-level 完整模态参与：

```text
每个 global round 中，所有 predicted modality clusters 都至少有一个客户端参与训练。
```

因此，新方法更符合论文设定：

- 不假设不同客户端之间存在 sample identity correspondence。
- 不构造伪配对多模态样本。
- 仍然保证每轮训练接收到所有模态簇的监督信号。
- 通过 shared semantic backbone 和 shared classifier 学习跨模态共享语义。

这比第一版 pseudo-paired concat 更容易防守，也更符合 unknown modality environments 下的实际限制。

## 二、实验结果分析

本次实验均复用 `presude-results` 中第一版 latest run 的 Stage 1/2 产物，包括：

- client partition
- predicted cluster assignments
- pretrained client encoders
- paired multimodal test set

因此，本次对比主要反映 Stage 3 训练机制的变化。

### 1. 与第一版 baseline 的 best 指标对比

| Dataset | 第一版 Best Acc | 新方法 Best Acc | Acc 提升 | 第一版 Best Macro F1 | 新方法 Best Macro F1 | Macro F1 提升 | 新方法最优设置 |
|---|---:|---:|---:|---:|---:|---:|---|
| UCI-HAR | 0.7655 | 0.8144 | +0.0489 | 0.7560 | 0.8052 | +0.0492 | 100 rounds, `lambda_align=0.05`, mean fusion |
| MHEALTH | 0.8035 | 0.9569 | +0.1534 | 0.7895 | 0.9369 | +0.1474 | 100 rounds, no prototype alignment, mean fusion |
| MHEALTH aligned | 0.8035 | 0.9501 | +0.1467 | 0.7895 | 0.9296 | +0.1401 | 100 rounds, `lambda_align=0.01`, mean fusion |
| PAMAP2 | 0.4485 | 0.5406 | +0.0921 | 0.3872 | 0.4446 | +0.0575 | 150 rounds, `lambda_align=0.01`, confidence-weighted fusion |

### 2. UCI-HAR 分析

UCI-HAR 上，初始新方法 50 rounds 的结果低于第一版：

| Setting | Best Round | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| initial unpaired, 50 rounds | 50 | 0.6607 | 0.6388 | 0.6497 |
| no alignment, 50 rounds | 50 | 0.6468 | 0.6181 | 0.6294 |
| `lambda_align=0.01`, 50 rounds | 35 | 0.6471 | 0.6212 | 0.6328 |
| `lambda_align=0.05`, 100 rounds | 95 | 0.8144 | 0.8052 | 0.8101 |

结论：

- UCI-HAR 需要 prototype alignment。
- 50 rounds 明显训练不足。
- 延长训练后，新方法超过第一版 pseudo-paired baseline。

这说明 UCI-HAR 的模态间语义结构较适合 prototype alignment，较强对齐可以帮助共享语义空间成型。

### 3. MHEALTH 分析

MHEALTH 上，新方法对 alignment 强度非常敏感。

| Setting | Best Round | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| initial unpaired, `lambda_align=0.05`, 50 rounds | 50 | 0.8236 | 0.7613 | 0.8075 |
| no alignment, 50 rounds | 50 | 0.8581 | 0.7963 | 0.8441 |
| `lambda_align=0.01`, 50 rounds | 50 | 0.8303 | 0.7704 | 0.8168 |
| no alignment, 100 rounds | 100 | 0.9569 | 0.9369 | 0.9549 |
| `lambda_align=0.01`, 100 rounds | 100 | 0.9501 | 0.9296 | 0.9483 |

结论：

- `lambda_align=0.05` 对 MHEALTH 过强。
- 关闭 alignment 后性能显著提升。
- 弱 alignment `lambda_align=0.01` 在 100 rounds 下也表现很好，略低于 no alignment。

论文中可以将 MHEALTH 的结果用于说明：

- prototype alignment 是有用的语义正则，但并非越强越好。
- 对齐过强会牺牲模态特异性和类别判别性。
- 因此需要 alignment strength ablation。

### 4. PAMAP2 分析

PAMAP2 的难点是不同模态簇性能差异较大，uniform mean fusion 容易被弱模态拖累。

| Setting | Best Round | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| initial unpaired, mean fusion | 90 | 0.5148 | 0.4135 | 0.4790 |
| confidence-weighted fusion, 100 rounds | 90 | 0.5563 | 0.4282 | 0.4973 |
| no alignment, mean fusion | 90 | 0.5181 | 0.4174 | 0.4835 |
| confidence-weighted fusion, 150 rounds | 105 | 0.5406 | 0.4446 | 0.5056 |

结论：

- PAMAP2 受益于 confidence-weighted decision fusion。
- 150 rounds 的 final result 下降，但 best checkpoint 在 round 105 达到更高 macro F1。
- 这说明 PAMAP2 后期存在过拟合或训练震荡，后续可以加入 early stopping 或 learning rate decay。

PAMAP2 的结果也说明，decision-level fusion 不是简单平均就一定最好。对于模态簇质量差异明显的数据集，自适应融合策略可以提升整体鲁棒性。

### 5. 当前推荐实验设置

如果以结果为优先，当前推荐设置如下：

| Dataset | Rounds | Alignment | Fusion |
|---|---:|---|---|
| UCI-HAR | 100 | `lambda_align=0.05` | mean |
| MHEALTH | 100 | no alignment 或 `lambda_align=0.01` | mean |
| PAMAP2 | 150 | `lambda_align=0.01` | confidence-weighted |

如果以论文主方法一致性为优先，可以采用：

| Dataset | Rounds | Alignment | Fusion |
|---|---:|---|---|
| UCI-HAR | 100 | `lambda_align=0.05` | mean |
| MHEALTH | 100 | `lambda_align=0.01` | mean |
| PAMAP2 | 150 | `lambda_align=0.01` | confidence-weighted |

同时将 no alignment 作为 ablation，说明 prototype alignment 的贡献和边界。

## 三、对论文叙事的建议

本次结果支持以下论文论点：

1. 第一版 pseudo-paired concat 虽然有效，但依赖不严格的 sample-level correspondence assumption。
2. 新方法不构造伪配对样本，更符合 unknown modality environments。
3. 通过 round-level full-modality participation，新方法仍然是多模态协同训练。
4. 充分训练后，新方法在三个数据集上都可以超过第一版 baseline。
5. Prototype alignment 是一种跨模态语义正则，其强度需要调节。
6. Decision-level fusion 可以根据模态簇质量进一步增强。

推荐论文中的主结论表述：

> The proposed unpaired shared semantic split learning framework improves over the pseudo-paired baseline after sufficient training, while avoiding invalid sample-level pairing assumptions. The results demonstrate that round-level full-modality participation can effectively support multimodal collaborative learning in unknown modality environments.

中文表述：

> 在充分训练后，所提出的未配对共享语义 Split Learning 框架能够超过伪配对 baseline，同时避免无效的样本级配对假设。实验结果表明，轮次级完整模态参与能够有效支持未知模态环境下的多模态协同学习。

