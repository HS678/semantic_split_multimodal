# Codex Instructions

## 1. Communication Rules

- 所有解释、计划、问题、总结均使用中文。
- 命令、代码、路径、配置字段、错误信息保持原文。
- 修改任何文件前必须说明：
  - 修改哪些文件；
  - 修改原因；
  - 影响范围。
- 如果存在多个方案，先说明方案差异。
- 不确定时必须询问，不允许自行改变实验设计。
- 不进行无关的大规模重构。

---

# 2. Project Identity

## Project

Semantic-aligned Distributed Split Multimodal Learning in Unknown Modality Environments

## Current Mainline

当前唯一正式实验路线：

```
MMBind-style Fusion Split Learning
```

代码实现：

```
mmbind_fusion_split_learning
```

ing

核心流程：

```
Modality Discovery

↓

Cluster-aware Scheduling

↓

Label-guided Semantic Pseudo Binding

↓

ClusterAdapter + Concat Fusion

↓

Split Learning

↓

Naturally Paired Evaluation
```

当前目标：

验证未知模态环境下，通过：

- 模态发现；
- 聚类调度；
- 语义伪绑定；
- Split Learning；

实现分布式多模态训练。

未经明确要求：

禁止：

- 恢复旧实验路线；
- 引入废弃模型；
- 修改论文核心方法。

---

# 3. Algorithm Constraints

未经明确要求，不允许修改以下核心算法协议。

## 3.1 Modality Discovery

当前Stage2流程：

```
Client Encoder Pretraining

↓

Fingerprint Extraction

↓

Clustering

↓

pred_cluster
```

训练流程使用：

```
pred_cluster
```

禁止使用：

```
true modality identity
```

hidden_modality_id：

只能用于：

- discovery metrics；
- evaluation audit。

禁止用于：

- fingerprint构造；
- clustering input；
- training scheduling。

## 3.2 Cluster-aware Scheduling

当前正式调度策略：

```
predicted-cluster coverage scheduling
```

要求：

scheduler依据：

```
pred_cluster
```

选择客户端。

目标：

每轮训练覆盖所有预测模态cluster。

禁止：

- 使用真实模态信息调度；
- 使用oracle scheduler作为主方法；
- 使用随机调度替代正式方法。

---

## 3.3 Semantic Pseudo Binding

当前正式binding方法：

```
label-guided semantic pseudo binding
```

作用：

利用标签提供语义一致性约束，构造训练阶段pseudo multimodal tuple。

注意：

label不是：

- 真实样本身份；
- 真实跨模态pair；
- 模态匹配信息。

label仅用于：

- 提供语义一致性；
- 辅助训练阶段pseudo binding。

禁止：

- 使用hidden_modality_id；
- 使用true modality name；
- 使用instance-level natural pairing。

---

## 3.4 Fusion Architecture

当前正式融合结构：

```
ClusterAdapter

+

Concat Fusion

+

Classifier
```

禁止随意修改：

- ClusterAdapter；
- fusion slot定义；
- cluster-to-slot mapping；
- concat fusion方式；
- classifier输入结构。

---

## 3.5 Split Learning

保持当前Split Learning流程。

客户端：

```
input

↓

encoder

↓

feature upload
```

服务器：

```
feature aggregation

↓

fusion

↓

classifier

↓

loss

↓

backward
```

客户端：

```
receive gradient

↓

encoder update
```

禁止修改：

- feature传输流程；
- server forward；
- server backward；
- gradient return逻辑；
- client update流程。

---

## 3.6 Evaluation Protocol

正式测试协议：

```
naturally paired multimodal evaluation
```

要求：

- 使用test_multimodal.pt；
- 使用自然配对多模态测试数据；
- label仅用于loss和metrics；
- oracle mapping仅用于evaluation。

禁止：

- 使用测试label构造输入；
- 使用hidden_modality_id参与训练；
- 修改evaluation协议。

---


## 3.7 No Modality Leakage

除evaluation audit外：

禁止任何训练流程读取：

- hidden_modality_id；
- true modality name；
- true modality count。

所有训练阶段模态信息必须来自：

- fingerprint；
- clustering；
- pred_cluster。

# 4. Coding Rules

## General Principles

修改代码时：

优先：

- 最小修改；
- 保持现有项目结构；
- 保持实验协议；
- 保持已有接口。

禁止：

- 无关重构；
- 引入未经验证的新依赖。

## Before Code Modification

修改代码前必须说明：

1. 修改文件：

例如：

```
src/MSL/learning/fusion_sl.py
```

2. 修改原因：

说明：

- 当前问题；
- 修改目标。

3. 影响范围：

说明：

- 是否影响训练；
- 是否影响evaluation；
- 是否影响已有实验结果。

---

## New Code Rules

新增代码必须说明：

- 文件位置；
- 模块职责；
- 与现有代码关系。

新增模块禁止：

- 重复已有功能；
- 引入旧路线代码；
- 破坏当前主线结构。

---

## Model and Training Rules

未经明确要求：

禁止修改：

- 模型结构；
- loss；
- optimizer；
- scheduler；
- learning rate策略；
- binding方式；
- fusion方式；
- evaluation协议。

如果修改可能影响实验结果：

必须先说明。

---

# 5. Experiment Rules

正式实验必须保证：

- 配置可追溯；
- 结果可复现；
- checkpoint可加载；
- evaluation协议一致。

实验结果统一保存：

```
local/results_msl/   # 当前正式方案
```

正式实验至少保存：

- config；
- metrics；
- checkpoint；
- evaluation结果。

禁止：

- 覆盖已有正式实验结果；
- 创建无法对应配置的实验目录；
- 分散保存checkpoint。

---

## Experiment Modification Rules

修改以下内容前必须说明：

- 数据划分；
- 模型结构；
- loss；
- optimizer；
- scheduler；
- binding；
- fusion；
- evaluation。

如果只是：

- 修复路径；
- 修复接口；
- 修复明显bug；

需要说明影响范围。

---

# 6. Validation Rules

代码修改后，根据影响范围执行验证。

## Basic Validation

至少执行：

```
compile/import check
```

例如：

```
python -m compileall
```

---

## Test Validation

涉及代码修改：

运行：

```
pytest
```

或者相关测试：

```
tests/
```

---

## Experiment Validation

涉及实验流程：

检查：

- config loading；
- script help；
- smoke run。

---

## Training Validation

涉及训练代码：

检查：

- training是否正常运行；
- checkpoint save；
- checkpoint reload；
- evaluation。

---

# 7. Git Rules

## Forbidden Automatic Actions

禁止自动执行：

```
git commit
git push
git reset
git checkout
```

禁止：

- 删除branch；
- 强制覆盖历史；
- 大规模删除文件。

---

## Require Confirmation

执行以下操作前必须询问：

- git commit；
- git push；
- 删除文件；
- 大规模移动文件。

必须说明：

- 原因；
- 风险；
- 影响范围。

---

## Allowed Operations

允许：

```
git status

git diff

git log
```

用于：

- 查看状态；
- 检查修改；
- 分析历史。

---

# 8. File Operation Rules

删除文件前：

必须确认：

- 文件用途；
- 是否被引用；
- 是否存在替代版本。

禁止直接执行：

```
rm -rf
```

删除大量目录。

移动文件时检查：

- import路径；
- 配置路径；
- 文档引用；
- 测试引用。

---

# 9. Completion Report

每次任务完成后必须使用中文总结。

必须包括：

## 修改文件

列出：

- 新增文件；
- 修改文件；
- 删除文件。

格式：

```
新增：
- xxx

修改：
- xxx

删除：
- xxx
```

---

## 修改内容

说明：

- 做了什么；
- 为什么修改；
- 是否影响算法协议；
- 是否影响实验结果。

---

## Validation

说明：

执行命令：

例如：

```
python scripts/stage3_train.py --config xxx.yaml
```

测试结果：

例如：

```
pytest:
passed

stage3:
success
```

---

## Risks

说明：

- 潜在风险；
- 是否需要人工确认；
- 是否可能影响已有实验。

---

## Git Commit Suggestion

如果修改tracked文件：

最后提供：

建议 commit message：

```
<commit message>
```

并说明：

- 提交目的；
- 修改范围；
- 是否影响实验协议。

不要自动执行：

```
git commit
```

---

# End of Codex Instructions
