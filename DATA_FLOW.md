# DATA_FLOW

## 1) Stage 1 聚类数据流（当前真实实现）
### 流程
`run_stage2_training.py`  
→ `Stage2Trainer.__init__` 创建所有 `SplitClient`  
→ `Stage2Trainer.cluster_clients()`  
→ 每个 client 调用 `SplitClient.cluster_representation()`（`clients/client_node.py`）  
→ `run_kmeans()`（`clustering/kmeans_cluster.py`）  
→ `evaluate_clustering()`（同文件）  
→ 生成 GT 池 / KMeans 池 / KMeans 映射模态池供调度器使用

### 当前 KMeans 输入到底是什么
- 输入是“每个 client 的 encoder 参数向量”：
  - `cluster_representation()` 将 `client.encoder.parameters()` 展平并拼接。
- 不是原始样本统计特征，也不是训练后中间 embedding。
- 由于 `cluster_clients()` 在 Stage 2 正式训练前调用，通常对应“初始化状态 encoder 参数表示”。

## 2) Stage 2 训练数据流（当前真实实现）
### 从数据到 server loss 的完整链路
`dataset`（synthetic/real/uci_har）  
→ `_prepare_dataset()`（`experiments/run_stage2_training.py`）  
→ `build_client_pool()`（`data/synthetic_dataset.py`）把 train paired 数据切为单模态客户端  
→ `SplitClient.sample_batch()`（`clients/client_node.py`，类均衡采样）  
→ `SplitClient.forward_to_server()`：
- `z_client = encoder(x)`
- `z_server = z_client.detach().requires_grad_(True)`  
→ `SemanticBatchBuilder.build()`（`server/server_core.py`）按 label 取各 client 交集标签并构造 `index_map`  
→ `SplitServer.train_step()`（`server/server_core.py`）：
- 每模态 `z_server[idx]` → `SemanticProjector`
- 得到 `stacked [B, M, D]`
- `SupervisedContrastiveLoss` 计算对齐损失
- `ConcatFusion` 融合
- `ClassifierHead` 分类
- `CE + lambda_align*align + lambda_proto*proto` 得总损失  
→ `loss.backward()`

## 3) Split learning 梯度流（当前真实实现）
### 协议链路
`server loss.backward()`  
→ 每个 client 原始 `z_server.grad` 产生  
→ `SplitServer._scatter_grad()` 按 `index_map` 回填 full-batch 形状梯度  
→ `grad_to_clients[cid]` 返回 trainer  
→ `SplitClient.backward_update(z_client, grad)`  
→ `z_client.backward(grad)`  
→ `optimizer.step()` 更新 client encoder

### 对应代码位置
- 客户端创建 `z_server`：`clients/client_node.py` `forward_to_server`
- semantic batch 构造：`server/server_core.py` `SemanticBatchBuilder.build`
- server backward 与梯度路由：`server/server_core.py` `SplitServer.train_step`
- 客户端反向更新：`clients/client_node.py` `backward_update` / `backward_from_server`

### 关键协议事实
- 当前实现没有对 `z_server[idx]` 做二次 `detach().requires_grad_()`，仅做索引选择。
- 若无 common labels，`train_step` 返回 `None`，该 local step 计为 skipped。

## 4) Evaluation 测试数据流（当前真实实现）
### 流程
`Stage2Trainer.run()` 每个 global round 后  
→ `evaluate_paired_test()`（`server/evaluation.py`）  
→ 直接读取 paired multimodal test set：`test_set["modalities"]` 与 `test_set["labels"]`  
→ 第 `m` 个模态测试张量输入第 `m` 个模态 encoder（当前是 `clients_by_modality[m][0]` 作为该模态 encoder 代理）  
→ server projector/fusion/classifier  
→ `pred` 与 `y` 计算 `acc`、`macro_f1`

### 与“是否用 label 配对/筛选”相关
- 测试阶段不使用 label 做配对或筛选；配对关系来自 test set 本身的样本对齐。
- label 仅用于最终指标计算（accuracy / macro-F1）。

## 5) 当前实现边界（与文档审计相关）
- 调度器主用 `FairRandomFullModalityScheduler`。
- 若配置 `clustering.use_oracle_clusters_for_training=false`，训练池来自 KMeans 映射模态池；如果某模态池为空会显式报错。
- Stub 组件存在但不参与主训练：ISODATA、PairedFullModalityScheduler、GlobalRandomScheduler、AttentionFusion、D2D。
