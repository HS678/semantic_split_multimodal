# 五数据集正式结果汇总

> 所有数值由 `final_metrics.json` 自动读取；缺失实验不会生成占位指标。

| 数据集 | 状态 | Accuracy | Balanced Acc/UA | Macro-F1 | Weighted-F1 | Binary-F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| uci_har | complete | 0.8751 | 0.8710 | 0.8711 | 0.8711 | — |
| mhealth | complete | 0.9492 | 0.9461 | 0.9419 | 0.9426 | — |
| pamap2 | complete | 0.4156 | 0.4113 | 0.3515 | 0.3737 | — |
| iemocap (5-fold) | complete | 0.5056 ± 0.0515 | 0.5077 ± 0.0555 | 0.4934 ± 0.0565 | 0.4953 ± 0.0553 | — |

## IEMOCAP 折状态

- 已完成 test Session：[1, 2, 3, 4, 5]
- 缺失 test Session：[]

## 运行记录

- `uci_har`：`/home/shuang/testWorkspace/semantic_split_multimodal/local/results/experiments/oracle_true_cluster/uci_har/enc-temporal_conv_gru__bind-label_random__loss-mmbind_weighted_contrastive__sched-balanced_cluster_round_robin__h-cec526fabb/seed-101/attempt-01`
- `mhealth`：`/home/shuang/testWorkspace/semantic_split_multimodal/local/results/experiments/oracle_true_cluster/mhealth/enc-temporal_conv_gru__bind-label_random__loss-mmbind_weighted_contrastive__sched-balanced_cluster_round_robin__h-0029ade1b6/seed-101/attempt-01`
- `pamap2`：`/home/shuang/testWorkspace/semantic_split_multimodal/local/results/experiments/oracle_true_cluster/pamap2/enc-temporal_conv_gru__bind-label_random__loss-mmbind_weighted_contrastive__sched-balanced_cluster_round_robin__h-6c7253f9e3/seed-101/attempt-01`
- `iemocap`：`/home/shuang/testWorkspace/semantic_split_multimodal/local/results/experiments/oracle_true_cluster/iemocap/enc-gru-mfcc_mobilevit_xs_distilbert_v1__bind-label_random__loss-mmbind_weighted_contrastive__sched-balanced_cluster_round_robin__h-b9c9c608ad/seed-101/attempt-01`
- `iemocap`：`/home/shuang/testWorkspace/semantic_split_multimodal/local/results/experiments/oracle_true_cluster/iemocap/enc-gru-mfcc_mobilevit_xs_distilbert_v1__bind-label_random__loss-mmbind_weighted_contrastive__sched-balanced_cluster_round_robin__h-c38868cb5d/seed-101/attempt-01`
- `iemocap`：`/home/shuang/testWorkspace/semantic_split_multimodal/local/results/experiments/oracle_true_cluster/iemocap/enc-gru-mfcc_mobilevit_xs_distilbert_v1__bind-label_random__loss-mmbind_weighted_contrastive__sched-balanced_cluster_round_robin__h-5a30e880bd/seed-101/attempt-01`
- `iemocap`：`/home/shuang/testWorkspace/semantic_split_multimodal/local/results/experiments/oracle_true_cluster/iemocap/enc-gru-mfcc_mobilevit_xs_distilbert_v1__bind-label_random__loss-mmbind_weighted_contrastive__sched-balanced_cluster_round_robin__h-39cbea5d8d/seed-101/attempt-01`
- `iemocap`：`/home/shuang/testWorkspace/semantic_split_multimodal/local/results/experiments/oracle_true_cluster/iemocap/enc-gru-mfcc_mobilevit_xs_distilbert_v1__bind-label_random__loss-mmbind_weighted_contrastive__sched-balanced_cluster_round_robin__h-6d77c019a2/seed-101/attempt-01`
