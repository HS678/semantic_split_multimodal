# 论文正式实验冻结协议常量与纯查询函数。

# 聚类参数（adaptive_isodata，四个数据集一致，内置）。
ADAPTIVE_ISODATA_PROTOCOL = {
    "seeds": [11, 23, 37, 53, 71],
    "max_iter": 20,
    "pca_variance": 0.95,
    "variance_epsilon": 1e-08,
    "q_max": 8,
    "min_cluster_size": 2,
    "min_cluster_size_fraction": None,
    "bic_improvement_min": 0.0,
    "min_split_silhouette": 0.2,
    "silhouette_patience": 2,
    "stability_min_ari": 0.75,
    "split_kmeans_restarts": 50,
}

FORMAL_SEEDS = (42, 123, 2025, 3407, 7777)


DATASET_PROTOCOLS = {
    "uci_har": {
        "num_classes": 6,
        "dataset": {
            "root": "./local/datasets/UCI-HAR",
            "split_protocol": "subject_disjoint_70_30",
        },
        "fold_count": None,
        "global_rounds": 200,
        "pretrain": {
            "objective": "classification",
            "epochs": 5,
            "batch_size": 64,
            "lr": 0.0005,
            "weight_decay": 0.0001,
            "class_weighting": "none",
            "max_grad_norm": 5.0,
        },
        "fingerprint_type": "hybrid",
        "encoder": {
            "type": "temporal_conv_gru",
            "conv_channels": [64, 128, 128],
            "kernel_sizes": [7, 5, 3],
            "gru_hidden_dim": 128,
            "gru_layers": 2,
            "bidirectional": True,
            "pooling": "attention",
            "dropout": 0.2,
        },
        "encoders": {},
        "training": {
            "clients_per_round": 4,
            "batch_size": 64,
            "eval_batch_size": 128,
            "client_lr": 0.0002,
            "server_lr": 0.0002,
            "class_weighting": "none",
        },
        "binding_batch_size": 64,
        "fusion_dropout": 0.2,
        "mmbind": {
            "temperature": 0.1,
            "contrastive_weight": 0.02,
            "heterogeneous_ce_weight": 0.1,
        },
        "cluster_adaptive": {
            "min_split_silhouette": 0.1,
        },
    },
    "iemocap": {
        "num_classes": 4,
        "dataset": {
            "name": "iemocap",
            "variant": "full",
            "root": "./local/datasets/IEMOCAP/IEMOCAP_full/IEMOCAP_full_release",
            "processed_root": "./local/datasets/IEMOCAP/processed/mfcc_mobilevit_xs_distilbert_v1",
            "feature_recipe": "mfcc_mobilevit_xs_distilbert_v1",
            "task": "emotion_4class",
            "label_protocol": "ang_hap_exc_sad_neu_v1",
            "split_protocol_template": "session_5fold_loso_fold{fold}",
        },
        "fold_count": 5,
        "global_rounds": 300,
        "pretrain": {
            "objective": "classification",
            "epochs": 5,
            "batch_size": 32,
            "lr": 0.0002,
            "weight_decay": 0.0001,
            "class_weighting": "inverse_sqrt",
            "max_grad_norm": 5.0,
        },
        "fingerprint_type": "hybrid",
        "encoder": {
            "type": "gru",
            "gru_hidden_dim": 256,
            "gru_layers": 2,
            "bidirectional": True,
            "pooling": "attention",
            "dropout": 0.2,
        },
        "encoders": {
            "conv_gru": {
                "conv_channels": [64, 128, 256],
                "kernel_size": 5,
                "gru_hidden_dim": 256,
                "gru_layers": 2,
                "bidirectional": True,
                "pooling": "attention",
                "dropout": 0.2,
            },
            "gru": {
                "gru_hidden_dim": 256,
                "gru_layers": 2,
                "bidirectional": True,
                "pooling": "attention",
                "dropout": 0.2,
            },
        },
        "training": {
            "clients_per_round": 6,
            "batch_size": 32,
            "eval_batch_size": 64,
            "client_lr": 0.0001,
            "server_lr": 0.0001,
            "class_weighting": "inverse_sqrt",
        },
        "binding_batch_size": 32,
        "fusion_dropout": 0.3,
        "mmbind": {
            "temperature": 0.1,
            "contrastive_weight": 0.03,
            "heterogeneous_ce_weight": 0.15,
        },
        "cluster_adaptive": {
            "bic_improvement_min": 15.0,
        },
    },
    "mhealth": {
        "num_classes": 12,
        "dataset": {
            "root": "./local/datasets/MHEALTH",
            "split_protocol_template": "subject_5fold_fold{fold}",
        },
        "fold_count": 5,
        "global_rounds": 200,
        "pretrain": {
            "objective": "classification",
            "epochs": 5,
            "batch_size": 64,
            "lr": 0.0005,
            "weight_decay": 0.0001,
            "class_weighting": "inverse_sqrt",
            "max_grad_norm": 5.0,
        },
        "fingerprint_type": "signal",
        "encoder": {
            "type": "temporal_conv_gru",
            "conv_channels": [64, 128, 128],
            "kernel_sizes": [7, 5, 3],
            "gru_hidden_dim": 128,
            "gru_layers": 2,
            "bidirectional": True,
            "pooling": "attention",
            "dropout": 0.15,
        },
        "encoders": {},
        "training": {
            "clients_per_round": 8,
            "batch_size": 64,
            "eval_batch_size": 128,
            "client_lr": 0.0002,
            "server_lr": 0.0002,
            "class_weighting": "inverse_sqrt",
        },
        "binding_batch_size": 64,
        "fusion_dropout": 0.15,
        "mmbind": {
            "temperature": 0.1,
            "contrastive_weight": 0.1,
            "heterogeneous_ce_weight": 0.5,
        },
        "cluster_adaptive": {
            "bic_improvement_min": 30.0,
        },
    },
    "pamap2": {
        "num_classes": 12,
        "dataset": {
            "root": "./local/datasets/PAMAP2",
            "split_protocol_template": "subject_8fold_loso_fold{fold}",
        },
        "fold_count": 8,
        "global_rounds": 300,
        "pretrain": {
            "objective": "classification",
            "epochs": 5,
            "batch_size": 64,
            "lr": 0.0003,
            "weight_decay": 0.0001,
            "class_weighting": "inverse_sqrt",
            "max_grad_norm": 5.0,
        },
        "fingerprint_type": "signal",
        "encoder": {
            "type": "temporal_conv_gru",
            "conv_channels": [64, 128, 256],
            "kernel_sizes": [7, 5, 3],
            "gru_hidden_dim": 128,
            "gru_layers": 2,
            "bidirectional": True,
            "pooling": "attention",
            "dropout": 0.2,
        },
        "encoders": {},
        "training": {
            "clients_per_round": 6,
            "batch_size": 64,
            "eval_batch_size": 128,
            "client_lr": 0.0001,
            "server_lr": 0.0001,
            "class_weighting": "inverse_sqrt",
        },
        "binding_batch_size": 64,
        "fusion_dropout": 0.2,
        "mmbind": {
            "temperature": 0.1,
            "contrastive_weight": 0.05,
            "heterogeneous_ce_weight": 0.25,
        },
        "cluster_adaptive": {
            "bic_improvement_min": 20.0,
        },
    },
}


FORMAL_CV_SEED = 42

DISCOVERY_METHODS = (
    "kmeans2",
    "kmeans3",
    "kmeans4",
    "kmeans5",
    "auto_kmeans",
    "gmm_bic",
    "adaptive_isodata",
)

TRAINING_METHODS = (
    "ours",
    "randomsl",
    "kmeans2",
    "kmeans3",
    "kmeans4",
    "kmeans5",
    "oracle",
)


def get_dataset_protocol(dataset: str) -> dict:
    key = str(dataset).strip().lower()
    if key not in DATASET_PROTOCOLS:
        raise ValueError(f"Unknown dataset type: {dataset!r}")
    return DATASET_PROTOCOLS[key]


def get_split_protocol(dataset: str, fold: int) -> str:
    protocol = get_dataset_protocol(dataset)
    fold_count = protocol.get("fold_count")
    if fold_count is None:
        raise ValueError(f"Dataset {dataset!r} does not define folds.")
    fold = int(fold)
    if fold < 1 or fold > int(fold_count):
        raise ValueError(f"Dataset {dataset!r} fold must be in [1, {fold_count}], got {fold}.")
    template = protocol.get("dataset", {}).get("split_protocol_template")
    if not template:
        raise ValueError(f"Dataset {dataset!r} does not define split_protocol_template.")
    return str(template).format(fold=fold)
