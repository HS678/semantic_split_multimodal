import torch
from torch import nn

from models.modules import SemanticProjector, ConcatFusion, ClassifierHead
from losses.semantic_losses import SupervisedContrastiveLoss, PrototypeLoss


class SemanticBatchBuilder:
    @staticmethod
    def build(selected_payloads):
        # payload: {client_id, modality_id, y, z_server}
        label_sets = [set(p["y"].tolist()) for p in selected_payloads]
        common_labels = sorted(set.intersection(*label_sets)) if label_sets else []
        if len(common_labels) == 0:
            return None

        indices_per_client = {p["client_id"]: [] for p in selected_payloads}
        semantic_labels = []
        for label in common_labels:
            per_client_indices = []
            min_count = None
            for p in selected_payloads:
                idx = (p["y"] == label).nonzero(as_tuple=False).squeeze(-1)
                per_client_indices.append(idx)
                count = idx.numel()
                min_count = count if min_count is None else min(min_count, count)
            if min_count is None or min_count == 0:
                continue
            for take_i in range(min_count):
                for ci, p in enumerate(selected_payloads):
                    # strictly index from original z_server path; no detach here.
                    indices_per_client[p["client_id"]].append(per_client_indices[ci][take_i].item())
                semantic_labels.append(label)

        if len(semantic_labels) == 0:
            return None

        semantic_labels = torch.tensor(semantic_labels, device=selected_payloads[0]["y"].device, dtype=torch.long)
        return {
            "common_labels": common_labels,
            "index_map": indices_per_client,
            "semantic_labels": semantic_labels,
        }


class SplitServer(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.num_modalities = cfg["num_modalities"]
        self.projectors = nn.ModuleDict(
            {str(m): SemanticProjector(cfg["encoder_hidden_dim"], cfg["projected_dim"]) for m in range(self.num_modalities)}
        )
        self.fusion = ConcatFusion()
        self.classifier = ClassifierHead(cfg["projected_dim"] * self.num_modalities, cfg["num_classes"])

        self.supcon = SupervisedContrastiveLoss(cfg["temperature"])
        self.prototype = PrototypeLoss()
        self.ce = nn.CrossEntropyLoss()

        self.opt = torch.optim.Adam(self.parameters(), lr=cfg["learning_rate"])

    def _scatter_grad(self, full_shape, used_indices, grad_subset):
        full = torch.zeros(full_shape, device=grad_subset.device)
        idx = torch.tensor(used_indices, dtype=torch.long, device=grad_subset.device)
        full[idx] = grad_subset
        return full

    def train_step(self, selected_payloads):
        build = SemanticBatchBuilder.build(selected_payloads)
        if build is None:
            return None

        common_labels = build["common_labels"]
        index_map = build["index_map"]
        y_sem = build["semantic_labels"]
        bsz = y_sem.shape[0]

        proj_list = []
        raw_grad_targets = {}
        print(f"common labels: {common_labels}")
        print(f"semantic batch size: {bsz}")

        self.opt.zero_grad()
        for p in selected_payloads:
            cid = p["client_id"]
            mid = p["modality_id"]
            idx = index_map[cid]
            z_server = p["z_server"]
            # protocol: selected z must come from original z_server via indexing only.
            z_subset = z_server[idx]
            assert z_subset.requires_grad, "selected z_subset must require grad from original z_server"
            proj = self.projectors[str(mid)](z_subset)
            proj_list.append(proj)
            raw_grad_targets[cid] = {
                "full_shape": z_server.shape,
                "idx": idx,
                "z_server": z_server,
            }
            print(f"feature shape [{cid}]: {tuple(z_server.shape)}")
            print(f"projected shape [{cid}]: {tuple(proj.shape)}")

        stacked = torch.stack(proj_list, dim=1)  # [B, M, D]
        loss_align = self.supcon(stacked, y_sem)

        fused = self.fusion(proj_list)
        logits = self.classifier(fused)
        loss_cls = self.ce(logits, y_sem)
        loss_proto = self.prototype()

        lambda_align = float(self.cfg.get("lambda_align", self.cfg.get("lambda_supcon", 0.0)))
        total_loss = (
            float(self.cfg["lambda_cls"]) * loss_cls
            + lambda_align * loss_align
            + float(self.cfg["lambda_proto"]) * loss_proto
        )
        total_loss.backward()
        self.opt.step()

        grad_to_clients = {}
        non_null = 0
        for p in selected_payloads:
            cid = p["client_id"]
            z_server = raw_grad_targets[cid]["z_server"]
            idx = raw_grad_targets[cid]["idx"]
            assert z_server.grad is not None, f"z_server.grad is None for {cid}"
            grad_subset = z_server.grad[idx]
            grad_full = self._scatter_grad(raw_grad_targets[cid]["full_shape"], idx, grad_subset)
            grad_to_clients[cid] = grad_full
            if grad_full is not None and torch.isfinite(grad_full).all().item():
                non_null += 1

        grad_non_null_ratio = non_null / max(1, len(selected_payloads))
        print(
            "loss dict: "
            f"cls={loss_cls.item():.4f}, align={loss_align.item():.4f}, proto={float(loss_proto):.4f}, total={total_loss.item():.4f}"
        )
        print(f"gradient non-null check: ratio={grad_non_null_ratio:.4f}")

        return {
            "grad_to_clients": grad_to_clients,
            "common_labels": common_labels,
            "common_label_count": int(len(common_labels)),
            "semantic_batch_size": int(bsz),
            "loss_total": float(total_loss.item()),
            "loss_cls": float(loss_cls.item()),
            "loss_align": float(loss_align.item()),
            "loss_proto": float(loss_proto.item() if hasattr(loss_proto, "item") else float(loss_proto)),
            "grad_non_null_ratio": float(grad_non_null_ratio),
        }
