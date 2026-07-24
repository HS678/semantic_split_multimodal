import torch
from torch import nn


class PrototypeBank:
    def __init__(self, momentum=0.9):
        self.momentum = float(momentum)
        self.bank = {}

    def alignment_loss(self, semantics, labels, cluster_ids):
        if semantics.numel() == 0:
            return semantics.new_tensor(0.0)
        current = {}
        for cluster_id in torch.unique(cluster_ids).detach().cpu().tolist():
            c_mask = cluster_ids == int(cluster_id)
            for class_id in torch.unique(labels[c_mask]).detach().cpu().tolist():
                mask = c_mask & (labels == int(class_id))
                current[(int(cluster_id), int(class_id))] = semantics[mask].mean(dim=0)

        targets = []
        preds = []
        for (cluster_id, class_id), proto in current.items():
            other = [
                value.to(proto.device)
                for (c, y), value in self.bank.items()
                if int(y) == int(class_id) and int(c) != int(cluster_id)
            ]
            if other:
                preds.append(proto)
                targets.append(torch.stack(other, dim=0).mean(dim=0).detach())
        if not preds:
            return semantics.new_tensor(0.0)
        return nn.functional.mse_loss(torch.stack(preds, dim=0), torch.stack(targets, dim=0))

    def update(self, semantics, labels, cluster_ids):
        with torch.no_grad():
            for cluster_id in torch.unique(cluster_ids).detach().cpu().tolist():
                c_mask = cluster_ids == int(cluster_id)
                for class_id in torch.unique(labels[c_mask]).detach().cpu().tolist():
                    mask = c_mask & (labels == int(class_id))
                    proto = semantics[mask].mean(dim=0).detach().cpu()
                    key = (int(cluster_id), int(class_id))
                    if key in self.bank:
                        self.bank[key] = self.momentum * self.bank[key] + (1.0 - self.momentum) * proto
                    else:
                        self.bank[key] = proto
