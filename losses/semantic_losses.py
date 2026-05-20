import torch
import torch.nn.functional as F


class SupervisedContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # features: [B, V, D]
        bsz, views, dim = features.shape
        x = F.normalize(features.reshape(bsz * views, dim), dim=1)
        y = labels.repeat_interleave(views)

        logits = torch.matmul(x, x.T) / self.temperature
        mask = torch.eq(y.unsqueeze(0), y.unsqueeze(1)).float()
        logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=mask.device)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-12)
        loss = -mean_log_prob_pos.mean()
        return loss


class PrototypeLoss(torch.nn.Module):
    def forward(self, *args, **kwargs):
        return torch.tensor(0.0)
