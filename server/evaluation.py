import torch
from sklearn.metrics import f1_score


@torch.no_grad()
def evaluate_paired_test(clients_by_modality, server_model, test_set, device):
    # full paired test, no label-guided pairing
    xs = [x.to(device) for x in test_set["modalities"]]
    y = test_set["labels"].to(device)

    proj_list = []
    for m, x in enumerate(xs):
        # use first client encoder per modality as modality encoder proxy
        client = clients_by_modality[m][0]
        z = client.encoder(x)
        proj = server_model.projectors[str(m)](z)
        proj_list.append(proj)

    fused = server_model.fusion(proj_list)
    logits = server_model.classifier(fused)
    pred = torch.argmax(logits, dim=1)
    acc = (pred == y).float().mean().item()
    macro_f1 = f1_score(y.cpu().numpy(), pred.cpu().numpy(), average="macro")
    return acc, macro_f1
