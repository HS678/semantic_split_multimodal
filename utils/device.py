import torch


def select_device(device_cfg: str):
    choice = str(device_cfg or "auto").lower().strip()
    if choice == "cpu":
        return torch.device("cpu")
    if choice in ("cuda", "gpu"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("device is set to cuda/gpu but CUDA is not available")
    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    # custom device string fallback, e.g. cuda:1
    return torch.device(choice)
