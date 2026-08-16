import copy
from collections import OrderedDict

import torch
from torch import nn


# 检查一组 encoder 的参数 key 和 shape 是否完全兼容。
def _validate_state_dicts(state_dicts: list[dict[str, torch.Tensor]]) -> None:
    if not state_dicts:
        raise ValueError("Cannot average an empty encoder state list.")
    reference_keys = list(state_dicts[0].keys())
    reference_shapes = {key: tuple(value.shape) for key, value in state_dicts[0].items()}
    for index, state in enumerate(state_dicts[1:], start=1):
        if list(state.keys()) != reference_keys:
            raise ValueError(f"Encoder state_dict keys differ at index {index}.")
        for key in reference_keys:
            if tuple(state[key].shape) != reference_shapes[key]:
                raise ValueError(
                    "Encoder state_dict shapes differ for "
                    f"{key}: expected={reference_shapes[key]}, got={tuple(state[key].shape)}"
                )


# 对兼容 encoder 的参数做 deterministic 平均。
def average_encoder_state_dicts(state_dicts: list[dict[str, torch.Tensor]]) -> OrderedDict:
    _validate_state_dicts(state_dicts)
    keys = list(state_dicts[0].keys())
    averaged = OrderedDict()
    for key in keys:
        values = [state[key].detach().cpu() for state in state_dicts]
        if not torch.is_floating_point(values[0]):
            if key.endswith("num_batches_tracked"):
                averaged[key] = torch.stack(values, dim=0).max(dim=0).values.clone()
                continue
            first = values[0]
            if any(not torch.equal(first, value) for value in values[1:]):
                raise ValueError(f"Non-floating encoder buffer differs for key {key}.")
            averaged[key] = first.clone()
        else:
            stacked = torch.stack([value.to(dtype=torch.float32) for value in values], dim=0)
            averaged[key] = stacked.mean(dim=0).to(dtype=values[0].dtype)
    return averaged


# 用同架构模板和平均参数构建 representative encoder。
def build_representative_encoder(template: nn.Module, state_dicts: list[dict[str, torch.Tensor]]) -> nn.Module:
    averaged = average_encoder_state_dicts(state_dicts)
    representative = copy.deepcopy(template)
    representative.load_state_dict(averaged)
    return representative


# 复制一个 encoder 模板并加载平均后的参数。
def clone_encoder_with_state(factory, state_dicts: list[dict[str, torch.Tensor]], device) -> nn.Module:
    encoder = factory().to(device)
    encoder.load_state_dict(average_encoder_state_dicts(state_dicts))
    return encoder
