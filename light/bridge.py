import torch

def to_torch_tensor(light_tensor):
    # TODO better support dtype
    assert(light_tensor.dtype() == 0)
    return torch.Tensor(light_tensor.tolist())
