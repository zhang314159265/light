import torch

def to_torch_tensor(light_tensor):
    if light_tensor.size() == []:
        return torch.Tensor([light_tensor.item()])[0]
    return torch.Tensor(light_tensor.tolist())
