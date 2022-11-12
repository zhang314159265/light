from light import Tensor

def Parameter(tensor):
    """
    PyTorch implement Parameter as a Tensor subclass. But in light, a parameter
    is just a normal Tensor with is_param flag to be true.
    """
    tensor.is_param = True
    tensor.requires_grad = True
    return tensor
