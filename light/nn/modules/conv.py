from light.nn.modules.module import Module
from light.nn.parameter import Parameter
import light
import itertools
import numpy as np
import math

def maybe_repeat(val, rep):
    if isinstance(val, (tuple, list)):
        assert len(val) == rep, "Invalid input"
        return val

    return list(itertools.repeat(val, rep))

class Conv2d(Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding, stride):
        super().__init__()
        kernel_size = maybe_repeat(kernel_size, 2)
        self.padding = maybe_repeat(padding, 2)
        self.stride = maybe_repeat(stride, 2)
        self.weight = Parameter(light.Tensor([out_channel, in_channel, kernel_size[0], kernel_size[1]]))
        self.bias = Parameter(light.Tensor([out_channel]))
        self.reset_parameters()

    @light.no_grad()
    def reset_parameters(self):
        bound = 1 / math.sqrt(np.prod(self.weight.size()[1:]))
        self.weight.uniform_(-bound, bound)
        self.bias.uniform_(-bound, bound)
        
    def forward(self, x):
        return light.conv2d(x, self.weight, self.bias, self.stride, self.padding)
