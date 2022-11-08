import light
import math
from light.nn.modules.module import Module
# from light.nn import Module # TODO this fail because of circular import

class Linear(Module):
    def __init__(self, in_features, out_features):
        self.weight = light.Tensor([out_features, in_features])
        self.weight.requires_grad = True
        self.bias = light.Tensor([out_features])
        self.bias.requires_grad = True

        self.reset_parameters()

    @light.no_grad()
    def reset_parameters(self):
        bound = 1 / math.sqrt(self.weight.size()[1])
        self.weight.uniform_(-bound, bound)
        self.bias.uniform_(-bound, bound)

    def forward(self, inp):
        return light.matmul(inp, self.weight.transpose(0, 1)) + self.bias
