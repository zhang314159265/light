import light

class SGD:
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

    @light.no_grad()
    def step(self):
        for param in self.params:
            assert param.grad
            param.add_(param.grad, -self.lr)
