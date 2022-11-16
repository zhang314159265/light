import light

class SGD:
    def __init__(self, params, lr):
        # remember to convert potential generator to list.
        # For a generator, only the first round of iteration will
        # get something.
        self.params = list(params)
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
