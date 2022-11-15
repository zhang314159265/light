import light

class CrossEntropyLoss:
    """
    PyTorch implements CrossEntropyLoss as an nn.Module, but here we just
    implement it as a callable.
    """
    def __call__(self, pred, label):
        prob = light.log_softmax(pred, 1)
        return light.nn.functional.nll_loss(prob, label)
