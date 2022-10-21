import os
import unittest

def use_pytorch():
    return os.environ.get("USE_PYTORCH", None) == "1"

def import_torch_light(use_pytorch = False):
    global torch, Tensor
    if use_pytorch:
        import torch
        from torch import Tensor
    else:
        import light as torch
        from light import Tensor

import_torch_light(use_pytorch=use_pytorch())

def parity_test(testCase, worker):
    """
    Worker returns a torch or light Tensor
    """
    import_torch_light(True)
    torch_tensor = worker()
    import_torch_light(False)
    light_tensor = worker()
    import_torch_light(use_pytorch())
    testCase.assertTrue(list(torch_tensor.size()) == light_tensor.size())
    testCase.assertEqual(torch_tensor.tolist(), light_tensor.tolist())

class TestLight(unittest.TestCase):
    def test_basic(self):
        a = Tensor(2, 3)
        b = Tensor(2, 3)
        c = Tensor(2, 3)

        a.fill_(2.0)
        b.fill_(3.0)
        c.fill_(5.0)

        res = a + b
        print(res)
        self.assertEqual(list(a.size()), [2, 3])
        self.assertTrue(c.equal(res))

    def test_rand(self):
        def f():
            torch.manual_seed(23)
            x = torch.rand(2, 3)
            return x
        parity_test(self, f)

    def test_nn_forward(self):
        torch.manual_seed(23)
        B = 2
        M = 4
        N = 2

        inp = torch.rand(B, M)
        weight0 = torch.rand(M, N)
        bias0 = torch.rand(N)
        weight1 = torch.rand(N, 1)
        bias1 = torch.rand(1)

        out = torch.matmul(inp, weight0)
        out = out + bias0
        out = torch.relu(out)

        out = torch.matmul(out, weight1)
        out = out + bias1
        out = torch.sigmoid(out)
        out = out.mean()
        print(out)
