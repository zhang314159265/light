import os
import unittest

from light.bridge import to_torch_tensor

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
        import light.optim

import_torch_light(use_pytorch=use_pytorch())

def parity_test(testCase, worker):
    """
    Worker returns a torch or light Tensor
    """
    import_torch_light(True)
    torch_tensor = worker()
    import_torch_light(False)
    light_tensor = worker()

    if torch_tensor is None and light_tensor is None:
        return
    testCase.assertTrue(list(torch_tensor.size()) == light_tensor.size())

    import torch as real_torch
    torch_tensor_from_light = to_torch_tensor(light_tensor)
    # When torch_tensor is a long tensor, it can not be compared with
    # torch_tensor_from_light with is a float tensor. Convert torch_tensor
    # to float tensor to ease the comparison.
    torch_tensor = torch_tensor.to(dtype=real_torch.float32) 
    testCase.assertTrue(real_torch.allclose(torch_tensor, torch_tensor_from_light), f"Torch tensor:\n{torch_tensor}\nlight tensor:\n{torch_tensor_from_light}")

    import_torch_light(use_pytorch())

class TestLight(unittest.TestCase):
    def test_basic(self):
        def f():
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
        parity_test(self, f)

    def test_rand(self):
        def f():
            torch.manual_seed(23)
            x = torch.rand(2, 3)
            return x
        parity_test(self, f)

    def test_randint(self):
        def f():
            torch.manual_seed(23)
            x = torch.randint(0, 10, (2, 3))
            print(x)
            return x
        parity_test(self, f)

    def test_matmul(self):
        def f():
            torch.manual_seed(23)
            x = torch.rand(2, 3)
            y = torch.rand(3, 2)
            out = torch.matmul(x, y)
            print(out)
            return out
        parity_test(self, f)

    def test_broadcast(self):
        # TODO merge f and f2 by returning a tuple of tensors
        def f():
            torch.manual_seed(23)
            x = torch.rand(2, 3)
            y = torch.rand(3)
            return x + y
        parity_test(self, f)

        def f2():
            torch.manual_seed(23)
            x = torch.rand(2, 3)
            y = torch.rand(3)
            return y + x
        parity_test(self, f2)

    def test_nn_forward(self):
        def f():
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
            return out

        parity_test(self, f)

    def test_simple_backward(self):
        def f():
            torch.manual_seed(23)
            inp = torch.rand(2, 2, requires_grad=True)
            y = torch.rand(2, 2)
            x = torch.sigmoid(inp)
            x = x + y
            x = torch.relu(x)
            out = x.mean()
            assert out.requires_grad
            out.backward()
            print(f"inp.grad {inp.grad}")
            assert not inp.grad.requires_grad
            return inp.grad
        parity_test(self, f)

    def test_nn_backward(self):
        def f():
            torch.manual_seed(23)
            B = 2
            M = 4
            N = 2
    
            inp = torch.rand(B, M)
            weight0 = torch.rand(M, N, requires_grad=True)
            bias0 = torch.rand(N, requires_grad=True)
            weight1 = torch.rand(N, 1, requires_grad=True)
            bias1 = torch.rand(1, requires_grad=True)
    
            out = torch.matmul(inp, weight0)
            out = out + bias0
            out = torch.relu(out)
    
            out = torch.matmul(out, weight1)
            out = out + bias1
            out = torch.sigmoid(out)
            out = out.mean()

            out.backward()
            print(weight0.grad)
            return weight0.grad

        parity_test(self, f)

    def test_linear(self):
        def f():
            torch.manual_seed(23)
            linear = torch.nn.Linear(2, 3)
            inp = torch.rand(10, 2)
            out = linear(inp)
            return out
        parity_test(self, f)

    def test_classifier(self):
        def f():
            torch.manual_seed(23)
            B = 2
            NF = 10
            NC = 3

            inp = torch.rand(B, NF)
            lin = torch.nn.Linear(NF, NC)
            sgd = torch.optim.SGD([lin.weight, lin.bias], lr=0.01)
            out = lin(inp)
            out = torch.log_softmax(out, 1)

            label = torch.randint(0, NC, (B,))
            assert len(label) == B
            loss = torch.nn.functional.nll_loss(out, label)
            # pytorch nll_loss already does reduction, but light nll_loss does not.
            # Thus add a mean here.
            loss = loss.mean()
            sgd.zero_grad()
            loss.backward()
            sgd.step()

            print(lin.weight)
            return lin.weight

        parity_test(self, f)
