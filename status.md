# Low-pri
- why there is no 100% numerical parity between torch.matmul and light.matmul. equal returns False while allclose returns True. Check `test_matmul`.
- implement something like pytree so we can return multi tensor in `test_nn_backward`
- automatically add the `create_backward_node` calls to ops

# Scratch
- end2end train the digit recognizer with light <++IMP
  - tensor equality

- digit recognizer trains too slow in light, improve that.

- improve the build system (do this after the digit recognizer works e2e)
  - make changing python file super easy [DONE]
  - recompile a file only if the dependencies is updates

## light dight recognizer
``` why the loss does not decrease
shunting@shunting-gpu-desktop:~/light$ time make dr
PYTHONPATH=build/install python3 model/digit_recognizer/mlp.py
Epoch 0/5
  Avg loss 2.3062849012361903
Epoch 1/5
  Avg loss 2.3062849012361903
Epoch 2/5
  Avg loss 2.3062849012361903
Epoch 3/5
  Avg loss 2.3062849012361903
Epoch 4/5
  Avg loss 2.3062849012361903
Traceback (most recent call last):
  File "/home/shunting/light/model/digit_recognizer/mlp.py", line 79, in <module>

  File "/home/shunting/light/model/digit_recognizer/mlp.py", line 75, in main
    model = Classifier()
  File "/home/shunting/cpython/build/install/lib/python3.9/contextlib.py", line 79, in inner
    return func(*args, **kwds)
  File "/home/shunting/light/model/digit_recognizer/mlp.py", line 55, in test_model
    # TODO support returning namedtuple
AttributeError: 'light._C.Tensor' object has no attribute 'max'
make: *** [Makefile:20: dr] Error 1

real    249m24.868s
user    249m25.223s
sys     0m3.992s

```
