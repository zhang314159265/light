import os
import inspect

def import_torch_light(use_pytorch = None):
    if use_pytorch is None:
        use_pytorch = (os.environ.get("USE_PYTORCH", None) == "1")
    if use_pytorch:
        import torch
        from torch import Tensor
    else:
        import light as torch
        from light import Tensor
        import light.optim

    # inject torch/Tensor etc to globals of caller
    symbols = ["torch", "Tensor"]
    caller_globals = inspect.stack()[1].frame.f_globals
    for sym in symbols:
        caller_globals[sym] = locals()[sym]
