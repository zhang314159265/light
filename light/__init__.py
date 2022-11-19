import contextlib
import numpy as np

@contextlib.contextmanager
def no_grad():
    try:
        disable_grad()
        yield
    finally:
        enable_grad()

from light._C import *
from . import nn

def flatten(t_in, start_dim, end_dim=-1):
    if end_dim < 0:
        end_dim = t_in.dim() + end_dim

    if start_dim == end_dim:
        return t_in

    assert start_dim >= 0
    assert start_dim < end_dim
    assert end_dim < t_in.dim()

    old_shape = t_in.shape
    new_shape = (old_shape[:start_dim]
        + [np.prod(old_shape[start_dim:(end_dim + 1)])]
        + old_shape[(end_dim + 1):])

    return _C.reshape(t_in, new_shape)
