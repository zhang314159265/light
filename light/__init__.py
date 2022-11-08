import contextlib

@contextlib.contextmanager
def no_grad():
    try:
        disable_grad()
        yield
    finally:
        enable_grad()

from light._C import *
from . import nn
