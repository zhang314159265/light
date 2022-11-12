from light import Tensor

class Module:
    def __init__(self):
        self._parameters = {}
        self._modules = {}

    def __setattr__(self, name, obj):
        if isinstance(obj, Module):
            self._modules[name] = obj
        elif isinstance(obj, Tensor) and obj.is_param:
            self._parameters[name] = obj
        else:
            object.__setattr__(self, name, obj)

    def __getattr__(self, name):
        if name in self._modules:
            return self._modules[name]
        elif name in self._parameters:
            return self._parameters[name]
        else:
            raise RuntimeError(f"Access non-existing attribute {name}")

    def descendants(self):
        """
        Yield descendant modules including the module itself.
        # TODO dedup?
        """
        yield self
        for mod in self._modules:
            yield from mod.descendants()

    def parameters(self):
        for mod in self.descendants():
            yield from mod._parameters.values() 

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
