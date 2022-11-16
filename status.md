# Low-pri
- why there is no 100% numerical parity between torch.matmul and light.matmul. equal returns False while allclose returns True. Check `test_matmul`.
- implement something like pytree so we can return multi tensor in `test_nn_backward`
- automatically add the `create_backward_node` calls to ops

# Scratch
- end2end train the digit recognizer with light <++IMP

- digit recognizer trains too slow in light, improve that.

- improve the build system (do this after the digit recognizer works e2e)
  - make changing python file super easy [DONE]
  - recompile a file only if the dependencies is updates

