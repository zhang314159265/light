# Low-pri
- why there is no 100% numerical parity between torch.matmul and light.matmul. equal returns False while allclose returns True. Check `test_matmul`.
- implement something like pytree so we can return multi tensor in `test_nn_backward`
- automatically add the `create_backward_node` calls to ops

# Scratch
- check how nn.Linear initialize parameters
- end2end train the digit recognizer and visualize the input and prediction <++ IMP
