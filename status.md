# Low-pri
- why there is no 100% numerical parity between torch.matmul and light.matmul. equal returns False while allclose returns True. Check `test_matmul`.
- implement something like pytree so we can return multi tensor in `test_nn_backward`
- automatically add the `create_backward_node` calls to ops

# Scratch
- TODO: use transformer to do machine translation.. (pytorch, not light yet)

- TODO: support batchnorm
  - understand how PyTorch implement batchnorm
    - check `native_batch_norm`
  - next: batchnorm backward

- TODO: support resnet18 in light

- TODO: dropout backward
- TODO: torch.flatten backward

- TODO: support alexnet in light

- TODO: conv backward <==
  - 2. check how pytorch implement conv backward and compare with my own implementation.

- end2end train the digit recognizer with light <++IMP
  - numerical correctness is guaranteed (88% accuracy)
  - but training too slow ( 254m v.s. 14s) <== 1000x slower (-O2?)
    ```
real    254m19.037s
user    254m18.274s
sys     0m3.600s

v.s.

real    0m13.886s
user    5m2.333s
sys     0m4.603s
    ```
  - TODO: create micro benchmarks
  - goal: at least 0.5 as fast


- improve the build system (do this after the digit recognizer works e2e)
  - make changing python file super easy [DONE]
  - recompile a file only if the dependencies is updates

