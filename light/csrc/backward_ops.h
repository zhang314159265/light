#pragma once

#include "light/csrc/backward_node.h"

class MeanBackward : public BackwardNode {
 public:
  using BackwardNode::BackwardNode;

  void run(Tensor out, Tensor out_grad) {
    assert(inputs_.size() == 1);
    Tensor& inp = inputs_[0];
    assert(inp.requires_grad());

    // calculate input gradient
    assert(out_grad.dim() == 0);
    using scalar_t = float; // TODO
    scalar_t val = 1.0 / inp.numel();
    Tensor inp_grad(inp.sizes(), inp.dtype());
    inp_grad.initWithScalar(val);

    propagate({inp_grad});
  }
};
