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

class MatmulBackward : public BackwardNode {
 public:
  using BackwardNode::BackwardNode;

  void run(Tensor out, Tensor out_grad) {
    assert(false && "MatmulBackward::run ni"); // TODO
  }
};

class AddBackward : public BackwardNode {
 public:
  using BackwardNode::BackwardNode;

  void run(Tensor out, Tensor out_grad) {
    assert(false && "AddBackward::run ni"); // TODO 
  }
};

class SubBackward : public BackwardNode {
 public:
  using BackwardNode::BackwardNode;

  void run(Tensor out, Tensor out_grad) {
    assert(false && "SubBackward::run ni"); // TODO
  }
};

class MulBackward : public BackwardNode {
 public:
  using BackwardNode::BackwardNode;

  void run(Tensor out, Tensor out_grad) {
    assert(false && "MulBackward::run ni"); // TODO
  }
};

class ReluBackward : public BackwardNode {
 public:
  using BackwardNode::BackwardNode;

  void run(Tensor out, Tensor out_grad) {
    assert(false && "ReluBackward::run ni"); // TODO 
  }
};

class SigmoidBackward : public BackwardNode {
 public:
  using BackwardNode::BackwardNode;

  void run(Tensor out, Tensor out_grad) {
    // TODO: allow automatic scalar to Tensor conversion
    Tensor inp_grad = out_grad * out * (Tensor::create_scalar_tensor(1.0f) - out);
    propagate({inp_grad});
  }
};
