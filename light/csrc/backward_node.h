#pragma once

#include "light/csrc/Tensor.h"
#include <iostream>

class BackwardNode {
 public:
  explicit BackwardNode(const std::vector<Tensor>& inputs) : inputs_(inputs) { }
  virtual void run(Tensor out, Tensor out_grad) = 0;
  // propagate the input grads backward
  void propagate(const std::vector<Tensor>& input_grads) {
    assert(inputs_.size() == input_grads.size());
    for (int i = 0; i < inputs_.size(); ++i) {
      Tensor inp = inputs_[i];
      Tensor inp_grad = input_grads[i];

      if (inp.requires_grad()) {
        if (inp.backward_node()) {
          // intermediate node
          inp.backward_node()->run(inp, inp_grad);
        } else {
          // leaf node
          inp.set_grad(inp_grad);
        }
      }
    }
  }
 protected:
  std::vector<Tensor> inputs_;
};
