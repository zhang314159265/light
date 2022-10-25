#pragma once

#include <string>
#include <vector>
#include "light/csrc/Tensor.h"

// TODO this cause a lot of dupliate binary code. Each BackwardNode subclass
// will have a copy of this function. Think about how to reduce
template <typename T>
static void create_backward_node(Tensor out, std::vector<Tensor> inputs) {
  assert(inputs.size() > 0);
  bool any_requires_grad = false;
  for (const auto& inp : inputs) {
    if (inp.requires_grad()) {
      any_requires_grad = true;
      break;
    }
  }

  if (!any_requires_grad) {
    return;
  }

  out.set_requires_grad(true);
  out.set_backward_node(new T(inputs));
}
