#pragma once

#include <string>
#include <vector>
#include "light/csrc/Tensor.h"

extern bool is_grad_enabled_;

static bool is_grad_enabled() {
  return is_grad_enabled_;
}

static void set_is_grad_enabled(bool newval) {
  is_grad_enabled_ = newval;
}

static void enable_grad() {
  is_grad_enabled_ = true;
}

static void disable_grad() {
  is_grad_enabled_ = false;
}

class DisableGradGuard {
 public:
  explicit DisableGradGuard() : prev_val_(is_grad_enabled()) {
    disable_grad();
  }
  ~DisableGradGuard() {
    set_is_grad_enabled(prev_val_);
  }
 private:
  bool prev_val_;
};

template <typename F>
static void create_backward_node(Tensor out, std::vector<Tensor> inputs, F f) {
  /*
   * f is a callable creating the backward node. Use this API if the backward node
   * takes extra arguments besides inputs tensors.
   */
  if (!is_grad_enabled()) {
    return;
  }
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
  out.set_backward_node(f());
}

// TODO this cause a lot of dupliate binary code. Each BackwardNode subclass
// will have a copy of this function. Think about how to reduce
template <typename T>
static void create_backward_node(Tensor out, std::vector<Tensor> inputs) {
  return create_backward_node(out, inputs, [&inputs]() {
    return new T(inputs);
  });
}
