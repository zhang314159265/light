#include "light/csrc/backward_ops.h"
#include "light/csrc/ops.h"

void MatmulBackward::run(Tensor out, Tensor out_grad) {
  Tensor lhs = inputs_[0];
  Tensor rhs = inputs_[1];
  Tensor lhs_grad = Tensor::dummy;
  Tensor rhs_grad = Tensor::dummy;
  if (lhs.requires_grad()) {
    lhs_grad = ops::matmul(out_grad, ops::transpose(rhs, 0, 1));
  }

  if (rhs.requires_grad()) {
    rhs_grad = ops::matmul(ops::transpose(lhs, 0, 1), out_grad);
  }
  propagate({lhs_grad, rhs_grad});
}
