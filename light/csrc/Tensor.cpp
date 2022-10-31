#include "light/csrc/Tensor.h"
#include "light/csrc/backward_node.h"
#include "light/csrc/ops.h"

Tensor Tensor::dummy = Tensor::create_scalar_tensor(0.0f);

std::vector<int> get_broadcast_shape(const std::vector<int>& lhs_shape, const std::vector<int>& rhs_shape) {
  int out_dim = std::max(lhs_shape.size(), rhs_shape.size());
  std::vector<int> out_shape(out_dim, 0);

  int lhs_off = out_dim - lhs_shape.size();
  int rhs_off = out_dim - rhs_shape.size();
  for (int i = 0; i < out_dim; ++i) {
    int lhs_size = i < lhs_off ? 1 : lhs_shape[i - lhs_off];
    int rhs_size = i < rhs_off ? 1 : rhs_shape[i - rhs_off];
    int out_size = -1;
    if (lhs_size == 1) {
      out_size = rhs_size;
    } else if (rhs_size == 1) {
      out_size = lhs_size;
    } else {
      assert(lhs_size == rhs_size);
      out_size = lhs_size;
    }
    out_shape[i] = out_size;
  }
  return out_shape;
}

std::vector<int> get_broadcast_shape(std::vector<Tensor> tensors) {
  assert(tensors.size() > 0);
  std::vector<int> shape = tensors[0].sizes();
  for (int i = 1; i < tensors.size(); ++i) {
    shape = get_broadcast_shape(shape, tensors[i].sizes());
  }
  return shape;
}

Tensor Tensor::add(const Tensor& lhs, const Tensor& rhs) {
  return ops::add(lhs, rhs);
}

Tensor operator-(const Tensor& lhs, const Tensor& rhs) {
  return ops::sub(lhs, rhs);
}

Tensor operator*(const Tensor& lhs, const Tensor& rhs) {
  return ops::mul(lhs, rhs);
}

Tensor Tensor::mean() const {
  return ops::mean(*this);
}

void Tensor::backward() {
  DisableGradGuard g;
  assert(requires_grad());
  assert(dim() == 0); // scalar tensor
  assert(impl_->backward_node_);
  impl_->backward_node_->run(*this, create_scalar_tensor(1.0f));
}

TensorImpl::~TensorImpl() {
  free(data_);
  if (backward_node_) {
    delete backward_node_;
  }
  if (grad_) {
    delete grad_;
  }
}
