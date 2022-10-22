#include "light/csrc/Tensor.h"

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
  std::vector<int> shape = get_broadcast_shape({lhs, rhs});
  Tensor out(shape, lhs.dtype());
  out.visit([&lhs, &rhs, &out](const std::vector<int>& indices) {
    using scalar_t = float; // TODO 
    auto lhs_ptr = (scalar_t*) lhs.locate(indices);
    auto rhs_ptr = (scalar_t*) rhs.locate(indices);
    auto out_ptr = (scalar_t*) out.locate(indices);
    *out_ptr = *lhs_ptr + *rhs_ptr;
    return true;
  });
  return out;
}
