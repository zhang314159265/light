#pragma once

#include <cassert>
#include "light/csrc/backward.h"
#include "light/csrc/backward_ops.h"

namespace ops {

static Tensor matmul(const Tensor& lhs, const Tensor& rhs) {
  assert(lhs.dim() == 2);
  assert(rhs.dim() == 2);
  assert(lhs.sizes()[1] == rhs.sizes()[0]);
  assert(lhs.dtype() == rhs.dtype()); // no type promotion so far

  int M = lhs.sizes()[0];
  int K = lhs.sizes()[1];
  int N = rhs.sizes()[1];

  Tensor out({M, N}, lhs.dtype());

  using scalar_t = float; // TODO
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      double sum = 0.0f;
      for (int k = 0; k < K; ++k) {
        sum += *(scalar_t*) lhs.locate(i, k) * *(scalar_t*) rhs.locate(k, j);
      }
      *(scalar_t*) out.locate(i, j) = sum;
    }
  }

  create_backward_node<MatmulBackward>(out, {lhs, rhs});
  return out;
}

static Tensor relu(const Tensor& inp) {
  Tensor out(inp.sizes(), inp.dtype());
  out.visit([&inp, &out](const std::vector<int>& indices) {
    using scalar_t = float; // TODO
    auto inp_ptr = (scalar_t*) inp.locate(indices);
    auto out_ptr = (scalar_t*) out.locate(indices);
    *out_ptr = std::max(*inp_ptr, 0.0f);
    return true;
  });
  create_backward_node<ReluBackward>(out, {inp});
  return out;
}

static Tensor sigmoid(const Tensor& inp) {
  Tensor out(inp.sizes(), inp.dtype());
  out.visit([&inp, &out](const std::vector<int>& indices) {
    using scalar_t = float; // TODO
    auto inp_ptr = (scalar_t*) inp.locate(indices);
    auto out_ptr = (scalar_t*) out.locate(indices);

    // use double for intermediate values
    double x = (double) *inp_ptr;
    double out = 1.0 / (1.0 + std::exp(-x));
    *out_ptr = (scalar_t) out;
    return true;
  });
  create_backward_node<SigmoidBackward>(out, {inp});
  return out;
}

static Tensor mean(const Tensor& inp) {
  double accum = 0;
  inp.visit([&accum, &inp](const std::vector<int>& indices) {
    using scalar_t = float; // TODO
    auto inp_ptr = (scalar_t*) inp.locate(indices);

    accum += *inp_ptr;
    return true;
  });
  int numel = inp.numel();
  assert(numel > 0);
  auto out = Tensor::create_scalar_tensor((float) (accum / numel));
  create_backward_node<MeanBackward>(out, {inp});
  return out;
}

static Tensor add(const Tensor& lhs, const Tensor& rhs) {
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
  create_backward_node<AddBackward>(out, {lhs, rhs});
  return out;
}

static Tensor sub(const Tensor& lhs, const Tensor& rhs) {
  std::vector<int> shape = get_broadcast_shape({lhs, rhs});
  Tensor out(shape, lhs.dtype());
  out.visit([&lhs, &rhs, &out](const std::vector<int>& indices) {
    using scalar_t = float; // TODO 
    auto lhs_ptr = (scalar_t*) lhs.locate(indices);
    auto rhs_ptr = (scalar_t*) rhs.locate(indices);
    auto out_ptr = (scalar_t*) out.locate(indices);
    *out_ptr = *lhs_ptr - *rhs_ptr;
    return true;
  });
  create_backward_node<SubBackward>(out, {lhs, rhs});
  return out;
}

static Tensor mul(const Tensor& lhs, const Tensor& rhs) {
  std::vector<int> shape = get_broadcast_shape({lhs, rhs});
  Tensor out(shape, lhs.dtype());
  out.visit([&lhs, &rhs, &out](const std::vector<int>& indices) {
    using scalar_t = float; // TODO 
    auto lhs_ptr = (scalar_t*) lhs.locate(indices);
    auto rhs_ptr = (scalar_t*) rhs.locate(indices);
    auto out_ptr = (scalar_t*) out.locate(indices);
    *out_ptr = *lhs_ptr * *rhs_ptr;
    return true;
  });
  create_backward_node<MulBackward>(out, {lhs, rhs});
  return out;
}

}
