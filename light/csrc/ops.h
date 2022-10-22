#pragma once

#include <cassert>

namespace ops {

Tensor matmul(const Tensor& lhs, const Tensor& rhs) {
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

  return out;
}

}
