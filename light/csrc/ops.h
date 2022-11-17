#pragma once

#include <cassert>
#include "light/csrc/backward.h"
#include "light/csrc/backward_ops.h"
#include "light/csrc/rand.h"

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

static std::tuple<Tensor, Tensor> max(const Tensor& inp, int dim) {
  assert(!inp.requires_grad());
  std::vector<int> out_size = inp.sizes();
  out_size.erase(out_size.begin() + dim);
  Tensor out_values(out_size, inp.dtype());
  Tensor out_pos_tensor(out_size, ScalarType::Int64);

  assert(inp.sizes()[dim] > 0);

  DISPATCH_DTYPE(inp.dtype(), [&]() {
    out_values.visit([&out_values, &out_pos_tensor, &inp, dim](const std::vector<int> out_indices) {
      int64_t& out_pos = *(int64_t*) out_pos_tensor.locate(out_indices);
      scalar_t& out_val = *(scalar_t*) out_values.locate(out_indices);
      int sz = inp.sizes()[dim];
      out_pos = 0;
      std::vector<int> in_indices = out_indices;
      in_indices.insert(in_indices.begin() + dim, 0);
      out_val = *(scalar_t*) inp.locate(in_indices);

      for (int i = 1; i < sz; ++i) {
        in_indices[dim] = i;
        scalar_t newval = *(scalar_t*) inp.locate(in_indices);
        if (newval > out_val) {
          out_val = newval;
          out_pos = i;
        }
      }
      return true;
    });
  });
  return std::make_tuple(out_values, out_pos_tensor);
}

static Tensor add(const Tensor& lhs, const Tensor& rhs) {
  std::vector<int> shape = get_broadcast_shape({lhs, rhs});
  Tensor out(shape, lhs.dtype());
  DISPATCH_DTYPE(lhs.dtype(), [&]() {
    out.visit([&lhs, &rhs, &out](const std::vector<int>& indices) {
      auto lhs_ptr = (scalar_t*) lhs.locate(indices);
      auto rhs_ptr = (scalar_t*) rhs.locate(indices);
      auto out_ptr = (scalar_t*) out.locate(indices);
      *out_ptr = *lhs_ptr + *rhs_ptr;
      return true;
    });
  });
  create_backward_node<AddBackward>(out, {lhs, rhs});
  return out;
}

// used in backward. Correspond to broadcast in forward.
static Tensor reduce(Tensor self, const std::vector<int>& reduced_size) {
  assert(!is_grad_enabled());
  assert(self.dim() >= reduced_size.size());
  if (self.dim() == reduced_size.size()) {
    return self;
  }
  int off = self.dim() - reduced_size.size();
  for (int i = 0; i < reduced_size.size(); ++i) {
    assert(self.sizes()[i + off] == reduced_size[i]);
  }
  Tensor out(reduced_size, self.dtype());
  DISPATCH_DTYPE(self.dtype(), [&]() {
    out.initWithScalar((scalar_t) 0);
    self.visit([&self, &out, off](const std::vector<int>& self_indices) {
      auto self_ptr = (scalar_t*) self.locate(self_indices);
      std::vector<int> out_indices(self_indices.begin() + off, self_indices.end());
      auto out_ptr = (scalar_t*) out.locate(out_indices);
      *out_ptr += *self_ptr;
      return true;
    });
  });
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

static Tensor eq(const Tensor& lhs, const Tensor& rhs) {
  assert(!lhs.requires_grad() && !rhs.requires_grad()); // TODO: no backward support right now
  Tensor out(lhs.sizes(), ScalarType::Bool);

  DISPATCH_DTYPE(lhs.dtype(), [&]() {
    out.visit([&lhs, &rhs, &out](const std::vector<int>& indices) {
      scalar_t* lhs_ptr = (scalar_t*) lhs.locate(indices);
      scalar_t* rhs_ptr = (scalar_t*) rhs.locate(indices);
      bool* out_ptr = (bool*) out.locate(indices);
      *out_ptr = (*lhs_ptr == *rhs_ptr);
      return true;
    });
  });
  return out;
}

static Tensor exp(const Tensor& in) {
  Tensor out(in.sizes(), in.dtype());
  out.visit([&in, &out](const std::vector<int>& indices) {
    using scalar_t = float; // TODO 
    auto in_ptr = (scalar_t*) in.locate(indices);
    auto out_ptr = (scalar_t*) out.locate(indices);
    *out_ptr = std::exp(*in_ptr);
    return true;
  });
  create_backward_node<ExpBackward>(out, {in});
  return out;
}

static Tensor log_softmax(const Tensor& inp, int dim) {
  assert(inp.dtype() == ScalarType::Float); // TODO
  using scalar_t = float;
  assert(inp.dim() == 2); // TODO only support 2 dim for now
  assert(dim == 1 || dim == -1); // TODO only support apply softmax to the lastdim for now

  int M = inp.sizes()[0];
  int N = inp.sizes()[1];
  Tensor out(inp.sizes(), inp.dtype());
  for (int r = 0; r < M; ++r) {
    scalar_t max_val = std::numeric_limits<scalar_t>::min();

    // pass 1: calculate max
    for (int c = 0; c < N; ++c) {
      scalar_t* inp_data_ptr = (scalar_t*) inp.locate(r, c);
      max_val = std::max(max_val, *inp_data_ptr);
    }

    // pass 2: calculate sum
    scalar_t sum = 0;
    for (int c = 0; c < N; ++c) {
      scalar_t* out_data_ptr = (scalar_t*) out.locate(r, c);
      scalar_t* inp_data_ptr = (scalar_t*) inp.locate(r, c);
      *out_data_ptr = *inp_data_ptr - max_val;
      sum += std::exp(*out_data_ptr);
    }
    sum = log(sum);

    // pass 3: divide by the sum
    for (int c = 0; c < N; ++c) {
      scalar_t* out_data_ptr = (scalar_t*) out.locate(r, c);
      *out_data_ptr -= sum;
    }
  }
  create_backward_node<LogSoftmaxBackward>(out, {inp});
  return out;
}

static Tensor sum(Tensor in) {
  assert(!in.requires_grad());

  // the sum of bool tensor should be int64 tensor
  ScalarType out_dtype = in.dtype();
  if (out_dtype == ScalarType::Bool) {
    out_dtype = ScalarType::Int64;
  }
  Tensor out({}, out_dtype);

  // NOTE: To sum a bool tensor, alternatively we could also convert the bool tensor to int64
  // tensor first. Thus the input/output tensor has the same dtype.
  DISPATCH_DTYPE(in.dtype(), [&]() {
    DISPATCH_DTYPE_WITH_NAME(out.dtype(), [&]() {
      out_scalar_t* out_ptr = (out_scalar_t*) out.data(); 
      *out_ptr = 0;
      in.visit([&in, out_ptr](const std::vector<int>& indices) {
        *out_ptr += *(scalar_t*) in.locate(indices);
        return true;
      });
    }, out_scalar_t);
  });
  return out;
}

static Tensor sum(Tensor in, int dim) {
  if (dim < 0) {
    dim += in.dim();
  }
  assert(dim == in.dim() - 1); // only support sum across the last dimension for now
  int M = in.sizes()[0], N = in.sizes()[1];
  Tensor out({M}, in.dtype());
  assert(in.dtype() == ScalarType::Float);
  using scalar_t = float; // TODO
  for (int r = 0; r < M; ++r) {
    scalar_t tot = 0.0f;
    for (int c = 0; c < N; ++c) {
      scalar_t* in_ptr = (scalar_t*) in.locate(r, c);
      tot += *in_ptr;
    }
    scalar_t* out_ptr = (scalar_t*) out.locate({r});
    *out_ptr = tot;
  }
  create_backward_node<SumBackward>(out, {in});
  return out;
}

// return a view of the input tensor
static Tensor unsqueeze(const Tensor& in, int dim) {
  assert(dim >= 0 && dim <= in.dim());
  std::vector<int> newsize = in.sizes();
  std::vector<int> newstride = in.strides();
  newsize.insert(newsize.begin() + dim, 1);
  newstride.insert(newstride.begin() + dim, 0);

  // TODO: this implementation has different semantic as pytorch since
  // Tensor `in' is inplace updated. Pytorch's implementation does not mutate
  // tensor `in'.
  Tensor out = in;
  out.resize(newsize, newstride);
  create_backward_node<UnsqueezeBackward>(out, {in});
  return out;
}

// TODO: implement transpose as a view and only tune strides?
static Tensor transpose(Tensor in, int dim1, int dim2) {
  assert(dim1 != dim2);
  std::vector<int> out_size = in.sizes();
  std::swap(out_size[dim1], out_size[dim2]);
  Tensor out(out_size, in.dtype());

  DISPATCH_DTYPE(in.dtype(), [&]() {
    out.visit([&in, &out, dim1, dim2](const std::vector<int>& out_indices) {
      std::vector<int> in_indices = out_indices;
      std::swap(in_indices[dim1], in_indices[dim2]);
      auto out_ptr = (scalar_t *) out.locate(out_indices);
      auto in_ptr = (scalar_t *) in.locate(in_indices);
      *out_ptr = *in_ptr;
      return true;
    });
  });
  create_backward_node(out, {in}, [&in, dim1, dim2]() {
    return new TransposeBackward({in}, dim1, dim2);
  });
  return out;
}

// pred is the result of log_softmax
static Tensor nll_loss(const Tensor& pred, const Tensor& label) {
  assert(pred.dim() == 2);
  assert(label.dim() == 1);
  int M = pred.sizes()[0];
  int N = pred.sizes()[1];
  assert(M == label.sizes()[0]);

  Tensor out({M}, pred.dtype());

  assert(label.dtype() == ScalarType::Int64);
  DISPATCH_DTYPE(pred.dtype(), [&]() {
    for (int i = 0; i < M; ++i) {
      uint64_t idx = *(int64_t*) label.locate({i}); 
      assert(idx < N);
      scalar_t logprob = *(scalar_t*) pred.locate(i, idx);
      *(scalar_t*) out.locate({i}) = -logprob;
    }
  });

  create_backward_node<NLLLossBackward>(out, {pred, label});
  // TODO: unlike pytorch, we don't do reduction here
  return out;
}

static void zero_(Tensor inp) {
  assert(!inp.requires_grad() && "Don't handle backward for inplace op yet");
  DISPATCH_DTYPE(inp.dtype(), [&]() {
    inp.visit([&inp](const std::vector<int>& indices) {
      auto inp_ptr = (scalar_t*) inp.locate(indices);
      *inp_ptr = 0;
      return true;
    });
  });
}

static void add_(Tensor self, Tensor other, double alpha) {
  // only used in optimizer step right now. So assume no_grad mode
  assert(!is_grad_enabled());
  DISPATCH_DTYPE(self.dtype(), [&]() {
    self.visit([&self, &other, alpha](const std::vector<int>& indices) {
      auto self_ptr = (scalar_t*) self.locate(indices);
      auto other_ptr = (scalar_t*) other.locate(indices);
      *self_ptr = *self_ptr + *other_ptr * alpha;
      return true;
    });
  });
}

static void uniform_(Tensor self, double lb, double ub) {
  assert(!is_grad_enabled());
  DISPATCH_DTYPE(self.dtype(), [&]() {
    self.visit([&self, lb, ub](const std::vector<int>& indices) {
      Generator gen;
      auto itemptr = (scalar_t*) self.locate(indices);
      *itemptr = gen.uniform(lb, ub);
      return true;
    });
  });
}

static Tensor divScalar(Tensor self, int other) {
  assert(!self.requires_grad()); // don't support grad yet
  Tensor out(self.sizes(), ScalarType::Float);
  DISPATCH_DTYPE(self.dtype(), [&]() {
    self.visit([&self, &out, other](const std::vector<int>& indices) {
      float other_f = other;
      auto self_val = *(scalar_t*) self.locate(indices);
      float self_val_f = (float) self_val;
      *(float*) out.locate(indices) = self_val_f / other_f;
      return true;
    });
  });
  return out;
}

static std::vector<int> calculate_indices_from_slice(int sz, py::slice slice_obj) {
  size_t start, stop, step, slice_len;
  bool status = slice_obj.compute(sz, &start, &stop, &step, &slice_len);
  assert(status);
  std::vector<int> indices(slice_len);
  size_t cur = start;
  for (int i = 0; i < slice_len; ++i) {
    indices[i] = cur;
    cur += step;
  }
  return indices;
}

static Tensor slice(Tensor self, py::slice slice_obj) {
  assert(!self.requires_grad()); // TODO: don't support backward for slice so far
  assert(self.dim() > 0);
  std::vector<int> dim0_index_list = calculate_indices_from_slice(self.sizes()[0], slice_obj);
  std::vector<int> out_sizes = self.sizes();
  out_sizes[0] = dim0_index_list.size();
  Tensor out(out_sizes, self.dtype());
  
  if (out.numel() > 0) {
    DISPATCH_DTYPE(self.dtype(), [&]() {
      for (int i = 0; i < dim0_index_list.size(); ++i) {
        int dim0_index = dim0_index_list[i];
        std::vector<int> in_indices = self.start_indices();
        in_indices[0] = dim0_index;
        std::vector<int> out_indices;
        do {
          auto in_val = *(scalar_t*) self.locate(in_indices);
          out_indices = in_indices;
          out_indices[0] = i;
          *(scalar_t*) out.locate(out_indices) = in_val;
        } while (self.next_indices(in_indices) && in_indices[0] == dim0_index);
      }
    });
  }
  return out;
}

}
