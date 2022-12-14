#pragma once

#include <cassert>
#include "light/csrc/backward.h"
#include "light/csrc/backward_ops.h"
#include "light/csrc/rand.h"
#include "light/csrc/adaptive_pool_helper.h"

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

static int calc_conv_out_size(int in_size, int padding, int kernel_size, int stride) {
  return (in_size + padding * 2 - kernel_size) / stride + 1;
}

static Tensor conv2d(Tensor in, Tensor weight, Tensor bias, std::vector<int> stride, std::vector<int> padding) {
  assert(stride.size() == 2);
  assert(padding.size() == 2);
  assert(in.dim() == 4);
  assert(weight.dim() == 4);
  assert(bias.dim() == 1);
  assert(bias.sizes()[0] == weight.sizes()[0]);
  assert(in.sizes()[1] == weight.sizes()[1]);

  int batch_size = in.sizes()[0];
  int out_channel = weight.sizes()[0];
  int in_channel = weight.sizes()[1];
  std::vector<int> out_sizes = in.sizes();

  out_sizes[1] = out_channel;

  for (int i = 0; i < 2; ++i) {
    out_sizes[i + 2] = calc_conv_out_size(in.sizes()[i + 2], padding[i], weight.sizes()[i + 2], stride[i]);
  }
  Tensor out(out_sizes, in.dtype());

  DISPATCH_DTYPE(in.dtype(), [&]() {
    for (int n = 0; n < batch_size; ++n) {
      for (int outc = 0; outc < out_channel; ++outc) {
        for (int outh_idx = 0; outh_idx < out.sizes()[2]; ++outh_idx) {
          for (int outw_idx = 0; outw_idx < out.sizes()[3]; ++outw_idx) {
            scalar_t& out_val = *(scalar_t*) out.locate({n, outc, outh_idx, outw_idx});
            scalar_t& bias_val = *(scalar_t*) bias.locate({outc});
            out_val = bias_val;
            
            for (int inc = 0; inc < in_channel; ++inc) {
              for (int kerh_idx = 0; kerh_idx < weight.sizes()[2]; ++kerh_idx) {
                for (int kerw_idx = 0; kerw_idx < weight.sizes()[3]; ++kerw_idx) {
                  // obtain weight
                  scalar_t& weight_val = *(scalar_t*) weight.locate({outc, inc, kerh_idx, kerw_idx});
                  // obtain in_val
                  int inh_idx = outh_idx * stride[0] + kerh_idx - padding[0];
                  int inw_idx = outw_idx * stride[1] + kerw_idx - padding[1];
                  scalar_t in_val = 0;
                  if (inh_idx >= 0 && inh_idx < in.sizes()[2] && inw_idx >= 0 && inw_idx < in.sizes()[3]) {
                    in_val = *(scalar_t*) in.locate({
                      n, inc, inh_idx, inw_idx
                    });
                  }
                  out_val += weight_val * in_val;
                }
              }
            }
          }
        }
      }
    }
  });

  create_backward_node(out, {in, weight, bias}, [&]() {
    return new Conv2dBackward({in, weight, bias}, stride, padding);
  });
  return out;
}

static Tensor max_pool2d(Tensor in, std::vector<int> kernel_size, std::vector<int> padding, std::vector<int> stride) {
  assert(kernel_size.size() == 2);
  assert(padding.size() == 2);
  assert(stride.size() == 2);
  assert(in.dim() == 4);
  int N = in.sizes()[0];
  int C = in.sizes()[1];
  std::vector<int> out_sizes = in.sizes();
  for (int i = 0; i < 2; ++i) {
    out_sizes[i + 2] = calc_conv_out_size(
      in.sizes()[i + 2], padding[i], kernel_size[i], stride[i]
    );
  }
  Tensor out(out_sizes, in.dtype());
  DISPATCH_DTYPE(in.dtype(), [&]() {
    for (int n = 0; n < N; ++n) {
      for (int c = 0; c < C; ++c) {
        for (int outh_idx = 0; outh_idx < out.sizes()[2]; ++outh_idx) {
          for (int outw_idx = 0; outw_idx < out.sizes()[3]; ++outw_idx) {
            scalar_t& out_val = *(scalar_t*) out.locate({n, c, outh_idx, outw_idx});
            out_val = std::numeric_limits<scalar_t>::min();

            for (int kerh_idx = 0; kerh_idx < kernel_size[0]; ++kerh_idx) {
              for (int kerw_idx = 0; kerw_idx < kernel_size[1]; ++kerw_idx) {
                // obtain in_val
                int inh_idx = outh_idx * stride[0] + kerh_idx - padding[0];
                int inw_idx = outw_idx * stride[1] + kerw_idx - padding[1];
                if (inh_idx >= 0 && inh_idx < in.sizes()[2] && inw_idx >= 0 && inw_idx < in.sizes()[3]) {
                  out_val = std::max(out_val, *(scalar_t*) in.locate({n, c, inh_idx, inw_idx}));
                }
              }
            }
          }
        }
      }
    }
  });

  create_backward_node(out, {in}, [&]() {
    return new MaxPool2dBackward({in}, kernel_size, padding, stride);
  }); 
  return out;
}

static Tensor dropout(Tensor in, bool train, double p) {
  if (!train) {
    return in;
  }
  assert(p > 0 && p < 1);
  Tensor out(in.sizes(), in.dtype());
  Generator gen;
  DISPATCH_DTYPE(in.dtype(), [&]() {
    double factor = 1.0 / (1 - p);
    out.visit([&](const std::vector<int>& indices) {
      scalar_t& out_val = *(scalar_t*) out.locate(indices);
      scalar_t& in_val = *(scalar_t*) in.locate(indices);
      // PyTorch compare the prob with (1-p) rather than p using a double
      // precision. We could compare with p and use single precision. But
      // to maintain numeric parity with PyTorch we compare with 1-p and
      // use double precision.
      if (gen.uniform64(0, 1) >= (1 - p)) {
        out_val = 0;
      } else {
        out_val = in_val * factor;
      }
      return true;
    });
  });
  create_backward_node<DropoutBackward>(out, {in});
  return out;
}

static Tensor adaptive_avg_pool2d(Tensor in, std::vector<int> outhw) {
  assert(outhw.size() == 2);
  assert(in.dim() == 4);

  int N = in.sizes()[0];
  int C = in.sizes()[1];
  int inh = in.sizes()[2];
  int inw = in.sizes()[3];

  int outh = outhw[0];
  int outw = outhw[1];
  assert(inh >= outh);
  assert(inw >= outw);
  Tensor out({N, C, outh, outw}, in.dtype());

  // TODO should we use double to do accumulation?
  DISPATCH_DTYPE(in.dtype(), [&]() {
    for (int n = 0; n < N; ++n) {
      for (int c = 0; c < C; ++c) {
        for (int outh_idx = 0; outh_idx < outh; ++outh_idx) {
          for (int outw_idx = 0; outw_idx < outw; ++outw_idx) {
            scalar_t& out_val = *(scalar_t*) out.locate({n, c, outh_idx, outw_idx});
            int inh_start = calc_start_idx(outh_idx, outh, inh);
            int inh_end = calc_end_idx(outh_idx, outh, inh);
            int inw_start = calc_start_idx(outw_idx, outw, inw);
            int inw_end = calc_end_idx(outw_idx, outw, inw);

            int cnt = (inh_end - inh_start) * (inw_end - inw_start);
            assert(cnt > 0);
            scalar_t sum = 0;
            for (int inh_idx = inh_start; inh_idx < inh_end; ++inh_idx) {
              for (int inw_idx = inw_start; inw_idx < inw_end; ++inw_idx) {
                scalar_t& in_val = *(scalar_t*) in.locate({n, c, inh_idx, inw_idx});
                sum += in_val;
              }
            }
            out_val = sum / cnt;
          }
        }
      }
    }
  });

  create_backward_node<AdaptiveAvgPool2dBackward>(out, {in});
  return out;
}

static Tensor reshape(Tensor in, std::vector<int> out_shape) {
  Tensor out(out_shape, in.dtype());
  assert(out.numel() == in.numel());

  DISPATCH_DTYPE(in.dtype(), [&]() {
    scalar_t* out_ptr = (scalar_t*) out.data();
    in.visit([&](const std::vector<int>& in_indices) {
      scalar_t in_val = *(scalar_t*) in.locate(in_indices);
      *out_ptr++ = in_val;
      return true;
    });
  });
  create_backward_node<ReshapeBackward>(out, {in});
  return out;
}

static Tensor batch_norm(Tensor in, Tensor running_mean, Tensor running_var, Tensor weight, Tensor bias, bool training, double momentum, double eps) {
  Tensor out(in.sizes(), in.dtype());

  int BS = in.sizes()[0];
  int C = in.sizes()[1];
  int H = in.sizes()[2];
  int W = in.sizes()[3];
  int N = in.numel() / C;

  DISPATCH_DTYPE(in.dtype(), [&]() {
    for (int c = 0; c < C; ++c) {
      #if 0
      // why this can not compile?
      scalar_t mean, var;
      #else
      scalar_t mean;
      scalar_t var;
      #endif

      // calculate mean & var
      if (training) {
        mean = 0;
        scalar_t var_sum = 0.0;

        for (int batch_idx = 0; batch_idx < BS; ++batch_idx) {
        for (int h_idx = 0; h_idx < H; ++h_idx) {
        for (int w_idx = 0; w_idx < W; ++w_idx) {
          mean += *(scalar_t*) in.locate({batch_idx, c, h_idx, w_idx});
        } } }
        mean /= N;
        for (int batch_idx = 0; batch_idx < BS; ++batch_idx) {
        for (int h_idx = 0; h_idx < H; ++h_idx) {
        for (int w_idx = 0; w_idx < W; ++w_idx) {
          scalar_t in_val = *(scalar_t*) in.locate({batch_idx, c, h_idx, w_idx});
          var_sum += (in_val - mean) * (in_val - mean);
        } } }

        // update the running mean/var
        scalar_t& running_mean_val = *(scalar_t*) running_mean.locate({c});
        scalar_t& running_var_val = *(scalar_t*) running_var.locate({c});
        running_mean_val = running_mean_val * (1 - momentum) + mean * momentum;
        running_var_val = running_var_val * (1 - momentum) + var_sum / (N - 1) * momentum;

        var = var_sum / N;
      } else {
        mean = *(scalar_t*) running_mean.locate({c});
        var = *(scalar_t*) running_var.locate({c});
      }

      // calculate out Tensor
      scalar_t weight_val = *(scalar_t*) weight.locate({c});
      scalar_t bias_val = *(scalar_t*) bias.locate({c});
      for (int batch_idx = 0; batch_idx < BS; ++batch_idx) {
      for (int h_idx = 0; h_idx < H; ++h_idx) {
      for (int w_idx = 0; w_idx < W; ++w_idx) {
        scalar_t& in_val = *(scalar_t*) in.locate({batch_idx, c, h_idx, w_idx});
        scalar_t& out_val = *(scalar_t*) out.locate({batch_idx, c, h_idx, w_idx});

        out_val = (in_val - mean) / sqrt(var + eps) * weight_val + bias_val;
      } } }
    }
  });

  create_backward_node<BatchNormBackward>(out, {in});
  return out;
}

}
