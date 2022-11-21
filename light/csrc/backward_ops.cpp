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

void MaxPool2dBackward::run(Tensor out, Tensor out_grad) {
  Tensor in = inputs_[0];
  assert(in.requires_grad());

  Tensor in_grad(in.sizes(), in.dtype());
  in_grad.zero_();
  int N = in.sizes()[0];
  int C = in.sizes()[1];

  DISPATCH_DTYPE(in_grad.dtype(), [&]() {
    for (int n = 0; n < N; ++n) {
      for (int c = 0; c < C; ++c) {
        for (int outh_idx = 0; outh_idx < out.sizes()[2]; ++outh_idx) {
          for (int outw_idx = 0; outw_idx < out.sizes()[3]; ++outw_idx) {
            scalar_t& out_grad_val = *(scalar_t*) out_grad.locate({n, c, outh_idx, outw_idx});

            scalar_t max_val = std::numeric_limits<scalar_t>::min();
            #if 0
            // TODO: why this cause build failure?
            int best_inh_idx = -1, best_inw_idx = -1;
            #else
            int best_inh_idx = -1;
            int best_inw_idx = -1;
            #endif
            scalar_t in_val;

            for (int kerh_idx = 0; kerh_idx < kernel_size_[0]; ++kerh_idx) {
              for (int kerw_idx = 0; kerw_idx < kernel_size_[1]; ++kerw_idx) {
                // obtain in_val
                int inh_idx = outh_idx * stride_[0] + kerh_idx - padding_[0];
                int inw_idx = outw_idx * stride_[1] + kerw_idx - padding_[1];
                if (inh_idx >= 0 && inh_idx < in.sizes()[2] && inw_idx >= 0 && inw_idx < in.sizes()[3]) {
                  in_val = *(scalar_t*) in.locate({n, c, inh_idx, inw_idx});
                  if (best_inh_idx < 0 || in_val > max_val) {
                    max_val = in_val;
                    best_inh_idx = inh_idx;
                    best_inw_idx = inw_idx;
                  }
                }
              }
            }

            if (best_inh_idx >= 0) {
              scalar_t& in_grad_val = *(scalar_t*) in_grad.locate({n, c, best_inh_idx, best_inw_idx});
              in_grad_val += out_grad_val;
            }
          }
        }
      }
    }
  });

  propagate({in_grad});
}

void AdaptiveAvgPool2dBackward::run(Tensor out, Tensor out_grad) {
  Tensor in = inputs_[0];
  assert(in.requires_grad());

  Tensor in_grad(in.sizes(), in.dtype());
  in_grad.zero_();

  int N = out.sizes()[0];
  int C = out.sizes()[1];
  int outh = out.sizes()[2];
  int outw = out.sizes()[3];

  int inh = in.sizes()[2];
  int inw = in.sizes()[3];

  DISPATCH_DTYPE(in_grad.dtype(), [&]() {
    for (int n = 0; n < N; ++n) {
      for (int c = 0; c < C; ++c) {
        for (int outh_idx = 0; outh_idx < outh; ++outh_idx) {
          for (int outw_idx = 0; outw_idx < outw; ++outw_idx) {
            scalar_t& out_grad_val = *(scalar_t*) out_grad.locate({n, c, outh_idx, outw_idx});
            int inh_start = calc_start_idx(outh_idx, outh, inh);
            int inh_end = calc_end_idx(outh_idx, outh, inh);
            int inw_start = calc_start_idx(outw_idx, outw, inw);
            int inw_end = calc_end_idx(outw_idx, outw, inw);

            int cnt = (inh_end - inh_start) * (inw_end - inw_start);
            assert(cnt > 0);
            scalar_t factor = 1.0 / cnt;
            for (int inh_idx = inh_start; inh_idx < inh_end; ++inh_idx) {
              for (int inw_idx = inw_start; inw_idx < inw_end; ++inw_idx) {
                scalar_t& in_grad_val = *(scalar_t*) in_grad.locate({n, c, inh_idx, inw_idx});
                in_grad_val += out_grad_val * factor;
              }
            }
          }
        }
      }
    }
  });
  propagate({in_grad});
}
