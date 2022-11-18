#pragma once

#include "light/csrc/backward_node.h"

class MeanBackward : public BackwardNode {
 public:
  using BackwardNode::BackwardNode;

  void run(Tensor out, Tensor out_grad) {
    assert(inputs_.size() == 1);
    Tensor& inp = inputs_[0];
    assert(inp.requires_grad());

    // calculate input gradient
    assert(out_grad.dim() == 0);
    using scalar_t = float; // TODO
    scalar_t val = 1.0 / inp.numel();
    Tensor inp_grad(inp.sizes(), inp.dtype());
    inp_grad.initWithScalar(val);

    propagate({inp_grad});
  }
};

class MatmulBackward : public BackwardNode {
 public:
  using BackwardNode::BackwardNode;

  void run(Tensor out, Tensor out_grad);
};

class AddBackward : public BackwardNode {
 public:
  using BackwardNode::BackwardNode;

  void run(Tensor out, Tensor out_grad) {
    Tensor lhs = inputs_[0];
    Tensor rhs = inputs_[1];
    // TODO: instead of clone out_grad, we should do accumulation
    // to handle the case that lhs and rhs are the same tensor.
    Tensor lhs_grad = Tensor::dummy;
    Tensor rhs_grad = Tensor::dummy;
    if (lhs.requires_grad()) {
      lhs_grad = out_grad.clone().reduce(lhs.sizes());
    }
    if (rhs.requires_grad()) {
      rhs_grad = out_grad.clone().reduce(rhs.sizes());
    }
    propagate({lhs_grad, rhs_grad});
  }
};

class SubBackward : public BackwardNode {
 public:
  using BackwardNode::BackwardNode;

  void run(Tensor out, Tensor out_grad) {
    assert(false && "SubBackward::run ni"); // TODO
  }
};

class MulBackward : public BackwardNode {
 public:
  using BackwardNode::BackwardNode;

  void run(Tensor out, Tensor out_grad) {
    assert(false && "MulBackward::run ni"); // TODO
  }
};

class TransposeBackward : public BackwardNode {
 public:
  explicit TransposeBackward(const std::vector<Tensor>& inputs, int dim1, int dim2)
    : BackwardNode(inputs), dim1_(dim1), dim2_(dim2) { }

  void run(Tensor out, Tensor out_grad) {
    Tensor in = inputs_[0];
    assert(in.requires_grad());

    Tensor in_grad = out_grad.transpose(dim1_, dim2_);
    propagate({in_grad});
  }
 private:
  int dim1_;
  int dim2_;
};

class ExpBackward : public BackwardNode {
 public:
  using BackwardNode::BackwardNode;

  void run(Tensor out, Tensor out_grad) {
    assert(false && "ExpBackward::run ni"); // TODO
  }
};

class SumBackward : public BackwardNode {
 public:
  using BackwardNode::BackwardNode;

  void run(Tensor out, Tensor out_grad) {
    assert(false && "SumBackward::run ni"); // TODO
  }
};

class UnsqueezeBackward : public BackwardNode {
 public:
  using BackwardNode::BackwardNode;

  void run(Tensor out, Tensor out_grad) {
    assert(false && "UnsqueezeBackward::run ni"); // TODO
  }
};

class NLLLossBackward : public BackwardNode {
 public:
  using BackwardNode::BackwardNode;

  void run(Tensor out, Tensor out_grad) {
    Tensor pred = inputs_[0]; 
    assert(pred.requires_grad());
    Tensor label = inputs_[1];

    Tensor pred_grad(pred.sizes(), pred.dtype());
    assert(pred.dim() == 2);
    int M = pred.sizes()[0];
    int N = pred.sizes()[1];
    DISPATCH_DTYPE(pred.dtype(), [&]() {
      for (int i = 0; i < M; ++i) {
        int idx = *(int64_t*) label.locate({i});
        scalar_t grad_val = *(scalar_t*) out_grad.locate({i});
        for (int j = 0; j < N; ++j) {
          scalar_t *grad_ptr = (scalar_t*) pred_grad.locate({i, j});    
          if (j == idx) {
            *grad_ptr = -grad_val;
          } else {
            *grad_ptr = 0;
          }
        }
      }
    });
    propagate({pred_grad, Tensor::dummy});
  }
};

class LogSoftmaxBackward : public BackwardNode {
 public:
  using BackwardNode::BackwardNode;

  /*
   * Consider calculating gradients for xi
   *   sigma = exp(x1) + exp(x2) + ... + exp(xn)
   * 
   *   yi = log(exp(xi) / sigma)
   *
   *   grad_xi(yi) -- the graidnet regarding xi. i.e. dyi/dxi
   *   = grad_xi(log(exp(xi) / sigma))
   *   = sigma / exp(xi) * [ exp(xi) * sigma - exp(xi) * exp(xi) ] / sigma / sigma
   *   = [sigma - exp(xi)] / sigma
   *   = 1 - exp(xi) / sigma
   *   = 1 - exp(yi)
   *
   *   grad_xi(yj) -- where j != i
   *   = grad_xi( log(exp(xj) / sigma) )
   *   = sigma / exp(xj) * exp(xj) * (-1) / sigma / sigma * exp(xi)
   *   = (-1) / sigma * exp(xi)
   *   = -exp(yi) -- it's interesting that this graident is the same for different value of j !
   *
   *   grad_xi(loss)
   *   = sum(for j in 1..n: grad_yj(loss) * grad_xi(yj)) 
   *   = grad_yi(loss) + sum(for j in 1..n: grad_yj(loss) * [-exp(yi)]
   *   = grad_yi(loss) - exp(yi) * sum(for j in 1..n: grad_yj(loss)) -- the final formula!
   */
  void run(Tensor out, Tensor out_grad) {
    assert(out.dim() == 2); // TODO: only support 2 dim for now
    #if 1
    // compose this op with existing ops
    Tensor in_grad = out_grad - out.exp() * out_grad.sum(-1).unsqueeze(1);
    #else
    Tensor in_grad(out_grad.sizes(), out_grad.dtype());
    int M = out_grad.sizes()[0];
    int N = out_grad.sizes()[1];
    using scalar_t = float; // TODO
    std::vector<scalar_t> sumvec(M, 0.0f);

    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        sumvec[i] += *(scalar_t*) out_grad.locate(i, j);
      }
      for (int j = 0; j < N; ++j) {
        scalar_t newval;
        newval = *(scalar_t*) out_grad.locate(i, j) - std::exp(*(scalar_t*) out.locate(i, j)) * sumvec[i];
        *(scalar_t*) in_grad.locate(i, j) = newval;
      }
    }
    #endif

    propagate({in_grad});
  }
};

class ReluBackward : public BackwardNode {
 public:
  using BackwardNode::BackwardNode;

  void run(Tensor out, Tensor out_grad) {
    // TODO: does it matter if we assign 0 or 1 as gradient when input is 0
    // TODO: it will be more general to implement as: inp_grad = (inp > 0) * out_grad
    Tensor inp = inputs_[0];
    Tensor inp_grad = out_grad.clone();
    inp_grad.visit([&inp, &inp_grad](const std::vector<int>& indices) {
      using scalar_t = float; // TODO
      auto inp_ptr = (scalar_t*) inp.locate(indices);
      auto inp_grad_ptr = (scalar_t*) inp_grad.locate(indices);
      if (*inp_ptr <= 0.0f) {
        *inp_grad_ptr = 0.0f;
      }
      return true;
    });
    propagate({inp_grad});
  }
};

class SigmoidBackward : public BackwardNode {
 public:
  using BackwardNode::BackwardNode;

  void run(Tensor out, Tensor out_grad) {
    // TODO: allow automatic scalar to Tensor conversion
    Tensor inp_grad = out_grad * out * (Tensor::create_scalar_tensor(1.0f) - out);
    propagate({inp_grad});
  }
};

class Conv2dBackward : public BackwardNode {
 public:
  using BackwardNode::BackwardNode;

  void run(Tensor out, Tensor out_grad) {
    assert(false && "Conv2dBackward::run ni"); // TODO
  }
};

class MaxPool2dBackward : public BackwardNode {
 public:
  using BackwardNode::BackwardNode;

  void run(Tensor out, Tensor out_grad) {
    assert(false && "MaxPool2dBackward::run ni"); // TODO
  }
};
