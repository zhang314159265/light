#pragma once

#include <vector>
#include <cassert>
#include <memory>
#include <numeric>
#include <functional>
#include <iostream>
#include <sstream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "light/csrc/rand.h"
#include "light/csrc/config.h"

namespace py = pybind11;

enum ScalarType {
  Float = 0,
  Double = 1,
  Int64 = 2,
  Bool = 3,
};

template <typename T>
static ScalarType scalarTypeFromCTypeToEnum() {
  assert(false && "must specialize");
}

template <> ScalarType scalarTypeFromCTypeToEnum<float>() { return Float; }
template <> ScalarType scalarTypeFromCTypeToEnum<double>() { return Double; }
template <> ScalarType scalarTypeFromCTypeToEnum<int64_t>() { return Int64; }

#define DISPATCH_DTYPE_WITH_NAME(dtype, worker, scalar_t) do { \
  switch (dtype) {\
  case ScalarType::Float: { \
    using scalar_t = float; \
    worker(); \
    break; \
  } \
  case ScalarType::Double: { \
    using scalar_t = double; \
    worker(); \
    break; \
  } \
  case ScalarType::Int64: { \
    using scalar_t = int64_t; \
    worker(); \
    break; \
  } \
  case ScalarType::Bool: { \
    using scalar_t = bool; \
    worker(); \
    break; \
  } \
  default: \
    assert(false && "unrecognized dtype"); \
  } \
} while(false)

#define DISPATCH_DTYPE(dtype, worker) DISPATCH_DTYPE_WITH_NAME(dtype, worker, scalar_t)

static inline int scalar_type_nbytes(ScalarType dtype) {
  switch (dtype) {
  case ScalarType::Float:
    return 4;
  case ScalarType::Double:
    return 8;
  case ScalarType::Int64:
    return 8;
  case ScalarType::Bool:
    return 1;
  default:
    assert(false && "element_size unhandled branch");
  }
}

static inline std::vector<int> contiguous_strides(const std::vector<int> sizes) {
  std::vector<int> strides(sizes.size(), 1);
  for (int i = sizes.size() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * sizes[i + 1];
  }
  return strides;
}

static inline int compute_numel(const std::vector<int>& sizes) {
  return std::accumulate(sizes.begin(), sizes.end(), 1, std::multiplies<int>());
}

class BackwardNode;
class Tensor;

class TensorImpl {
 public:
  explicit TensorImpl(const std::vector<int>& sizes, ScalarType dtype=Float)
      : sizes_(sizes), dtype_(dtype) {
    strides_ = contiguous_strides(sizes);
    numel_ = compute_numel(sizes);
    capacity_ = numel_ * scalar_type_nbytes(dtype);
    data_ = malloc(capacity_);
  }

  ~TensorImpl();

 private:
  std::vector<int> sizes_;
  std::vector<int> strides_;
  int numel_;
  void *data_;
  int capacity_;
  ScalarType dtype_;
  bool requires_grad_ = false;
  bool is_param_ = false;

  // Can not define backward_node_ as unique_ptr since in that case
  // 1. TensorImpl will require complete definition of BackwardNode
  // 2. BackwardNode requires complete definition of Tensor
  // The depecencies will be hard to manage
  // std::unique_ptr<BackwardNode> backward_node_;
  BackwardNode* backward_node_ = nullptr;
  Tensor* grad_ = nullptr;

  friend class Tensor;
};

class Tensor {
 public:
  explicit Tensor(const std::vector<int>& sizes, ScalarType dtype)
    : impl_(new TensorImpl(sizes, dtype)) {
  }

  // TODO support type other than float
  static Tensor create_scalar_tensor(float val) {
    Tensor out({}, ScalarType::Float);
    out.set_item(val);
    return out;
  }

  Tensor clone() const {
    Tensor out = Tensor(sizes(), dtype());
    assert(out.impl_->capacity_ == impl_->capacity_);
    memcpy(out.impl_->data_, impl_->data_, impl_->capacity_);
    return out;
  }

  static Tensor dummy; // a dummy tensor. The value does not matter

  // TODO support type other than float
  void set_item(float val) {
    assert(dtype() == ScalarType::Float);
    assert(dim() == 0);
    *((float *) data()) = val;
  }

  template <typename T>
  T item() const {
    assert(dim() == 0);
    assert(scalarTypeFromCTypeToEnum<T>() == dtype());
    return *((T *) data());
  }

  bool requires_grad() const {
    return impl_->requires_grad_;
  }

  void set_requires_grad(bool requires_grad) {
    impl_->requires_grad_ = requires_grad;
  }

  bool is_param() const {
    return impl_->is_param_;
  }

  void set_is_param(bool is_param) {
    impl_->is_param_ = is_param;
  }

  void resize(const std::vector<int>& newsizes, const std::vector<int>& newstrides) {
    assert(numel() == compute_numel(newsizes));
    impl_->sizes_ = newsizes;
    impl_->strides_ = newstrides;
  }

  BackwardNode* backward_node() {
    return impl_->backward_node_;
  }

  void set_backward_node(BackwardNode* node) {
    impl_->backward_node_ = node;
  }

  Tensor grad() {
    assert(impl_->grad_);
    return *(impl_->grad_);
  }

  Tensor* grad_ptr() {
    return impl_->grad_;
  }

  void set_grad(Tensor grad) {
    if (!config_keep_grad_for_nonleaf()) {
      // only set gradient for leaf node
      assert(!backward_node());
    }

    // accumulate the grad
    if (impl_->grad_) {
      *impl_->grad_ = *impl_->grad_ + grad;
    } else {
      impl_->grad_ = new Tensor(grad);
    }
  }

  const std::vector<int>& sizes() const {
    return impl_->sizes_;
  }

  const std::vector<int>& strides() const {
    return impl_->strides_;
  }

  int numel() const {
    return impl_->numel_;
  }

  ScalarType dtype() const {
    return impl_->dtype_;
  }

  int element_size() const {
    return scalar_type_nbytes(dtype());
  }

  int dim() const {
    return sizes().size();
  }

  void *data() {
    return impl_->data_;
  }

  void *data() const {
    return impl_->data_;
  }

  void *locate(int i, int j) const {
    return locate(std::vector<int>{i, j});
  }

  void *locate(const std::vector<int>& indices) {
    // indices.size() > dim() for broadcasting
    assert(indices.size() >= dim());
    int off = indices.size() - dim();
    int idx = 0;
    for (int i = 0; i < dim(); ++i) {
      idx += indices[i + off] * strides()[i];
    }
    return data() + idx * element_size();
  }

  void *locate(const std::vector<int>& indices) const {
    return ((Tensor*) this)->locate(indices);
  }

  /*
   * visitor return false to early terminate
   */
  template <typename T>
  void visit(T visitor) const {
    auto indices = start_indices();
    do {
      if (!visitor(indices)) {
        break;
      }
    } while (next_indices(indices));
  }

  template <typename T>
  void initWithScalar(T val) {
    // std::cout << "initWithScalar type " << typeid(val).name() << std::endl;
    visit([&val, this](const std::vector<int>& indices) {
      *(T *) locate(indices) = val;
      return true;
    });
  }

  std::string to_string() const {
    std::stringstream ss;
    ss << "Print tensor" << std::endl;
    DISPATCH_DTYPE(dtype(), [&]() {
      visit([this, &ss](const std::vector<int>& indices) {
        ss << *(scalar_t *) locate(indices) << std::endl;
        return true;
      });
    });
    return ss.str();
  }

  void print() const {
    std::cout << to_string();
  }

  friend Tensor operator+(const Tensor& lhs, const Tensor& rhs);
  friend Tensor operator-(const Tensor& lhs, const Tensor& rhs);
  friend Tensor operator*(const Tensor& lhs, const Tensor& rhs);
  Tensor mean() const;
  std::tuple<Tensor, Tensor> max(int dim) const;
  Tensor exp() const;
  // returns a scalar tensor for the sum
  Tensor sum() const;
  Tensor sum(int dim) const;
  Tensor unsqueeze(int dim) const;
  Tensor transpose(int dim1, int dim2) const;
  // broadcast in fwd is reduce in bwd
  Tensor reduce(const std::vector<int>& reduced_size) const;
  Tensor divScalar(int scalar) const;
  Tensor slice(py::slice slice_obj) const;

  void zero_();
  void add_(Tensor other, double alpha);
  void uniform_(double lb, double ub);

  bool equal(const Tensor& other) const {
    bool ans = true;
    visit([this, &other, &ans](const std::vector<int>& indices) {
      using scalar_t = float; // TODO 
      auto me_ptr = (scalar_t*) locate(indices);
      auto other_ptr = (scalar_t*) other.locate(indices);
      if (*me_ptr != *other_ptr) {
        ans = false;
        return false;
      } else {
        return true;
      }
    });
    return ans;
  }

  std::vector<int> start_indices() const {
    return std::vector<int>(dim(), 0);
  }

  bool next_indices(std::vector<int>& indices) const {
    for (int i = dim() - 1; i >= 0; --i) {
      ++indices[i];
      if (indices[i] == sizes()[i]) {
        indices[i] = 0;
      } else {
        return true;
      }
    }
    return false;
  }

  py::list tolist(std::vector<int>& indices) const {
    assert(dim() >= 1);
    assert(indices.size() < dim());

    py::list out;
    indices.push_back(-1);
    if (indices.size() < dim()) {
      for (int i = 0; i < sizes()[indices.size() - 1]; ++i) {
        indices.back() = i;
        out.append(tolist(indices));
      }
    } else {
      assert(indices.size() == dim());
      DISPATCH_DTYPE(dtype(), [&]() {
        for (int i = 0; i < sizes()[indices.size() - 1]; ++i) {
          indices.back() = i;
          auto val = *(scalar_t*) locate(indices);
          out.append(val);
        }
      });
    }
    indices.pop_back();
    return out;
  }

  void backward();
 private:
  std::shared_ptr<TensorImpl> impl_;
};

static inline Tensor createRandTensor(const std::vector<int>& sizes, ScalarType dtype) {
  Tensor out(sizes, dtype);
  Generator gen;
  out.visit([&gen, &out](const std::vector<int>& indices) {
    using scalar_t = float; // TODO
    auto itemptr = (scalar_t*) out.locate(indices);
    *itemptr = gen.uniform(0.0f, 1.0f);
    return true;
  });
  return out;
}

static inline Tensor createRandIntTensor(int low, int high, const std::vector<int>& sizes) {
  Tensor out(sizes, ScalarType::Int64);
  Generator gen;
  DISPATCH_DTYPE(out.dtype(), [&]() {
    out.visit([&gen, &out, &low, &high](const std::vector<int>& indices) {
      auto itemptr = (scalar_t*) out.locate(indices);
      *itemptr = gen.uniformInt(low, high);
      return true;
    });
  });
  return out;
}

// TODO can we borrow the storage from the np array directly to avoid a copy
template <typename scalar_t>
static inline Tensor createFromNpArray(py::array_t<scalar_t> ar) {
  py::buffer_info np_buf = ar.request();
  int ndim = np_buf.ndim;
  std::vector<int> shape(ndim);
  int size = 1;
  for (int i = 0; i < ndim; ++i) {
    shape[i] = np_buf.shape[i];
    size *= shape[i];
  }
  assert(size == np_buf.size);
  Tensor out(shape, scalarTypeFromCTypeToEnum<scalar_t>());

  scalar_t* np_ptr = (scalar_t*) np_buf.ptr;
  scalar_t* data_ptr = (scalar_t*) out.data();
  for (int i = 0; i < size; ++i) {
    data_ptr[i] = np_ptr[i];
  }
  return out;
}

std::vector<int> get_broadcast_shape(const std::vector<int>& lhs_shape, const std::vector<int>& rhs_shape);
std::vector<int> get_broadcast_shape(std::vector<Tensor> tensors);
