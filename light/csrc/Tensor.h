#pragma once

#include <vector>
#include <cassert>
#include <memory>
#include <numeric>
#include <functional>
#include <iostream>
#include <sstream>

enum ScalarType {
  Float = 0,
  Double = 1,
};

int scalar_type_nbytes(ScalarType dtype) {
  switch (dtype) {
  case ScalarType::Float:
    return 4;
  case ScalarType::Double:
    return 8;
  default:
    assert(false && "element_size unhandled branch");
  }
}

std::vector<int> contiguous_strides(const std::vector<int> sizes) {
  std::vector<int> strides(sizes.size(), 1);
  for (int i = sizes.size() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * sizes[i + 1];
  }
  return strides;
}

int compute_numel(const std::vector<int>& sizes) {
  return std::accumulate(sizes.begin(), sizes.end(), 1, std::multiplies<int>());
}

class TensorImpl {
 public:
  explicit TensorImpl(const std::vector<int>& sizes, ScalarType dtype)
      : sizes_(sizes), dtype_(dtype) {
    strides_ = contiguous_strides(sizes);
    numel_ = compute_numel(sizes);
    capacity_ = numel_ * scalar_type_nbytes(dtype);
    data_ = malloc(capacity_);
  }

  ~TensorImpl() {
    free(data_);
  }

 private:
  std::vector<int> sizes_;
  std::vector<int> strides_;
  int numel_;
  void *data_;
  int capacity_;
  ScalarType dtype_;

  friend class Tensor;
};

class Tensor {
 public:
  explicit Tensor(const std::vector<int>& sizes, ScalarType dtype)
    : impl_(new TensorImpl(sizes, dtype)) {
  }

  const std::vector<int>& sizes() const {
    return impl_->sizes_;
  }

  const std::vector<int>& strides() const {
    return impl_->strides_;
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

  void *locate(const std::vector<int>& indices) {
    int idx = 0;
    for (int i = 0; i < dim(); ++i) {
      idx += indices[i] * strides()[i];
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
    visit([this, &ss](const std::vector<int>& indices) {
      using T = float; // TODO
      ss << *(T *) locate(indices) << std::endl;
      return true;
    });
    return ss.str();
  }

  void print() const {
    std::cout << to_string();
  }

  static Tensor add(const Tensor& lhs, const Tensor& rhs) {
    Tensor out(lhs.sizes(), lhs.dtype());
    lhs.visit([&lhs, &rhs, &out](const std::vector<int>& indices) {
      using scalar_t = float; // TODO 
      auto lhs_ptr = (scalar_t*) lhs.locate(indices);
      auto rhs_ptr = (scalar_t*) rhs.locate(indices);
      auto out_ptr = (scalar_t*) out.locate(indices);
      *out_ptr = *lhs_ptr + *rhs_ptr;
      return true;
    });
    return out;
  }

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

 private:
  std::shared_ptr<TensorImpl> impl_;
};
