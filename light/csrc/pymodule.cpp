#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <memory>
#include "light/csrc/Tensor.h"
#include "light/csrc/rand.h"
#include "light/csrc/ops.h"

namespace py = pybind11;

PYBIND11_MODULE(_C, m) {
  py::class_<Tensor>(m, "Tensor")
    .def(py::init([](const std::vector<int>& sizes, int dtype) {
      return std::make_unique<Tensor>(sizes, (ScalarType) dtype);
    }), py::arg("sizes"), py::arg("dtype") = (int) ScalarType::Float)
    // TODO support dimensions other than 2
    .def(py::init([](int size0, int size1) {
      return std::make_unique<Tensor>(std::vector<int>({size0, size1}), ScalarType::Float);
    }))
    .def(py::init([](py::array_t<float> ar) {
      return createFromNpArray(ar);
    }))
    .def("size", &Tensor::sizes)
    .def("stride", &Tensor::strides)
    .def("dtype", [](Tensor self) {
      return (int) self.dtype();
    })
    .def("fill_", &Tensor::initWithScalar<float>)
    .def("__str__", &Tensor::to_string)
    .def("__repr__", &Tensor::to_string)
    .def("__add__", [](Tensor lhs, Tensor rhs) { return lhs + rhs; })
    .def("__radd__", [](Tensor rhs, int lhs_ival) {
      Tensor lhs = Tensor::create_scalar_tensor((int64_t) lhs_ival);
      return lhs + rhs;
    })
    // TODO can we use C++ overloaded '==' here? Can cpp '==' returns a Tensor
    // rather than bool
    .def("__eq__", [](Tensor lhs, Tensor rhs) { return ops::eq(lhs, rhs); })
    .def("__eq__", [](Tensor lhs, Tensor rhs) { return ops::eq(lhs, rhs); })
    .def("__truediv__", &Tensor::divScalar)
    .def("__len__", [](Tensor self) {
      assert(self.dim() > 0);
      return self.sizes()[0];
    })
    .def("__getitem__", &Tensor::slice)
    .def("mean", &Tensor::mean)
    .def("max", &Tensor::max, py::arg("dim"))
    .def("sum", [](Tensor self) {
      return self.sum();
    })
    .def("sum", [](Tensor self, int dim) {
      return self.sum(dim);
    })
    .def("transpose", &Tensor::transpose)
    .def("equal", &Tensor::equal)
    .def("tolist", [](Tensor self) {
      std::vector<int> indices;
      return self.tolist(indices);
    })
    .def("item", [](Tensor self) {
      py::object ret;
      DISPATCH_DTYPE(self.dtype(), [&]() {
        ret = py::cast(self.item<scalar_t>());
      });
      return ret;
    })
    .def_property("requires_grad", &Tensor::requires_grad, &Tensor::set_requires_grad)
    .def_property("is_param", &Tensor::is_param, &Tensor::set_is_param)
    .def_property("grad", [](Tensor self) -> py::object {
      Tensor* grad_ptr = self.grad_ptr();
      if (grad_ptr) {
        return py::cast(*grad_ptr);
      } else {
        return py::none();
      }
    }, nullptr)
    .def("backward", &Tensor::backward)
    .def("zero_", &Tensor::zero_)
    .def("add_", &Tensor::add_)
    .def("uniform_", &Tensor::uniform_)
    ;

  // in pytorch LongTensor is implemented as a class, here we implement it as
  // a method
  m.def("LongTensor", [](py::array_t<int64_t> ar) {
    return createFromNpArray(ar);
  });

  m.def("manual_seed", [](int seed) {
    set_seed(seed);
  });
  m.def("rand", [](int size0, bool requires_grad) {
    auto out = createRandTensor({size0}, ScalarType::Float);
    if (requires_grad) {
      out.set_requires_grad(true);
    }
    return out;
  }, py::arg("size0"), py::arg("requires_grad") = false);
  m.def("rand", [](int size0, int size1, bool requires_grad) {
    auto out = createRandTensor({size0, size1}, ScalarType::Float);
    if (requires_grad) {
      out.set_requires_grad(true);
    }
    return out;
  }, py::arg("size0"), py::arg("size1"), py::arg("requires_grad") = false);

  m.def("randint", [](int low, int high, std::vector<int> sizes) {
    return createRandIntTensor(low, high, sizes);
  });

  m.def("matmul", &ops::matmul);
  m.def("relu", &ops::relu);
  m.def("sigmoid", &ops::sigmoid);
  m.def("log_softmax", &ops::log_softmax);
  m.def("nll_loss", &ops::nll_loss);
  m.def("disable_grad", &disable_grad);
  m.def("enable_grad", &enable_grad);
}
