#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
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
    .def("size", &Tensor::sizes)
    .def("stride", &Tensor::strides)
    .def("dtype", [](Tensor self) {
      return (int) self.dtype();
    })
    .def("fill_", &Tensor::initWithScalar<float>)
    .def("__str__", &Tensor::to_string)
    .def("__repr__", &Tensor::to_string)
    .def("__add__", [](Tensor lhs, Tensor rhs) { return lhs + rhs; })
    .def("mean", &Tensor::mean)
    .def("equal", &Tensor::equal)
    .def("tolist", [](Tensor self) {
      std::vector<int> indices;
      return self.tolist(indices);
    })
    .def("item", [](Tensor self) {
      assert(self.dtype() == ScalarType::Float); // TODO
      return self.item<float>();
    })
    .def_property("requires_grad", &Tensor::requires_grad, &Tensor::set_requires_grad)
    .def_property("grad", &Tensor::grad, nullptr)
    .def("backward", &Tensor::backward)
    ;

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

  m.def("matmul", &ops::matmul);
  m.def("relu", &ops::relu);
  m.def("sigmoid", &ops::sigmoid);
  m.def("log_softmax", &ops::log_softmax);
}
