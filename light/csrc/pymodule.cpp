#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>
#include "Tensor.h"

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
    .def("fill_", &Tensor::initWithScalar<float>)
    .def("__str__", &Tensor::to_string)
    .def("__repr__", &Tensor::to_string)
    .def("__add__", &Tensor::add)
    .def("equal", &Tensor::equal)
    // in pytorch __eq__ returns a tensor with elementwise result
    // .def("__eq__", &Tensor::equal)
    ;
}
