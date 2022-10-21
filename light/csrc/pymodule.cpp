#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>
#include "Tensor.h"

namespace py = pybind11;

PYBIND11_MODULE(_C, m) {
  py::class_<Tensor>(m, "Tensor")
    .def(py::init([](const std::vector<int>& sizes, int dtype) {
      return std::make_unique<Tensor>(sizes, (ScalarType) dtype);
    }))
    .def("size", &Tensor::sizes)
    .def("stride", &Tensor::strides)
    .def("initWithScalar", &Tensor::initWithScalar<float>)
    .def("__str__", &Tensor::to_string)
    .def("__repr__", &Tensor::to_string)
    .def("__add__", &Tensor::add)
    .def("__eq__", &Tensor::equal)
    ;
}
