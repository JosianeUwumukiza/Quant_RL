#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "AlphaEnv.hpp"
#include "MarketReplay.hpp"

namespace py = pybind11;

PYBIND11_MODULE(qrl_bindings, m) {
    py::class_<AlphaEnv>(m, "AlphaEnv")
        .def(py::init<int,double>(), py::arg("window")=32, py::arg("fee_bps")=1.0)
        .def("load_data", &AlphaEnv::load_data)
        .def("reset", &AlphaEnv::reset, py::arg("seed")=0)
        .def("step", &AlphaEnv::step)
        .def("obs", &AlphaEnv::obs)
        .def("obs_dim", &AlphaEnv::obs_dim)
        .def("action_dim", &AlphaEnv::action_dim)
        .def("t", &AlphaEnv::t)
        .def("position", &AlphaEnv::position)
        .def("equity", &AlphaEnv::equity);
}
