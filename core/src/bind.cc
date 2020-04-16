#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "bind.h"

namespace py = pybind11;

PYBIND11_MODULE(wnn, m){

    py::class_<Discriminator>(m, "Discriminator")
      .def(py::init<int,int>())
      .def("train", (void (Discriminator::*)(const vector<bool>&)) &Discriminator::train)
      .def("rank", (int (Discriminator::*)(const vector<bool>&)) &Discriminator::rank)
      .def("info", (void (Discriminator::*)()) &Discriminator::info)
    ;

    py::class_<BloomDiscriminator>(m, "BloomDiscriminator")
      .def(py::init<int, int, long int, py::kwargs>())
      .def("train", (void (BloomDiscriminator::*)(const vector<bool>&)) &BloomDiscriminator::train)
      .def("rank", (int (BloomDiscriminator::*)(const vector<bool>&)) &BloomDiscriminator::rank)
      .def("info", (void (BloomDiscriminator::*)()) &BloomDiscriminator::info)
    ;

    py::class_<Wisard>(m, "Wisard")
      .def(py::init<int, int, int>())
      .def("train", (void (Wisard::*)(const vector<vector<bool>>&, const vector<int>&)) &Wisard::train)
      //.def("rank", (int (Wisard::*)(const vector<bool>&)) &Wisard::rank)
      .def("rank", (py::array_t<int> (Wisard::*)(const vector<vector<bool>>&)) &Wisard::rank)
      .def("info", (void (Wisard::*)()) &Wisard::info)
      .def("stats", (py::array_t<unsigned long int> (Wisard::*)()) &Wisard::stats)
      .def("reset", (void (Wisard::*)()) &Wisard::reset)
    ;

    py::class_<BloomWisard>(m, "BloomWisard")
      .def(py::init<int, int, int, long int, py::kwargs>())
      .def("train", (void (BloomWisard::*)(const vector<vector<bool>>&, const vector<int>&)) &BloomWisard::train)
      //.def("rank", (int (BloomWisard::*)(const vector<bool>&)) &BloomWisard::rank)
      .def("rank", (py::array_t<int> (BloomWisard::*)(const vector<vector<bool>>&)) &BloomWisard::rank)
      .def("info", (void (BloomWisard::*)()) &BloomWisard::info)
      .def("stats", (py::array_t<unsigned long int> (BloomWisard::*)()) &BloomWisard::stats)
      .def("reset", (void (BloomWisard::*)()) &BloomWisard::reset)
      .def("error", (float (BloomWisard::*)()) &BloomWisard::getError)
    ;

    py::class_<DictWisard>(m, "DictWisard")
      .def(py::init<int, int, int>())
      .def("train", (void (DictWisard::*)(const vector<vector<bool>>&, const vector<int>&)) &DictWisard::train)
      //.def("rank", (int (DictWisard::*)(const vector<bool>&)) &DictWisard::rank)
      .def("rank", (py::array_t<int> (DictWisard::*)(const vector<vector<bool>>&)) &DictWisard::rank)
      .def("info", (void (DictWisard::*)()) &DictWisard::info)
      .def("stats", (py::array_t<unsigned long int> (DictWisard::*)()) &DictWisard::stats)
    ;
}