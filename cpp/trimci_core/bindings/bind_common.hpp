#pragma once
// Shared header for pybind11 binding files.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/operators.h>

namespace py = pybind11;

// Forward declarations of all bind_* functions
void bind_determinants(py::module& m);
void bind_hamiltonian(py::module& m);
void bind_screening(py::module& m);
void bind_trim(py::module& m);
void bind_rdm(py::module& m);
void bind_rdm_full(py::module& m);
void bind_orbital_opt(py::module& m);
void bind_iterative_workflow(py::module& m);
void bind_direct_builders(py::module& m);
void bind_davidson_gep(py::module& m);
void bind_matfree_davidson(py::module& m);
void bind_fast_expansion(py::module& m);
void bind_davidson_utils(py::module& m);
