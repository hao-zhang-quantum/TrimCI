#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <sstream>
#include <bitset>
#include <array>

#include "determinant_scalable.hpp"
#include "hamiltonian_scalable.hpp"
#include "screening_scalable.hpp"
#include "trim_scalable.hpp"

namespace py = pybind11;
using namespace trimci_core;

// Helper function to create string representation for array-based determinants
template<size_t N>
std::string array_to_bitstring(const std::array<uint64_t, N>& arr) {
    std::ostringstream oss;
    for (size_t i = 0; i < N; ++i) {
        if (i > 0) oss << "|";
        oss << std::bitset<64>(arr[i]);
    }
    return oss.str();
}

void bind_scalable_determinants(py::module& m) {
    // Determinant64 (uint64_t storage)
    py::class_<Determinant64>(m, "Determinant64")
        .def(py::init<uint64_t, uint64_t>())
        .def_readwrite("alpha", &Determinant64::alpha)
        .def_readwrite("beta",  &Determinant64::beta)
        .def("__repr__", [](const Determinant64& d) {
            std::ostringstream oss;
            oss << "Determinant64(alpha=" << std::bitset<64>(d.alpha)
                << ", beta=" << std::bitset<64>(d.beta) << ")";
            return oss.str();
        })
        .def(py::pickle(
            [](const Determinant64& d) {
                return py::make_tuple(d.alpha, d.beta);
            },
            [](py::tuple t) {
                if (t.size() != 2)
                    throw std::runtime_error("Invalid state for Determinant64");
                return new Determinant64(
                    t[0].cast<uint64_t>(),
                    t[1].cast<uint64_t>()
                );
            }
        ));

    // Determinant128 (std::array<uint64_t, 2> storage)
    py::class_<Determinant128>(m, "Determinant128")
        .def(py::init<std::array<uint64_t, 2>, std::array<uint64_t, 2>>())
        .def_readwrite("alpha", &Determinant128::alpha)
        .def_readwrite("beta",  &Determinant128::beta)
        .def("__repr__", [](const Determinant128& d) {
            std::ostringstream oss;
            oss << "Determinant128(alpha=" << array_to_bitstring(d.alpha)
                << ", beta=" << array_to_bitstring(d.beta) << ")";
            return oss.str();
        })
        .def(py::pickle(
            [](const Determinant128& d) {
                return py::make_tuple(d.alpha, d.beta);
            },
            [](py::tuple t) {
                if (t.size() != 2)
                    throw std::runtime_error("Invalid state for Determinant128");
                return new Determinant128(
                    t[0].cast<std::array<uint64_t, 2>>(),
                    t[1].cast<std::array<uint64_t, 2>>()
                );
            }
        ));

    // Determinant192 (std::array<uint64_t, 3> storage)
    py::class_<Determinant192>(m, "Determinant192")
        .def(py::init<std::array<uint64_t, 3>, std::array<uint64_t, 3>>())
        .def_readwrite("alpha", &Determinant192::alpha)
        .def_readwrite("beta",  &Determinant192::beta)
        .def("__repr__", [](const Determinant192& d) {
            std::ostringstream oss;
            oss << "Determinant192(alpha=" << array_to_bitstring(d.alpha)
                << ", beta=" << array_to_bitstring(d.beta) << ")";
            return oss.str();
        })
        .def(py::pickle(
            [](const Determinant192& d) {
                return py::make_tuple(d.alpha, d.beta);
            },
            [](py::tuple t) {
                if (t.size() != 2)
                    throw std::runtime_error("Invalid state for Determinant192");
                return new Determinant192(
                    t[0].cast<std::array<uint64_t, 3>>(),
                    t[1].cast<std::array<uint64_t, 3>>()
                );
            }
        ));

    // Template-based helper functions for Determinant64
    m.def("generate_reference_det_64", 
          [](int n_alpha, int n_beta) { return generate_reference_det_t<uint64_t>(n_alpha, n_beta); },
          py::arg("n_alpha"), py::arg("n_beta"));
    m.def("generate_excitations_64", 
          [](const DeterminantT<uint64_t>& det, int n_orb) { 
              return generate_excitations_t<uint64_t>(det, n_orb); 
          },
          py::arg("det"), py::arg("n_orb"));

    // Template-based helper functions for Determinant128
    m.def("generate_reference_det_128", 
          [](int n_alpha, int n_beta) { return generate_reference_det_t<std::array<uint64_t, 2>>(n_alpha, n_beta); },
          py::arg("n_alpha"), py::arg("n_beta"));
    m.def("generate_excitations_128", 
          [](const DeterminantT<std::array<uint64_t, 2>>& det, int n_orb) { 
              return generate_excitations_t<std::array<uint64_t, 2>>(det, n_orb); 
          },
          py::arg("det"), py::arg("n_orb"));

    // Template-based helper functions for Determinant192
    m.def("generate_reference_det_192", 
          [](int n_alpha, int n_beta) { return generate_reference_det_t<std::array<uint64_t, 3>>(n_alpha, n_beta); },
          py::arg("n_alpha"), py::arg("n_beta"));
    m.def("generate_excitations_192", 
          [](const DeterminantT<std::array<uint64_t, 3>>& det, int n_orb) { 
              return generate_excitations_t<std::array<uint64_t, 3>>(det, n_orb); 
          },
          py::arg("det"), py::arg("n_orb"));
}

void bind_scalable_hamiltonian(py::module& m) {
    // Hamiltonian functions for Determinant64
    m.def("load_or_create_Hij_cache_64", &load_or_create_Hij_cache_t<uint64_t>,
          py::arg("mol_name"), py::arg("n_elec"), py::arg("n_orb"),
          py::arg("cache_dir") = std::string("cache"));
    m.def("compute_H_ij_64", &compute_H_ij_t<uint64_t>,
          py::arg("det_i"), py::arg("det_j"),
          py::arg("h1"), py::arg("eri"));

    // Hamiltonian functions for Determinant128
    m.def("load_or_create_Hij_cache_128", &load_or_create_Hij_cache_t<std::array<uint64_t, 2>>,
          py::arg("mol_name"), py::arg("n_elec"), py::arg("n_orb"),
          py::arg("cache_dir") = std::string("cache"));
    m.def("compute_H_ij_128", &compute_H_ij_t<std::array<uint64_t, 2>>,
          py::arg("det_i"), py::arg("det_j"),
          py::arg("h1"), py::arg("eri"));

    // Hamiltonian functions for Determinant192
    m.def("load_or_create_Hij_cache_192", &load_or_create_Hij_cache_t<std::array<uint64_t, 3>>,
          py::arg("mol_name"), py::arg("n_elec"), py::arg("n_orb"),
          py::arg("cache_dir") = std::string("cache"));
    m.def("compute_H_ij_192", &compute_H_ij_t<std::array<uint64_t, 3>>,
          py::arg("det_i"), py::arg("det_j"),
          py::arg("h1"), py::arg("eri"));
}

void bind_scalable_screening(py::module& m) {
    // Screening functions for Determinant64
    m.def("pool_build_64", [](const std::vector<Determinant64>& initial_pool,
                              const std::vector<double>& initial_coeff,
                              int n_orb,
                              const std::vector<std::vector<double>>& h1,
                              const std::vector<std::vector<std::vector<std::vector<double>>>>& eri,
                              double threshold,
                              size_t target_size,
                              HijCacheT<uint64_t>& cache,
                              const std::string& cache_file,
                              int max_rounds) {
        auto result = pool_build_t(initial_pool, initial_coeff, n_orb, h1, eri,
                                   threshold, target_size, cache, cache_file, max_rounds);
        return py::make_tuple(result.first, result.second);
    }, py::arg("initial_pool"), py::arg("initial_coeff"), py::arg("n_orb"),
       py::arg("h1"), py::arg("eri"),
       py::arg("threshold"), py::arg("target_size"),
       py::arg("cache"), py::arg("cache_file"),
       py::arg("max_rounds") = -1);

    // Screening functions for Determinant128
    m.def("pool_build_128", [](const std::vector<Determinant128>& initial_pool,
                               const std::vector<double>& initial_coeff,
                               int n_orb,
                               const std::vector<std::vector<double>>& h1,
                               const std::vector<std::vector<std::vector<std::vector<double>>>>& eri,
                               double threshold,
                               size_t target_size,
                               HijCacheT<std::array<uint64_t, 2>>& cache,
                               const std::string& cache_file,
                               int max_rounds) {
        auto result = pool_build_t(initial_pool, initial_coeff, n_orb, h1, eri,
                                   threshold, target_size, cache, cache_file, max_rounds);
        return py::make_tuple(result.first, result.second);
    }, py::arg("initial_pool"), py::arg("initial_coeff"), py::arg("n_orb"),
       py::arg("h1"), py::arg("eri"),
       py::arg("threshold"), py::arg("target_size"),
       py::arg("cache"), py::arg("cache_file"),
       py::arg("max_rounds") = -1);

    // Screening functions for Determinant192
    m.def("pool_build_192", [](const std::vector<Determinant192>& initial_pool,
                               const std::vector<double>& initial_coeff,
                               int n_orb,
                               const std::vector<std::vector<double>>& h1,
                               const std::vector<std::vector<std::vector<std::vector<double>>>>& eri,
                               double threshold,
                               size_t target_size,
                               HijCacheT<std::array<uint64_t, 3>>& cache,
                               const std::string& cache_file,
                               int max_rounds) {
        auto result = pool_build_t(initial_pool, initial_coeff, n_orb, h1, eri,
                                   threshold, target_size, cache, cache_file, max_rounds);
        return py::make_tuple(result.first, result.second);
    }, py::arg("initial_pool"), py::arg("initial_coeff"), py::arg("n_orb"),
       py::arg("h1"), py::arg("eri"),
       py::arg("threshold"), py::arg("target_size"),
       py::arg("cache"), py::arg("cache_file"),
       py::arg("max_rounds") = -1);
}

void bind_scalable_trim(py::module& m) {
    // Trim functions for Determinant64
    m.def("diagonalize_subspace_davidson_64", &diagonalize_subspace_davidson_t<uint64_t>,
          py::arg("dets"), py::arg("h1"), py::arg("eri"),
          py::arg("cache"), py::arg("quantization"),
          py::arg("max_iter") = 100, py::arg("tol") = 1e-6,
          py::arg("verbose") = false, py::arg("n_orb") = 0);

    m.def("select_top_k_dets_64", &select_top_k_dets_t<uint64_t>,
          py::arg("dets"), py::arg("coeffs"), py::arg("k"),
          py::arg("core_set") = std::vector<Determinant64>{},
          py::arg("keep_core") = true);

    m.def("run_trim_64", &run_trim_t<uint64_t>,
          py::arg("pool"), py::arg("h1"), py::arg("eri"),
          py::arg("mol_name"), py::arg("n_elec"), py::arg("n_orb"),
          py::arg("group_sizes"), 
          py::arg("keep_sizes"),
          py::arg("quantization") = false, py::arg("save_cache") = true, py::arg("verbose") = true,
          py::arg("external_core_dets") = std::vector<Determinant64>{});

    // Trim functions for Determinant128
    m.def("diagonalize_subspace_davidson_128", &diagonalize_subspace_davidson_t<std::array<uint64_t, 2>>,
          py::arg("dets"), py::arg("h1"), py::arg("eri"),
          py::arg("cache"), py::arg("quantization"),
          py::arg("max_iter") = 100, py::arg("tol") = 1e-6,
          py::arg("verbose") = false, py::arg("n_orb") = 0);

    m.def("select_top_k_dets_128", &select_top_k_dets_t<std::array<uint64_t, 2>>,
          py::arg("dets"), py::arg("coeffs"), py::arg("k"),
          py::arg("core_set") = std::vector<Determinant128>{},
          py::arg("keep_core") = true);

    m.def("run_trim_128", &run_trim_t<std::array<uint64_t, 2>>,
          py::arg("pool"), py::arg("h1"), py::arg("eri"),
          py::arg("mol_name"), py::arg("n_elec"), py::arg("n_orb"),
          py::arg("group_sizes"), 
          py::arg("keep_sizes"),
          py::arg("quantization") = false, py::arg("save_cache") = true, py::arg("verbose") = true,
          py::arg("external_core_dets") = std::vector<Determinant128>{});

    // Trim functions for Determinant192
    m.def("diagonalize_subspace_davidson_192", &diagonalize_subspace_davidson_t<std::array<uint64_t, 3>>,
          py::arg("dets"), py::arg("h1"), py::arg("eri"),
          py::arg("cache"), py::arg("quantization"),
          py::arg("max_iter") = 100, py::arg("tol") = 1e-6,
          py::arg("verbose") = false, py::arg("n_orb") = 0);

    m.def("select_top_k_dets_192", &select_top_k_dets_t<std::array<uint64_t, 3>>,
          py::arg("dets"), py::arg("coeffs"), py::arg("k"),
          py::arg("core_set") = std::vector<Determinant192>{},
          py::arg("keep_core") = true);

    m.def("run_trim_192", &run_trim_t<std::array<uint64_t, 3>>,
          py::arg("pool"), py::arg("h1"), py::arg("eri"),
          py::arg("mol_name"), py::arg("n_elec"), py::arg("n_orb"),
          py::arg("group_sizes"), 
          py::arg("keep_sizes"),
          py::arg("quantization") = false, py::arg("save_cache") = true, py::arg("verbose") = true,
          py::arg("external_core_dets") = std::vector<Determinant192>{});
}

// Remove the PYBIND11_MODULE definition to avoid duplicate module definitions
// The functions will be called from pybind_wrapper.cpp