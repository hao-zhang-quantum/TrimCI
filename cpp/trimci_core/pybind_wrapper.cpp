#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <sstream>
#include <bitset>

#include "determinant.hpp"
#include "hamiltonian.hpp"
#include "screening.hpp"
#include "trim.hpp"

// Forward declarations for scalable functions
void bind_scalable_determinants(pybind11::module& m);
void bind_scalable_hamiltonian(pybind11::module& m);
void bind_scalable_screening(pybind11::module& m);
void bind_scalable_trim(pybind11::module& m);

namespace py = pybind11;
using namespace trimci_core;

PYBIND11_MODULE(trimci_core, m) {
    m.doc() = "TrimCI core: Determinant, Hamiltonian, Screening, Trim";

    // Determinant
    py::class_<Determinant>(m, "Determinant")
        .def(py::init<uint64_t, uint64_t>())
        .def_readwrite("alpha", &Determinant::alpha)
        .def_readwrite("beta",  &Determinant::beta)
        .def("__repr__", [](Determinant const &d) {
            std::ostringstream oss;
            oss << "Determinant(alpha=" << std::bitset<64>(d.alpha)
                << ", beta=" << std::bitset<64>(d.beta) << ")";
            return oss.str();
        })
        .def(py::pickle(
            [](Determinant const &d) {
                return py::make_tuple(d.alpha, d.beta);
            },
            [](py::tuple t) {
                if (t.size() != 2)
                    throw std::runtime_error("Invalid state for Determinant");
                return new Determinant(
                    t[0].cast<uint64_t>(),
                    t[1].cast<uint64_t>()
                );
            }
        ));

    m.def("generate_reference_det", &generate_reference_det,
          py::arg("n_alpha"), py::arg("n_beta"));
    m.def("generate_excitations",      &generate_excitations,
          py::arg("det"), py::arg("n_orb"));

    // Hamiltonian
    m.def("extract_mol_name",          &extract_mol_name,
          py::arg("atom_str"));
    m.def("load_or_create_Hij_cache",  &load_or_create_Hij_cache,
          py::arg("mol_name"), py::arg("n_elec"), py::arg("n_orb"),
          py::arg("cache_dir") = std::string("cache"));
    m.def("pair_key",                  &pair_key,
          py::arg("d1"), py::arg("d2"));
    m.def("compute_H_ij",              &compute_H_ij,
          py::arg("det_i"), py::arg("det_j"),
          py::arg("h1"), py::arg("eri"));

    // Screening
    m.def("pool_build", [](const std::vector<Determinant>& initial_pool,
                           const std::vector<double>& initial_coeff,
                           int n_orb,
                           const std::vector<std::vector<double>>& h1,
                           const std::vector<std::vector<std::vector<std::vector<double>>>>& eri,
                           double threshold,
                           size_t target_size,
                           HijCache& cache,
                           const std::string& cache_file,
                           int max_rounds) {
        auto result = pool_build(initial_pool, initial_coeff, n_orb, h1, eri,
                                threshold, target_size, cache, cache_file, max_rounds);
        return py::make_tuple(result.first, result.second);
    }, py::arg("initial_pool"), py::arg("initial_coeff"), py::arg("n_orb"),
       py::arg("h1"), py::arg("eri"),
       py::arg("threshold"), py::arg("target_size"),
       py::arg("cache"), py::arg("cache_file"),
       py::arg("max_rounds") = -1);

    // Trim

    
    m.def("diagonalize_subspace_davidson", &diagonalize_subspace_davidson,
          py::arg("dets"), py::arg("h1"), py::arg("eri"),
          py::arg("cache"), py::arg("quantization"),
          py::arg("max_iter") = 100, py::arg("tol") = 1e-6,
          py::arg("verbose") = false, py::arg("n_orb") = 0);

      m.def("select_top_k_dets", &select_top_k_dets,
            py::arg("dets"), py::arg("coeffs"), py::arg("k"),
            py::arg("core_set") = std::vector<Determinant>{},
            py::arg("keep_core") = true);

          
    // MODIFIED run_trim binding with external_core_dets support
    m.def("run_trim",                  &run_trim,
          py::arg("pool"), py::arg("h1"), py::arg("eri"),
          py::arg("mol_name"), py::arg("n_elec"), py::arg("n_orb"),
          py::arg("group_sizes"), 
          py::arg("keep_sizes"),
          py::arg("quantization") = false, py::arg("save_cache") = true, py::arg("verbose") = true,
          py::arg("external_core_dets") = std::vector<Determinant>{});

    // Bind scalable functions
    bind_scalable_determinants(m);
    bind_scalable_hamiltonian(m);
    bind_scalable_screening(m);
    bind_scalable_trim(m);
}