#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/operators.h>

#include <sstream>
#include <bitset>
#include <array>
#include <chrono>

#include "determinant.hpp"
#include "hamiltonian.hpp"
#include "screening.hpp"
#include "trim.hpp"
#include "iterative_workflow.hpp"
#include "davidson_gep.hpp"
#include "matfree_davidson.hpp"

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

// Template helpers for scalable bindings
template<size_t N>
void bind_scalable_determinant(py::module& m, const std::string& suffix) {
    using DetType = DeterminantT<std::array<uint64_t, N>>;
    std::string className = "Determinant" + suffix;
    py::class_<DetType>(m, className.c_str())
        .def(py::init<std::array<uint64_t, N>, std::array<uint64_t, N>>())
        .def_readwrite("alpha", &DetType::alpha)
        .def_readwrite("beta",  &DetType::beta)
        .def(py::self == py::self)
        .def("__repr__", [className](const DetType& d) {
            std::ostringstream oss;
            oss << className << "(alpha=" << array_to_bitstring(d.alpha)
                << ", beta=" << array_to_bitstring(d.beta) << ")";
            return oss.str();
        })
        .def(py::pickle(
            [](const DetType& d) { return py::make_tuple(d.alpha, d.beta); },
            [](py::tuple t) {
                if (t.size() != 2) throw std::runtime_error("Invalid state");
                return new DetType(t[0].cast<std::array<uint64_t, N>>(), t[1].cast<std::array<uint64_t, N>>());
            }
        ));
        
    m.def(("generate_reference_det_" + suffix).c_str(), 
          [](int n_alpha, int n_beta) { return generate_reference_det_t<std::array<uint64_t, N>>(n_alpha, n_beta); },
          py::arg("n_alpha"), py::arg("n_beta"));

    m.def(("generate_excitations_" + suffix).c_str(), 
          [](const DetType& det, int n_orb) { return generate_excitations_t<std::array<uint64_t, N>>(det, n_orb); },
          py::arg("det"), py::arg("n_orb"));
}

template<size_t N>
void bind_scalable_hamiltonian(py::module& m, const std::string& suffix) {
    m.def(("load_or_create_Hij_cache_" + suffix).c_str(), &load_or_create_Hij_cache_t<std::array<uint64_t, N>>,
          py::arg("mol_name"), py::arg("n_elec"), py::arg("n_orb"), py::arg("cache_dir") = std::string("cache"));
    using HijFunc = double(*)(const DeterminantT<std::array<uint64_t, N>>&,
                              const DeterminantT<std::array<uint64_t, N>>&,
                              const std::vector<std::vector<double>>&,
                              const std::vector<double>&);
    m.def(("compute_H_ij_" + suffix).c_str(), static_cast<HijFunc>(&compute_H_ij_t<std::array<uint64_t, N>>),
          py::arg("det_i"), py::arg("det_j"), py::arg("h1"), py::arg("eri"));
}

template<size_t N>
void bind_scalable_screening(py::module& m, const std::string& suffix) {
    using DetType = DeterminantT<std::array<uint64_t, N>>;
    using CacheType = HijCacheT<std::array<uint64_t, N>>;
    m.def(("pool_build_" + suffix).c_str(), 
          [](const std::vector<DetType>& initial_pool, const std::vector<double>& initial_coeff, int n_orb,
             const std::vector<std::vector<double>>& h1, const std::vector<double>& eri,
             double threshold, size_t target_size, CacheType& cache, const std::string& cache_file, 
             int max_rounds, double threshold_decay, const std::vector<int>& attentive_orbitals, int verbosity,
             const std::string& screening_mode, double e0, int strategy_factor) {
              // Build params struct
              PoolBuildParams params;
              params.screening_mode = screening_mode;
              params.e0 = e0;
              params.max_rounds = max_rounds;
              params.threshold_decay = threshold_decay;
              params.strategy_factor = strategy_factor;
              
              auto result = pool_build_t(initial_pool, initial_coeff, n_orb, h1, eri, threshold, target_size, 
                                        cache, cache_file, attentive_orbitals, verbosity, params);
              return py::make_tuple(result.first, result.second);
          },
          py::arg("initial_pool"), py::arg("initial_coeff"), py::arg("n_orb"), py::arg("h1"), py::arg("eri"),
          py::arg("threshold"), py::arg("target_size"), py::arg("cache"), py::arg("cache_file"), 
          py::arg("max_rounds") = 2, py::arg("threshold_decay") = 0.5, py::arg("attentive_orbitals") = std::vector<int>{},
          py::arg("verbosity") = 1, py::arg("screening_mode") = "heat_bath", py::arg("e0") = 0.0,
          py::arg("strategy_factor") = -1);
}

template<size_t N>
void bind_scalable_trim(py::module& m, const std::string& suffix) {
    using DetType = DeterminantT<std::array<uint64_t, N>>;
    m.def(("diagonalize_subspace_davidson_" + suffix).c_str(), &diagonalize_subspace_davidson_t<std::array<uint64_t, N>>,
          py::arg("dets"), py::arg("h1"), py::arg("eri"), py::arg("cache"), py::arg("quantization"),
          py::arg("max_iter") = 100, py::arg("tol") = 1e-3, py::arg("verbosity") = 0, py::arg("n_orb") = 0,
          py::arg("initial_guess") = std::vector<double>{},
          py::arg("sparsity") = nullptr,
          py::arg("init_strategy") = std::string("lowest_diag_noise"));

    m.def(("select_top_k_dets_" + suffix).c_str(), &select_top_k_dets_t<std::array<uint64_t, N>>,
          py::arg("dets"), py::arg("coeffs"), py::arg("k"), py::arg("core_set") = std::vector<DetType>{}, py::arg("keep_core") = true);

    m.def(("run_trim_" + suffix).c_str(), &run_trim_t<std::array<uint64_t, N>>,
          py::arg("pool"), py::arg("h1"), py::arg("eri"), py::arg("mol_name"), py::arg("n_elec"), py::arg("n_orb"),
          py::arg("group_sizes"), py::arg("keep_sizes"),
          py::arg("quantization") = false, py::arg("save_cache") = true,
          py::arg("external_core_dets") = std::vector<DetType>{}, py::arg("tol") = 1e-3, py::arg("verbosity") = 1,
          py::arg("davidson_init") = std::string("lowest_diag_noise"));
}

template<size_t N>
void bind_scalable_workflow(py::module& m, const std::string& suffix) {
    using DetType = DeterminantT<std::array<uint64_t, N>>;
    using ResultType = IterativeWorkflowResult<std::array<uint64_t, N>>;
    
    // Bind the Result class for this bit width
    std::string resultClassName = "IterativeWorkflowResult" + suffix;
    py::class_<ResultType>(m, resultClassName.c_str())
        .def(py::init<>())
        .def_readonly("final_energy", &ResultType::final_energy)
        .def_readonly("final_dets", &ResultType::final_dets)
        .def_readonly("final_coeffs", &ResultType::final_coeffs)
        .def_readonly("iteration_history", &ResultType::iteration_history)
        .def_readonly("total_time", &ResultType::total_time)
        .def_readonly("total_iterations", &ResultType::total_iterations)
        .def_readonly("success", &ResultType::success)
        .def_readonly("error_message", &ResultType::error_message);
    
    // Bind the workflow function (original list-based version)
    std::string fnName = "iterative_workflow_cpp_" + suffix;
    m.def(fnName.c_str(),
          &iterative_workflow_t<std::array<uint64_t, N>>,
          py::arg("h1"), py::arg("eri"),
          py::arg("n_alpha"), py::arg("n_beta"), py::arg("n_orb"),
          py::arg("system_name"),
          py::arg("initial_dets"), py::arg("initial_coeffs"),
          py::arg("nuclear_repulsion"), py::arg("params"),
          py::call_guard<py::gil_scoped_release>());
    
    // Bind numpy-optimized version for large systems
    std::string fnNameNp = "iterative_workflow_cpp_np_" + suffix;
    m.def(fnNameNp.c_str(),
          [](py::array_t<double, py::array::c_style | py::array::forcecast> h1_np,
             py::array_t<double, py::array::c_style | py::array::forcecast> eri_np,
             int n_alpha, int n_beta, int n_orb,
             const std::string& system_name,
             const std::vector<DetType>& initial_dets,
             const std::vector<double>& initial_coeffs,
             double nuclear_repulsion,
             const IterativeWorkflowParams& params) {
              
              // Convert h1 from 2D numpy array to vector<vector<double>>
              auto h1_buf = h1_np.request();
              if (h1_buf.ndim != 2) {
                  throw std::runtime_error("h1 must be 2D array");
              }
              size_t h1_rows = h1_buf.shape[0];
              size_t h1_cols = h1_buf.shape[1];
              double* h1_ptr = static_cast<double*>(h1_buf.ptr);
              
              std::vector<std::vector<double>> h1(h1_rows, std::vector<double>(h1_cols));
              for (size_t i = 0; i < h1_rows; ++i) {
                  for (size_t j = 0; j < h1_cols; ++j) {
                      h1[i][j] = h1_ptr[i * h1_cols + j];
                  }
              }
              
              // Convert eri from 1D numpy array to vector<double>
              auto eri_buf = eri_np.request();
              double* eri_ptr = static_cast<double*>(eri_buf.ptr);
              size_t eri_size = eri_buf.size;
              std::vector<double> eri(eri_ptr, eri_ptr + eri_size);
              
              // Note: Not releasing GIL since workflow may access Python objects
              
              // Call the actual workflow
              return iterative_workflow_t<std::array<uint64_t, N>>(
                  h1, eri, n_alpha, n_beta, n_orb, 
                  system_name, initial_dets, initial_coeffs,
                  nuclear_repulsion, params);
          },
          py::arg("h1"), py::arg("eri"),
          py::arg("n_alpha"), py::arg("n_beta"), py::arg("n_orb"),
          py::arg("system_name"),
          py::arg("initial_dets"), py::arg("initial_coeffs"),
          py::arg("nuclear_repulsion"), py::arg("params"));
          // Note: GIL is released manually inside lambda after numpy conversion
}

void bind_determinants(py::module& m) {
    // Determinant64 (uint64_t storage) - Exposed as standard Determinant for backward compatibility
    py::class_<Determinant64>(m, "Determinant") // Also Determinant64 logic
        .def(py::init<uint64_t, uint64_t>())
        .def_readwrite("alpha", &Determinant64::alpha)
        .def_readwrite("beta",  &Determinant64::beta)
        .def(py::self == py::self)
        .def("__repr__", [](const Determinant64& d) {
            std::ostringstream oss;
            oss << "Determinant(alpha=" << std::bitset<64>(d.alpha)
                << ", beta=" << std::bitset<64>(d.beta) << ")";
            return oss.str();
        })
        .def(py::pickle(
            [](const Determinant64& d) {
                return py::make_tuple(d.alpha, d.beta);
            },
            [](py::tuple t) {
                if (t.size() != 2)
                    throw std::runtime_error("Invalid state for Determinant");
                return new Determinant64(
                    t[0].cast<uint64_t>(),
                    t[1].cast<uint64_t>()
                );
            }
        ));
    
    // Explicit alias or separate binding if needed? 
    // m.attr("Determinant64") = m.attr("Determinant"); // Simple alias

    // Helper functions for Determinant64 (Default)
    m.def("generate_reference_det", &generate_reference_det,
          py::arg("n_alpha"), py::arg("n_beta"));
    m.def("generate_excitations",   &generate_excitations,
          py::arg("det"), py::arg("n_orb"));

    // Aliases for 64-bit to match scalable naming convention
    m.attr("Determinant64") = m.attr("Determinant");
    m.def("generate_reference_det_64", &generate_reference_det, py::arg("n_alpha"), py::arg("n_beta"));
    m.def("generate_excitations_64", &generate_excitations, py::arg("det"), py::arg("n_orb"));

    // Scalable determinants bindings
    bind_scalable_determinant<2>(m, "128");
    bind_scalable_determinant<3>(m, "192");
    bind_scalable_determinant<4>(m, "256");
    bind_scalable_determinant<5>(m, "320");
    bind_scalable_determinant<6>(m, "384");
    bind_scalable_determinant<7>(m, "448");
    bind_scalable_determinant<8>(m, "512");
}

void bind_hamiltonian(py::module& m) {
    m.def("extract_mol_name", &extract_mol_name, py::arg("atom_str"));

    // Hamiltonian functions for Determinant64 (Default)
    m.def("load_or_create_Hij_cache", &load_or_create_Hij_cache,
          py::arg("mol_name"), py::arg("n_elec"), py::arg("n_orb"),
          py::arg("cache_dir") = std::string("cache"));
    m.def("compute_H_ij", &compute_H_ij,
          py::arg("det_i"), py::arg("det_j"),
          py::arg("h1"), py::arg("eri"));
          
    m.def("pair_key", &pair_key, py::arg("d1"), py::arg("d2"));

    // Aliases for 64-bit
    m.def("load_or_create_Hij_cache_64", &load_or_create_Hij_cache,
          py::arg("mol_name"), py::arg("n_elec"), py::arg("n_orb"),
          py::arg("cache_dir") = std::string("cache"));
    m.def("compute_H_ij_64", &compute_H_ij,
          py::arg("det_i"), py::arg("det_j"),
          py::arg("h1"), py::arg("eri"));
          
    // Scalable hamiltonian bindings
    bind_scalable_hamiltonian<2>(m, "128");
    bind_scalable_hamiltonian<3>(m, "192");
    bind_scalable_hamiltonian<4>(m, "256");
    bind_scalable_hamiltonian<5>(m, "320");
    bind_scalable_hamiltonian<6>(m, "384");
    bind_scalable_hamiltonian<7>(m, "448");
    bind_scalable_hamiltonian<8>(m, "512");
    
    // CI Energy Evaluation
    m.def("evaluate_ci_energy", &evaluate_ci_energy,
          py::arg("dets_alpha"), py::arg("dets_beta"), py::arg("coeffs"),
          py::arg("h1"), py::arg("eri"), py::arg("n_orb"),
          py::call_guard<py::gil_scoped_release>(),
          R"doc(
Evaluate CI energy given determinants and coefficients.

Computes E = Σ_ij c_i c_j ⟨D_i|H|D_j⟩ using Slater-Condon rules.
OpenMP parallelized for efficiency.

Args:
    dets_alpha: Alpha bitstrings for each determinant
    dets_beta: Beta bitstrings for each determinant
    coeffs: CI coefficients
    h1: One-body integrals (n_orb x n_orb, as vector<vector>)
    eri: Two-body integrals (flattened n_orb^4)
    n_orb: Number of orbitals

Returns:
    CI energy (float)
)doc");
}

void bind_screening(py::module& m) {
    // Screening functions for Determinant64 (Default)
    m.def("pool_build", [](const std::vector<Determinant>& initial_pool,
                           const std::vector<double>& initial_coeff,
                           int n_orb,
                           const std::vector<std::vector<double>>& h1,
                           const std::vector<double>& eri,
                           double threshold,
                           size_t target_size,
                           HijCache& cache,
                           const std::string& cache_file,
                           int max_rounds,
                           double threshold_decay,
                           const std::vector<int>& attentive_orbitals,
                           int verbosity,
                           const std::string& screening_mode,
                           double e0,
                           int strategy_factor) {
        PoolBuildParams params;
        params.screening_mode = screening_mode;
        params.e0 = e0;
        params.max_rounds = max_rounds;
        params.threshold_decay = threshold_decay;
        params.strategy_factor = strategy_factor;
        
        auto result = pool_build_t<uint64_t>(initial_pool, initial_coeff, n_orb, h1, eri,
                                threshold, target_size, cache, cache_file,
                                attentive_orbitals, verbosity, params);
        return py::make_tuple(result.first, result.second);
    }, py::arg("initial_pool"), py::arg("initial_coeff"), py::arg("n_orb"),
       py::arg("h1"), py::arg("eri"),
       py::arg("threshold"), py::arg("target_size"),
       py::arg("cache"), py::arg("cache_file"),
       py::arg("max_rounds") = 2, py::arg("threshold_decay") = 0.5,
       py::arg("attentive_orbitals") = std::vector<int>{},
       py::arg("verbosity") = 1,
       py::arg("screening_mode") = "heat_bath", py::arg("e0") = 0.0,
       py::arg("strategy_factor") = -1);

    // Alias for 64-bit
    m.def("pool_build_64", [](const std::vector<Determinant>& initial_pool,
                              const std::vector<double>& initial_coeff,
                              int n_orb,
                              const std::vector<std::vector<double>>& h1,
                              const std::vector<double>& eri,
                              double threshold,
                              size_t target_size,
                              HijCache& cache,
                              const std::string& cache_file,
                              int max_rounds,
                              double threshold_decay,
                              const std::vector<int>& attentive_orbitals,
                              int verbosity,
                              const std::string& screening_mode,
                              double e0,
                              int strategy_factor) {
        PoolBuildParams params;
        params.screening_mode = screening_mode;
        params.e0 = e0;
        params.max_rounds = max_rounds;
        params.threshold_decay = threshold_decay;
        params.strategy_factor = strategy_factor;
        
        auto result = pool_build_t<uint64_t>(initial_pool, initial_coeff, n_orb, h1, eri,
                                threshold, target_size, cache, cache_file,
                                attentive_orbitals, verbosity, params);
        return py::make_tuple(result.first, result.second);
    }, py::arg("initial_pool"), py::arg("initial_coeff"), py::arg("n_orb"),
       py::arg("h1"), py::arg("eri"),
       py::arg("threshold"), py::arg("target_size"),
       py::arg("cache"), py::arg("cache_file"),
       py::arg("max_rounds") = 2, py::arg("threshold_decay") = 0.5,
       py::arg("attentive_orbitals") = std::vector<int>{},
       py::arg("verbosity") = 1,
       py::arg("screening_mode") = "heat_bath", py::arg("e0") = 0.0,
       py::arg("strategy_factor") = -1);


    // Scalable screening bindings
    bind_scalable_screening<2>(m, "128");
    bind_scalable_screening<3>(m, "192");
    bind_scalable_screening<4>(m, "256");
    bind_scalable_screening<5>(m, "320");
    bind_scalable_screening<6>(m, "384");
    bind_scalable_screening<7>(m, "448");
    bind_scalable_screening<8>(m, "512");
}

void bind_trim(py::module& m) {
    // Trim functions for Determinant64 (Default)
    m.def("diagonalize_subspace_davidson", &diagonalize_subspace_davidson,
          py::arg("dets"), py::arg("h1"), py::arg("eri"),
          py::arg("cache"), py::arg("quantization"),
          py::arg("max_iter") = 100, py::arg("tol") = 1e-3,
          py::arg("verbosity") = 0, py::arg("n_orb") = 0,
          py::arg("initial_guess") = std::vector<double>{},
          py::arg("init_strategy") = std::string("lowest_diag_noise"),
          py::call_guard<py::gil_scoped_release>());

    m.def("select_top_k_dets", &select_top_k_dets,
          py::arg("dets"), py::arg("coeffs"), py::arg("k"),
          py::arg("core_set") = std::vector<Determinant>{},
          py::arg("keep_core") = true);

    m.def("run_trim",
        [](const std::vector<Determinant>& pool,
           const std::vector<std::vector<double>>& h1,
           const std::vector<double>& eri,
           const std::string& mol_name,
           int n_elec, int n_orb,
           const std::vector<int>& group_sizes,
           const std::vector<int>& keep_sizes,
           bool quantization, bool save_cache,
           const std::vector<Determinant>& external_core_dets,
           double tol, int verbosity,
           const std::string& davidson_init) {
               return run_trim(pool, h1, eri, mol_name, n_elec, n_orb, group_sizes, keep_sizes, quantization, save_cache, external_core_dets, tol, verbosity, davidson_init);
           },
          py::arg("pool"), py::arg("h1"), py::arg("eri"),
          py::arg("mol_name"), py::arg("n_elec"), py::arg("n_orb"),
          py::arg("group_sizes"),
          py::arg("keep_sizes"),
          py::arg("quantization") = false, py::arg("save_cache") = true,
          py::arg("external_core_dets") = std::vector<Determinant>{},
          py::arg("tol") = 1e-3, py::arg("verbosity") = 1,
          py::arg("davidson_init") = std::string("lowest_diag_noise"));

    // Aliases for 64-bit
    m.def("diagonalize_subspace_davidson_64", &diagonalize_subspace_davidson,
          py::arg("dets"), py::arg("h1"), py::arg("eri"),
          py::arg("cache"), py::arg("quantization"),
          py::arg("max_iter") = 100, py::arg("tol") = 1e-3,
          py::arg("verbosity") = 0, py::arg("n_orb") = 0,
          py::arg("initial_guess") = std::vector<double>{},
          py::arg("init_strategy") = std::string("lowest_diag_noise"),
          py::call_guard<py::gil_scoped_release>());

    m.def("select_top_k_dets_64", &select_top_k_dets,
          py::arg("dets"), py::arg("coeffs"), py::arg("k"),
          py::arg("core_set") = std::vector<Determinant>{},
          py::arg("keep_core") = true);

    m.def("run_trim_64",
        [](const std::vector<Determinant>& pool,
           const std::vector<std::vector<double>>& h1,
           const std::vector<double>& eri,
           const std::string& mol_name,
           int n_elec, int n_orb,
           const std::vector<int>& group_sizes,
           const std::vector<int>& keep_sizes,
           bool quantization, bool save_cache,
           const std::vector<Determinant>& external_core_dets,
           double tol, int verbosity,
           const std::string& davidson_init) {
               return run_trim(pool, h1, eri, mol_name, n_elec, n_orb, group_sizes, keep_sizes, quantization, save_cache, external_core_dets, tol, verbosity, davidson_init);
           },
          py::arg("pool"), py::arg("h1"), py::arg("eri"),
          py::arg("mol_name"), py::arg("n_elec"), py::arg("n_orb"),
          py::arg("group_sizes"),
          py::arg("keep_sizes"),
          py::arg("quantization") = false, py::arg("save_cache") = true,
          py::arg("external_core_dets") = std::vector<Determinant>{},
          py::arg("tol") = 1e-3, py::arg("verbosity") = 1,
          py::arg("davidson_init") = std::string("lowest_diag_noise"));


    // Scalable trim bindings
    bind_scalable_trim<2>(m, "128");
    bind_scalable_trim<3>(m, "192");
    bind_scalable_trim<4>(m, "256");
    bind_scalable_trim<5>(m, "320");
    bind_scalable_trim<6>(m, "384");
    bind_scalable_trim<7>(m, "448");
    bind_scalable_trim<8>(m, "512");
    
    // New function: Integral Transformation (with attentive optimization)
    m.def("transform_integrals", &transform_integrals,
          py::arg("h1"), py::arg("eri"), py::arg("U"),
          py::arg("attentive_orbitals") = std::vector<int>{},
          py::call_guard<py::gil_scoped_release>());
    
    // Benchmark function: measure pure C++ transform time (no data conversion)
    m.def("benchmark_transform_integrals", [](int n, int k, int n_runs) {
        // Create test data in C++
        std::vector<std::vector<double>> h1(n, std::vector<double>(n, 0.0));
        std::vector<double> eri(n*n*n*n, 0.1);
        std::vector<std::vector<double>> U(n, std::vector<double>(n, 0.0));
        
        // Identity U with small rotation in attentive block
        for (int i = 0; i < n; ++i) {
            U[i][i] = 1.0;
        }
        // Add small rotation in attentive block
        for (int i = 0; i < k && i < n; ++i) {
            for (int j = i+1; j < k && j < n; ++j) {
                U[i][j] = 0.01;
                U[j][i] = -0.01;
            }
        }
        
        std::vector<int> att;
        for (int i = 0; i < k; ++i) att.push_back(i);
        
        // Warmup
        transform_integrals(h1, eri, U, att);
        
        // Benchmark BD
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < n_runs; ++i) {
            transform_integrals(h1, eri, U, att);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double t_bd = std::chrono::duration<double, std::milli>(t1 - t0).count() / n_runs;
        
        // Benchmark Full
        std::vector<int> empty_att;
        t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < n_runs; ++i) {
            transform_integrals(h1, eri, U, empty_att);
        }
        t1 = std::chrono::high_resolution_clock::now();
        double t_full = std::chrono::duration<double, std::milli>(t1 - t0).count() / n_runs;
        
        return py::make_tuple(t_full, t_bd);
    }, py::arg("n"), py::arg("k"), py::arg("n_runs") = 10,
    "Benchmark pure C++ ERI transform (no Python data conversion). Returns (full_ms, bd_ms).");
    
    // New function: Parallel FD Gradient Computation
    m.def("compute_fd_gradient_parallel", &compute_fd_gradient_parallel,
          py::arg("dets"), py::arg("h1"), py::arg("eri"), py::arg("cache"),
          py::arg("x"), py::arg("active_indices"),
          py::arg("n_orb"), py::arg("n_elec"),
          py::arg("eps") = 1e-5, py::arg("davidson_tol") = 1e-6, py::arg("davidson_max_iter") = 500,
          py::call_guard<py::gil_scoped_release>(),
          "Compute FD gradient in parallel using OpenMP. Only computes for active_indices.");
}

void bind_iterative_workflow(py::module& m) {
    // Bind IterativeWorkflowParams
    py::class_<IterativeWorkflowParams>(m, "IterativeWorkflowParams")
        .def(py::init<>())
        .def_readwrite("max_iterations", &IterativeWorkflowParams::max_iterations)
        .def_readwrite("energy_threshold", &IterativeWorkflowParams::energy_threshold)
        .def_readwrite("max_final_dets", &IterativeWorkflowParams::max_final_dets)
        .def_readwrite("core_set_ratio", &IterativeWorkflowParams::core_set_ratio)
        .def_readwrite("initial_pool_size", &IterativeWorkflowParams::initial_pool_size)
        .def_readwrite("core_set_schedule", &IterativeWorkflowParams::core_set_schedule)
        .def_readwrite("first_cycle_keep_size", &IterativeWorkflowParams::first_cycle_keep_size)
        .def_readwrite("pool_core_ratio", &IterativeWorkflowParams::pool_core_ratio)
        .def_readwrite("pool_build_strategy", &IterativeWorkflowParams::pool_build_strategy)
        .def_readwrite("threshold", &IterativeWorkflowParams::threshold)
        .def_readwrite("first_cycle_threshold", &IterativeWorkflowParams::first_cycle_threshold)
        .def_readwrite("threshold_decay", &IterativeWorkflowParams::threshold_decay)
        .def_readwrite("max_rounds", &IterativeWorkflowParams::max_rounds)
        .def_readwrite("attentive_orbitals", &IterativeWorkflowParams::attentive_orbitals)
        .def_readwrite("strategy_factor", &IterativeWorkflowParams::strategy_factor)
        .def_readwrite("e0", &IterativeWorkflowParams::e0)
        .def_readwrite("pool_strict_target_size", &IterativeWorkflowParams::pool_strict_target_size)
        .def_readwrite("max_pool_size", &IterativeWorkflowParams::max_pool_size)
        .def_readwrite("stagnation_limit", &IterativeWorkflowParams::stagnation_limit)
        .def_readwrite("noise_strength", &IterativeWorkflowParams::noise_strength)
        .def_readwrite("num_groups", &IterativeWorkflowParams::num_groups)
        .def_readwrite("num_groups_ratio", &IterativeWorkflowParams::num_groups_ratio)
        .def_readwrite("local_trim_keep_ratio", &IterativeWorkflowParams::local_trim_keep_ratio)
        .def_readwrite("keep_ratio", &IterativeWorkflowParams::keep_ratio)
        .def_readwrite("verbosity", &IterativeWorkflowParams::verbosity)
        .def_readwrite("davidson_init", &IterativeWorkflowParams::davidson_init)
        .def_readwrite("save_period", &IterativeWorkflowParams::save_period)
        .def_readwrite("save_pool", &IterativeWorkflowParams::save_pool)
        .def_readwrite("save_initial", &IterativeWorkflowParams::save_initial)
        .def_readwrite("output_dir", &IterativeWorkflowParams::output_dir);
    
    // Bind IterationInfo
    py::class_<IterationInfo>(m, "IterationInfo")
        .def(py::init<>())
        .def_readonly("iteration", &IterationInfo::iteration)
        .def_readonly("core_set_size_before", &IterationInfo::core_set_size_before)
        .def_readonly("target_pool_size", &IterationInfo::target_pool_size)
        .def_readonly("actual_pool_size", &IterationInfo::actual_pool_size)
        .def_readonly("final_threshold", &IterationInfo::final_threshold)
        .def_readonly("pool_building_time", &IterationInfo::pool_building_time)
        .def_readonly("trim_m", &IterationInfo::trim_m)
        .def_readonly("trim_k", &IterationInfo::trim_k)
        .def_readonly("raw_dets_count", &IterationInfo::raw_dets_count)
        .def_readonly("raw_energy", &IterationInfo::raw_energy)
        .def_readonly("energy_change", &IterationInfo::energy_change)
        .def_readonly("converged", &IterationInfo::converged)
        .def_readonly("core_set_size_after", &IterationInfo::core_set_size_after)
        .def_readonly("iteration_time", &IterationInfo::iteration_time)
        .def_readonly("cumulative_time", &IterationInfo::cumulative_time)
        .def_readonly("stop_reason", &IterationInfo::stop_reason);
    
    // Bind IterativeWorkflowResult64
    py::class_<IterativeWorkflowResult64>(m, "IterativeWorkflowResult")
        .def(py::init<>())
        .def_readonly("final_energy", &IterativeWorkflowResult64::final_energy)
        .def_readonly("final_dets", &IterativeWorkflowResult64::final_dets)
        .def_readonly("final_coeffs", &IterativeWorkflowResult64::final_coeffs)
        .def_readonly("iteration_history", &IterativeWorkflowResult64::iteration_history)
        .def_readonly("total_time", &IterativeWorkflowResult64::total_time)
        .def_readonly("total_iterations", &IterativeWorkflowResult64::total_iterations)
        .def_readonly("success", &IterativeWorkflowResult64::success)
        .def_readonly("error_message", &IterativeWorkflowResult64::error_message);
    
    // Bind iterative_workflow function
    m.def("iterative_workflow_cpp", &iterative_workflow,
          py::arg("h1"), py::arg("eri"),
          py::arg("n_alpha"), py::arg("n_beta"), py::arg("n_orb"),
          py::arg("system_name"),
          py::arg("initial_dets"), py::arg("initial_coeffs"),
          py::arg("nuclear_repulsion"),
          py::arg("params"),
          py::call_guard<py::gil_scoped_release>(),
          R"doc(
C++ implementation of iterative_workflow.

This is a pure C++ implementation that avoids the overhead of 
C++ -> Python object conversion for intermediate results.
Only the final result is converted to Python objects.

Args:
    h1: One-body integrals (n_orb x n_orb)
    eri: Two-body integrals (flattened n_orb^4)
    n_alpha, n_beta: Number of alpha/beta electrons
    n_orb: Number of orbitals
    system_name: System identifier
    initial_dets: Initial determinants
    initial_coeffs: Initial coefficients
    nuclear_repulsion: Nuclear repulsion energy
    params: IterativeWorkflowParams configuration

Returns:
    IterativeWorkflowResult with final energy, dets, coeffs, and history
)doc");

    // Numpy-optimized version: accepts numpy arrays directly to avoid list conversion overhead
    // This can provide ~2-3x speedup for large systems (256+ orbitals)
    m.def("iterative_workflow_cpp_np", 
          [](py::array_t<double, py::array::c_style | py::array::forcecast> h1_np,
             py::array_t<double, py::array::c_style | py::array::forcecast> eri_np,
             int n_alpha, int n_beta, int n_orb,
             const std::string& system_name,
             const std::vector<Determinant64>& initial_dets,
             const std::vector<double>& initial_coeffs,
             double nuclear_repulsion,
             const IterativeWorkflowParams& params) {
              
              // Convert h1 from 2D numpy array to vector<vector<double>>
              auto h1_buf = h1_np.request();
              if (h1_buf.ndim != 2) {
                  throw std::runtime_error("h1 must be 2D array");
              }
              size_t h1_rows = h1_buf.shape[0];
              size_t h1_cols = h1_buf.shape[1];
              double* h1_ptr = static_cast<double*>(h1_buf.ptr);
              
              std::vector<std::vector<double>> h1(h1_rows, std::vector<double>(h1_cols));
              for (size_t i = 0; i < h1_rows; ++i) {
                  for (size_t j = 0; j < h1_cols; ++j) {
                      h1[i][j] = h1_ptr[i * h1_cols + j];
                  }
              }
              
              // Convert eri from 1D numpy array to vector<double>
              auto eri_buf = eri_np.request();
              double* eri_ptr = static_cast<double*>(eri_buf.ptr);
              size_t eri_size = eri_buf.size;
              std::vector<double> eri(eri_ptr, eri_ptr + eri_size);
              
              // Note: Not releasing GIL here since iterative_workflow may
              // indirectly access Python objects through pybind11
              
              // Call the actual workflow
              return iterative_workflow(h1, eri, n_alpha, n_beta, n_orb, 
                                        system_name, initial_dets, initial_coeffs,
                                        nuclear_repulsion, params);
          },
          py::arg("h1"), py::arg("eri"),
          py::arg("n_alpha"), py::arg("n_beta"), py::arg("n_orb"),
          py::arg("system_name"),
          py::arg("initial_dets"), py::arg("initial_coeffs"),
          py::arg("nuclear_repulsion"),
          py::arg("params"),
          // Note: GIL is released manually inside lambda after numpy conversion
          R"doc(
Numpy-optimized C++ implementation of iterative_workflow.

Same as iterative_workflow_cpp, but accepts numpy arrays directly
instead of Python lists. This avoids the expensive list-to-vector
conversion and provides ~2-3x speedup for large systems.

Args:
    h1: One-body integrals as 2D numpy array (n_orb x n_orb)
    eri: Two-body integrals as 1D numpy array (flattened n_orb^4)
    n_alpha, n_beta: Number of alpha/beta electrons
    n_orb: Number of orbitals
    system_name: System identifier
    initial_dets: Initial determinants
    initial_coeffs: Initial coefficients
    nuclear_repulsion: Nuclear repulsion energy
    params: IterativeWorkflowParams configuration

Returns:
    IterativeWorkflowResult with final energy, dets, coeffs, and history
)doc");

    // Bind scalable workflow versions (128-512 bit) - all return unified Result objects
    bind_scalable_workflow<2>(m, "128");
    bind_scalable_workflow<3>(m, "192");
    bind_scalable_workflow<4>(m, "256");
    bind_scalable_workflow<5>(m, "320");
    bind_scalable_workflow<6>(m, "384");
    bind_scalable_workflow<7>(m, "448");
    bind_scalable_workflow<8>(m, "512");
}

PYBIND11_MODULE(trimci_core, m) {
    m.doc() = "TrimCI core: Scalable Determinant, Hamiltonian, Screening, Trim, Iterative Workflow";

    bind_determinants(m);
    bind_hamiltonian(m);
    bind_screening(m);
    bind_trim(m);
    bind_iterative_workflow(m);

    // Davidson GEP solver
    m.def("davidson_gep", [](
        Eigen::Ref<const Eigen::MatrixXd> H,
        Eigen::Ref<const Eigen::MatrixXd> S,
        int max_iter,
        double tol,
        int max_subspace,
        int verbose) {
            // Run solver without GIL
            DavidsonGEPResult result;
            {
                py::gil_scoped_release release;
                result = davidson_gep(H, S, max_iter, tol, max_subspace, verbose);
            }
            // Build Python objects with GIL held
            py::dict info;
            info["converged"] = result.converged;
            info["iterations"] = result.iterations;
            info["residual_norm"] = result.residual_norm;
            return py::make_tuple(result.eigenvalue, result.eigenvector, info);
        },
        py::arg("H"), py::arg("S"),
        py::arg("max_iter") = 200,
        py::arg("tol") = 1e-8,
        py::arg("max_subspace") = 40,
        py::arg("verbose") = 0,
        "Davidson solver for generalized eigenvalue problem Hc = ESc");

    // Matrix-free Davidson GEP solver
    m.def("matfree_davidson_gep", [](
        py::array_t<uint64_t, py::array::c_style> ref_alpha,
        py::array_t<uint64_t, py::array::c_style> ref_beta,
        py::array_t<double, py::array::c_style> ref_coeffs,
        py::array_t<int, py::array::c_style> exc_types,
        py::array_t<int, py::array::c_style> exc_indices,
        py::array_t<double, py::array::c_style> h1,
        py::array_t<double, py::array::c_style> eri,
        int n_orb,
        int n_basis,
        int max_iter,
        double tol,
        int max_subspace,
        int verbose) {
            int n_ref = static_cast<int>(ref_alpha.size());
            int n_exc = n_basis - 1;

            MatfreeDavidsonResult result;
            {
                py::gil_scoped_release release;
                result = matfree_davidson_gep(
                    ref_alpha.data(), ref_beta.data(), ref_coeffs.data(), n_ref,
                    exc_types.data(), exc_indices.data(), n_exc, n_basis,
                    h1.data(), eri.data(), n_orb,
                    max_iter, tol, max_subspace, verbose);
            }

            py::dict info;
            info["converged"] = result.converged;
            info["iterations"] = result.iterations;
            info["residual_norm"] = result.residual_norm;
            info["h_diag_0"] = result.h_diag_0;
            return py::make_tuple(result.eigenvalue, result.eigenvector, info);
        },
        py::arg("ref_alpha"), py::arg("ref_beta"), py::arg("ref_coeffs"),
        py::arg("exc_types"), py::arg("exc_indices"),
        py::arg("h1"), py::arg("eri"),
        py::arg("n_orb"), py::arg("n_basis"),
        py::arg("max_iter") = 100,
        py::arg("tol") = 1e-6,
        py::arg("max_subspace") = 30,
        py::arg("verbose") = 0,
        "Matrix-free Davidson solver for LVCC generalized eigenvalue problem.\n"
        "Computes H*v and S*v on-the-fly using inverse map + Slater-Condon rules.\n"
        "Memory: O(n_basis × n_subspace) vs O(n_basis²) for dense Davidson.");
}