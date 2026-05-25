// Bindings for screening and trim algorithms
#include "bind_common.hpp"
#include <chrono>

#include "determinant.hpp"
#include "hamiltonian.hpp"
#include "screening.hpp"
#include "trim.hpp"

using namespace trimci_core;

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
          py::arg("verbosity") = 1, py::arg("screening_mode") = "hb", py::arg("e0") = 0.0,
          py::arg("strategy_factor") = -1);
}

template<size_t N>
void bind_scalable_trim(py::module& m, const std::string& suffix) {
    using DetType = DeterminantT<std::array<uint64_t, N>>;
    // Hide IntegralSparsityInfo* from Python (same pattern as run_trim).
    using CacheType = HijCacheT<std::array<uint64_t, N>>;
    m.def(("diagonalize_subspace_davidson_" + suffix).c_str(),
          [](const std::vector<DetType>& dets,
             const std::vector<std::vector<double>>& h1,
             const std::vector<double>& eri,
             CacheType& cache, bool quantization,
             int max_iter, double tol, int verbosity, int n_orb,
             const std::vector<double>& initial_guess) {
                return diagonalize_subspace_davidson_t<std::array<uint64_t, N>>(
                    dets, h1, eri, cache, quantization,
                    max_iter, tol, verbosity, n_orb, initial_guess, nullptr);
          },
          py::arg("dets"), py::arg("h1"), py::arg("eri"),
          py::arg("cache"), py::arg("quantization"),
          py::arg("max_iter") = 100, py::arg("tol") = 1e-3,
          py::arg("verbosity") = 0, py::arg("n_orb") = 0,
          py::arg("initial_guess") = std::vector<double>{});

    m.def(("select_top_k_dets_" + suffix).c_str(), &select_top_k_dets_t<std::array<uint64_t, N>>,
          py::arg("dets"), py::arg("coeffs"), py::arg("k"), py::arg("core_set") = std::vector<DetType>{}, py::arg("keep_core") = true);

    // NOTE:
    // Keep Python-visible signature aligned with run_trim_64.
    // Exposing `IntegralSparsityInfo*` directly here breaks calls when that type
    // is not bound in Python (TypeError on all invocations, even with defaults).
    m.def(("run_trim_" + suffix).c_str(),
          [](const std::vector<DetType>& pool,
             const std::vector<std::vector<double>>& h1,
             const std::vector<double>& eri,
             const std::string& mol_name,
             int n_elec, int n_orb,
             const std::vector<int>& group_sizes,
             const std::vector<int>& keep_sizes,
             bool quantization, bool save_cache,
             const std::vector<DetType>& external_core_dets,
             double tol, int verbosity) {
                return run_trim_t<std::array<uint64_t, N>>(
                    pool, h1, eri, mol_name, n_elec, n_orb, group_sizes, keep_sizes,
                    quantization, save_cache, external_core_dets, tol, verbosity,
                    nullptr, std::vector<double>{});
          },
          py::arg("pool"), py::arg("h1"), py::arg("eri"),
          py::arg("mol_name"), py::arg("n_elec"), py::arg("n_orb"),
          py::arg("group_sizes"), py::arg("keep_sizes"),
          py::arg("quantization") = false, py::arg("save_cache") = true,
          py::arg("external_core_dets") = std::vector<DetType>{},
          py::arg("tol") = 1e-3, py::arg("verbosity") = 1);
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
       py::arg("screening_mode") = "hb", py::arg("e0") = 0.0,
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
       py::arg("screening_mode") = "hb", py::arg("e0") = 0.0,
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
           double tol, int verbosity) {
               return run_trim(pool, h1, eri, mol_name, n_elec, n_orb, group_sizes, keep_sizes, quantization, save_cache, external_core_dets, tol, verbosity);
           },
          py::arg("pool"), py::arg("h1"), py::arg("eri"),
          py::arg("mol_name"), py::arg("n_elec"), py::arg("n_orb"),
          py::arg("group_sizes"),
          py::arg("keep_sizes"),
          py::arg("quantization") = false, py::arg("save_cache") = true,
          py::arg("external_core_dets") = std::vector<Determinant>{},
          py::arg("tol") = 1e-3, py::arg("verbosity") = 1);

    // Aliases for 64-bit
    m.def("diagonalize_subspace_davidson_64", &diagonalize_subspace_davidson,
          py::arg("dets"), py::arg("h1"), py::arg("eri"),
          py::arg("cache"), py::arg("quantization"),
          py::arg("max_iter") = 100, py::arg("tol") = 1e-3,
          py::arg("verbosity") = 0, py::arg("n_orb") = 0,
          py::arg("initial_guess") = std::vector<double>{},
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
           double tol, int verbosity) {
               return run_trim(pool, h1, eri, mol_name, n_elec, n_orb, group_sizes, keep_sizes, quantization, save_cache, external_core_dets, tol, verbosity);
           },
          py::arg("pool"), py::arg("h1"), py::arg("eri"),
          py::arg("mol_name"), py::arg("n_elec"), py::arg("n_orb"),
          py::arg("group_sizes"),
          py::arg("keep_sizes"),
          py::arg("quantization") = false, py::arg("save_cache") = true,
          py::arg("external_core_dets") = std::vector<Determinant>{},
          py::arg("tol") = 1e-3, py::arg("verbosity") = 1);

    // Scalable trim bindings
    bind_scalable_trim<2>(m, "128");
    bind_scalable_trim<3>(m, "192");
    bind_scalable_trim<4>(m, "256");
    bind_scalable_trim<5>(m, "320");
    bind_scalable_trim<6>(m, "384");
    bind_scalable_trim<7>(m, "448");
    bind_scalable_trim<8>(m, "512");

    // Integral Transformation
    m.def("transform_integrals", &transform_integrals,
          py::arg("h1"), py::arg("eri"), py::arg("U"),
          py::arg("attentive_orbitals") = std::vector<int>{},
          py::call_guard<py::gil_scoped_release>());

    // Benchmark function
    m.def("benchmark_transform_integrals", [](int n, int k, int n_runs) {
        std::vector<std::vector<double>> h1(n, std::vector<double>(n, 0.0));
        std::vector<double> eri(n*n*n*n, 0.1);
        std::vector<std::vector<double>> U(n, std::vector<double>(n, 0.0));
        for (int i = 0; i < n; ++i) U[i][i] = 1.0;
        for (int i = 0; i < k && i < n; ++i)
            for (int j = i+1; j < k && j < n; ++j) { U[i][j] = 0.01; U[j][i] = -0.01; }
        std::vector<int> att;
        for (int i = 0; i < k; ++i) att.push_back(i);
        transform_integrals(h1, eri, U, att);
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < n_runs; ++i) transform_integrals(h1, eri, U, att);
        auto t1 = std::chrono::high_resolution_clock::now();
        double t_bd = std::chrono::duration<double, std::milli>(t1 - t0).count() / n_runs;
        std::vector<int> empty_att;
        t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < n_runs; ++i) transform_integrals(h1, eri, U, empty_att);
        t1 = std::chrono::high_resolution_clock::now();
        double t_full = std::chrono::duration<double, std::milli>(t1 - t0).count() / n_runs;
        return py::make_tuple(t_full, t_bd);
    }, py::arg("n"), py::arg("k"), py::arg("n_runs") = 10,
    "Benchmark pure C++ ERI transform. Returns (full_ms, bd_ms).");

    // Parallel FD Gradient
    m.def("compute_fd_gradient_parallel", &compute_fd_gradient_parallel,
          py::arg("dets"), py::arg("h1"), py::arg("eri"), py::arg("cache"),
          py::arg("x"), py::arg("active_indices"),
          py::arg("n_orb"), py::arg("n_elec"),
          py::arg("eps") = 1e-5, py::arg("davidson_tol") = 1e-6, py::arg("davidson_max_iter") = 500,
          py::call_guard<py::gil_scoped_release>(),
          "Compute FD gradient in parallel using OpenMP. Only computes for active_indices.");
}
