#pragma once

#include <vector>
#include <tuple>
#include <string>
#include <vector>
#include "determinant.hpp"
#include "hamiltonian.hpp"

namespace trimci_core {

// =================================================================================
// 1) Template: partition_pool_t
// =================================================================================
template<typename StorageType>
std::vector<std::vector<DeterminantT<StorageType>>>
partition_pool_t(const std::vector<DeterminantT<StorageType>>& pool, int m);

// Wrapper
std::vector<std::vector<Determinant>>
partition_pool(const std::vector<Determinant>& pool, int m);

// =================================================================================
// 2) Template: diagonalize_subspace_davidson_t
// =================================================================================
template<typename StorageType>
std::tuple<double, std::vector<double>>
diagonalize_subspace_davidson_t(
    const std::vector<DeterminantT<StorageType>>& dets,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    HijCacheT<StorageType>& cache,
    bool quantization,
    int max_iter = 100,
    double tol = 1e-3,
    int verbosity = 0,  // 0=silent, 1=basic, 2=detailed
    int n_orb = 0,
    const std::vector<double>& initial_guess = {},
    const IntegralSparsityInfo* sparsity = nullptr,
    const std::string& init_strategy = "lowest_diag_noise"
);

// Wrapper
std::tuple<double, std::vector<double>>
diagonalize_subspace_davidson(
    const std::vector<Determinant>& dets,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    HijCache& cache,
    bool quantization,
    int max_iter = 100,
    double tol = 1e-3,
    int verbosity = 0,  // 0=silent, 1=basic, 2=detailed
    int n_orb = 0,
    const std::vector<double>& initial_guess = {},
    const std::string& init_strategy = "lowest_diag_noise"
);

// =================================================================================
// 3) Template: select_top_k_dets_t
// =================================================================================
template<typename StorageType>
std::vector<DeterminantT<StorageType>>
select_top_k_dets_t(
    const std::vector<DeterminantT<StorageType>>& dets,
    const std::vector<double>& coeffs,
    size_t k,
    const std::vector<DeterminantT<StorageType>>& core_vec = {},
    bool keep_core = true
);

// Wrapper
std::vector<Determinant>
select_top_k_dets(
    const std::vector<Determinant>& dets,
    const std::vector<double>& coeffs,
    size_t k,
    const std::vector<Determinant>& core_vec = {},
    bool keep_core = true
);

// =================================================================================
// 4) Template: run_trim_t (full multi-round Trim algorithm)
// =================================================================================
template<typename StorageType>
std::tuple<double, std::vector<DeterminantT<StorageType>>, std::vector<double>>
run_trim_t(
    const std::vector<DeterminantT<StorageType>>& pool,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    const std::string& mol_name,
    int n_elec,
    int n_orb,
    const std::vector<int>& group_sizes,
    const std::vector<int>& keep_sizes,
    bool quantization,
    bool save_cache,
    const std::vector<DeterminantT<StorageType>>& external_core_dets = {},
    double tol = 1e-3,
    int verbosity = 1,  // 0=silent, 1=basic, 2=detailed
    const std::string& davidson_init = "lowest_diag_noise"
);

// Wrapper
std::tuple<double, std::vector<Determinant>, std::vector<double>>
run_trim(
    const std::vector<Determinant>& pool,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    const std::string& mol_name,
    int n_elec,
    int n_orb,
    const std::vector<int>& group_sizes,
    const std::vector<int>& keep_sizes,
    bool quantization,
    bool save_cache,
    const std::vector<Determinant>& external_core_dets,
    double tol = 1e-3,
    int verbosity = 1,  // 0=silent, 1=basic, 2=detailed
    const std::string& davidson_init = "lowest_diag_noise"
);

std::pair<std::vector<std::vector<double>>, std::vector<double>>
transform_integrals(
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    const std::vector<std::vector<double>>& U,
    const std::vector<int>& attentive_orbitals = {}
);

// =================================================================================
// Finite Difference Gradient for Orbital Optimization
// =================================================================================
/**
 * @brief Computes forward finite difference gradient for orbital rotation parameters.
 * 
 * Evaluates grad[k] = (E(x + eps*e_k) - E(x)) / eps for each active index k,
 * where E(x) is the CI energy with orbitals rotated by U = expm(K(x)).
 * 
 * All integral transformations and energy evaluations are performed internally
 * in C++ to avoid Python↔C++ data conversion overhead across FD iterations.
 * Uses OpenMP for parallel computation across active parameter indices.
 * 
 * @param dets       Determinant basis set
 * @param h1         One-electron integrals in original (unrotated) MO basis
 * @param eri        Two-electron integrals (flattened) in original MO basis
 * @param cache      Hamiltonian element cache (used for center point only)
 * @param x          Current orbital rotation parameters (kappa vector)
 * @param active_indices  Parameter indices to compute gradient for
 * @param n_orb      Number of spatial orbitals
 * @param n_elec     Number of electrons
 * @param eps        Finite difference step size
 * @param davidson_tol    Davidson solver convergence tolerance
 * @param davidson_max_iter  Maximum Davidson iterations
 * @return Gradient vector (zero for inactive indices)
 */
template<typename StorageType>
std::vector<double> compute_fd_gradient_parallel_t(
    const std::vector<DeterminantT<StorageType>>& dets,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    HijCacheT<StorageType>& cache,
    const std::vector<double>& x,
    const std::vector<int>& active_indices,
    int n_orb,
    int n_elec,
    double eps,
    double davidson_tol,
    int davidson_max_iter
);

/// @brief Non-templated wrapper using default 64-bit storage type.
std::vector<double> compute_fd_gradient_parallel(
    const std::vector<Determinant>& dets,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    HijCache& cache,
    const std::vector<double>& x,
    const std::vector<int>& active_indices,
    int n_orb,
    int n_elec,
    double eps,
    double davidson_tol,
    int davidson_max_iter
);

} // namespace trimci_core