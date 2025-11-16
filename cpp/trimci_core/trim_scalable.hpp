// =================================================================================
// FILE: trim_scalable.hpp
// =================================================================================
#ifndef TRIMCI_CORE_TRIM_SCALABLE_HPP
#define TRIMCI_CORE_TRIM_SCALABLE_HPP

#include <vector>
#include <tuple>
#include <string>
#include <array>
#include <unordered_set>
#include "determinant.hpp"
#include "hamiltonian.hpp"

namespace trimci_core {

// =================================================================================
// 1) Template: partition_pool_t
// =================================================================================
template<typename StorageType>
std::vector<std::vector<DeterminantT<StorageType>>>
partition_pool_t(const std::vector<DeterminantT<StorageType>>& pool, int m);

// =================================================================================
// 2) Template: diagonalize_subspace_davidson_t
// =================================================================================
template<typename StorageType>
std::tuple<double, std::vector<double>>
diagonalize_subspace_davidson_t(
    const std::vector<DeterminantT<StorageType>>& dets,
    const std::vector<std::vector<double>>& h1,
    const std::vector<std::vector<std::vector<std::vector<double>>>>& eri,
    HijCacheT<StorageType>& cache,
    bool quantization,
    int max_iter = 100,
    double tol = 1e-6,
    bool verbose = false,
    int n_orb = 0
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

// =================================================================================
// 4) Template: run_trim_t (full multi-round Trim algorithm)
// =================================================================================
template<typename StorageType>
std::tuple<double, std::vector<DeterminantT<StorageType>>, std::vector<double>>
run_trim_t(
    const std::vector<DeterminantT<StorageType>>& pool,
    const std::vector<std::vector<double>>& h1,
    const std::vector<std::vector<std::vector<std::vector<double>>>>& eri,
    const std::string& mol_name,
    int n_elec,
    int n_orb,
    const std::vector<int>& group_sizes,
    const std::vector<int>& keep_sizes,
    bool quantization,
    bool save_cache,
    bool verbose,
    const std::vector<DeterminantT<StorageType>>& external_core_dets = {}
);

} // namespace trimci_core

#endif // TRIMCI_CORE_TRIM_SCALABLE_HPP
