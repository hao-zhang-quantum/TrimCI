// =================================================================================
// FILE: trim.hpp
// =================================================================================
#ifndef TRIMCI_CORE_TRIM_HPP
#define TRIMCI_CORE_TRIM_HPP

#include <vector>
#include <tuple>
#include <string>
#include "determinant.hpp"
#include "hamiltonian.hpp"
#include <unordered_set>

// Include scalable trim for template-based functionality
#include "trim_scalable.hpp"

namespace trimci_core {

// 1) Randomly partition the pool of determinants
std::vector<std::vector<Determinant>>
partition_pool(const std::vector<Determinant>& pool, int m);

// 2) Diagonally Preconditioned Davidson Method
std::tuple<double, std::vector<double>>
diagonalize_subspace_davidson(
    const std::vector<Determinant>& dets,
    const std::vector<std::vector<double>>& h1,
    const std::vector<std::vector<std::vector<std::vector<double>>>>& eri,
    HijCache& cache,
    bool quantization,
    int max_iter = 100,
    double tol = 1e-6,
    bool verbose = false,
    int n_orb = 0
);

// 3) Select top-k determinants based on coefficient magnitude
std::vector<Determinant>
select_top_k_dets(
    const std::vector<Determinant>& dets,
    const std::vector<double>& coeffs,
    size_t k,
    const std::vector<Determinant>& core_vec = {},
    bool keep_core = true
);

// 4) Main Trim workflow with core_dets support
// 该函数现在支持多个核心行列式(core_dets)，包括HF态和其他重要行列式
// 这些核心行列式会被添加到每个子空间中，并在选择top-k时考虑
// 用户可以提供外部的core_dets，如果不提供，函数会自动生成
std::tuple<double, std::vector<Determinant>, std::vector<double>>
run_trim(
    const std::vector<Determinant>& pool,
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
    const std::vector<Determinant>& external_core_dets
);

} // namespace trimci_core

#endif // TRIMCI_CORE_TRIM_HPP