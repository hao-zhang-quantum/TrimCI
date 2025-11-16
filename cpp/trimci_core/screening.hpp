// screening.hpp
#pragma once

#include <vector>
#include <unordered_map>
#include <tuple>
#include "determinant.hpp"
#include "hamiltonian.hpp"

namespace trimci_core {

// 双激发表的 key 和哈希
using ExcTableKey = std::pair<int,int>;
struct PairHash {
    size_t operator()(ExcTableKey const& p) const noexcept {
        return std::hash<int>()(p.first)
             ^ (std::hash<int>()(p.second) << 1);
    }
};

// (<i,j> -> list of (p,q,h_val))
using DoubleExcTable =
    std::unordered_map<ExcTableKey,
                       std::vector<std::tuple<int,int,double>>,
                       PairHash>;

// 预计算双激发表：筛选 |h|>thr，按 |h| 降序
DoubleExcTable precompute_double_exc_table(
    int n_orb,
    const std::vector<std::vector<std::vector<std::vector<double>>>>& eri,
    double thr
);

// 处理单个父行列式，返回所有 |H_ij|>thr 的 (det, hij)
std::vector<std::pair<Determinant,double>>
process_parent_worker(
    const Determinant& det,
    int n_orb,
    double thr,
    const DoubleExcTable& table,
    const std::vector<std::vector<double>>& h1,
    const std::vector<std::vector<std::vector<std::vector<double>>>>& eri
);

// 优化后的并行筛选主流程：直接返回完整的池和调整后的threshold，避免Python端的列表拼接
std::pair<std::vector<Determinant>, double>
pool_build(
    const std::vector<Determinant>& initial_pool,
    const std::vector<double>& initial_coeff,
    int n_orb,
    const std::vector<std::vector<double>>& h1,
    const std::vector<std::vector<std::vector<std::vector<double>>>>& eri,
    double threshold,
    size_t target_size,
    HijCache& cache,
    const std::string& cache_file,
    int max_rounds = -1
);

} // namespace trimci_core
