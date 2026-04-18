#pragma once

#include <vector>
#include <unordered_map>
#include <tuple>
#include <array>
#include <string>
#include "determinant.hpp"
#include "hamiltonian.hpp"

namespace trimci_core {

// Pool building parameters structure
// Contains tunable parameters for the pool building algorithm
struct PoolBuildParams {
    // Screening strategy
    std::string screening_mode = "heat_bath";  // "heat_bath", "heat_bath_pt2", "pt2"
    double e0 = 0.0;  // reference energy for PT2 denominator
    
    // Expansion control
    int max_rounds = 2;           // rounds per threshold level
    double threshold_decay = 0.5; // threshold reduction factor
    double min_threshold = 1e-10; // minimum threshold
    int max_stagnant_rounds = 1000; // max consecutive threshold reductions without progress
    
    // Strategy factor: controls pre-filter size for PT2 modes
    // -1 = automatic (1 for heat_bath, 20 for PT2 modes)
    // User can override to any positive value
    int strategy_factor = -1;

    // Mutual-information weights for "mi" screening mode.
    // Flat n_orb × n_orb: mi_weights[p * mi_n_orb + q] = |C(p,q)|.
    // Score = |H_ij * c_i| * (1 + mi_alpha * mi_weight(hole, particle)).
    std::vector<double> mi_weights;
    int mi_n_orb = 0;
    double mi_alpha = 10.0;  // weight scaling factor
    
    // PT2 specific
    double pt2_denom_min = 1e-10; // minimum denominator to avoid intruder states
    
    // Strict target size: if true, truncate pool to exactly target_size
    // This gives better time control at the cost of potentially missing important dets
    bool strict_target_size = false;

    // Early stop: if true, stop processing chunks as soon as pool >= target_size.
    // If false, process all frontier chunks before truncation, giving a larger
    // candidate pool and better strict truncation quality at the cost of more compute.
    // Default true preserves original behavior.
    bool early_stop = true;

    // Max pool size: hard cap on pool_map size during merge.
    // When pool_map reaches this size, stop merging new candidates.
    // Candidates from high-weight parents (merged first) are kept.
    // 0 = no limit (default).
    int max_pool_size = 0;
};

// Screening mode (string-based for extensibility)
// Supported modes:
// - "heat_bath": Use |H_ij * c_i| for screening (default, fast)
// - "heat_bath_pt2": Use |H_ij * c_i| / |E_0 - H_jj| for screening (includes PT2 denominator)
// - "pt2": Use |sum_i H_ij * c_i|^2 / |E_0 - H_jj| for screening (full PT2 estimation)
// Additional modes can be added in the future without ABI changes

// Double excitation table key and hash
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

// Precompute double excitation table: filter |h|>thr, sort by |h| desc
// If attentive_orbitals is non-empty, only include orbitals in that set
DoubleExcTable precompute_double_exc_table(
    int n_orb,
    const std::vector<double>& eri,
    double thr,
    const std::vector<int>& attentive_orbitals = {}
);

namespace detail {

template<typename StorageType>
int single_phase_t(const StorageType& storage, int i, int p) {
    // Use Hamiltonian's cre_des_sign_t
    return trimci_core::detail::cre_des_sign_t(p, i, storage);
}

template<typename StorageType>
int double_phase_t(const StorageType& storage, int i, int j, int p, int q) {
    // i, j are occupied; p, q are virtual
    // Phase for a_p^dag a_i (removing i, adding p)
    int ph1 = trimci_core::detail::cre_des_sign_t(p, i, storage);
    
    // Intermediate storage
    StorageType tmp = storage;
    BitOps<StorageType>::clear_bit(tmp, i);
    BitOps<StorageType>::set_bit(tmp, p);
    
    // Phase for a_q^dag a_j (removing j, adding q)
    int ph2 = trimci_core::detail::cre_des_sign_t(q, j, tmp);
    
    return ph1 * ph2;
}

} // namespace detail

// Template function declarations
// If attentive_orbitals is non-empty, excitations are restricted to those orbitals
template<typename StorageType>
std::vector<std::pair<DeterminantT<StorageType>, double>>
process_parent_worker_t(
    const DeterminantT<StorageType>& det,
    int n_orb,
    double thr,
    const DoubleExcTable& table,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    const std::unordered_set<int>& attentive_set = {},
    const IntegralSparsityInfo* sparsity = nullptr
);

// Wrapper for backward compatibility
std::vector<std::pair<Determinant,double>>
process_parent_worker(
    const Determinant& det,
    int n_orb,
    double thr,
    const DoubleExcTable& table,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri
);

// Template-based pool building function
// attentive_orbitals: if non-empty, excitations are restricted to these orbital indices
// params: PoolBuildParams struct containing algorithm parameters
template<typename StorageType>
std::pair<std::vector<DeterminantT<StorageType>>, double>
pool_build_t(
    const std::vector<DeterminantT<StorageType>>& initial_pool,
    std::vector<double> initial_coeff,  // P8: by-value for post-loop release
    int n_orb,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    double threshold,
    size_t target_size,
    HijCacheT<StorageType>& cache,
    const std::string& cache_file,
    const std::vector<int>& attentive_orbitals,
    int verbosity,
    const PoolBuildParams& params = {},
    const IntegralSparsityInfo* precomputed_sparsity = nullptr,
    const DoubleExcTable* precomputed_table = nullptr
);

// Wrapper for backward compatibility
std::pair<std::vector<Determinant>, double>
pool_build(
    const std::vector<Determinant>& initial_pool,
    const std::vector<double>& initial_coeff,
    int n_orb,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    double threshold,
    size_t target_size,
    HijCache& cache,
    const std::string& cache_file,
    int max_rounds = -1,
    double threshold_decay = 0.9,
    const std::vector<int>& attentive_orbitals = {},
    int verbosity = 1
);

} // namespace trimci_core
