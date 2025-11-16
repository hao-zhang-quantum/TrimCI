#pragma once

#include <vector>
#include <unordered_map>
#include <tuple>
#include <array>
#include "determinant_scalable.hpp"
#include "hamiltonian_scalable.hpp"
#include "screening.hpp"
#include "bit_compat.hpp"

namespace trimci_core {

// Forward declarations for types from screening.hpp
using ExcTableKey = std::pair<int,int>;

// Template-based double excitation table
using DoubleExcTable = std::unordered_map<ExcTableKey,
                                         std::vector<std::tuple<int,int,double>>,
                                         PairHash>;

// Precompute double excitation table (same implementation for all types)
// Template function declarations
template<typename StorageType>
std::vector<std::pair<DeterminantT<StorageType>, double>>
process_parent_worker_t(
    const DeterminantT<StorageType>& det,
    int n_orb,
    double thr,
    const DoubleExcTable& table,
    const std::vector<std::vector<double>>& h1,
    const std::vector<std::vector<std::vector<std::vector<double>>>>& eri
);

// Template-based pool building function
template<typename StorageType>
std::pair<std::vector<DeterminantT<StorageType>>, double>
pool_build_t(
    const std::vector<DeterminantT<StorageType>>& initial_pool,
    const std::vector<double>& initial_coeff,
    int n_orb,
    const std::vector<std::vector<double>>& h1,
    const std::vector<std::vector<std::vector<std::vector<double>>>>& eri,
    double threshold,
    size_t target_size,
    HijCacheT<StorageType>& cache,
    const std::string& cache_file,
    int max_rounds = -1
);

// Backward compatible version
std::pair<std::vector<Determinant>, double>
pool_build_compat(
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

// Helper functions for template implementation
namespace detail {

// Template-based phase calculation for single excitations
template<typename StorageType>
int single_phase_t(const StorageType& mask, int i, int p) {
    if constexpr (std::is_same_v<StorageType, uint64_t>) {
        int n_i = __builtin_popcountll(mask & ((1ULL<<i)-1));
        uint64_t m1 = mask & ~(1ULL<<i);
        int n_p = __builtin_popcountll(m1 & ((1ULL<<p)-1));
        return ((n_i + n_p) % 2) ? -1 : 1;
    } else {
        // For array storage, count bits manually
        int n_i = 0, n_p = 0;
        
        // Count bits before position i
        for (int bit = 0; bit < i; ++bit) {
            if (BitOps<StorageType>::get_bit(mask, bit)) n_i++;
        }
        
        // Create mask with bit i cleared
        StorageType m1 = mask;
        BitOps<StorageType>::clear_bit(m1, i);
        
        // Count bits before position p in modified mask
        for (int bit = 0; bit < p; ++bit) {
            if (BitOps<StorageType>::get_bit(m1, bit)) n_p++;
        }
        
        return ((n_i + n_p) % 2) ? -1 : 1;
    }
}

// Template-based phase calculation for double excitations
template<typename StorageType>
int double_phase_t(const StorageType& mask, int i, int j, int p, int q) {
    if constexpr (std::is_same_v<StorageType, uint64_t>) {
        int n_i = __builtin_popcountll(mask & ((1ULL<<i)-1));
        uint64_t m1 = mask & ~(1ULL<<i);
        int n_j = __builtin_popcountll(m1 & ((1ULL<<j)-1));
        uint64_t m2 = m1 & ~(1ULL<<j);
        int n_p = __builtin_popcountll(m2 & ((1ULL<<p)-1));
        uint64_t m3 = m2 | (1ULL<<p);
        int n_q = __builtin_popcountll(m3 & ((1ULL<<q)-1));
        return ((n_i + n_j + n_p + n_q) % 2) ? -1 : 1;
    } else {
        // For array storage, count bits manually
        int n_i = 0, n_j = 0, n_p = 0, n_q = 0;
        
        // Count bits before position i
        for (int bit = 0; bit < i; ++bit) {
            if (BitOps<StorageType>::get_bit(mask, bit)) n_i++;
        }
        
        // Create mask with bit i cleared
        StorageType m1 = mask;
        BitOps<StorageType>::clear_bit(m1, i);
        
        // Count bits before position j in m1
        for (int bit = 0; bit < j; ++bit) {
            if (BitOps<StorageType>::get_bit(m1, bit)) n_j++;
        }
        
        // Create mask with bit j cleared
        StorageType m2 = m1;
        BitOps<StorageType>::clear_bit(m2, j);
        
        // Count bits before position p in m2
        for (int bit = 0; bit < p; ++bit) {
            if (BitOps<StorageType>::get_bit(m2, bit)) n_p++;
        }
        
        // Create mask with bit p set
        StorageType m3 = m2;
        BitOps<StorageType>::set_bit(m3, p);
        
        // Count bits before position q in m3
        for (int bit = 0; bit < q; ++bit) {
            if (BitOps<StorageType>::get_bit(m3, bit)) n_q++;
        }
        
        return ((n_i + n_j + n_p + n_q) % 2) ? -1 : 1;
    }
}

} // namespace detail

} // namespace trimci_core