#pragma once

#include <string>
#include <tuple>
#include <map>
#include <vector>
#include <cstdint>
#include "determinant.hpp"
#include "bit_compat.hpp"

namespace trimci_core {
namespace detail {

// Template-based bit manipulation utilities
// Template-based bit manipulation utilities moved to bit_compat.hpp


/// Fermionic anticommutation sign: (-1)^(occupied orbitals strictly between p and a).
///
/// This determines the phase when creating/destroying a fermion at orbital p
/// given the current occupation. The sign counts how many occupied orbitals
/// lie between positions p and a in the bitstring, and returns +1 (even) or -1 (odd).
///
/// For uint64_t: single mask + popcount (~2 cycles).
/// For std::array<uint64_t, N>: word-wise masked popcount (~N cycles).
///   Previous implementation used bit-by-bit loop (~high-low cycles, up to 100× slower).
template<typename StorageType>
int cre_des_sign_t(int p, int a, const StorageType& bitstring) {
    if (p == a) return 1;
    int low = std::min(p, a) + 1;
    int high = std::max(p, a);

    int count = 0;
    if constexpr (std::is_same_v<StorageType, uint64_t>) {
        // Single-word fast path: 1 mask + 1 popcount
        uint64_t mask = ((uint64_t(1) << high) - 1) ^ ((uint64_t(1) << low) - 1);
        count = __builtin_popcountll(bitstring & mask);
    } else {
        // Multi-word fast path: word-wise masked popcount
        // Split [low, high) into: partial first word | full middle words | partial last word
        int lo_word = low / 64;
        int lo_bit  = low % 64;
        int hi_word = high / 64;
        int hi_bit  = high % 64;

        if (lo_word == hi_word) {
            // Both endpoints in the same word
            uint64_t mask = ((uint64_t(1) << hi_bit) - 1) ^ ((uint64_t(1) << lo_bit) - 1);
            count = __builtin_popcountll(bitstring[lo_word] & mask);
        } else {
            // First partial word: bits [lo_bit, 64) in word lo_word
            uint64_t first_mask = ~((uint64_t(1) << lo_bit) - 1);  // bits lo_bit and above
            count += __builtin_popcountll(bitstring[lo_word] & first_mask);

            // Full middle words: all 64 bits count
            for (int w = lo_word + 1; w < hi_word; ++w)
                count += __builtin_popcountll(bitstring[w]);

            // Last partial word: bits [0, hi_bit) in word hi_word
            if (hi_bit > 0) {
                uint64_t last_mask = (uint64_t(1) << hi_bit) - 1;  // bits below hi_bit
                count += __builtin_popcountll(bitstring[hi_word] & last_mask);
            }
        }
    }
    return (count % 2 == 0) ? 1 : -1;
}

} // namespace detail

// Extract molecular formula from atom string
std::string extract_mol_name(const std::string& atom_str);

// Template-based Hij cache for different determinant types
template<typename StorageType>
using HijCacheT = std::map<std::pair<DeterminantT<StorageType>, DeterminantT<StorageType>>, double>;

// Backward compatibility
using HijCache = HijCacheT<uint64_t>;

// Load or create disk cache (template version)
template<typename StorageType>
std::tuple<HijCacheT<StorageType>, std::string>
load_or_create_Hij_cache_t(const std::string& mol_name,
                           int n_elec, int n_orb,
                           const std::string& cache_dir = "cache");

// Wrapper for backward compatibility
std::tuple<HijCache, std::string>
load_or_create_Hij_cache(const std::string& mol_name,
                         int n_elec, int n_orb,
                         const std::string& cache_dir = "cache");

// Template-based pair key function for cache
template<typename StorageType>
std::pair<DeterminantT<StorageType>, DeterminantT<StorageType>>
pair_key_t(const DeterminantT<StorageType>& d1, const DeterminantT<StorageType>& d2);

// Wrapper for backward compatibility
std::pair<Determinant, Determinant>
pair_key(const Determinant& d1, const Determinant& d2);

// Helper to access flattened ERI
// eri[i][j][k][l] -> eri[((i*n + j)*n + k)*n + l]
// Note: Uses size_t to avoid integer overflow for n_orb >= 256 (256^4 > INT_MAX)
inline double get_eri(const std::vector<double>& eri, int n, int i, int j, int k, int l) {
    size_t idx = ((static_cast<size_t>(i) * n + j) * n + k) * n + l;
    return eri[idx];
}


// =============================================================================
// Integral Sparsity Info (auto-detected, transparent optimization)
// =============================================================================
struct IntegralSparsityInfo {
    // h1_neighbors[i] = sorted list of j where |h1[i][j]| > threshold (i != j)
    std::vector<std::vector<int>> h1_neighbors;

    // ab_exc_table[i] = list of (j, p, q, eri(i,p,j,q)) for non-zero entries
    // i = alpha orbital index; j = beta occ, p = alpha virt, q = beta virt
    using ABEntry = std::tuple<int, int, int, double>;
    std::vector<std::vector<ABEntry>> ab_exc_table;

    bool eri_is_diagonal = false;  // true if only (ii|ii) ERIs are non-zero
    bool is_sparse = false;        // true if sparsity paths should be used
    double h1_sparsity = 1.0;     // fraction of non-zero h1 off-diagonal
    double eri_sparsity = 1.0;    // fraction of non-zero ERI
};

/// Build sparsity info from integrals (one-time O(n^4) cost).
IntegralSparsityInfo build_sparsity_info(
    int n_orb,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    double threshold = 1e-14);

// Template-based Slater-Condon matrix element computation
template<typename StorageType>
double compute_H_ij_t(const DeterminantT<StorageType>& det_i,
                      const DeterminantT<StorageType>& det_j,
                      const std::vector<std::vector<double>>& h1,
                      const std::vector<double>& eri);

// Overload with sparsity fast paths
template<typename StorageType>
double compute_H_ij_t(const DeterminantT<StorageType>& det_i,
                      const DeterminantT<StorageType>& det_j,
                      const std::vector<std::vector<double>>& h1,
                      const std::vector<double>& eri,
                      const IntegralSparsityInfo* sparsity);

// Wrapper for backward compatibility
double compute_H_ij(const Determinant& det_i,
                    const Determinant& det_j,
                    const std::vector<std::vector<double>>& h1,
                    const std::vector<double>& eri);

// =============================================================================
// CI Energy Evaluation
// =============================================================================

/**
 * @brief Evaluate CI energy given determinants and coefficients.
 * 
 * Computes E = Σ_ij c_i c_j ⟨D_i|H|D_j⟩ using Slater-Condon rules.
 * 
 * @param dets_alpha Alpha bitstrings for each determinant
 * @param dets_beta Beta bitstrings for each determinant
 * @param coeffs CI coefficients
 * @param h1 One-body integrals (n_orb x n_orb)
 * @param eri Two-body integrals (flattened n_orb^4)
 * @param n_orb Number of orbitals
 * @return CI energy
 */
double evaluate_ci_energy(
    const std::vector<uint64_t>& dets_alpha,
    const std::vector<uint64_t>& dets_beta,
    const std::vector<double>& coeffs,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    int n_orb);

/**
 * @brief Template version of evaluate_ci_energy for different storage types.
 */
template<typename StorageType>
double evaluate_ci_energy_t(
    const std::vector<DeterminantT<StorageType>>& dets,
    const std::vector<double>& coeffs,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri);

} // namespace trimci_core
