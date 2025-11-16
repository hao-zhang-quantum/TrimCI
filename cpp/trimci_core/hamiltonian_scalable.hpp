#ifndef TRIMCI_CORE_HAMILTONIAN_SCALABLE_HPP
#define TRIMCI_CORE_HAMILTONIAN_SCALABLE_HPP

#include <string>
#include <tuple>
#include <map>
#include <vector>
#include <cstdint>
#include "determinant_scalable.hpp"
#include "determinant.hpp"
#include "bit_compat.hpp"

namespace trimci_core {

// Extract molecular formula from atom string
std::string extract_mol_name(const std::string& atom_str);

// Template-based Hij cache for different determinant types
template<typename StorageType>
using HijCacheT = std::map<std::pair<DeterminantT<StorageType>, DeterminantT<StorageType>>, double>;

// Note: We don't redefine HijCache here to avoid conflicts with the original definition
// The original HijCache in hamiltonian.hpp will be used for backward compatibility

// Load or create disk cache (template version)
template<typename StorageType>
std::tuple<HijCacheT<StorageType>, std::string>
load_or_create_Hij_cache_t(const std::string& mol_name,
                           int n_elec, int n_orb,
                           const std::string& cache_dir = "cache");

// Backward compatible version (renamed to avoid conflict)
std::tuple<HijCacheT<uint64_t>, std::string>
load_or_create_Hij_cache_compat(const std::string& mol_name,
                                 int n_elec, int n_orb,
                                 const std::string& cache_dir = "cache");

// Template-based pair key function for cache
template<typename StorageType>
std::pair<DeterminantT<StorageType>, DeterminantT<StorageType>>
pair_key_t(const DeterminantT<StorageType>& d1, const DeterminantT<StorageType>& d2);

// Template-based Slater-Condon matrix element computation
template<typename StorageType>
double compute_H_ij_t(const DeterminantT<StorageType>& det_i,
                      const DeterminantT<StorageType>& det_j,
                      const std::vector<std::vector<double>>& h1,
                      const std::vector<std::vector<std::vector<std::vector<double>>>>& eri);

// Helper functions for template implementation
namespace detail {

// Template-based bit manipulation utilities
template<typename StorageType>
struct HamiltonianBitOps {
    using BitOpsType = BitOps<StorageType>;
    
    // Count differences between two storage types
    static int count_differences(const StorageType& s1, const StorageType& s2) {
        if constexpr (std::is_same_v<StorageType, uint64_t>) {
            return __builtin_popcountll(s1 ^ s2);
        } else {
            int count = 0;
            for (size_t i = 0; i < s1.size(); ++i) {
                count += __builtin_popcountll(s1[i] ^ s2[i]);
            }
            return count;
        }
    }
    
    // Get XOR result
    static StorageType xor_storage(const StorageType& s1, const StorageType& s2) {
        if constexpr (std::is_same_v<StorageType, uint64_t>) {
            return s1 ^ s2;
        } else {
            StorageType result = s1;
            for (size_t i = 0; i < s1.size(); ++i) {
                result[i] ^= s2[i];
            }
            return result;
        }
    }
    
    // Get AND result with NOT
    static StorageType and_not(const StorageType& s1, const StorageType& s2) {
        if constexpr (std::is_same_v<StorageType, uint64_t>) {
            return s1 & ~s2;
        } else {
            StorageType result{};
            for (size_t i = 0; i < s1.size(); ++i) {
                result[i] = s1[i] & ~s2[i];
            }
            return result;
        }
    }
    
    // Convert storage to indices (for occupied orbitals)
    static std::vector<int> storage_to_indices(const StorageType& storage) {
        std::vector<int> indices;
        if constexpr (std::is_same_v<StorageType, uint64_t>) {
            StorageType mask = storage;
            int idx = 0;
            while (mask) {
                if (mask & 1) indices.push_back(idx);
                mask >>= 1;
                ++idx;
            }
        } else {
            for (size_t unit = 0; unit < storage.size(); ++unit) {
                uint64_t mask = storage[unit];
                int base_idx = unit * 64;
                int idx = 0;
                while (mask) {
                    if (mask & 1) indices.push_back(base_idx + idx);
                    mask >>= 1;
                    ++idx;
                }
            }
        }
        return indices;
    }
    
    // Extract indices from mask (inline version for small arrays)
    static int storage_to_indices_inline(const StorageType& storage, int* out, int max_count) {
        int count = 0;
        if constexpr (std::is_same_v<StorageType, uint64_t>) {
            StorageType mask = storage;
            while (mask && count < max_count) {
                int idx = __builtin_ctzll(mask);
                out[count++] = idx;
                mask &= mask - 1;
            }
        } else {
            for (size_t unit = 0; unit < storage.size() && count < max_count; ++unit) {
                uint64_t mask = storage[unit];
                int base_idx = unit * 64;
                while (mask && count < max_count) {
                    int idx = __builtin_ctzll(mask);
                    out[count++] = base_idx + idx;
                    mask &= mask - 1;
                }
            }
        }
        return count;
    }
};

// Template-based creation/destruction sign calculation
template<typename StorageType>
int cre_des_sign_t(int p, int a, const StorageType& bitstring) {
    if (p == a) return 1;
    int low = std::min(p, a) + 1;
    int high = std::max(p, a);
    
    int count = 0;
    if constexpr (std::is_same_v<StorageType, uint64_t>) {
        uint64_t mask = ((uint64_t(1) << high) - 1) ^ ((uint64_t(1) << low) - 1);
        count = __builtin_popcountll(bitstring & mask);
    } else {
        // For array storage, count bits in the range [low, high)
        for (int i = low; i < high; ++i) {
            if (BitOps<StorageType>::get_bit(bitstring, i)) {
                count++;
            }
        }
    }
    return (count % 2 == 0) ? 1 : -1;
}

} // namespace detail

} // namespace trimci_core

#endif // TRIMCI_CORE_HAMILTONIAN_SCALABLE_HPP