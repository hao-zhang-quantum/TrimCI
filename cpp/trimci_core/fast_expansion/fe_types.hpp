#pragma once
/**
 * Fast Expansion: shared type aliases and hash utilities.
 *
 * Provides:
 *   - High-performance hash map/set selection (robin_hood / std fallback)
 *   - SplitMix64 hash for determinant bitstrings
 *   - Common type aliases used across all FE modules
 */

#include <cstdint>
#include <cstddef>
#include <vector>
#include <array>
#include <utility>

#ifdef USE_ROBIN_HOOD
#include <robin_hood.h>
#endif

#include "determinant.hpp"
#include "bit_compat.hpp"

namespace trimci_core {
namespace fe {

// ============================================================================
// Hash map / set selection (mirrors vrci_types.hpp pattern)
// ============================================================================
#ifdef USE_ROBIN_HOOD

template<typename K, typename V, typename Hash = robin_hood::hash<K>>
using fe_map = robin_hood::unordered_flat_map<K, V, Hash>;

template<typename K, typename Hash = robin_hood::hash<K>>
using fe_set = robin_hood::unordered_flat_set<K, Hash>;

#else

template<typename K, typename V, typename Hash = std::hash<K>>
using fe_map = std::unordered_map<K, V, Hash>;

template<typename K, typename Hash = std::hash<K>>
using fe_set = std::unordered_set<K, Hash>;

#endif

// ============================================================================
// SplitMix64 hash  (mandatory — std::hash<uint64_t> can be identity)
// ============================================================================
struct SplitMix64Hash {
    static uint64_t mix(uint64_t x) noexcept {
        x ^= x >> 33;
        x *= 0xff51afd7ed558ccdULL;
        x ^= x >> 33;
        x *= 0xc4ceb9fe1a85ec53ULL;
        x ^= x >> 33;
        return x;
    }

    // Hash a single uint64_t
    size_t operator()(uint64_t x) const noexcept {
        return static_cast<size_t>(mix(x));
    }

    // Hash an array<uint64_t, N>
    template<size_t N>
    size_t operator()(const std::array<uint64_t, N>& a) const noexcept {
        size_t h = mix(a[0]);
        for (size_t i = 1; i < N; ++i) {
            uint64_t m = mix(a[i]);
            h ^= m + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        }
        return h;
    }
};

/// Hash a determinant (combines alpha and beta hashes).
template<typename StorageType>
struct DetHash {
    size_t operator()(const DeterminantT<StorageType>& d) const noexcept {
        SplitMix64Hash h;
        size_t h1 = h(d.alpha);
        size_t h2 = h(d.beta);
        return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
    }
};

// ============================================================================
// Excitation type enum (for spin-aware H_ij dispatch)
// ============================================================================
enum ExcType : int {
    SAME_ALPHA_S = 0,  // Δα=0, Δβ=1 (beta single)
    SAME_ALPHA_D = 1,  // Δα=0, Δβ=2 (beta double)
    SAME_BETA_S  = 2,  // Δα=1, Δβ=0 (alpha single)
    SAME_BETA_D  = 3,  // Δα=2, Δβ=0 (alpha double)
    MIXED_D      = 4,  // Δα=1, Δβ=1 (mixed double)
};

}  // namespace fe
}  // namespace trimci_core
