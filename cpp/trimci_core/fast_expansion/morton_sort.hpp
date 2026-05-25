#pragma once
/**
 * Fast Expansion: Morton (Z-order) determinant sorting.
 *
 * Determinants live in 2D space (alpha_id, beta_id) but must be stored in 1D.
 * Alpha-first lexicographic order gives perfect locality for same-alpha
 * connections but O(N) scatter for same-beta connections.
 *
 * Morton (Z-order) interleaving maps 2D proximity to 1D proximity for BOTH
 * dimensions simultaneously, reducing same-beta access distance from O(N)
 * to O(sqrt(N)).  This is the key to unlocking memory bandwidth for matvec.
 *
 * Cost: O(N log N) sort per expansion round, negligible vs ABIndex build.
 */

#include <vector>
#include <cstdint>
#include "determinant.hpp"
#include "fe_types.hpp"

namespace trimci_core {
namespace fe {

/// Portable Morton interleave: interleaves bits of a and b.
/// Result: a0 b0 a1 b1 a2 b2 ... (a in even bits, b in odd bits).
/// Works on both ARM (Apple M-series) and x86.
inline uint64_t morton_interleave(uint32_t a, uint32_t b) {
    auto spread = [](uint32_t x) -> uint64_t {
        uint64_t v = x;
        v = (v | (v << 16)) & 0x0000FFFF0000FFFFULL;
        v = (v | (v <<  8)) & 0x00FF00FF00FF00FFULL;
        v = (v | (v <<  4)) & 0x0F0F0F0F0F0F0F0FULL;
        v = (v | (v <<  2)) & 0x3333333333333333ULL;
        v = (v | (v <<  1)) & 0x5555555555555555ULL;
        return v;
    };
    return spread(a) | (spread(b) << 1);
}

/// Sort determinants by Morton (Z-order) key derived from (alpha_id, beta_id).
/// Reorders dets in-place.  All existing matvec functions automatically benefit
/// because v[] indices follow the new ordering.
template<typename StorageType>
void morton_sort_dets(std::vector<DeterminantT<StorageType>>& dets, int n_orb);

}  // namespace fe
}  // namespace trimci_core
