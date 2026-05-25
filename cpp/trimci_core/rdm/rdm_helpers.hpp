#pragma once
/**
 * Shared helper functions for RDM computation.
 *
 * Inline utilities for phase calculations and bit manipulation.
 * Used by all rdm_*.cpp files.
 */

#include "determinant.hpp"
#include "bit_compat.hpp"
#include <vector>
#include <algorithm>

namespace trimci_core {
namespace rdm_detail {

// Count bits between positions lo (exclusive) and hi (exclusive)
inline int count_bits_between(uint64_t bits, int lo, int hi) {
    if (lo >= hi) return 0;
    uint64_t mask = ((1ULL << hi) - 1) & ~((1ULL << (lo + 1)) - 1);
    return __builtin_popcountll(bits & mask);
}

// Phase for single excitation a†_a a_i on alpha string
inline int single_phase_alpha(const Determinant& det, int i, int a) {
    int lo = std::min(i, a);
    int hi = std::max(i, a);
    int count = count_bits_between(det.alpha, lo, hi);
    return (count % 2 == 0) ? 1 : -1;
}

inline int single_phase_beta(const Determinant& det, int i, int a) {
    int lo = std::min(i, a);
    int hi = std::max(i, a);
    int count = count_bits_between(det.beta, lo, hi);
    return (count % 2 == 0) ? 1 : -1;
}

// Phase for double excitation a†_{p1} a†_{p2} a_{q2} a_{q1} on alpha string
inline int double_phase_alpha(uint64_t alpha, int q1, int q2, int p1, int p2) {
    int phase = 0;
    int pos_q1 = __builtin_popcountll(alpha & ((1ULL << q1) - 1));
    phase ^= (pos_q1 & 1);
    alpha &= ~(1ULL << q1);
    int pos_q2 = __builtin_popcountll(alpha & ((1ULL << q2) - 1));
    phase ^= (pos_q2 & 1);
    alpha &= ~(1ULL << q2);
    int count_p2 = __builtin_popcountll(alpha & ((1ULL << p2) - 1));
    phase ^= (count_p2 & 1);
    alpha |= (1ULL << p2);
    int count_p1 = __builtin_popcountll(alpha & ((1ULL << p1) - 1));
    phase ^= (count_p1 & 1);
    return phase ? -1 : 1;
}

inline int double_phase_beta(uint64_t beta, int q1, int q2, int p1, int p2) {
    int phase = 0;
    int pos_q1 = __builtin_popcountll(beta & ((1ULL << q1) - 1));
    phase ^= (pos_q1 & 1);
    beta &= ~(1ULL << q1);
    int pos_q2 = __builtin_popcountll(beta & ((1ULL << q2) - 1));
    phase ^= (pos_q2 & 1);
    beta &= ~(1ULL << q2);
    int count_p2 = __builtin_popcountll(beta & ((1ULL << p2) - 1));
    phase ^= (count_p2 & 1);
    beta |= (1ULL << p2);
    int count_p1 = __builtin_popcountll(beta & ((1ULL << p1) - 1));
    phase ^= (count_p1 & 1);
    return phase ? -1 : 1;
}

// Phase for quadruple excitation a†_{p1} a†_{p2} a†_{p3} a†_{p4} a_{q4} a_{q3} a_{q2} a_{q1}
// Must modify bit string sequentially for correct phase
inline int quadruple_phase_alpha(uint64_t alpha, int q1, int q2, int q3, int q4, 
                                  int p1, int p2, int p3, int p4) {
    int phase = 0;
    // Annihilate q1, q2, q3, q4 (in sequence)
    int pos = __builtin_popcountll(alpha & ((1ULL << q1) - 1));
    phase ^= (pos & 1);
    alpha &= ~(1ULL << q1);
    pos = __builtin_popcountll(alpha & ((1ULL << q2) - 1));
    phase ^= (pos & 1);
    alpha &= ~(1ULL << q2);
    pos = __builtin_popcountll(alpha & ((1ULL << q3) - 1));
    phase ^= (pos & 1);
    alpha &= ~(1ULL << q3);
    pos = __builtin_popcountll(alpha & ((1ULL << q4) - 1));
    phase ^= (pos & 1);
    alpha &= ~(1ULL << q4);
    // Create p4, p3, p2, p1 (reverse order)
    pos = __builtin_popcountll(alpha & ((1ULL << p4) - 1));
    phase ^= (pos & 1);
    alpha |= (1ULL << p4);
    pos = __builtin_popcountll(alpha & ((1ULL << p3) - 1));
    phase ^= (pos & 1);
    alpha |= (1ULL << p3);
    pos = __builtin_popcountll(alpha & ((1ULL << p2) - 1));
    phase ^= (pos & 1);
    alpha |= (1ULL << p2);
    pos = __builtin_popcountll(alpha & ((1ULL << p1) - 1));
    phase ^= (pos & 1);
    return phase ? -1 : 1;
}

inline int quadruple_phase_beta(uint64_t beta, int q1, int q2, int q3, int q4,
                                 int p1, int p2, int p3, int p4) {
    int phase = 0;
    int pos = __builtin_popcountll(beta & ((1ULL << q1) - 1));
    phase ^= (pos & 1);
    beta &= ~(1ULL << q1);
    pos = __builtin_popcountll(beta & ((1ULL << q2) - 1));
    phase ^= (pos & 1);
    beta &= ~(1ULL << q2);
    pos = __builtin_popcountll(beta & ((1ULL << q3) - 1));
    phase ^= (pos & 1);
    beta &= ~(1ULL << q3);
    pos = __builtin_popcountll(beta & ((1ULL << q4) - 1));
    phase ^= (pos & 1);
    beta &= ~(1ULL << q4);
    pos = __builtin_popcountll(beta & ((1ULL << p4) - 1));
    phase ^= (pos & 1);
    beta |= (1ULL << p4);
    pos = __builtin_popcountll(beta & ((1ULL << p3) - 1));
    phase ^= (pos & 1);
    beta |= (1ULL << p3);
    pos = __builtin_popcountll(beta & ((1ULL << p2) - 1));
    phase ^= (pos & 1);
    beta |= (1ULL << p2);
    pos = __builtin_popcountll(beta & ((1ULL << p1) - 1));
    phase ^= (pos & 1);
    return phase ? -1 : 1;
}

// Phase for triple excitation a†_{p1} a†_{p2} a†_{p3} a_{q3} a_{q2} a_{q1}
inline int triple_phase_alpha(uint64_t alpha, int q1, int q2, int q3,
                               int p1, int p2, int p3) {
    int phase = 0;
    // Annihilate q1, q2, q3
    int pos = __builtin_popcountll(alpha & ((1ULL << q1) - 1));
    phase ^= (pos & 1);
    alpha &= ~(1ULL << q1);
    pos = __builtin_popcountll(alpha & ((1ULL << q2) - 1));
    phase ^= (pos & 1);
    alpha &= ~(1ULL << q2);
    pos = __builtin_popcountll(alpha & ((1ULL << q3) - 1));
    phase ^= (pos & 1);
    alpha &= ~(1ULL << q3);
    // Create p3, p2, p1 (reverse order)
    pos = __builtin_popcountll(alpha & ((1ULL << p3) - 1));
    phase ^= (pos & 1);
    alpha |= (1ULL << p3);
    pos = __builtin_popcountll(alpha & ((1ULL << p2) - 1));
    phase ^= (pos & 1);
    alpha |= (1ULL << p2);
    pos = __builtin_popcountll(alpha & ((1ULL << p1) - 1));
    phase ^= (pos & 1);
    return phase ? -1 : 1;
}

inline int triple_phase_beta(uint64_t beta, int q1, int q2, int q3,
                              int p1, int p2, int p3) {
    int phase = 0;
    int pos = __builtin_popcountll(beta & ((1ULL << q1) - 1));
    phase ^= (pos & 1);
    beta &= ~(1ULL << q1);
    pos = __builtin_popcountll(beta & ((1ULL << q2) - 1));
    phase ^= (pos & 1);
    beta &= ~(1ULL << q2);
    pos = __builtin_popcountll(beta & ((1ULL << q3) - 1));
    phase ^= (pos & 1);
    beta &= ~(1ULL << q3);
    pos = __builtin_popcountll(beta & ((1ULL << p3) - 1));
    phase ^= (pos & 1);
    beta |= (1ULL << p3);
    pos = __builtin_popcountll(beta & ((1ULL << p2) - 1));
    phase ^= (pos & 1);
    beta |= (1ULL << p2);
    pos = __builtin_popcountll(beta & ((1ULL << p1) - 1));
    phase ^= (pos & 1);
    return phase ? -1 : 1;
}

// Get indices of set bits
inline std::vector<int> get_set_bits(uint64_t bits) {
    std::vector<int> result;
    while (bits) {
        int idx = __builtin_ctzll(bits);
        result.push_back(idx);
        bits &= bits - 1;
    }
    return result;
}

} // namespace rdm_detail
} // namespace trimci_core
