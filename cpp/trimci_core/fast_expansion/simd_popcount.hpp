#pragma once
/**
 * SIMD batch popcount filter for fast connection enumeration.
 *
 * Scans sequential arrays of bit-strings (alpha/beta determinant strings),
 * XORs each with a reference, popcounts the result, and invokes callbacks
 * for entries matching target popcount values (2 = single excitation,
 * 4 = double excitation).
 *
 * Platform dispatch (compile-time, via -march=native):
 *   - AVX-512 VPOPCNTDQ (Zen 4 / Ice Lake+): 8-wide uint64_t
 *   - ARM NEON (Apple M-series): 2-wide uint64_t
 *   - Scalar fallback: all platforms
 */

#include <cstdint>
#include <cstddef>
#include <array>
#include "bit_compat.hpp"

// Platform detection
#if defined(__AVX512VPOPCNTDQ__) && defined(__AVX512F__)
  #include <immintrin.h>
  #define TRIMCI_SIMD_AVX512 1
#elif defined(__ARM_NEON)
  #include <arm_neon.h>
  #define TRIMCI_SIMD_NEON 1
#endif

namespace trimci_core {
namespace fe {
namespace simd {

// ============================================================================
// scan_popcount_sd: find single (popcount==2) and double (popcount==4)
// Used by Channel 1 (same-alpha) and Channel 3 (same-beta)
// ============================================================================
template<typename CbSingle, typename CbDouble>
inline void scan_popcount_sd(
    const uint64_t* strings,
    size_t count,
    uint64_t reference,
    CbSingle&& cb_single,
    CbDouble&& cb_double)
{
#if defined(TRIMCI_SIMD_AVX512)
    const __m512i ref_vec  = _mm512_set1_epi64(static_cast<long long>(reference));
    const __m512i two_vec  = _mm512_set1_epi64(2LL);
    const __m512i four_vec = _mm512_set1_epi64(4LL);

    size_t i = 0;
    const size_t vec_end = count & ~size_t(7);

    for (; i < vec_end; i += 8) {
        __m512i data  = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(strings + i));
        __m512i xored = _mm512_xor_si512(data, ref_vec);
        __m512i popc  = _mm512_popcnt_epi64(xored);

        __mmask8 mask_s = _mm512_cmpeq_epi64_mask(popc, two_vec);
        __mmask8 mask_d = _mm512_cmpeq_epi64_mask(popc, four_vec);

        while (mask_s) {
            int bit = __builtin_ctz(mask_s);
            cb_single(i + bit);
            mask_s &= mask_s - 1;
        }
        while (mask_d) {
            int bit = __builtin_ctz(mask_d);
            cb_double(i + bit);
            mask_d &= mask_d - 1;
        }
    }
    // Scalar tail
    for (; i < count; ++i) {
        int pc = popcount64(strings[i] ^ reference);
        if (pc == 2) cb_single(i);
        else if (pc == 4) cb_double(i);
    }

#elif defined(TRIMCI_SIMD_NEON)
    const uint64x2_t ref_vec  = vdupq_n_u64(reference);
    const uint64x2_t two_vec  = vdupq_n_u64(2ULL);
    const uint64x2_t four_vec = vdupq_n_u64(4ULL);

    size_t i = 0;
    const size_t vec_end = count & ~size_t(1);

    for (; i < vec_end; i += 2) {
        uint64x2_t data  = vld1q_u64(strings + i);
        uint64x2_t xored = veorq_u64(data, ref_vec);

        // Byte-level popcount → horizontal sum to 64-bit lanes
        uint8x16_t byte_pc = vcntq_u8(vreinterpretq_u8_u64(xored));
        uint64x2_t sum64   = vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(byte_pc)));

        uint64x2_t is_s = vceqq_u64(sum64, two_vec);
        uint64x2_t is_d = vceqq_u64(sum64, four_vec);

        if (vgetq_lane_u64(is_s, 0)) cb_single(i);
        if (vgetq_lane_u64(is_s, 1)) cb_single(i + 1);
        if (vgetq_lane_u64(is_d, 0)) cb_double(i);
        if (vgetq_lane_u64(is_d, 1)) cb_double(i + 1);
    }
    // Scalar tail
    for (; i < count; ++i) {
        int pc = popcount64(strings[i] ^ reference);
        if (pc == 2) cb_single(i);
        else if (pc == 4) cb_double(i);
    }

#else
    // Scalar fallback
    for (size_t i = 0; i < count; ++i) {
        int pc = popcount64(strings[i] ^ reference);
        if (pc == 2) cb_single(i);
        else if (pc == 4) cb_double(i);
    }
#endif
}

// ============================================================================
// scan_popcount_eq: find entries matching a single target popcount
// Used by Channel 2 (mixed doubles, target=2)
// ============================================================================
template<typename Callback>
inline void scan_popcount_eq(
    const uint64_t* strings,
    size_t count,
    uint64_t reference,
    int target,
    Callback&& cb)
{
#if defined(TRIMCI_SIMD_AVX512)
    const __m512i ref_vec    = _mm512_set1_epi64(static_cast<long long>(reference));
    const __m512i target_vec = _mm512_set1_epi64(static_cast<long long>(target));

    size_t i = 0;
    const size_t vec_end = count & ~size_t(7);

    for (; i < vec_end; i += 8) {
        __m512i data  = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(strings + i));
        __m512i xored = _mm512_xor_si512(data, ref_vec);
        __m512i popc  = _mm512_popcnt_epi64(xored);
        __mmask8 mask = _mm512_cmpeq_epi64_mask(popc, target_vec);

        while (mask) {
            int bit = __builtin_ctz(mask);
            cb(i + bit);
            mask &= mask - 1;
        }
    }
    for (; i < count; ++i) {
        if (popcount64(strings[i] ^ reference) == target)
            cb(i);
    }

#elif defined(TRIMCI_SIMD_NEON)
    const uint64x2_t ref_vec    = vdupq_n_u64(reference);
    const uint64x2_t target_vec = vdupq_n_u64(static_cast<uint64_t>(target));

    size_t i = 0;
    const size_t vec_end = count & ~size_t(1);

    for (; i < vec_end; i += 2) {
        uint64x2_t data  = vld1q_u64(strings + i);
        uint64x2_t xored = veorq_u64(data, ref_vec);

        uint8x16_t byte_pc = vcntq_u8(vreinterpretq_u8_u64(xored));
        uint64x2_t sum64   = vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(byte_pc)));

        uint64x2_t match = vceqq_u64(sum64, target_vec);
        if (vgetq_lane_u64(match, 0)) cb(i);
        if (vgetq_lane_u64(match, 1)) cb(i + 1);
    }
    for (; i < count; ++i) {
        if (popcount64(strings[i] ^ reference) == target)
            cb(i);
    }

#else
    for (size_t i = 0; i < count; ++i) {
        if (popcount64(strings[i] ^ reference) == target)
            cb(i);
    }
#endif
}

// ============================================================================
// Array StorageType overloads: scalar-only (no SIMD for multi-word strings)
// ============================================================================
template<size_t N, typename CbSingle, typename CbDouble>
inline void scan_popcount_sd(
    const std::array<uint64_t, N>* strings,
    size_t count,
    const std::array<uint64_t, N>& reference,
    CbSingle&& cb_single,
    CbDouble&& cb_double)
{
    using BitOpsType = detail::HamiltonianBitOps<std::array<uint64_t, N>>;
    for (size_t i = 0; i < count; ++i) {
        int pc = BitOpsType::count_differences(reference, strings[i]);
        if (pc == 2) cb_single(i);
        else if (pc == 4) cb_double(i);
    }
}

template<size_t N, typename Callback>
inline void scan_popcount_eq(
    const std::array<uint64_t, N>* strings,
    size_t count,
    const std::array<uint64_t, N>& reference,
    int target,
    Callback&& cb)
{
    using BitOpsType = detail::HamiltonianBitOps<std::array<uint64_t, N>>;
    for (size_t i = 0; i < count; ++i) {
        if (BitOpsType::count_differences(reference, strings[i]) == target)
            cb(i);
    }
}

}  // namespace simd
}  // namespace fe
}  // namespace trimci_core
