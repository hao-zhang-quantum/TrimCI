#include "direct_s_builder.hpp"

#include <vector>
#include <utility>
#include <algorithm>
#include <cstring>
#include <cstddef>
#include <type_traits>

#include "bit_compat.hpp"

// High-performance hash map selection (matching project convention)
#ifdef USE_ABSL
#include <absl/container/flat_hash_map.h>
template<typename K, typename V, typename H>
using hash_map_t = absl::flat_hash_map<K, V, H>;
#elif defined(USE_ROBIN_HOOD)
#include <robin_hood.h>
template<typename K, typename V, typename H>
using hash_map_t = robin_hood::unordered_map<K, V, H>;
#else
#include <unordered_map>
template<typename K, typename V, typename H>
using hash_map_t = std::unordered_map<K, V, H>;
#endif

namespace trimci_core {

// ── Excitation type IDs (matching Python convention) ──
enum ExcType : int {
    EXC_S_ALPHA = 0,
    EXC_S_BETA  = 1,
    EXC_D_AA    = 2,
    EXC_D_AB    = 3,
    EXC_D_BB    = 4
};

// ── Determinant key: (alpha, beta) bitstrings ──
using Det = std::pair<uint64_t, uint64_t>;

struct DetHash {
    size_t operator()(const Det& d) const noexcept {
        // Same hash as DeterminantT<uint64_t>
        return std::hash<uint64_t>()(d.first)
             ^ (std::hash<uint64_t>()(d.second) << 1);
    }
};

// ── Phase computation ──

/**
 * Count occupied orbitals strictly between positions pos1 and pos2
 * (excluding both endpoints) in a bitstring.
 *
 * This computes the fermionic phase factor: (-1)^count when moving
 * a creation/annihilation operator past occupied orbitals.
 */
inline int count_between(uint64_t bits, int pos1, int pos2) {
    int lo = (pos1 < pos2) ? pos1 : pos2;
    int hi = (pos1 < pos2) ? pos2 : pos1;
    if (hi - lo <= 1) return 0;
    // Mask for bits in positions [lo+1, hi-1]
    uint64_t mask = ((1ULL << hi) - 1) & ~((1ULL << (lo + 1)) - 1);
    return popcount64(bits & mask);
}

// ── Excitation application with phase ──

struct ExcResult {
    uint64_t alpha;
    uint64_t beta;
    int phase;  // +1 or -1, or 0 for invalid
};

/**
 * Apply excitation operator τ̂_μ to a determinant and compute fermionic phase.
 *
 * Phase convention matches Python _apply_excitation_with_phase():
 *   - Single: count electrons between i and a (excluding i)
 *   - D_aa/D_bb: sequential application, first i→a then j→b on modified string
 *   - D_ab: independent alpha and beta counts
 */
inline ExcResult apply_excitation(uint64_t alpha, uint64_t beta,
                                  int exc_type, const int* idx) {
    ExcResult r;
    r.phase = 0;  // default: invalid

    switch (exc_type) {
    case EXC_S_ALPHA: {
        int i = idx[0], a = idx[1];
        uint64_t mi = 1ULL << i, ma = 1ULL << a;
        if (!(alpha & mi) || (alpha & ma)) return r;
        r.alpha = (alpha ^ mi) | ma;
        r.beta = beta;
        r.phase = (count_between(alpha, i, a) & 1) ? -1 : 1;
        return r;
    }
    case EXC_S_BETA: {
        int i = idx[0], a = idx[1];
        uint64_t mi = 1ULL << i, ma = 1ULL << a;
        if (!(beta & mi) || (beta & ma)) return r;
        r.alpha = alpha;
        r.beta = (beta ^ mi) | ma;
        r.phase = (count_between(beta, i, a) & 1) ? -1 : 1;
        return r;
    }
    case EXC_D_AA: {
        // τ̂^{ij→ab}_{αα} = a†_a a†_b a_j a_i (applied right-to-left)
        int i = idx[0], j = idx[1], a = idx[2], b = idx[3];
        uint64_t mi = 1ULL << i, mj = 1ULL << j;
        uint64_t ma = 1ULL << a, mb = 1ULL << b;
        if (!(alpha & mi) || !(alpha & mj)) return r;
        if ((alpha & ma) || (alpha & mb)) return r;
        r.alpha = (alpha ^ mi ^ mj) | ma | mb;
        r.beta = beta;
        // Phase: sequential a_i first (→a), then a_j (→b)
        int cnt1 = count_between(alpha, i, a);
        uint64_t modified = (alpha ^ mi) | ma;
        int cnt2 = count_between(modified, j, b);
        r.phase = ((cnt1 + cnt2) & 1) ? -1 : 1;
        return r;
    }
    case EXC_D_BB: {
        int i = idx[0], j = idx[1], a = idx[2], b = idx[3];
        uint64_t mi = 1ULL << i, mj = 1ULL << j;
        uint64_t ma = 1ULL << a, mb = 1ULL << b;
        if (!(beta & mi) || !(beta & mj)) return r;
        if ((beta & ma) || (beta & mb)) return r;
        r.alpha = alpha;
        r.beta = (beta ^ mi ^ mj) | ma | mb;
        int cnt1 = count_between(beta, i, a);
        uint64_t modified = (beta ^ mi) | ma;
        int cnt2 = count_between(modified, j, b);
        r.phase = ((cnt1 + cnt2) & 1) ? -1 : 1;
        return r;
    }
    case EXC_D_AB: {
        // τ̂^{ij→ab}_{αβ}: i,a are α; j,b are β
        int i = idx[0], j = idx[1], a = idx[2], b = idx[3];
        uint64_t mi = 1ULL << i, mj = 1ULL << j;
        uint64_t ma = 1ULL << a, mb = 1ULL << b;
        if (!(alpha & mi) || !(beta & mj)) return r;
        if ((alpha & ma) || (beta & mb)) return r;
        r.alpha = (alpha ^ mi) | ma;
        r.beta = (beta ^ mj) | mb;
        // Alpha and beta are independent spin channels
        int cnt_a = count_between(alpha, i, a);
        int cnt_b = count_between(beta, j, b);
        r.phase = ((cnt_a + cnt_b) & 1) ? -1 : 1;
        return r;
    }
    default:
        return r;
    }
}

// ── Main build function ──

void build_S_direct(
    const uint64_t* ref_alpha,
    const uint64_t* ref_beta,
    const double* ref_coeffs,
    int n_ref,
    const int* exc_types,
    const int* exc_indices,
    int n_exc,
    int n_basis,
    double* S_out)
{
    // Inverse map: det → vector of (row_index, merged_coeff)
    // Each entry is guaranteed to have unique row indices (pre-merged).
    using ContribVec = std::vector<std::pair<int, double>>;
    hash_map_t<Det, ContribVec, DetHash> inverse_map;
    inverse_map.reserve(static_cast<size_t>(n_exc) * 2);

    // Temporary map for per-excitation coefficient accumulation.
    // Reused across excitations to avoid repeated allocation.
    hash_map_t<Det, double, DetHash> tmp_map;
    tmp_map.reserve(static_cast<size_t>(n_ref) * 2);

    // ── Row 0: reference state ──
    for (int I = 0; I < n_ref; ++I) {
        Det d = {ref_alpha[I], ref_beta[I]};
        tmp_map[d] += ref_coeffs[I];
    }
    for (auto& [det, coeff] : tmp_map) {
        inverse_map[det].emplace_back(0, coeff);
    }

    // ── Rows 1..n_exc: excitations ──
    for (int mu = 0; mu < n_exc; ++mu) {
        const int* idx = &exc_indices[mu * 4];
        int etype = exc_types[mu];
        int row = mu + 1;

        // Per-excitation accumulation (small map, at most n_ref entries)
        tmp_map.clear();
        for (int I = 0; I < n_ref; ++I) {
            ExcResult res = apply_excitation(
                ref_alpha[I], ref_beta[I], etype, idx);
            if (res.phase != 0) {
                Det d = {res.alpha, res.beta};
                tmp_map[d] += ref_coeffs[I] * res.phase;
            }
        }

        // Merge into inverse map
        for (auto& [det, coeff] : tmp_map) {
            inverse_map[det].emplace_back(row, coeff);
        }
    }

    // ── Accumulate S via outer products per determinant ──
    // For each unique det d with contributions [(r_i, c_i)]:
    //   S[r_i, r_j] += c_i * c_j
    for (auto& [det, contribs] : inverse_map) {
        const int k = static_cast<int>(contribs.size());
        if (k == 1) {
            // Common case: single contribution → diagonal only
            int r = contribs[0].first;
            double c = contribs[0].second;
            S_out[r * n_basis + r] += c * c;
        } else {
            // General case: full outer product
            for (int i = 0; i < k; ++i) {
                int ri = contribs[i].first;
                double ci = contribs[i].second;
                S_out[ri * n_basis + ri] += ci * ci;
                for (int j = i + 1; j < k; ++j) {
                    int rj = contribs[j].first;
                    double cj = contribs[j].second;
                    double val = ci * cj;
                    S_out[ri * n_basis + rj] += val;
                    S_out[rj * n_basis + ri] += val;
                }
            }
        }
    }
}

} // namespace trimci_core
