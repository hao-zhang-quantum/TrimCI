#include "direct_h_detspace.hpp"

#include <vector>
#include <utility>
#include <algorithm>
#include <cstring>
#include <cstddef>
#include <chrono>
#include <cstdio>
#include <cstdlib>

#include "bit_compat.hpp"

#include "omp_compat.hpp"
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
        return std::hash<uint64_t>()(d.first)
             ^ (std::hash<uint64_t>()(d.second) << 1);
    }
};

// ── Phase computation ──

inline int count_between(uint64_t bits, int pos1, int pos2) {
    int lo = (pos1 < pos2) ? pos1 : pos2;
    int hi = (pos1 < pos2) ? pos2 : pos1;
    if (hi - lo <= 1) return 0;
    uint64_t mask = ((1ULL << hi) - 1) & ~((1ULL << (lo + 1)) - 1);
    return popcount64(bits & mask);
}

// ── Excitation application with phase (same as direct_h_builder.cpp) ──

struct ExcResult {
    uint64_t alpha;
    uint64_t beta;
    int phase;  // +1 or -1, or 0 for invalid
};

inline ExcResult apply_excitation(uint64_t alpha, uint64_t beta,
                                  int exc_type, const int* idx) {
    ExcResult r;
    r.phase = 0;

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
        int i = idx[0], j = idx[1], a = idx[2], b = idx[3];
        uint64_t mi = 1ULL << i, mj = 1ULL << j;
        uint64_t ma = 1ULL << a, mb = 1ULL << b;
        if (!(alpha & mi) || !(alpha & mj)) return r;
        if ((alpha & ma) || (alpha & mb)) return r;
        r.alpha = (alpha ^ mi ^ mj) | ma | mb;
        r.beta = beta;
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
        int i = idx[0], j = idx[1], a = idx[2], b = idx[3];
        uint64_t mi = 1ULL << i, mj = 1ULL << j;
        uint64_t ma = 1ULL << a, mb = 1ULL << b;
        if (!(alpha & mi) || !(beta & mj)) return r;
        if ((alpha & ma) || (beta & mb)) return r;
        r.alpha = (alpha ^ mi) | ma;
        r.beta = (beta ^ mj) | mb;
        int cnt_a = count_between(alpha, i, a);
        int cnt_b = count_between(beta, j, b);
        r.phase = ((cnt_a + cnt_b) & 1) ? -1 : 1;
        return r;
    }
    default:
        return r;
    }
}

// ── ERI / h1 access helpers ──

inline double get_eri(const double* eri, int n, int i, int j, int k, int l) {
    size_t idx = (((static_cast<size_t>(i) * n + j) * n + k) * n + l);
    return eri[idx];
}

inline double get_h1(const double* h1, int n, int p, int q) {
    return h1[static_cast<size_t>(p) * n + q];
}

// ── Extract occupied orbital indices from bitstring ──

inline void get_occupied(uint64_t bits, std::vector<int>& occ) {
    occ.clear();
    uint64_t temp = bits;
    while (temp) {
        int p = ctz64(temp);
        occ.push_back(p);
        temp &= temp - 1;
    }
}

// ── Direct SC element computation (when we know the excitation type) ──

/**
 * Diagonal energy ⟨d|H|d⟩.
 */
inline double compute_diagonal(uint64_t alpha, uint64_t beta,
                                const double* h1, const double* eri, int n_orb) {
    thread_local std::vector<int> occ_a, occ_b;
    get_occupied(alpha, occ_a);
    get_occupied(beta, occ_b);

    double E = 0.0;

    // One-electron
    for (int i : occ_a) E += get_h1(h1, n_orb, i, i);
    for (int i : occ_b) E += get_h1(h1, n_orb, i, i);

    // Two-electron α-α
    for (size_t ii = 0; ii < occ_a.size(); ++ii) {
        int i = occ_a[ii];
        for (size_t jj = ii + 1; jj < occ_a.size(); ++jj) {
            int j = occ_a[jj];
            E += get_eri(eri, n_orb, i, i, j, j) - get_eri(eri, n_orb, i, j, j, i);
        }
    }

    // Two-electron β-β
    for (size_t ii = 0; ii < occ_b.size(); ++ii) {
        int i = occ_b[ii];
        for (size_t jj = ii + 1; jj < occ_b.size(); ++jj) {
            int j = occ_b[jj];
            E += get_eri(eri, n_orb, i, i, j, j) - get_eri(eri, n_orb, i, j, j, i);
        }
    }

    // Two-electron α-β (Coulomb only)
    for (int i : occ_a) {
        for (int j : occ_b) {
            E += get_eri(eri, n_orb, i, i, j, j);
        }
    }

    return E;
}

/**
 * Single α excitation element: ⟨d'|H|d⟩ where d' = d with α orbital i→a.
 * Returns phase * [h_{ai} + Σ_k J/K terms].
 * Phase is computed from the ket determinant d.
 */
inline double compute_single_alpha(uint64_t alpha, uint64_t beta,
                                    int i, int a,
                                    const double* h1, const double* eri, int n_orb) {
    int phase = (count_between(alpha, i, a) & 1) ? -1 : 1;

    double elem = get_h1(h1, n_orb, a, i);

    // J-K with other α electrons (excluding i)
    uint64_t temp_a = alpha;
    while (temp_a) {
        int k = ctz64(temp_a);
        temp_a &= temp_a - 1;
        if (k != i) {
            elem += get_eri(eri, n_orb, a, i, k, k) - get_eri(eri, n_orb, a, k, k, i);
        }
    }

    // J with β electrons (Coulomb only)
    uint64_t temp_b = beta;
    while (temp_b) {
        int k = ctz64(temp_b);
        temp_b &= temp_b - 1;
        elem += get_eri(eri, n_orb, a, i, k, k);
    }

    return phase * elem;
}

/**
 * Single β excitation element: ⟨d'|H|d⟩ where d' = d with β orbital i→a.
 */
inline double compute_single_beta(uint64_t alpha, uint64_t beta,
                                   int i, int a,
                                   const double* h1, const double* eri, int n_orb) {
    int phase = (count_between(beta, i, a) & 1) ? -1 : 1;

    double elem = get_h1(h1, n_orb, a, i);

    // J-K with other β electrons (excluding i)
    uint64_t temp_b = beta;
    while (temp_b) {
        int k = ctz64(temp_b);
        temp_b &= temp_b - 1;
        if (k != i) {
            elem += get_eri(eri, n_orb, a, i, k, k) - get_eri(eri, n_orb, a, k, k, i);
        }
    }

    // J with α electrons (Coulomb only)
    uint64_t temp_a = alpha;
    while (temp_a) {
        int k = ctz64(temp_a);
        temp_a &= temp_a - 1;
        elem += get_eri(eri, n_orb, a, i, k, k);
    }

    return phase * elem;
}

/**
 * Double αα excitation: ⟨d'|H|d⟩ where d' = d with α (i,j)→(a,b).
 * i < j, a < b (canonical ordering).
 */
inline double compute_double_aa(uint64_t alpha,
                                 int i, int j, int a, int b,
                                 const double* eri, int n_orb) {
    // Phase: sequential i→a then j→b
    int cnt1 = count_between(alpha, i, a);
    uint64_t modified = (alpha & ~(1ULL << i)) | (1ULL << a);
    int cnt2 = count_between(modified, j, b);
    int phase = ((cnt1 + cnt2) & 1) ? -1 : 1;

    double elem = get_eri(eri, n_orb, a, i, b, j) - get_eri(eri, n_orb, a, j, b, i);
    return phase * elem;
}

/**
 * Double ββ excitation: ⟨d'|H|d⟩ where d' = d with β (i,j)→(a,b).
 */
inline double compute_double_bb(uint64_t beta,
                                 int i, int j, int a, int b,
                                 const double* eri, int n_orb) {
    int cnt1 = count_between(beta, i, a);
    uint64_t modified = (beta & ~(1ULL << i)) | (1ULL << a);
    int cnt2 = count_between(modified, j, b);
    int phase = ((cnt1 + cnt2) & 1) ? -1 : 1;

    double elem = get_eri(eri, n_orb, a, i, b, j) - get_eri(eri, n_orb, a, j, b, i);
    return phase * elem;
}

/**
 * Double αβ excitation: ⟨d'|H|d⟩ where d' = d with α i→a, β j→b.
 */
inline double compute_double_ab(uint64_t alpha, uint64_t beta,
                                 int i_a, int a_a, int i_b, int a_b,
                                 const double* eri, int n_orb) {
    int cnt_a = count_between(alpha, i_a, a_a);
    int cnt_b = count_between(beta, i_b, a_b);
    int phase = ((cnt_a + cnt_b) & 1) ? -1 : 1;

    double elem = get_eri(eri, n_orb, a_a, i_a, a_b, i_b);
    return phase * elem;
}

// ── Outer-product accumulation helpers ──

// Thread-local version (no synchronization, fast)
inline void accumulate_outer_product(
    double h_elem,
    const std::vector<std::pair<int, double>>& contribs_d,
    const std::vector<std::pair<int, double>>& contribs_dp,
    int n_basis, double* H_out)
{
    for (const auto& [ri, ci] : contribs_d) {
        for (const auto& [rj, cj] : contribs_dp) {
            double val = ci * h_elem * cj;
            H_out[ri * n_basis + rj] += val;
            H_out[rj * n_basis + ri] += val;
        }
    }
}

inline void accumulate_outer_product_diag(
    double h_elem,
    const std::vector<std::pair<int, double>>& contribs,
    int n_basis, double* H_out)
{
    const int k = static_cast<int>(contribs.size());
    for (int i = 0; i < k; ++i) {
        int ri = contribs[i].first;
        double ci = contribs[i].second;
        H_out[ri * n_basis + ri] += ci * h_elem * ci;
        for (int j = i + 1; j < k; ++j) {
            int rj = contribs[j].first;
            double cj = contribs[j].second;
            double val = ci * h_elem * cj;
            H_out[ri * n_basis + rj] += val;
            H_out[rj * n_basis + ri] += val;
        }
    }
}

// Atomic version (for shared H matrix, avoids thread-local copies)
inline void accumulate_outer_product_atomic(
    double h_elem,
    const std::vector<std::pair<int, double>>& contribs_d,
    const std::vector<std::pair<int, double>>& contribs_dp,
    int n_basis, double* H_out)
{
    for (const auto& [ri, ci] : contribs_d) {
        for (const auto& [rj, cj] : contribs_dp) {
            double val = ci * h_elem * cj;
            #pragma omp atomic
            H_out[ri * n_basis + rj] += val;
            #pragma omp atomic
            H_out[rj * n_basis + ri] += val;
        }
    }
}

inline void accumulate_outer_product_diag_atomic(
    double h_elem,
    const std::vector<std::pair<int, double>>& contribs,
    int n_basis, double* H_out)
{
    const int k = static_cast<int>(contribs.size());
    for (int i = 0; i < k; ++i) {
        int ri = contribs[i].first;
        double ci = contribs[i].second;
        #pragma omp atomic
        H_out[ri * n_basis + ri] += ci * h_elem * ci;
        for (int j = i + 1; j < k; ++j) {
            int rj = contribs[j].first;
            double cj = contribs[j].second;
            double val = ci * h_elem * cj;
            #pragma omp atomic
            H_out[ri * n_basis + rj] += val;
            #pragma omp atomic
            H_out[rj * n_basis + ri] += val;
        }
    }
}

// ── Main build function ──

void build_H_detspace(
    const uint64_t* ref_alpha,
    const uint64_t* ref_beta,
    const double* ref_coeffs,
    int n_ref,
    const int* exc_types,
    const int* exc_indices,
    int n_exc,
    int n_basis,
    const double* h1,
    const double* eri,
    int n_orb,
    double* H_out)
{
    using clock_t = std::chrono::high_resolution_clock;
    bool timing = (std::getenv("TRIMCI_TIMING") != nullptr);
    auto t_start = clock_t::now();

    // ════════════════════════════════════════════════════════════════════
    // Step 1: Build inverse map  det → [(row, coeff)]
    //         Same algorithm as direct_s_builder.cpp
    // ════════════════════════════════════════════════════════════════════

    using ContribVec = std::vector<std::pair<int, double>>;
    hash_map_t<Det, ContribVec, DetHash> inverse_map;
    inverse_map.reserve(static_cast<size_t>(n_exc) * 2);

    // Temp map for per-row coefficient accumulation (reused)
    hash_map_t<Det, double, DetHash> tmp_map;
    tmp_map.reserve(static_cast<size_t>(n_ref) * 2);

    // Row 0: reference state
    for (int I = 0; I < n_ref; ++I) {
        Det d = {ref_alpha[I], ref_beta[I]};
        tmp_map[d] += ref_coeffs[I];
    }
    for (auto& [det, coeff] : tmp_map) {
        inverse_map[det].emplace_back(0, coeff);
    }

    // Rows 1..n_exc: excitations
    for (int mu = 0; mu < n_exc; ++mu) {
        const int* idx = &exc_indices[mu * 4];
        int etype = exc_types[mu];
        int row = mu + 1;

        tmp_map.clear();
        for (int I = 0; I < n_ref; ++I) {
            ExcResult res = apply_excitation(
                ref_alpha[I], ref_beta[I], etype, idx);
            if (res.phase != 0) {
                Det d = {res.alpha, res.beta};
                tmp_map[d] += ref_coeffs[I] * res.phase;
            }
        }

        for (auto& [det, coeff] : tmp_map) {
            inverse_map[det].emplace_back(row, coeff);
        }
    }

    auto t_invmap = clock_t::now();

    // ════════════════════════════════════════════════════════════════════
    // Step 2: Collect all unique dets into a vector for iteration
    // ════════════════════════════════════════════════════════════════════

    struct DetEntry {
        uint64_t alpha;
        uint64_t beta;
        const ContribVec* contribs;
    };

    std::vector<DetEntry> all_dets;
    all_dets.reserve(inverse_map.size());
    for (auto& [det, contribs] : inverse_map) {
        all_dets.push_back({det.first, det.second, &contribs});
    }

    auto t_collect = clock_t::now();

    // ════════════════════════════════════════════════════════════════════
    // Step 3: For each det, compute diagonal + enumerate connected dets
    // ════════════════════════════════════════════════════════════════════

    const size_t n_dets = all_dets.size();

    // Determine number of threads
    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif

    // Memory-aware strategy: use thread-local H if affordable, else atomic
    const size_t H_size = static_cast<size_t>(n_basis) * n_basis;
    const size_t bytes_per_copy = H_size * sizeof(double);
    const size_t total_thread_local_bytes = static_cast<size_t>(n_threads) * bytes_per_copy;

    // Cap thread-local memory at 64 GB; above that, use atomic writes
    const size_t MAX_THREAD_LOCAL_BYTES = 64ULL * 1024 * 1024 * 1024;
    const bool use_atomic = (total_thread_local_bytes > MAX_THREAD_LOCAL_BYTES);

    if (use_atomic) {
        fprintf(stderr, "[detspace] n_basis=%d, thread-local would need %.1f GB "
                "(%d threads × %.1f GB); using atomic mode\n",
                n_basis,
                total_thread_local_bytes / 1e9, n_threads,
                bytes_per_copy / 1e9);
    } else {
        fprintf(stderr, "[detspace] n_basis=%d, using %d thread-local copies "
                "(%.1f GB total)\n",
                n_basis, n_threads, total_thread_local_bytes / 1e9);
    }

    if (use_atomic) {
        // ── Atomic mode: all threads write directly to H_out ──
#ifdef _OPENMP
        #pragma omp parallel
#endif
        {
            std::vector<int> occ_a, occ_b;

#ifdef _OPENMP
            #pragma omp for schedule(dynamic, 64)
#endif
            for (int d_idx = 0; d_idx < static_cast<int>(n_dets); ++d_idx) {
                const auto& entry = all_dets[d_idx];
                uint64_t alpha_d = entry.alpha;
                uint64_t beta_d = entry.beta;
                const ContribVec& contribs_d = *entry.contribs;

                get_occupied(alpha_d, occ_a);
                get_occupied(beta_d, occ_b);

                double h_dd = compute_diagonal(alpha_d, beta_d, h1, eri, n_orb);
                accumulate_outer_product_diag_atomic(h_dd, contribs_d, n_basis, H_out);

                for (int i : occ_a) {
                    for (int a = 0; a < n_orb; ++a) {
                        if (alpha_d & (1ULL << a)) continue;
                        uint64_t alpha_dp = (alpha_d & ~(1ULL << i)) | (1ULL << a);
                        Det dp = {alpha_dp, beta_d};
                        if (dp <= Det{alpha_d, beta_d}) continue;
                        auto it = inverse_map.find(dp);
                        if (it == inverse_map.end()) continue;
                        double h_elem = compute_single_alpha(alpha_d, beta_d, i, a, h1, eri, n_orb);
                        accumulate_outer_product_atomic(h_elem, contribs_d, it->second, n_basis, H_out);
                    }
                }

                for (int i : occ_b) {
                    for (int a = 0; a < n_orb; ++a) {
                        if (beta_d & (1ULL << a)) continue;
                        uint64_t beta_dp = (beta_d & ~(1ULL << i)) | (1ULL << a);
                        Det dp = {alpha_d, beta_dp};
                        if (dp <= Det{alpha_d, beta_d}) continue;
                        auto it = inverse_map.find(dp);
                        if (it == inverse_map.end()) continue;
                        double h_elem = compute_single_beta(alpha_d, beta_d, i, a, h1, eri, n_orb);
                        accumulate_outer_product_atomic(h_elem, contribs_d, it->second, n_basis, H_out);
                    }
                }

                for (size_t ii = 0; ii < occ_a.size(); ++ii) {
                    int i = occ_a[ii];
                    for (size_t jj = ii + 1; jj < occ_a.size(); ++jj) {
                        int j = occ_a[jj];
                        for (int a = 0; a < n_orb; ++a) {
                            if (alpha_d & (1ULL << a)) continue;
                            for (int b = a + 1; b < n_orb; ++b) {
                                if (alpha_d & (1ULL << b)) continue;
                                uint64_t alpha_dp = (alpha_d & ~(1ULL << i) & ~(1ULL << j))
                                                  | (1ULL << a) | (1ULL << b);
                                Det dp = {alpha_dp, beta_d};
                                if (dp <= Det{alpha_d, beta_d}) continue;
                                auto it = inverse_map.find(dp);
                                if (it == inverse_map.end()) continue;
                                double h_elem = compute_double_aa(alpha_d, i, j, a, b, eri, n_orb);
                                accumulate_outer_product_atomic(h_elem, contribs_d, it->second, n_basis, H_out);
                            }
                        }
                    }
                }

                for (size_t ii = 0; ii < occ_b.size(); ++ii) {
                    int i = occ_b[ii];
                    for (size_t jj = ii + 1; jj < occ_b.size(); ++jj) {
                        int j = occ_b[jj];
                        for (int a = 0; a < n_orb; ++a) {
                            if (beta_d & (1ULL << a)) continue;
                            for (int b = a + 1; b < n_orb; ++b) {
                                if (beta_d & (1ULL << b)) continue;
                                uint64_t beta_dp = (beta_d & ~(1ULL << i) & ~(1ULL << j))
                                                 | (1ULL << a) | (1ULL << b);
                                Det dp = {alpha_d, beta_dp};
                                if (dp <= Det{alpha_d, beta_d}) continue;
                                auto it = inverse_map.find(dp);
                                if (it == inverse_map.end()) continue;
                                double h_elem = compute_double_bb(beta_d, i, j, a, b, eri, n_orb);
                                accumulate_outer_product_atomic(h_elem, contribs_d, it->second, n_basis, H_out);
                            }
                        }
                    }
                }

                for (int i_a : occ_a) {
                    for (int a_a = 0; a_a < n_orb; ++a_a) {
                        if (alpha_d & (1ULL << a_a)) continue;
                        uint64_t alpha_dp = (alpha_d & ~(1ULL << i_a)) | (1ULL << a_a);
                        for (int i_b : occ_b) {
                            for (int a_b = 0; a_b < n_orb; ++a_b) {
                                if (beta_d & (1ULL << a_b)) continue;
                                uint64_t beta_dp = (beta_d & ~(1ULL << i_b)) | (1ULL << a_b);
                                Det dp = {alpha_dp, beta_dp};
                                if (dp <= Det{alpha_d, beta_d}) continue;
                                auto it = inverse_map.find(dp);
                                if (it == inverse_map.end()) continue;
                                double h_elem = compute_double_ab(alpha_d, beta_d,
                                                                   i_a, a_a, i_b, a_b,
                                                                   eri, n_orb);
                                accumulate_outer_product_atomic(h_elem, contribs_d, it->second, n_basis, H_out);
                            }
                        }
                    }
                }
            }
        }

    } else {
        // ── Thread-local mode: each thread accumulates into private copy ──
        std::vector<std::vector<double>> thread_H(n_threads, std::vector<double>(H_size, 0.0));

#ifdef _OPENMP
        #pragma omp parallel
#endif
        {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            double* local_H = thread_H[tid].data();

            std::vector<int> occ_a, occ_b;

#ifdef _OPENMP
            #pragma omp for schedule(dynamic, 64)
#endif
            for (int d_idx = 0; d_idx < static_cast<int>(n_dets); ++d_idx) {
                const auto& entry = all_dets[d_idx];
                uint64_t alpha_d = entry.alpha;
                uint64_t beta_d = entry.beta;
                const ContribVec& contribs_d = *entry.contribs;

                get_occupied(alpha_d, occ_a);
                get_occupied(beta_d, occ_b);

                double h_dd = compute_diagonal(alpha_d, beta_d, h1, eri, n_orb);
                accumulate_outer_product_diag(h_dd, contribs_d, n_basis, local_H);

                for (int i : occ_a) {
                    for (int a = 0; a < n_orb; ++a) {
                        if (alpha_d & (1ULL << a)) continue;
                        uint64_t alpha_dp = (alpha_d & ~(1ULL << i)) | (1ULL << a);
                        Det dp = {alpha_dp, beta_d};
                        if (dp <= Det{alpha_d, beta_d}) continue;
                        auto it = inverse_map.find(dp);
                        if (it == inverse_map.end()) continue;
                        double h_elem = compute_single_alpha(alpha_d, beta_d, i, a, h1, eri, n_orb);
                        accumulate_outer_product(h_elem, contribs_d, it->second, n_basis, local_H);
                    }
                }

                for (int i : occ_b) {
                    for (int a = 0; a < n_orb; ++a) {
                        if (beta_d & (1ULL << a)) continue;
                        uint64_t beta_dp = (beta_d & ~(1ULL << i)) | (1ULL << a);
                        Det dp = {alpha_d, beta_dp};
                        if (dp <= Det{alpha_d, beta_d}) continue;
                        auto it = inverse_map.find(dp);
                        if (it == inverse_map.end()) continue;
                        double h_elem = compute_single_beta(alpha_d, beta_d, i, a, h1, eri, n_orb);
                        accumulate_outer_product(h_elem, contribs_d, it->second, n_basis, local_H);
                    }
                }

                for (size_t ii = 0; ii < occ_a.size(); ++ii) {
                    int i = occ_a[ii];
                    for (size_t jj = ii + 1; jj < occ_a.size(); ++jj) {
                        int j = occ_a[jj];
                        for (int a = 0; a < n_orb; ++a) {
                            if (alpha_d & (1ULL << a)) continue;
                            for (int b = a + 1; b < n_orb; ++b) {
                                if (alpha_d & (1ULL << b)) continue;
                                uint64_t alpha_dp = (alpha_d & ~(1ULL << i) & ~(1ULL << j))
                                                  | (1ULL << a) | (1ULL << b);
                                Det dp = {alpha_dp, beta_d};
                                if (dp <= Det{alpha_d, beta_d}) continue;
                                auto it = inverse_map.find(dp);
                                if (it == inverse_map.end()) continue;
                                double h_elem = compute_double_aa(alpha_d, i, j, a, b, eri, n_orb);
                                accumulate_outer_product(h_elem, contribs_d, it->second, n_basis, local_H);
                            }
                        }
                    }
                }

                for (size_t ii = 0; ii < occ_b.size(); ++ii) {
                    int i = occ_b[ii];
                    for (size_t jj = ii + 1; jj < occ_b.size(); ++jj) {
                        int j = occ_b[jj];
                        for (int a = 0; a < n_orb; ++a) {
                            if (beta_d & (1ULL << a)) continue;
                            for (int b = a + 1; b < n_orb; ++b) {
                                if (beta_d & (1ULL << b)) continue;
                                uint64_t beta_dp = (beta_d & ~(1ULL << i) & ~(1ULL << j))
                                                 | (1ULL << a) | (1ULL << b);
                                Det dp = {alpha_d, beta_dp};
                                if (dp <= Det{alpha_d, beta_d}) continue;
                                auto it = inverse_map.find(dp);
                                if (it == inverse_map.end()) continue;
                                double h_elem = compute_double_bb(beta_d, i, j, a, b, eri, n_orb);
                                accumulate_outer_product(h_elem, contribs_d, it->second, n_basis, local_H);
                            }
                        }
                    }
                }

                for (int i_a : occ_a) {
                    for (int a_a = 0; a_a < n_orb; ++a_a) {
                        if (alpha_d & (1ULL << a_a)) continue;
                        uint64_t alpha_dp = (alpha_d & ~(1ULL << i_a)) | (1ULL << a_a);
                        for (int i_b : occ_b) {
                            for (int a_b = 0; a_b < n_orb; ++a_b) {
                                if (beta_d & (1ULL << a_b)) continue;
                                uint64_t beta_dp = (beta_d & ~(1ULL << i_b)) | (1ULL << a_b);
                                Det dp = {alpha_dp, beta_dp};
                                if (dp <= Det{alpha_d, beta_d}) continue;
                                auto it = inverse_map.find(dp);
                                if (it == inverse_map.end()) continue;
                                double h_elem = compute_double_ab(alpha_d, beta_d,
                                                                   i_a, a_a, i_b, a_b,
                                                                   eri, n_orb);
                                accumulate_outer_product(h_elem, contribs_d, it->second, n_basis, local_H);
                            }
                        }
                    }
                }
            }
        }

        // Reduce thread-local H matrices into H_out
        for (int t = 0; t < n_threads; ++t) {
            const double* src = thread_H[t].data();
            for (size_t i = 0; i < H_size; ++i) {
                H_out[i] += src[i];
            }
        }
    } // end thread-local mode

    auto t_end = clock_t::now();

    if (timing) {
        auto ms = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        fprintf(stderr, "[detspace timing] n_dets=%zu, n_exc=%d, n_ref=%d, n_orb=%d\n",
                n_dets, n_exc, n_ref, n_orb);
        fprintf(stderr, "  Step 1 (inverse map): %8.2f ms\n", ms(t_start, t_invmap));
        fprintf(stderr, "  Step 2 (collect):     %8.2f ms\n", ms(t_invmap, t_collect));
        fprintf(stderr, "  Step 3+4 (SC+accum):  %8.2f ms  [%s]\n",
                ms(t_collect, t_end), use_atomic ? "atomic" : "thread-local");
        fprintf(stderr, "  Total:                %8.2f ms\n", ms(t_start, t_end));
    }
}

} // namespace trimci_core
