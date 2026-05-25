#include "matfree_davidson.hpp"

#include <vector>
#include <utility>
#include <algorithm>
#include <cstring>
#include <cstddef>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <numeric>

#include "bit_compat.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

// High-performance hash map (same as detspace)
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
namespace {

// ════════════════════════════════════════════════════════════════════
// Shared helpers (same as direct_h_detspace.cpp)
// ════════════════════════════════════════════════════════════════════

enum ExcType : int {
    EXC_S_ALPHA = 0, EXC_S_BETA = 1,
    EXC_D_AA = 2, EXC_D_AB = 3, EXC_D_BB = 4
};

using Det = std::pair<uint64_t, uint64_t>;

struct DetHash {
    size_t operator()(const Det& d) const noexcept {
        return std::hash<uint64_t>()(d.first)
             ^ (std::hash<uint64_t>()(d.second) << 1);
    }
};

inline int count_between(uint64_t bits, int pos1, int pos2) {
    int lo = (pos1 < pos2) ? pos1 : pos2;
    int hi = (pos1 < pos2) ? pos2 : pos1;
    if (hi - lo <= 1) return 0;
    uint64_t mask = ((1ULL << hi) - 1) & ~((1ULL << (lo + 1)) - 1);
    return popcount64(bits & mask);
}

struct ExcResult {
    uint64_t alpha, beta;
    int phase;
};

inline ExcResult apply_excitation(uint64_t alpha, uint64_t beta,
                                   int exc_type, const int* idx) {
    ExcResult r; r.phase = 0;
    switch (exc_type) {
    case EXC_S_ALPHA: {
        int i = idx[0], a = idx[1];
        uint64_t mi = 1ULL << i, ma = 1ULL << a;
        if (!(alpha & mi) || (alpha & ma)) return r;
        r.alpha = (alpha ^ mi) | ma; r.beta = beta;
        r.phase = (count_between(alpha, i, a) & 1) ? -1 : 1;
        return r;
    }
    case EXC_S_BETA: {
        int i = idx[0], a = idx[1];
        uint64_t mi = 1ULL << i, ma = 1ULL << a;
        if (!(beta & mi) || (beta & ma)) return r;
        r.alpha = alpha; r.beta = (beta ^ mi) | ma;
        r.phase = (count_between(beta, i, a) & 1) ? -1 : 1;
        return r;
    }
    case EXC_D_AA: {
        int i = idx[0], j = idx[1], a = idx[2], b = idx[3];
        uint64_t mi = 1ULL << i, mj = 1ULL << j, ma = 1ULL << a, mb = 1ULL << b;
        if (!(alpha & mi) || !(alpha & mj) || (alpha & ma) || (alpha & mb)) return r;
        r.alpha = (alpha ^ mi ^ mj) | ma | mb; r.beta = beta;
        int c1 = count_between(alpha, i, a);
        int c2 = count_between((alpha ^ mi) | ma, j, b);
        r.phase = ((c1 + c2) & 1) ? -1 : 1;
        return r;
    }
    case EXC_D_BB: {
        int i = idx[0], j = idx[1], a = idx[2], b = idx[3];
        uint64_t mi = 1ULL << i, mj = 1ULL << j, ma = 1ULL << a, mb = 1ULL << b;
        if (!(beta & mi) || !(beta & mj) || (beta & ma) || (beta & mb)) return r;
        r.alpha = alpha; r.beta = (beta ^ mi ^ mj) | ma | mb;
        int c1 = count_between(beta, i, a);
        int c2 = count_between((beta ^ mi) | ma, j, b);
        r.phase = ((c1 + c2) & 1) ? -1 : 1;
        return r;
    }
    case EXC_D_AB: {
        int i = idx[0], j = idx[1], a = idx[2], b = idx[3];
        if (!(alpha & (1ULL << i)) || !(beta & (1ULL << j)) ||
            (alpha & (1ULL << a)) || (beta & (1ULL << b))) return r;
        r.alpha = (alpha ^ (1ULL << i)) | (1ULL << a);
        r.beta = (beta ^ (1ULL << j)) | (1ULL << b);
        r.phase = ((count_between(alpha, i, a) + count_between(beta, j, b)) & 1) ? -1 : 1;
        return r;
    }
    default: return r;
    }
}

inline double get_eri(const double* eri, int n, int i, int j, int k, int l) {
    return eri[(((size_t)i * n + j) * n + k) * n + l];
}
inline double get_h1(const double* h1, int n, int p, int q) {
    return h1[(size_t)p * n + q];
}

inline void get_occupied(uint64_t bits, std::vector<int>& occ) {
    occ.clear();
    uint64_t t = bits;
    while (t) { occ.push_back(ctz64(t)); t &= t - 1; }
}

// ── Slater-Condon matrix element helpers ──

inline double compute_diagonal(uint64_t alpha, uint64_t beta,
                                const double* h1, const double* eri, int n_orb) {
    thread_local std::vector<int> oa, ob;
    get_occupied(alpha, oa); get_occupied(beta, ob);
    double E = 0.0;
    for (int i : oa) E += get_h1(h1, n_orb, i, i);
    for (int i : ob) E += get_h1(h1, n_orb, i, i);
    for (size_t ii = 0; ii < oa.size(); ++ii)
        for (size_t jj = ii+1; jj < oa.size(); ++jj)
            E += get_eri(eri, n_orb, oa[ii], oa[ii], oa[jj], oa[jj])
               - get_eri(eri, n_orb, oa[ii], oa[jj], oa[jj], oa[ii]);
    for (size_t ii = 0; ii < ob.size(); ++ii)
        for (size_t jj = ii+1; jj < ob.size(); ++jj)
            E += get_eri(eri, n_orb, ob[ii], ob[ii], ob[jj], ob[jj])
               - get_eri(eri, n_orb, ob[ii], ob[jj], ob[jj], ob[ii]);
    for (int i : oa) for (int j : ob)
        E += get_eri(eri, n_orb, i, i, j, j);
    return E;
}

inline double compute_single_alpha(uint64_t alpha, uint64_t beta,
                                    int i, int a, const double* h1,
                                    const double* eri, int n_orb) {
    int phase = (count_between(alpha, i, a) & 1) ? -1 : 1;
    double elem = get_h1(h1, n_orb, a, i);
    uint64_t t = alpha;
    while (t) { int k = ctz64(t); t &= t-1;
        if (k != i) elem += get_eri(eri,n_orb,a,i,k,k) - get_eri(eri,n_orb,a,k,k,i); }
    t = beta;
    while (t) { int k = ctz64(t); t &= t-1;
        elem += get_eri(eri,n_orb,a,i,k,k); }
    return phase * elem;
}

inline double compute_single_beta(uint64_t alpha, uint64_t beta,
                                   int i, int a, const double* h1,
                                   const double* eri, int n_orb) {
    int phase = (count_between(beta, i, a) & 1) ? -1 : 1;
    double elem = get_h1(h1, n_orb, a, i);
    uint64_t t = beta;
    while (t) { int k = ctz64(t); t &= t-1;
        if (k != i) elem += get_eri(eri,n_orb,a,i,k,k) - get_eri(eri,n_orb,a,k,k,i); }
    t = alpha;
    while (t) { int k = ctz64(t); t &= t-1;
        elem += get_eri(eri,n_orb,a,i,k,k); }
    return phase * elem;
}

inline double compute_double_aa(uint64_t alpha, int i, int j, int a, int b,
                                 const double* eri, int n_orb) {
    int c1 = count_between(alpha, i, a);
    int c2 = count_between((alpha & ~(1ULL<<i)) | (1ULL<<a), j, b);
    int phase = ((c1+c2) & 1) ? -1 : 1;
    return phase * (get_eri(eri,n_orb,a,i,b,j) - get_eri(eri,n_orb,a,j,b,i));
}

inline double compute_double_bb(uint64_t beta, int i, int j, int a, int b,
                                 const double* eri, int n_orb) {
    int c1 = count_between(beta, i, a);
    int c2 = count_between((beta & ~(1ULL<<i)) | (1ULL<<a), j, b);
    int phase = ((c1+c2) & 1) ? -1 : 1;
    return phase * (get_eri(eri,n_orb,a,i,b,j) - get_eri(eri,n_orb,a,j,b,i));
}

inline double compute_double_ab(uint64_t alpha, uint64_t beta,
                                 int ia, int aa, int ib, int ab,
                                 const double* eri, int n_orb) {
    int phase = ((count_between(alpha,ia,aa) + count_between(beta,ib,ab)) & 1) ? -1 : 1;
    return phase * get_eri(eri,n_orb,aa,ia,ab,ib);
}

// ════════════════════════════════════════════════════════════════════
// Inverse map data structure
// ════════════════════════════════════════════════════════════════════

using ContribVec = std::vector<std::pair<int, double>>;

struct DetEntry {
    uint64_t alpha, beta;
    const ContribVec* contribs;
};

struct SCConnection {
    uint32_t d1, d2;   // indices into all_dets
    double h_elem;     // ⟨d1|H|d2⟩
};

struct InverseMapData {
    hash_map_t<Det, ContribVec, DetHash> map;
    std::vector<DetEntry> all_dets;
    std::vector<double> h_diag_det;      // per-det diagonal SC element
    std::vector<SCConnection> connections; // precomputed off-diagonal SC pairs
    size_t n_conn = 0;
};

void build_inverse_map(InverseMapData& data,
                       const uint64_t* ref_alpha, const uint64_t* ref_beta,
                       const double* ref_coeffs, int n_ref,
                       const int* exc_types, const int* exc_indices,
                       int n_exc) {
    auto& inv = data.map;
    inv.reserve(static_cast<size_t>(n_exc) * 2);

    hash_map_t<Det, double, DetHash> tmp;
    tmp.reserve(static_cast<size_t>(n_ref) * 2);

    // Row 0: reference
    for (int I = 0; I < n_ref; ++I)
        tmp[{ref_alpha[I], ref_beta[I]}] += ref_coeffs[I];
    for (auto& [det, coeff] : tmp)
        inv[det].emplace_back(0, coeff);

    // Rows 1..n_exc
    for (int mu = 0; mu < n_exc; ++mu) {
        const int* idx = &exc_indices[mu * 4];
        int etype = exc_types[mu];
        tmp.clear();
        for (int I = 0; I < n_ref; ++I) {
            ExcResult res = apply_excitation(ref_alpha[I], ref_beta[I], etype, idx);
            if (res.phase != 0)
                tmp[{res.alpha, res.beta}] += ref_coeffs[I] * res.phase;
        }
        for (auto& [det, coeff] : tmp)
            inv[det].emplace_back(mu + 1, coeff);
    }

    // Collect dets for iteration
    data.all_dets.reserve(inv.size());
    for (auto& [det, contribs] : inv)
        data.all_dets.push_back({det.first, det.second, &contribs});
}

// ════════════════════════════════════════════════════════════════════
// Precompute SC connection graph (done once, used every matvec)
// ════════════════════════════════════════════════════════════════════

void precompute_connections(InverseMapData& data,
                            const double* h1, const double* eri, int n_orb) {
    const size_t n_dets = data.all_dets.size();

    // Build det→index map
    hash_map_t<Det, uint32_t, DetHash> det_to_idx;
    det_to_idx.reserve(n_dets * 2);
    for (size_t d = 0; d < n_dets; ++d)
        det_to_idx[{data.all_dets[d].alpha, data.all_dets[d].beta}] = (uint32_t)d;

    // Compute diagonal SC element per det
    data.h_diag_det.resize(n_dets);
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 256)
    #endif
    for (size_t d = 0; d < n_dets; ++d)
        data.h_diag_det[d] = compute_diagonal(
            data.all_dets[d].alpha, data.all_dets[d].beta, h1, eri, n_orb);

    // Enumerate SC connections (parallel, thread-local buffers)
    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif
    std::vector<std::vector<SCConnection>> thread_conns(n_threads);

#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        auto& local_conns = thread_conns[tid];
        std::vector<int> occ_a, occ_b;

#ifdef _OPENMP
        #pragma omp for schedule(dynamic, 64)
#endif
        for (size_t d_idx = 0; d_idx < n_dets; ++d_idx) {
            uint64_t alpha_d = data.all_dets[d_idx].alpha;
            uint64_t beta_d = data.all_dets[d_idx].beta;
            Det det_d = {alpha_d, beta_d};
            get_occupied(alpha_d, occ_a);
            get_occupied(beta_d, occ_b);

            // Single alpha: i → a
            for (int i : occ_a) {
                for (int a = 0; a < n_orb; ++a) {
                    if (alpha_d & (1ULL << a)) continue;
                    Det dp = {(alpha_d & ~(1ULL << i)) | (1ULL << a), beta_d};
                    if (dp <= det_d) continue;
                    auto it = det_to_idx.find(dp);
                    if (it != det_to_idx.end())
                        local_conns.push_back({(uint32_t)d_idx, it->second,
                            compute_single_alpha(alpha_d, beta_d, i, a, h1, eri, n_orb)});
                }
            }
            // Single beta
            for (int i : occ_b) {
                for (int a = 0; a < n_orb; ++a) {
                    if (beta_d & (1ULL << a)) continue;
                    Det dp = {alpha_d, (beta_d & ~(1ULL << i)) | (1ULL << a)};
                    if (dp <= det_d) continue;
                    auto it = det_to_idx.find(dp);
                    if (it != det_to_idx.end())
                        local_conns.push_back({(uint32_t)d_idx, it->second,
                            compute_single_beta(alpha_d, beta_d, i, a, h1, eri, n_orb)});
                }
            }
            // Double alpha-alpha
            for (size_t ii = 0; ii < occ_a.size(); ++ii) {
                int i = occ_a[ii];
                for (size_t jj = ii+1; jj < occ_a.size(); ++jj) {
                    int j = occ_a[jj];
                    for (int a = 0; a < n_orb; ++a) {
                        if (alpha_d & (1ULL << a)) continue;
                        for (int b = a+1; b < n_orb; ++b) {
                            if (alpha_d & (1ULL << b)) continue;
                            uint64_t adp = (alpha_d & ~(1ULL<<i) & ~(1ULL<<j)) | (1ULL<<a) | (1ULL<<b);
                            Det dp = {adp, beta_d};
                            if (dp <= det_d) continue;
                            auto it = det_to_idx.find(dp);
                            if (it != det_to_idx.end())
                                local_conns.push_back({(uint32_t)d_idx, it->second,
                                    compute_double_aa(alpha_d, i, j, a, b, eri, n_orb)});
                        }
                    }
                }
            }
            // Double beta-beta
            for (size_t ii = 0; ii < occ_b.size(); ++ii) {
                int i = occ_b[ii];
                for (size_t jj = ii+1; jj < occ_b.size(); ++jj) {
                    int j = occ_b[jj];
                    for (int a = 0; a < n_orb; ++a) {
                        if (beta_d & (1ULL << a)) continue;
                        for (int b = a+1; b < n_orb; ++b) {
                            if (beta_d & (1ULL << b)) continue;
                            uint64_t bdp = (beta_d & ~(1ULL<<i) & ~(1ULL<<j)) | (1ULL<<a) | (1ULL<<b);
                            Det dp = {alpha_d, bdp};
                            if (dp <= det_d) continue;
                            auto it = det_to_idx.find(dp);
                            if (it != det_to_idx.end())
                                local_conns.push_back({(uint32_t)d_idx, it->second,
                                    compute_double_bb(beta_d, i, j, a, b, eri, n_orb)});
                        }
                    }
                }
            }
            // Double alpha-beta
            for (int i_a : occ_a) {
                for (int a_a = 0; a_a < n_orb; ++a_a) {
                    if (alpha_d & (1ULL << a_a)) continue;
                    uint64_t adp = (alpha_d & ~(1ULL << i_a)) | (1ULL << a_a);
                    for (int i_b : occ_b) {
                        for (int a_b = 0; a_b < n_orb; ++a_b) {
                            if (beta_d & (1ULL << a_b)) continue;
                            uint64_t bdp = (beta_d & ~(1ULL << i_b)) | (1ULL << a_b);
                            Det dp = {adp, bdp};
                            if (dp <= det_d) continue;
                            auto it = det_to_idx.find(dp);
                            if (it != det_to_idx.end())
                                local_conns.push_back({(uint32_t)d_idx, it->second,
                                    compute_double_ab(alpha_d, beta_d, i_a, a_a, i_b, a_b, eri, n_orb)});
                        }
                    }
                }
            }
        } // end parallel for
    } // end parallel

    // Merge thread-local connections
    size_t total = 0;
    for (auto& tc : thread_conns) total += tc.size();
    data.connections.reserve(total);
    for (auto& tc : thread_conns) {
        data.connections.insert(data.connections.end(), tc.begin(), tc.end());
        tc.clear(); tc.shrink_to_fit();
    }
    data.n_conn = data.connections.size();
}

// ════════════════════════════════════════════════════════════════════
// Matrix-vector products: H*v and S*v
// ════════════════════════════════════════════════════════════════════

// S*v: for each det d, sigma_d = sum c_mu * v[mu], then result[mu] += c_mu * sigma_d
void compute_Sv(const double* v, double* result, int n_basis,
                const InverseMapData& data) {
    std::memset(result, 0, sizeof(double) * n_basis);
    const size_t n_dets = data.all_dets.size();

    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif

    // Thread-local result vectors (tiny: n_basis doubles)
    std::vector<std::vector<double>> thread_results(n_threads, std::vector<double>(n_basis, 0.0));

#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        double* local = thread_results[tid].data();

#ifdef _OPENMP
        #pragma omp for schedule(dynamic, 256)
#endif
        for (size_t d = 0; d < n_dets; ++d) {
            const ContribVec& contribs = *data.all_dets[d].contribs;
            // Gather: sigma = sum c_mu * v[mu]
            double sigma = 0.0;
            for (const auto& [mu, c_mu] : contribs)
                sigma += c_mu * v[mu];
            // Scatter: result[mu] += c_mu * sigma
            for (const auto& [mu, c_mu] : contribs)
                local[mu] += c_mu * sigma;
        }
    }

    // Reduce
    for (int t = 0; t < n_threads; ++t)
        for (int i = 0; i < n_basis; ++i)
            result[i] += thread_results[t][i];
}

// H*v using precomputed SC connection graph — no hash lookups during matvec
void compute_Hv(const double* v, double* result, int n_basis,
                const InverseMapData& data) {
    std::memset(result, 0, sizeof(double) * n_basis);
    const size_t n_dets = data.all_dets.size();
    const size_t n_conn = data.connections.size();

    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif

    // Gather sigma for all dets
    std::vector<double> sigma(n_dets);
    for (size_t d = 0; d < n_dets; ++d) {
        double s = 0.0;
        for (const auto& [mu, c_mu] : *data.all_dets[d].contribs)
            s += c_mu * v[mu];
        sigma[d] = s;
    }

    // Thread-local result vectors (only n_basis doubles each — fits in cache)
    std::vector<std::vector<double>> thread_results(n_threads, std::vector<double>(n_basis, 0.0));

#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        double* local = thread_results[tid].data();

        // Diagonal contributions
#ifdef _OPENMP
        #pragma omp for schedule(static)
#endif
        for (size_t d = 0; d < n_dets; ++d) {
            double val = data.h_diag_det[d] * sigma[d];
            for (const auto& [mu, c_mu] : *data.all_dets[d].contribs)
                local[mu] += c_mu * val;
        }

        // Off-diagonal: iterate over precomputed connections (no hash lookups!)
#ifdef _OPENMP
        #pragma omp for schedule(static)
#endif
        for (size_t c = 0; c < n_conn; ++c) {
            const auto& conn = data.connections[c];
            double h = conn.h_elem;
            double s1 = sigma[conn.d1], s2 = sigma[conn.d2];
            for (const auto& [mu, c_mu] : *data.all_dets[conn.d1].contribs)
                local[mu] += c_mu * h * s2;
            for (const auto& [nu, c_nu] : *data.all_dets[conn.d2].contribs)
                local[nu] += c_nu * h * s1;
        }
    }

    // Reduce
    for (int t = 0; t < n_threads; ++t)
        for (int i = 0; i < n_basis; ++i)
            result[i] += thread_results[t][i];
}

// ════════════════════════════════════════════════════════════════════
// Diagonal elements H_ii / S_ii for preconditioner
// ════════════════════════════════════════════════════════════════════

// Approximate diagonal using only diagonal SC elements (no off-diagonal corrections).
// This is sufficient for the Davidson preconditioner. Exact H[0,0] is computed
// separately via a single Hv call.
void compute_diagonals(Eigen::VectorXd& H_diag, Eigen::VectorXd& S_diag,
                       int n_basis, const InverseMapData& data) {
    H_diag.setZero(n_basis);
    S_diag.setZero(n_basis);

    const size_t n_dets = data.all_dets.size();
    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif

    std::vector<std::vector<double>> thr_Hdiag(n_threads, std::vector<double>(n_basis, 0.0));
    std::vector<std::vector<double>> thr_Sdiag(n_threads, std::vector<double>(n_basis, 0.0));

#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        double* lH = thr_Hdiag[tid].data();
        double* lS = thr_Sdiag[tid].data();

#ifdef _OPENMP
        #pragma omp for schedule(dynamic, 256)
#endif
        for (size_t d = 0; d < n_dets; ++d) {
            const ContribVec& contribs = *data.all_dets[d].contribs;
            double h_dd = data.h_diag_det[d];
            for (const auto& [mu, c_mu] : contribs) {
                lH[mu] += c_mu * c_mu * h_dd;
                lS[mu] += c_mu * c_mu;
            }
        }
    }

    // Reduce
    for (int t = 0; t < n_threads; ++t)
        for (int i = 0; i < n_basis; ++i) {
            H_diag(i) += thr_Hdiag[t][i];
            S_diag(i) += thr_Sdiag[t][i];
        }
}

} // anonymous namespace

// ════════════════════════════════════════════════════════════════════
// Main entry: matrix-free Davidson GEP solver
// ════════════════════════════════════════════════════════════════════

MatfreeDavidsonResult matfree_davidson_gep(
    const uint64_t* ref_alpha, const uint64_t* ref_beta,
    const double* ref_coeffs, int n_ref,
    const int* exc_types, const int* exc_indices, int n_exc, int n_basis,
    const double* h1, const double* eri, int n_orb,
    int max_iter, double tol, int max_subspace, int verbose)
{
    using clock_t = std::chrono::high_resolution_clock;
    auto t_start = clock_t::now();

    MatfreeDavidsonResult result;
    result.converged = false;
    result.iterations = 0;
    result.h_diag_0 = 0.0;

    // Step 1: Build inverse map
    if (verbose >= 1)
        fprintf(stderr, "  [matfree] Building inverse map (n_basis=%d, n_ref=%d)...\n",
                n_basis, n_ref);

    InverseMapData imap;
    build_inverse_map(imap, ref_alpha, ref_beta, ref_coeffs, n_ref,
                      exc_types, exc_indices, n_exc);

    auto t_invmap = clock_t::now();
    double dt_invmap = std::chrono::duration<double>(t_invmap - t_start).count();

    size_t n_dets = imap.all_dets.size();
    if (verbose >= 1)
        fprintf(stderr, "  [matfree] Inverse map: %zu unique dets, %.1fs\n",
                n_dets, dt_invmap);

    // Step 2: Precompute SC connection graph (done once, reused every matvec)
    if (verbose >= 1)
        fprintf(stderr, "  [matfree] Precomputing SC connections...\n");
    precompute_connections(imap, h1, eri, n_orb);

    auto t_precomp = clock_t::now();
    double dt_precomp = std::chrono::duration<double>(t_precomp - t_invmap).count();
    if (verbose >= 1)
        fprintf(stderr, "  [matfree] SC graph: %zu connections, %.1fs\n",
                imap.n_conn, dt_precomp);

    // Step 3: Compute approximate diagonal preconditioner (parallel, fast)
    Eigen::VectorXd H_diag, S_diag;
    compute_diagonals(H_diag, S_diag, n_basis, imap);

    Eigen::VectorXd precond(n_basis);
    for (int i = 0; i < n_basis; ++i)
        precond(i) = (std::abs(S_diag(i)) > 1e-14) ? H_diag(i) / S_diag(i) : H_diag(i);

    // Compute exact H[0,0] via a single Hv(e_0) call (already multithreaded)
    {
        Eigen::VectorXd e0 = Eigen::VectorXd::Zero(n_basis);
        e0(0) = 1.0;
        Eigen::VectorXd He0(n_basis);
        compute_Hv(e0.data(), He0.data(), n_basis, imap);
        result.h_diag_0 = He0(0);
    }

    auto t_diag = clock_t::now();
    double dt_diag = std::chrono::duration<double>(t_diag - t_precomp).count();
    if (verbose >= 1)
        fprintf(stderr, "  [matfree] Diagonals computed, H[0,0]=%.10f (exact via Hv), %.1fs\n",
                result.h_diag_0, dt_diag);

    // Step 3: Davidson iteration
    int idx = 0;
    precond.minCoeff(&idx);

    // Subspace basis V (columns are S-orthonormal)
    Eigen::MatrixXd V(n_basis, 1);
    V.col(0).setZero();
    V(idx, 0) = 1.0;

    // S-normalize initial vector
    Eigen::VectorXd Sv(n_basis);
    compute_Sv(V.col(0).data(), Sv.data(), n_basis, imap);
    double snorm = std::sqrt(std::abs(V.col(0).dot(Sv)));
    if (snorm > 1e-14) V.col(0) /= snorm;

    // Cached AV = H*V and BV = S*V columns
    Eigen::MatrixXd AV(n_basis, 1), BV(n_basis, 1);
    compute_Hv(V.col(0).data(), AV.col(0).data(), n_basis, imap);
    compute_Sv(V.col(0).data(), BV.col(0).data(), n_basis, imap);

    double E_old = 0.0;

    for (int iter = 0; iter < max_iter; ++iter) {
        int k = V.cols();

        // Project: H_sub = V^T * AV, S_sub = V^T * BV
        Eigen::MatrixXd H_sub = V.transpose() * AV;
        Eigen::MatrixXd S_sub = V.transpose() * BV;
        H_sub = 0.5 * (H_sub + H_sub.transpose());
        S_sub = 0.5 * (S_sub + S_sub.transpose());

        // Solve small GEP via canonical orthogonalization
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> S_solver(S_sub);
        Eigen::VectorXd s_eigs = S_solver.eigenvalues();
        Eigen::MatrixXd s_vecs = S_solver.eigenvectors();

        double s_thresh = 1e-10;
        int n_valid = 0;
        for (int i = 0; i < k; ++i)
            if (s_eigs(i) > s_thresh) ++n_valid;

        if (n_valid == 0) {
            if (verbose >= 1) fprintf(stderr, "  [matfree] No valid subspace modes\n");
            break;
        }

        Eigen::MatrixXd X(k, n_valid);
        int col = 0;
        for (int i = 0; i < k; ++i)
            if (s_eigs(i) > s_thresh)
                X.col(col++) = s_vecs.col(i) / std::sqrt(s_eigs(i));

        Eigen::MatrixXd H_orth = X.transpose() * H_sub * X;
        H_orth = 0.5 * (H_orth + H_orth.transpose());

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> H_solver(H_orth);
        double E = H_solver.eigenvalues()(0);
        Eigen::VectorXd alpha = X * H_solver.eigenvectors().col(0);

        // Ritz vector
        Eigen::VectorXd x = V * alpha;

        // Residual: r = H*x - E*S*x = AV*alpha - E*BV*alpha
        Eigen::VectorXd Hx = AV * alpha;
        Eigen::VectorXd Sx = BV * alpha;
        Eigen::VectorXd r = Hx - E * Sx;
        double r_norm = r.norm();

        if (verbose >= 2)
            fprintf(stderr, "  [matfree] iter %3d: E=%.10f, |r|=%.2e, k=%d\n",
                    iter+1, E, r_norm, k);

        result.iterations = iter + 1;
        result.residual_norm = r_norm;

        if (std::abs(E - E_old) < tol && r_norm < tol * 100) {
            result.converged = true;
            result.eigenvalue = E;
            double norm_sq = x.dot(Sx);
            if (norm_sq > 0) x /= std::sqrt(norm_sq);
            if (x(0) < 0) x = -x;
            result.eigenvector = x;

            auto t_end = clock_t::now();
            double dt_total = std::chrono::duration<double>(t_end - t_start).count();
            if (verbose >= 1)
                fprintf(stderr, "  [matfree] Converged in %d iterations, E=%.10f, |r|=%.2e, "
                        "total=%.1fs (invmap=%.1fs, precomp=%.1fs, davidson=%.1fs)\n",
                        iter+1, E, r_norm, dt_total, dt_invmap, dt_precomp,
                        dt_total - dt_invmap - dt_precomp - dt_diag);
            return result;
        }
        E_old = E;

        // Preconditioned correction
        Eigen::VectorXd t(n_basis);
        for (int i = 0; i < n_basis; ++i) {
            double denom = precond(i) - E;
            t(i) = (std::abs(denom) > 1e-12) ? -r(i) / denom : -r(i);
        }

        // S-orthogonalize t against V (double Gram-Schmidt using cached BV)
        for (int pass = 0; pass < 2; ++pass) {
            Eigen::VectorXd St(n_basis);
            compute_Sv(t.data(), St.data(), n_basis, imap);
            Eigen::VectorXd coeffs = V.transpose() * St;
            t -= V * coeffs;
        }

        // S-normalize t
        Eigen::VectorXd St_final(n_basis);
        compute_Sv(t.data(), St_final.data(), n_basis, imap);
        double t_snorm = std::sqrt(std::abs(t.dot(St_final)));
        if (t_snorm < 1e-14) {
            if (verbose >= 2) fprintf(stderr, "  [matfree] Expansion vector too small at iter %d\n", iter+1);
            continue;
        }
        t /= t_snorm;

        // Restart or expand subspace
        if (k >= max_subspace) {
            int n_keep = std::min(3, k);
            Eigen::MatrixXd best(n_basis, n_keep);
            for (int j = 0; j < n_keep; ++j)
                best.col(j) = V * (X * H_solver.eigenvectors().col(j));

            // Re-S-orthonormalize
            for (int j = 0; j < n_keep; ++j) {
                for (int ii = 0; ii < j; ++ii) {
                    Eigen::VectorXd S_bj(n_basis);
                    compute_Sv(best.col(j).data(), S_bj.data(), n_basis, imap);
                    double overlap = best.col(ii).dot(S_bj);
                    best.col(j) -= overlap * best.col(ii);
                }
                Eigen::VectorXd S_bj(n_basis);
                compute_Sv(best.col(j).data(), S_bj.data(), n_basis, imap);
                double sn = std::sqrt(std::abs(best.col(j).dot(S_bj)));
                if (sn > 1e-14) best.col(j) /= sn;
            }

            V.resize(n_basis, n_keep + 1);
            V.leftCols(n_keep) = best;
            V.col(n_keep) = t;

            // Recompute AV and BV for new V
            AV.resize(n_basis, n_keep + 1);
            BV.resize(n_basis, n_keep + 1);
            for (int j = 0; j <= n_keep; ++j) {
                compute_Hv(V.col(j).data(), AV.col(j).data(), n_basis, imap);
                compute_Sv(V.col(j).data(), BV.col(j).data(), n_basis, imap);
            }

            if (verbose >= 2)
                fprintf(stderr, "  [matfree] Subspace restart %d -> %d\n", k, n_keep + 1);
        } else {
            V.conservativeResize(n_basis, k + 1);
            V.col(k) = t;

            AV.conservativeResize(n_basis, k + 1);
            BV.conservativeResize(n_basis, k + 1);
            compute_Hv(t.data(), AV.col(k).data(), n_basis, imap);
            compute_Sv(t.data(), BV.col(k).data(), n_basis, imap);
        }
    }

    // Not converged: return best estimate
    result.eigenvalue = E_old;
    result.eigenvector = Eigen::VectorXd::Zero(n_basis);
    result.eigenvector(0) = 1.0;

    if (verbose >= 1)
        fprintf(stderr, "  [matfree] NOT converged after %d iterations, E=%.10f, |r|=%.2e\n",
                max_iter, result.eigenvalue, result.residual_norm);

    return result;
}

} // namespace trimci_core
