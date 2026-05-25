#include "detspace_matvec.hpp"
#include "hamiltonian.hpp"
#include "omp_compat.hpp"
#include <cstring>
#include <algorithm>
#include <chrono>
#include <cstdio>

namespace trimci_core {
namespace fe {

// ============================================================================
// compute_diagonals: batch H_ii computation
// ============================================================================
template<typename StorageType>
std::vector<double> compute_diagonals(
    const std::vector<DeterminantT<StorageType>>& dets,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    int n_orb)
{
    const size_t N = dets.size();
    std::vector<double> diag(N);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(N); ++i) {
        diag[i] = compute_H_ij_t<StorageType>(dets[i], dets[i], h1, eri);
    }

    return diag;
}

// ============================================================================
// compute_H_ij_strings: spin-aware fast path, takes string references directly
//
// Skips the XOR+popcount classification that compute_H_ij_t performs,
// since ABIndex already provides the excitation type.
// Takes individual alpha/beta strings instead of DeterminantT, avoiding
// random access to dets[] when called from for_each_connection with
// inline CSR strings.
// ============================================================================
template<typename StorageType>
double compute_H_ij_strings(
    const StorageType& ai, const StorageType& bi,
    const StorageType& aj, const StorageType& bj,
    ExcType exc_type,
    const std::vector<std::vector<double>>& h1,
    const double* eri_data,
    int n_orb)
{
    using BitOpsType = detail::HamiltonianBitOps<StorageType>;

    const size_t N = n_orb;
    const size_t N2 = N * N;
    const size_t N3 = N2 * N;

    switch (exc_type) {

    // ------------------------------------------------------------------
    // Alpha single excitation (same beta string, 1 alpha difference)
    // ------------------------------------------------------------------
    case SAME_BETA_S: {
        int da_rem[2], da_add[2];
        BitOpsType::storage_to_indices_inline(BitOpsType::and_not(ai, aj), da_rem, 2);
        BitOpsType::storage_to_indices_inline(BitOpsType::and_not(aj, ai), da_add, 2);
        int m = da_rem[0], p = da_add[0];
        int phase = detail::cre_des_sign_t(m, p, aj);

        double Hij = h1[m][p];
        const size_t base_mp = m * N3 + p * N2;  // eri[m,p,:,:]
        const size_t base_m  = m * N3;            // eri[m,:,:,:]
        const size_t Np1 = N + 1;
        const size_t N2pN = N2 + N;

        // Same-spin (alpha-alpha) Coulomb-exchange
        int occ_a[64];
        int n_a = BitOpsType::storage_to_indices_inline(ai, occ_a, 64);
        for (int k = 0; k < n_a; ++k) {
            int n = occ_a[k];
            if (n != m) Hij += eri_data[base_mp + n * Np1] - eri_data[base_m + n * N2pN + p];
        }

        // Opposite-spin (alpha-beta) Coulomb only
        int occ_b[64];
        int n_b = BitOpsType::storage_to_indices_inline(bi, occ_b, 64);
        for (int k = 0; k < n_b; ++k) {
            Hij += eri_data[base_mp + occ_b[k] * Np1];
        }

        return Hij * phase;
    }

    // ------------------------------------------------------------------
    // Beta single excitation (same alpha string, 1 beta difference)
    // ------------------------------------------------------------------
    case SAME_ALPHA_S: {
        int db_rem[2], db_add[2];
        BitOpsType::storage_to_indices_inline(BitOpsType::and_not(bi, bj), db_rem, 2);
        BitOpsType::storage_to_indices_inline(BitOpsType::and_not(bj, bi), db_add, 2);
        int m = db_rem[0], p = db_add[0];
        int phase = detail::cre_des_sign_t(m, p, bj);

        double Hij = h1[m][p];
        const size_t base_mp = m * N3 + p * N2;
        const size_t base_m  = m * N3;
        const size_t Np1 = N + 1;
        const size_t N2pN = N2 + N;

        // Same-spin (beta-beta) Coulomb-exchange
        int occ_b[64];
        int n_b = BitOpsType::storage_to_indices_inline(bi, occ_b, 64);
        for (int k = 0; k < n_b; ++k) {
            int n = occ_b[k];
            if (n != m) Hij += eri_data[base_mp + n * Np1] - eri_data[base_m + n * N2pN + p];
        }

        // Opposite-spin (beta-alpha) Coulomb only
        int occ_a[64];
        int n_a = BitOpsType::storage_to_indices_inline(ai, occ_a, 64);
        for (int k = 0; k < n_a; ++k) {
            Hij += eri_data[base_mp + occ_a[k] * Np1];
        }

        return Hij * phase;
    }

    // ------------------------------------------------------------------
    // Alpha double excitation (same beta string, 2 alpha differences)
    // ------------------------------------------------------------------
    case SAME_BETA_D: {
        int da_rem[2], da_add[2];
        BitOpsType::storage_to_indices_inline(BitOpsType::and_not(ai, aj), da_rem, 2);
        BitOpsType::storage_to_indices_inline(BitOpsType::and_not(aj, ai), da_add, 2);
        int m = std::min(da_rem[0], da_rem[1]);
        int n = std::max(da_rem[0], da_rem[1]);
        int p = std::min(da_add[0], da_add[1]);
        int q = std::max(da_add[0], da_add[1]);

        int phase1 = detail::cre_des_sign_t(m, p, aj);
        auto new_a = aj;
        BitOps<StorageType>::set_bit(new_a, m);
        BitOps<StorageType>::clear_bit(new_a, p);
        int phase2 = detail::cre_des_sign_t(n, q, new_a);

        return phase1 * phase2 *
            (eri_data[m * N3 + p * N2 + n * N + q] -
             eri_data[m * N3 + q * N2 + n * N + p]);
    }

    // ------------------------------------------------------------------
    // Beta double excitation (same alpha string, 2 beta differences)
    // ------------------------------------------------------------------
    case SAME_ALPHA_D: {
        int db_rem[2], db_add[2];
        BitOpsType::storage_to_indices_inline(BitOpsType::and_not(bi, bj), db_rem, 2);
        BitOpsType::storage_to_indices_inline(BitOpsType::and_not(bj, bi), db_add, 2);
        int m = std::min(db_rem[0], db_rem[1]);
        int n = std::max(db_rem[0], db_rem[1]);
        int p = std::min(db_add[0], db_add[1]);
        int q = std::max(db_add[0], db_add[1]);

        int phase1 = detail::cre_des_sign_t(m, p, bj);
        auto new_b = bj;
        BitOps<StorageType>::set_bit(new_b, m);
        BitOps<StorageType>::clear_bit(new_b, p);
        int phase2 = detail::cre_des_sign_t(n, q, new_b);

        return phase1 * phase2 *
            (eri_data[m * N3 + p * N2 + n * N + q] -
             eri_data[m * N3 + q * N2 + n * N + p]);
    }

    // ------------------------------------------------------------------
    // Mixed double excitation (1 alpha + 1 beta)
    // ------------------------------------------------------------------
    case MIXED_D: {
        int da_rem[2], da_add[2], db_rem[2], db_add[2];
        BitOpsType::storage_to_indices_inline(BitOpsType::and_not(ai, aj), da_rem, 2);
        BitOpsType::storage_to_indices_inline(BitOpsType::and_not(aj, ai), da_add, 2);
        BitOpsType::storage_to_indices_inline(BitOpsType::and_not(bi, bj), db_rem, 2);
        BitOpsType::storage_to_indices_inline(BitOpsType::and_not(bj, bi), db_add, 2);
        int m = da_rem[0], p = da_add[0];
        int n = db_rem[0], q = db_add[0];
        int phase = detail::cre_des_sign_t(m, p, aj) * detail::cre_des_sign_t(n, q, bj);
        return phase * eri_data[m * N3 + p * N2 + n * N + q];
    }

    }  // switch

    return 0.0;
}

// Thin wrapper preserving old DeterminantT interface
template<typename StorageType>
double compute_H_ij_by_type(
    const DeterminantT<StorageType>& di,
    const DeterminantT<StorageType>& dj,
    ExcType exc_type,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    int n_orb)
{
    return compute_H_ij_strings<StorageType>(
        di.alpha, di.beta, dj.alpha, dj.beta,
        exc_type, h1, eri.data(), n_orb);
}

// ============================================================================
// matvec: sigma = H * v (matrix-free)
//
// Upper-triangle with per-call calloc sigma buffers + parallel reduce.
// calloc gets COW zero pages from OS (no memset needed).
// Spin-aware H_ij dispatch skips XOR+popcount classification.
// v[i] hoisted and sigma_i accumulated locally to reduce memory traffic.
// ============================================================================
template<typename StorageType>
void matvec(const MatvecContext<StorageType>& ctx,
            const double* v,
            double* sigma,
            size_t N)
{
    // Diagonal contribution
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(N); ++i) {
        sigma[i] = ctx.diag[i] * v[i];
    }

    const int n_threads = omp_get_max_threads();

    // Per-thread sigma buffers: calloc provides COW zero pages from OS
    std::vector<double*> thread_sigma(n_threads);
    #pragma omp parallel
    {
        thread_sigma[omp_get_thread_num()] = (double*)std::calloc(N, sizeof(double));
    }

    #pragma omp parallel
    {
        double* my_sigma = thread_sigma[omp_get_thread_num()];

        #pragma omp for schedule(dynamic, 64)
        for (int i = 0; i < static_cast<int>(N); ++i) {
            const double vi = v[i];
            double sigma_i = 0.0;

            const auto& ai = ctx.dets[i].alpha;
            const auto& bi = ctx.dets[i].beta;

            ctx.ab_index.for_each_connection(i,
                [&](size_t j, ExcType exc_type,
                    const StorageType& alpha_j, const StorageType& beta_j) {
                if (j <= i) return;  // upper triangle only

                double H_ij = compute_H_ij_strings<StorageType>(
                    ai, bi, alpha_j, beta_j, exc_type,
                    ctx.h1, ctx.eri.data(), ctx.n_orb);

                sigma_i += H_ij * v[j];
                my_sigma[j] += H_ij * vi;
            });

            my_sigma[i] += sigma_i;
        }

        // Parallel reduce
        #pragma omp for schedule(static)
        for (int i = 0; i < static_cast<int>(N); ++i) {
            double sum = 0.0;
            for (int t = 0; t < n_threads; ++t) {
                sum += thread_sigma[t][i];
            }
            sigma[i] += sum;
        }
    }

    // Free thread-local buffers
    for (int t = 0; t < n_threads; ++t) {
        std::free(thread_sigma[t]);
    }
}

// ============================================================================
// build_connection_cache: two-pass parallel CSR construction
//
// Pass 1: count upper-triangle connections per det (parallel).
// Pass 2: fill (j, H_ij) values (parallel, each det writes its own CSR segment).
// Build cost ≈ 2 matvec iterations; amortized over ~20 Davidson iterations.
// ============================================================================
template<typename StorageType>
ConnectionCache<StorageType> build_connection_cache(
    const ABIndex<StorageType>& ab_index,
    const std::vector<DeterminantT<StorageType>>& dets,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    int n_orb)
{
    const size_t N = dets.size();
    ConnectionCache<StorageType> cache;
    cache.n_dets = N;

    // Pass 1: count upper-triangle connections per det
    cache.ptr.resize(N + 1, 0);

    #pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < static_cast<int>(N); ++i) {
        size_t count = 0;
        ab_index.for_each_connection(i,
            [&](size_t j, ExcType, const StorageType&, const StorageType&) {
                if (j > i) count++;
            });
        cache.ptr[i + 1] = count;
    }

    // Prefix sum
    for (size_t i = 1; i <= N; ++i) {
        cache.ptr[i] += cache.ptr[i - 1];
    }

    const size_t nnz = cache.ptr[N];
    cache.col.resize(nnz);
    cache.val.resize(nnz);

    // Pass 2: fill col[] and val[]
    const double* eri_data = eri.data();

    #pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < static_cast<int>(N); ++i) {
        const auto& ai = dets[i].alpha;
        const auto& bi = dets[i].beta;
        size_t pos = cache.ptr[i];

        ab_index.for_each_connection(i,
            [&](size_t j, ExcType exc_type,
                const StorageType& alpha_j, const StorageType& beta_j) {
                if (j > i) {
                    cache.col[pos] = static_cast<uint32_t>(j);
                    cache.val[pos] = compute_H_ij_strings<StorageType>(
                        ai, bi, alpha_j, beta_j, exc_type,
                        h1, eri_data, n_orb);
                    pos++;
                }
            });
    }

    return cache;
}

// ============================================================================
// matvec_cached: sigma = H * v using precomputed connection cache
//
// Inner loop is a simple cached SpMV: sequential read of (j, H_ij) pairs.
// No connection enumeration, no H_ij computation — all precomputed.
// Upper-triangle with symmetric sigma update + parallel reduce.
// ============================================================================
template<typename StorageType>
void matvec_cached(const ConnectionCache<StorageType>& cache,
                   const std::vector<double>& diag,
                   const double* v,
                   double* sigma,
                   size_t N)
{
    // Diagonal contribution
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(N); ++i) {
        sigma[i] = diag[i] * v[i];
    }

    const int n_threads = omp_get_max_threads();

    // Per-thread sigma buffers: calloc provides COW zero pages from OS
    std::vector<double*> thread_sigma(n_threads);
    #pragma omp parallel
    {
        thread_sigma[omp_get_thread_num()] = (double*)std::calloc(N, sizeof(double));
    }

    #pragma omp parallel
    {
        double* my_sigma = thread_sigma[omp_get_thread_num()];

        #pragma omp for schedule(dynamic, 64)
        for (int i = 0; i < static_cast<int>(N); ++i) {
            const double vi = v[i];
            double sigma_i = 0.0;

            // Sequential read from cached CSR
            const size_t row_begin = cache.ptr[i];
            const size_t row_end   = cache.ptr[i + 1];
            const uint32_t* col_ptr = cache.col.data() + row_begin;
            const double*   val_ptr = cache.val.data() + row_begin;
            const size_t row_len = row_end - row_begin;

            for (size_t k = 0; k < row_len; ++k) {
                const uint32_t j = col_ptr[k];
                const double H_ij = val_ptr[k];
                sigma_i += H_ij * v[j];
                my_sigma[j] += H_ij * vi;
            }

            my_sigma[i] += sigma_i;
        }

        // Parallel reduce
        #pragma omp for schedule(static)
        for (int i = 0; i < static_cast<int>(N); ++i) {
            double sum = 0.0;
            for (int t = 0; t < n_threads; ++t) {
                sum += thread_sigma[t][i];
            }
            sigma[i] += sum;
        }
    }

    // Free thread-local buffers
    for (int t = 0; t < n_threads; ++t) {
        std::free(thread_sigma[t]);
    }
}

// ============================================================================
// precompute_J_alpha: alpha-group Coulomb table
//
// J_alpha[m][p] = Σ_{n∈occ_α} eri[m,p,n,n]
// Constant for all dets sharing the same alpha string.
// Table size: n_orb² × 8 bytes (~10 KB for n_orb=36), fits L1 cache.
// ============================================================================
template<typename StorageType>
void precompute_J_alpha(
    const StorageType& alpha,
    const double* eri_data,
    int n_orb,
    double* J_alpha)  // [n_orb * n_orb] output, row-major
{
    using BitOpsType = detail::HamiltonianBitOps<StorageType>;

    const size_t N = n_orb;
    const size_t N2 = N * N;
    const size_t N3 = N2 * N;

    std::memset(J_alpha, 0, N * N * sizeof(double));

    int occ_a[64];
    int n_a = BitOpsType::storage_to_indices_inline(alpha, occ_a, 64);

    for (int k = 0; k < n_a; ++k) {
        const int n = occ_a[k];
        const size_t n_stride = static_cast<size_t>(n) * N + n;  // n*N+n for eri[m,p,n,n]
        for (int m = 0; m < n_orb; ++m) {
            const size_t base_m = static_cast<size_t>(m) * N3;
            for (int p = 0; p < n_orb; ++p) {
                J_alpha[m * N + p] += eri_data[base_m + static_cast<size_t>(p) * N2 + n_stride];
            }
        }
    }
}

// ============================================================================
// compute_H_ij_same_alpha_s_dressed: SAME_ALPHA_S with J_alpha table
//
// Replaces the alpha Coulomb loop (Σ_{n∈occ_α} eri[m,p,n,n]) with a single
// J_alpha[m][p] table lookup. The beta Coulomb-exchange loop remains.
// ============================================================================
template<typename StorageType>
double compute_H_ij_same_alpha_s_dressed(
    const StorageType& bi, const StorageType& bj,
    const std::vector<std::vector<double>>& h1,
    const double* eri_data,
    const double* J_alpha,
    int n_orb)
{
    using BitOpsType = detail::HamiltonianBitOps<StorageType>;

    const size_t N = n_orb;
    const size_t N2 = N * N;
    const size_t N3 = N2 * N;
    const size_t Np1 = N + 1;
    const size_t N2pN = N2 + N;

    // Extract excitation orbitals: beta single m→p
    int db_rem[2], db_add[2];
    BitOpsType::storage_to_indices_inline(BitOpsType::and_not(bi, bj), db_rem, 2);
    BitOpsType::storage_to_indices_inline(BitOpsType::and_not(bj, bi), db_add, 2);
    int m = db_rem[0], p = db_add[0];
    int phase = detail::cre_des_sign_t(m, p, bj);

    // One-electron + alpha Coulomb (table lookup)
    double Hij = h1[m][p] + J_alpha[m * N + p];

    // Same-spin (beta-beta) Coulomb-exchange: loop over occ_β(i), skip n=m
    const size_t base_mp = static_cast<size_t>(m) * N3 + static_cast<size_t>(p) * N2;
    const size_t base_m  = static_cast<size_t>(m) * N3;

    int occ_b[64];
    int n_b = BitOpsType::storage_to_indices_inline(bi, occ_b, 64);
    for (int k = 0; k < n_b; ++k) {
        int n = occ_b[k];
        if (n != m) Hij += eri_data[base_mp + n * Np1] - eri_data[base_m + n * N2pN + p];
    }

    return Hij * phase;
}

// ============================================================================
// matvec_dressed: on-the-fly sigma = H * v with alpha-group Coulomb sharing
//
// Same structure as matvec(), but per-thread tracks the current alpha group
// and precomputes J_alpha[m][p] when it changes. With alpha-sorted dets,
// this table is reused across ~200 consecutive dets per alpha group.
//
// For SAME_ALPHA_S connections: uses compute_H_ij_same_alpha_s_dressed
// For all other exc types: falls back to standard compute_H_ij_strings
// ============================================================================
template<typename StorageType>
void matvec_dressed(const MatvecContext<StorageType>& ctx,
                    const double* v,
                    double* sigma,
                    size_t N)
{
    // Diagonal contribution
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(N); ++i) {
        sigma[i] = ctx.diag[i] * v[i];
    }

    const int n_threads = omp_get_max_threads();
    const int n_orb = ctx.n_orb;
    const double* eri_data = ctx.eri.data();

    // Per-thread sigma buffers: calloc provides COW zero pages from OS
    std::vector<double*> thread_sigma(n_threads);
    #pragma omp parallel
    {
        thread_sigma[omp_get_thread_num()] = (double*)std::calloc(N, sizeof(double));
    }

    // Channel profiling counters (per-thread to avoid atomics)
    static int matvec_call_count = 0;
    const bool do_profile = (matvec_call_count == 0);  // profile first call only
    matvec_call_count++;

    std::vector<uint64_t> thread_cnt_as(n_threads, 0), thread_cnt_ad(n_threads, 0);
    std::vector<uint64_t> thread_cnt_bs(n_threads, 0), thread_cnt_bd(n_threads, 0);
    std::vector<uint64_t> thread_cnt_mx(n_threads, 0);

    auto t_main_start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel
    {
        double* my_sigma = thread_sigma[omp_get_thread_num()];
        const int tid = omp_get_thread_num();

        // Per-thread J_alpha table (n_orb² doubles, ~10 KB, fits L1)
        std::vector<double> J_alpha(n_orb * n_orb, 0.0);
        uint32_t prev_alpha_id = UINT32_MAX;  // sentinel: force first computation

        uint64_t cnt_as = 0, cnt_ad = 0, cnt_bs = 0, cnt_bd = 0, cnt_mx = 0;

        #pragma omp for schedule(dynamic, 64)
        for (int i = 0; i < static_cast<int>(N); ++i) {
            const double vi = v[i];
            double sigma_i = 0.0;

            const auto& ai = ctx.dets[i].alpha;
            const auto& bi = ctx.dets[i].beta;

            // Update J_alpha table if alpha group changed
            uint32_t cur_alpha_id = ctx.ab_index.det_alpha_id(i);
            if (cur_alpha_id != prev_alpha_id) {
                precompute_J_alpha<StorageType>(ai, eri_data, n_orb, J_alpha.data());
                prev_alpha_id = cur_alpha_id;
            }

            ctx.ab_index.for_each_connection(i,
                [&](size_t j, ExcType exc_type,
                    const StorageType& alpha_j, const StorageType& beta_j) {
                if (j <= i) return;  // upper triangle only

                if (do_profile) {
                    switch(exc_type) {
                        case SAME_ALPHA_S: cnt_as++; break;
                        case SAME_ALPHA_D: cnt_ad++; break;
                        case SAME_BETA_S:  cnt_bs++; break;
                        case SAME_BETA_D:  cnt_bd++; break;
                        case MIXED_D:      cnt_mx++; break;
                    }
                }

                double H_ij;
                if (exc_type == SAME_ALPHA_S) {
                    // Dressed path: J_alpha table replaces alpha Coulomb loop
                    H_ij = compute_H_ij_same_alpha_s_dressed<StorageType>(
                        bi, beta_j, ctx.h1, eri_data, J_alpha.data(), n_orb);
                } else {
                    // Standard path for all other excitation types
                    H_ij = compute_H_ij_strings<StorageType>(
                        ai, bi, alpha_j, beta_j, exc_type,
                        ctx.h1, eri_data, n_orb);
                }

                sigma_i += H_ij * v[j];
                my_sigma[j] += H_ij * vi;
            });

            my_sigma[i] += sigma_i;
        }

        if (do_profile) {
            thread_cnt_as[tid] = cnt_as;
            thread_cnt_ad[tid] = cnt_ad;
            thread_cnt_bs[tid] = cnt_bs;
            thread_cnt_bd[tid] = cnt_bd;
            thread_cnt_mx[tid] = cnt_mx;
        }

        // Parallel reduce
        #pragma omp for schedule(static)
        for (int i = 0; i < static_cast<int>(N); ++i) {
            double sum = 0.0;
            for (int t = 0; t < n_threads; ++t) {
                sum += thread_sigma[t][i];
            }
            sigma[i] += sum;
        }
    }

    auto t_main_end = std::chrono::high_resolution_clock::now();

    if (do_profile) {
        uint64_t tot_as=0, tot_ad=0, tot_bs=0, tot_bd=0, tot_mx=0;
        for (int t = 0; t < n_threads; ++t) {
            tot_as += thread_cnt_as[t]; tot_ad += thread_cnt_ad[t];
            tot_bs += thread_cnt_bs[t]; tot_bd += thread_cnt_bd[t];
            tot_mx += thread_cnt_mx[t];
        }
        uint64_t total = tot_as + tot_ad + tot_bs + tot_bd + tot_mx;
        double ms = std::chrono::duration<double, std::milli>(t_main_end - t_main_start).count();
        std::fprintf(stderr,
            "\n[matvec_dressed profile] N=%zu, %d threads, main_loop=%.1f ms\n"
            "  Connections (upper-tri): total=%llu\n"
            "    Ch1 SAME_ALPHA_S: %llu (%.1f%%)\n"
            "    Ch1 SAME_ALPHA_D: %llu (%.1f%%)\n"
            "    Ch3 SAME_BETA_S:  %llu (%.1f%%)\n"
            "    Ch3 SAME_BETA_D:  %llu (%.1f%%)\n"
            "    Ch2 MIXED_D:      %llu (%.1f%%)\n"
            "  Avg connections/det: %.1f\n",
            N, n_threads, ms,
            (unsigned long long)total,
            (unsigned long long)tot_as, 100.0*tot_as/total,
            (unsigned long long)tot_ad, 100.0*tot_ad/total,
            (unsigned long long)tot_bs, 100.0*tot_bs/total,
            (unsigned long long)tot_bd, 100.0*tot_bd/total,
            (unsigned long long)tot_mx, 100.0*tot_mx/total,
            (double)total / N);
    }

    // Free thread-local buffers
    for (int t = 0; t < n_threads; ++t) {
        std::free(thread_sigma[t]);
    }
}

// ============================================================================
// matvec_dressed_full: row-only accumulation, no per-thread sigma buffers
//
// Processes all connections (not just upper triangle) and accumulates only
// sigma[i] per row. This doubles H_ij computation but eliminates:
//   - O(N × T) per-thread sigma buffers (245 GB at 320M × 96 threads)
//   - O(N × T) parallel reduce phase
//   - Random writes to my_sigma[j]
//
// Each row is fully independent → perfect parallel scaling, zero contention.
// Preferred for large N where per-thread buffers exceed physical memory.
// ============================================================================
template<typename StorageType>
void matvec_dressed_full(const MatvecContext<StorageType>& ctx,
                         const double* v,
                         double* sigma,
                         size_t N)
{
    const int n_orb = ctx.n_orb;
    const double* eri_data = ctx.eri.data();

    #pragma omp parallel
    {
        // Per-thread J_alpha table (n_orb² doubles, ~10 KB, fits L1)
        std::vector<double> J_alpha(n_orb * n_orb, 0.0);
        uint32_t prev_alpha_id = UINT32_MAX;

        #pragma omp for schedule(dynamic, 64)
        for (int i = 0; i < static_cast<int>(N); ++i) {
            double sigma_i = ctx.diag[i] * v[i];

            const auto& ai = ctx.dets[i].alpha;
            const auto& bi = ctx.dets[i].beta;

            // Update J_alpha table if alpha group changed
            uint32_t cur_alpha_id = ctx.ab_index.det_alpha_id(i);
            if (cur_alpha_id != prev_alpha_id) {
                precompute_J_alpha<StorageType>(ai, eri_data, n_orb, J_alpha.data());
                prev_alpha_id = cur_alpha_id;
            }

            ctx.ab_index.for_each_connection(i,
                [&](size_t j, ExcType exc_type,
                    const StorageType& alpha_j, const StorageType& beta_j) {
                double H_ij;
                if (exc_type == SAME_ALPHA_S) {
                    H_ij = compute_H_ij_same_alpha_s_dressed<StorageType>(
                        bi, beta_j, ctx.h1, eri_data, J_alpha.data(), n_orb);
                } else {
                    H_ij = compute_H_ij_strings<StorageType>(
                        ai, bi, alpha_j, beta_j, exc_type,
                        ctx.h1, eri_data, n_orb);
                }

                sigma_i += H_ij * v[j];
            });

            sigma[i] = sigma_i;
        }
    }
}

// ============================================================================
// matvec_row_sliced: compute sigma[row_start:row_end) for distributed computation.
//
// Identical to matvec_dressed_full but restricted to rows [row_start, row_end).
// Output sigma has size (row_end - row_start), indexed from 0.
// Input v is the full coefficient vector of size N.
//
// Each row is independent → perfect for distributing across workers.
// Worker k computes: sigma_k = H[row_start_k:row_end_k, :] @ v
// Coordinator concatenates: sigma = [sigma_0, sigma_1, ..., sigma_{K-1}]
// ============================================================================
template<typename StorageType>
void matvec_row_sliced(const MatvecContext<StorageType>& ctx,
                       const double* v,
                       double* sigma,
                       size_t N,
                       size_t row_start,
                       size_t row_end)
{
    const int n_orb = ctx.n_orb;
    const double* eri_data = ctx.eri.data();

    if (row_end > N) row_end = N;
    if (row_start >= row_end) return;

    #pragma omp parallel
    {
        std::vector<double> J_alpha(n_orb * n_orb, 0.0);
        uint32_t prev_alpha_id = UINT32_MAX;

        #pragma omp for schedule(dynamic, 64)
        for (int i = row_start; i < static_cast<int>(row_end); ++i) {
            double sigma_i = ctx.diag[i] * v[i];

            const auto& ai = ctx.dets[i].alpha;
            const auto& bi = ctx.dets[i].beta;

            uint32_t cur_alpha_id = ctx.ab_index.det_alpha_id(i);
            if (cur_alpha_id != prev_alpha_id) {
                precompute_J_alpha<StorageType>(ai, eri_data, n_orb, J_alpha.data());
                prev_alpha_id = cur_alpha_id;
            }

            ctx.ab_index.for_each_connection(i,
                [&](size_t j, ExcType exc_type,
                    const StorageType& alpha_j, const StorageType& beta_j) {
                double H_ij;
                if (exc_type == SAME_ALPHA_S) {
                    H_ij = compute_H_ij_same_alpha_s_dressed<StorageType>(
                        bi, beta_j, ctx.h1, eri_data, J_alpha.data(), n_orb);
                } else {
                    H_ij = compute_H_ij_strings<StorageType>(
                        ai, bi, alpha_j, beta_j, exc_type,
                        ctx.h1, eri_data, n_orb);
                }

                sigma_i += H_ij * v[j];
            });

            sigma[i - row_start] = sigma_i;
        }
    }
}

// ============================================================================
// matvec_dual_order: split alpha pass (Ch1+Ch2) + beta pass (Ch3)
//
// Alpha pass: row-only accumulation using for_each_connection_alpha_mixed.
//   v[j] access benefits from Morton ordering (O(sqrt(N)) locality).
//   J_alpha table shared within alpha groups as in dressed_full.
//
// Beta pass: gather v → v_beta (beta-sorted), then scan within beta groups.
//   v_beta[pos] is sequential within groups → eliminates cache misses.
//   Uses SIMD scan on beta_alpha_strings for single/double detection.
//
// Memory: 2 × N × 8 bytes (v_beta + sigma_beta).
// ============================================================================
template<typename StorageType>
void matvec_dual_order(const MatvecContext<StorageType>& ctx,
                       const double* v,
                       double* sigma,
                       size_t N)
{
    const int n_orb = ctx.n_orb;
    const double* eri_data = ctx.eri.data();
    const auto& ab = ctx.ab_index;

    // ---- Alpha pass: diagonal + Channel 1 + Channel 2 ----
    #pragma omp parallel
    {
        std::vector<double> J_alpha(n_orb * n_orb, 0.0);
        uint32_t prev_alpha_id = UINT32_MAX;

        #pragma omp for schedule(dynamic, 64)
        for (int i = 0; i < static_cast<int>(N); ++i) {
            double sigma_i = ctx.diag[i] * v[i];

            const auto& ai = ctx.dets[i].alpha;
            const auto& bi = ctx.dets[i].beta;

            uint32_t cur_alpha_id = ab.det_alpha_id(i);
            if (cur_alpha_id != prev_alpha_id) {
                precompute_J_alpha<StorageType>(ai, eri_data, n_orb, J_alpha.data());
                prev_alpha_id = cur_alpha_id;
            }

            ab.for_each_connection_alpha_mixed(i,
                [&](size_t j, ExcType exc_type,
                    const StorageType& alpha_j, const StorageType& beta_j) {
                double H_ij;
                if (exc_type == SAME_ALPHA_S) {
                    H_ij = compute_H_ij_same_alpha_s_dressed<StorageType>(
                        bi, beta_j, ctx.h1, eri_data, J_alpha.data(), n_orb);
                } else {
                    H_ij = compute_H_ij_strings<StorageType>(
                        ai, bi, alpha_j, beta_j, exc_type,
                        ctx.h1, eri_data, n_orb);
                }
                sigma_i += H_ij * v[j];
            });

            sigma[i] = sigma_i;
        }
    }

    // ---- Gather: v → v_beta (beta-CSR order) ----
    std::vector<double> v_beta(N);
    {
        const uint32_t* beta_flat = ab.beta_det_flat_data();
        #pragma omp parallel for schedule(static)
        for (int pos = 0; pos < static_cast<int>(N); ++pos) {
            v_beta[pos] = v[beta_flat[pos]];
        }
    }

    // ---- Beta pass: Channel 3 (same-beta) in beta-sorted order ----
    std::vector<double> sigma_beta(N, 0.0);
    {
        const uint32_t n_beta = static_cast<uint32_t>(ab.n_unique_beta());
        const StorageType* alpha_strings = ab.beta_alpha_strings_data();
        const uint32_t* beta_flat = ab.beta_det_flat_data();

        #pragma omp parallel for schedule(dynamic, 4)
        for (int b = 0; b < static_cast<int>(n_beta); ++b) {
            const size_t begin = ab.beta_group_begin(b);
            const size_t end   = ab.beta_group_end(b);
            const size_t group_size = end - begin;
            if (group_size <= 1) continue;

            const StorageType& beta_b = ab.unique_beta(b);
            const StorageType* group_alpha = alpha_strings + begin;
            const double* group_v = v_beta.data() + begin;

            for (size_t li = 0; li < group_size; ++li) {
                const StorageType& alpha_i = group_alpha[li];
                double row_sum = 0.0;

                simd::scan_popcount_sd(
                    group_alpha,
                    group_size,
                    alpha_i,
                    [&](size_t lj) {
                        // single excitation (Δα=1)
                        if (lj != li) {
                            double H_ij = compute_H_ij_strings<StorageType>(
                                alpha_i, beta_b, group_alpha[lj], beta_b,
                                SAME_BETA_S, ctx.h1, eri_data, n_orb);
                            row_sum += H_ij * group_v[lj];
                        }
                    },
                    [&](size_t lj) {
                        // double excitation (Δα=2)
                        if (lj != li) {
                            double H_ij = compute_H_ij_strings<StorageType>(
                                alpha_i, beta_b, group_alpha[lj], beta_b,
                                SAME_BETA_D, ctx.h1, eri_data, n_orb);
                            row_sum += H_ij * group_v[lj];
                        }
                    }
                );

                sigma_beta[begin + li] += row_sum;
            }
        }
    }

    // ---- Scatter: sigma += sigma_beta (permuted back to original order) ----
    {
        const uint32_t* beta_flat = ab.beta_det_flat_data();
        #pragma omp parallel for schedule(static)
        for (int pos = 0; pos < static_cast<int>(N); ++pos) {
            // Each pos maps to exactly one original det_id → no race
            sigma[beta_flat[pos]] += sigma_beta[pos];
        }
    }
}

template void matvec_dual_order<uint64_t>(
    const MatvecContext<uint64_t>&,
    const double*, double*, size_t);

// ============================================================================
// Explicit template instantiations
// ============================================================================
template std::vector<double> compute_diagonals<uint64_t>(
    const std::vector<DeterminantT<uint64_t>>&,
    const std::vector<std::vector<double>>&,
    const std::vector<double>&, int);

template std::vector<double> compute_diagonals<std::array<uint64_t, 2>>(
    const std::vector<DeterminantT<std::array<uint64_t, 2>>>&,
    const std::vector<std::vector<double>>&,
    const std::vector<double>&, int);

template double compute_H_ij_strings<uint64_t>(
    const uint64_t&, const uint64_t&,
    const uint64_t&, const uint64_t&,
    ExcType, const std::vector<std::vector<double>>&,
    const double*, int);

template double compute_H_ij_by_type<uint64_t>(
    const DeterminantT<uint64_t>&, const DeterminantT<uint64_t>&,
    ExcType, const std::vector<std::vector<double>>&,
    const std::vector<double>&, int);

template void matvec<uint64_t>(
    const MatvecContext<uint64_t>&, const double*, double*, size_t);

template ConnectionCache<uint64_t> build_connection_cache<uint64_t>(
    const ABIndex<uint64_t>&,
    const std::vector<DeterminantT<uint64_t>>&,
    const std::vector<std::vector<double>>&,
    const std::vector<double>&, int);

template void matvec_cached<uint64_t>(
    const ConnectionCache<uint64_t>&,
    const std::vector<double>&,
    const double*, double*, size_t);

template void matvec_dressed<uint64_t>(
    const MatvecContext<uint64_t>&,
    const double*, double*, size_t);

template void matvec_dressed_full<uint64_t>(
    const MatvecContext<uint64_t>&,
    const double*, double*, size_t);

template void matvec_row_sliced<uint64_t>(
    const MatvecContext<uint64_t>&,
    const double*, double*, size_t, size_t, size_t);

template void precompute_J_alpha<uint64_t>(
    const uint64_t&, const double*, int, double*);

template double compute_H_ij_same_alpha_s_dressed<uint64_t>(
    const uint64_t&, const uint64_t&,
    const std::vector<std::vector<double>>&,
    const double*, const double*, int);

// ============================================================================
// matvec_dressed_full_block: batch matvec sharing H_ij across n_vecs vectors
//
// For each connection (i, j), compute H_ij once and apply to all vectors:
//   sigma_b[i] += H_ij * v_b[j]  for b = 0..n_vecs-1
//
// Amortizes ABIndex traversal + H_ij computation over n_vecs vectors.
// Used by Block Davidson to reduce effective matvec cost per trial vector.
// ============================================================================
template<typename StorageType>
void matvec_dressed_full_block(const MatvecContext<StorageType>& ctx,
                               const double* const* v_batch,
                               double** sigma_batch,
                               size_t N,
                               int n_vecs)
{
    if (n_vecs == 1) {
        matvec_dressed_full(ctx, v_batch[0], sigma_batch[0], N);
        return;
    }

    const int n_orb = ctx.n_orb;
    const double* eri_data = ctx.eri.data();

    #pragma omp parallel
    {
        std::vector<double> J_alpha(n_orb * n_orb, 0.0);
        std::vector<double> sigma_i_buf(n_vecs);
        uint32_t prev_alpha_id = UINT32_MAX;

        #pragma omp for schedule(dynamic, 64)
        for (int i = 0; i < static_cast<int>(N); ++i) {
            // Initialize per-row accumulators for all vectors
            for (int b = 0; b < n_vecs; ++b)
                sigma_i_buf[b] = ctx.diag[i] * v_batch[b][i];

            const auto& ai = ctx.dets[i].alpha;
            const auto& bi = ctx.dets[i].beta;

            uint32_t cur_alpha_id = ctx.ab_index.det_alpha_id(i);
            if (cur_alpha_id != prev_alpha_id) {
                precompute_J_alpha<StorageType>(ai, eri_data, n_orb, J_alpha.data());
                prev_alpha_id = cur_alpha_id;
            }

            ctx.ab_index.for_each_connection(i,
                [&](size_t j, ExcType exc_type,
                    const StorageType& alpha_j, const StorageType& beta_j) {
                double H_ij;
                if (exc_type == SAME_ALPHA_S) {
                    H_ij = compute_H_ij_same_alpha_s_dressed<StorageType>(
                        bi, beta_j, ctx.h1, eri_data, J_alpha.data(), n_orb);
                } else {
                    H_ij = compute_H_ij_strings<StorageType>(
                        ai, bi, alpha_j, beta_j, exc_type,
                        ctx.h1, eri_data, n_orb);
                }

                // Apply shared H_ij to all vectors in batch
                for (int b = 0; b < n_vecs; ++b) {
                    sigma_i_buf[b] += H_ij * v_batch[b][j];
                }
            });

            for (int b = 0; b < n_vecs; ++b)
                sigma_batch[b][i] = sigma_i_buf[b];
        }
    }
}

template void matvec_dressed_full_block<uint64_t>(
    const MatvecContext<uint64_t>&,
    const double* const*, double**, size_t, int);

// 128-bit instantiations
using Bit128 = std::array<uint64_t, 2>;

template void matvec_dual_order<Bit128>(
    const MatvecContext<Bit128>&,
    const double*, double*, size_t);

template double compute_H_ij_strings<Bit128>(
    const Bit128&, const Bit128&,
    const Bit128&, const Bit128&,
    ExcType, const std::vector<std::vector<double>>&,
    const double*, int);

template double compute_H_ij_by_type<Bit128>(
    const DeterminantT<Bit128>&, const DeterminantT<Bit128>&,
    ExcType, const std::vector<std::vector<double>>&,
    const std::vector<double>&, int);

template void matvec<Bit128>(
    const MatvecContext<Bit128>&, const double*, double*, size_t);

template ConnectionCache<Bit128> build_connection_cache<Bit128>(
    const ABIndex<Bit128>&,
    const std::vector<DeterminantT<Bit128>>&,
    const std::vector<std::vector<double>>&,
    const std::vector<double>&, int);

template void matvec_cached<Bit128>(
    const ConnectionCache<Bit128>&,
    const std::vector<double>&,
    const double*, double*, size_t);

template void matvec_dressed<Bit128>(
    const MatvecContext<Bit128>&,
    const double*, double*, size_t);

template void matvec_dressed_full<Bit128>(
    const MatvecContext<Bit128>&,
    const double*, double*, size_t);

template void matvec_row_sliced<Bit128>(
    const MatvecContext<Bit128>&,
    const double*, double*, size_t, size_t, size_t);

template void precompute_J_alpha<Bit128>(
    const Bit128&, const double*, int, double*);

template double compute_H_ij_same_alpha_s_dressed<Bit128>(
    const Bit128&, const Bit128&,
    const std::vector<std::vector<double>>&,
    const double*, const double*, int);

template void matvec_dressed_full_block<Bit128>(
    const MatvecContext<Bit128>&,
    const double* const*, double**, size_t, int);

}  // namespace fe
}  // namespace trimci_core
