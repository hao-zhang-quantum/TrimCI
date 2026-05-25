#include "sparse_update_davidson.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>

#ifdef _OPENMP
#include "omp_compat.hpp"
#endif

namespace trimci_core {
namespace fe {

// ============================================================================
// BLAS-like helpers
// ============================================================================
namespace {

double dot(const double* a, const double* b, size_t N) {
    double s = 0.0;
    #pragma omp parallel for reduction(+:s) schedule(static)
    for (int i = 0; i < static_cast<int>(N); ++i) s += a[i] * b[i];
    return s;
}

void axpy(double alpha, const double* x, double* y, size_t N) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(N); ++i) y[i] += alpha * x[i];
}

double vec_norm(const double* x, size_t N) {
    return std::sqrt(dot(x, x, N));
}

double normalize(double* x, size_t N) {
    double n = vec_norm(x, N);
    if (n > 1e-15) {
        double inv = 1.0 / n;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < static_cast<int>(N); ++i) x[i] *= inv;
    }
    return n;
}

} // anonymous namespace

// ============================================================================
// sparse_matvec: compute delta_sigma = H * delta_c for sparse delta_c
//
// Column-scatter: for each active det i, enumerate all connections j
// and scatter H_ij * dc[i] to delta_sigma[j] using atomic writes.
//
// NOTE (exp09d v3 benchmark, Job 811234): alpha-group + J_alpha precomputation
// was tested but found to be 4x SLOWER at K/N=5% (mv: 16.2s vs 3.79s).
// Root cause: precompute_J_alpha reads n_occ×n_orb² = 34,992 ERI entries per
// alpha group. At K/N=5%, each group averages ~0.8 active dets; the
// precomputation cost (34,992 ERI reads) dominates the savings from J_alpha
// table lookups (saves ~27 ERI reads per SAME_ALPHA_S connection × ~243 conns
// = 6,561 reads). Net: 28K MORE ERI reads per active det. Not worth it.
// J_alpha pays off only when G >> 5 active dets per group (K/N >> 0.5).
// ============================================================================
template<typename StorageType>
static void sparse_matvec(
    const MatvecContext<StorageType>& ctx,
    const uint32_t* active_ids, size_t K,
    const double* delta_c,
    double* delta_sigma,
    size_t N)
{
    const double* eri_data = ctx.eri.data();
    const int n_orb = ctx.n_orb;

    std::memset(delta_sigma, 0, N * sizeof(double));

    #pragma omp parallel for schedule(dynamic, 1)
    for (int k = 0; k < static_cast<int>(K); ++k) {
        const uint32_t i = active_ids[k];
        const double dc_i = delta_c[i];
        if (std::abs(dc_i) < 1e-300) continue;

        const auto& ai = ctx.dets[i].alpha;
        const auto& bi = ctx.dets[i].beta;

        // Diagonal
        {
            double contrib = ctx.diag[i] * dc_i;
            #pragma omp atomic
            delta_sigma[i] += contrib;
        }

        // Off-diagonal: scatter H_ij * dc_i to all connected j
        ctx.ab_index.for_each_connection(i,
            [&](size_t j, ExcType exc_type,
                const StorageType& alpha_j, const StorageType& beta_j) {
            double H_ij = compute_H_ij_strings<StorageType>(
                ai, bi, alpha_j, beta_j, exc_type,
                ctx.h1, eri_data, n_orb);
            double contrib = H_ij * dc_i;
            #pragma omp atomic
            delta_sigma[j] += contrib;
        });
    }
}

// ============================================================================
// sparse_block_matvec: compute delta_sigma_b = H * delta_c_b for b=0..n_vecs-1
//
// Block sparse matvec that shares H_ij computation across multiple trial vectors.
// For each source det i in active_union: enumerate connections ONCE, compute H_ij
// ONCE, scatter H_ij * dc_b[i] to all n_vecs output vectors simultaneously.
//
// Amortization benefit: if active sets of the B trial vectors have high overlap
// (typical when generated from the same residual via multi-shift Jacobi), the
// union |active_union| << B*K, reducing total H_ij computations by factor B.
//
// Expected overlap: at K/N=5%, multi-shift trial vectors share ~80% of active dets
// (same residual, similar |t_b[i]| ordering), yielding K_union ≈ 1.6K for B=4.
// H_ij savings: B*K / K_union = 4 / 1.6 = 2.5×.
// ============================================================================
template<typename StorageType>
static void sparse_block_matvec(
    const MatvecContext<StorageType>& ctx,
    const uint32_t* active_union, size_t K_union,
    const double* const* delta_c_batch,   // [n_vecs] pointers, each length N
    double** delta_sigma_batch,            // [n_vecs] output pointers, each length N
    size_t N,
    int n_vecs)
{
    const double* eri_data = ctx.eri.data();
    const int n_orb = ctx.n_orb;

    for (int b = 0; b < n_vecs; ++b)
        std::memset(delta_sigma_batch[b], 0, N * sizeof(double));

    #pragma omp parallel for schedule(dynamic, 1)
    for (int k = 0; k < static_cast<int>(K_union); ++k) {
        const uint32_t i = active_union[k];
        const auto& ai = ctx.dets[i].alpha;
        const auto& bi = ctx.dets[i].beta;

        // Diagonal: scatter ctx.diag[i] * dc_b[i] for each vector b
        for (int b = 0; b < n_vecs; ++b) {
            double dc = delta_c_batch[b][i];
            if (std::abs(dc) < 1e-300) continue;
            double contrib = ctx.diag[i] * dc;
            #pragma omp atomic
            delta_sigma_batch[b][i] += contrib;
        }

        // Off-diagonal: enumerate connections ONCE, H_ij computed ONCE,
        // scatter H_ij * dc_b[i] to all n_vecs output vectors.
        ctx.ab_index.for_each_connection(i,
            [&](size_t j, ExcType exc_type,
                const StorageType& alpha_j, const StorageType& beta_j) {
            double H_ij = compute_H_ij_strings<StorageType>(
                ai, bi, alpha_j, beta_j, exc_type,
                ctx.h1, eri_data, n_orb);
            for (int b = 0; b < n_vecs; ++b) {
                double dc = delta_c_batch[b][i];
                if (std::abs(dc) < 1e-300) continue;
                #pragma omp atomic
                delta_sigma_batch[b][j] += H_ij * dc;
            }
        });
    }
}

// ============================================================================
// select_top_K: find indices of K largest |v[i]| components
// Returns active indices and the capture ratio (fraction of ||v||² kept).
// ============================================================================
static std::vector<uint32_t> select_top_K(
    const double* v, size_t N, size_t K, double* capture_ratio_out = nullptr)
{
    if (K >= N) {
        std::vector<uint32_t> all(N);
        std::iota(all.begin(), all.end(), 0);
        if (capture_ratio_out) *capture_ratio_out = 1.0;
        return all;
    }

    std::vector<size_t> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    std::nth_element(idx.begin(), idx.begin() + K, idx.end(),
        [v](size_t a, size_t b) { return std::abs(v[a]) > std::abs(v[b]); });

    std::vector<uint32_t> active(K);
    double active_norm2 = 0.0, total_norm2 = 0.0;
    for (size_t i = 0; i < K; ++i) {
        active[i] = static_cast<uint32_t>(idx[i]);
        active_norm2 += v[idx[i]] * v[idx[i]];
    }
    if (capture_ratio_out) {
        total_norm2 = dot(v, v, N);
        *capture_ratio_out = (total_norm2 > 1e-30) ? active_norm2 / total_norm2 : 1.0;
    }
    return active;
}

// ============================================================================
// Sparse-Davidson solver
//
// True Davidson with Krylov subspace acceleration + hybrid full/sparse.
//
// Key optimizations vs v1:
//   1. Incremental H_small: only compute new row/column each iter (O(k*N))
//   2. Buffer reuse: all N-dimensional work vectors allocated once
//   3. Hybrid: first n_warm_iters use full_matvec for better convergence
//   4. Adaptive switch: auto-detect when truncation captures enough of ||t||²
// ============================================================================
// Get K for a given step from K_schedule.
// step 0 = initial H*c0, step i >= 1 = iteration (i-1).
// Returns 0 when K_schedule is empty (signal to use old ratio-based logic).
inline size_t get_K_for_step(const SparseUpdateParams& params, int step, size_t N) {
    if (params.K_schedule.empty()) return 0;
    size_t idx = static_cast<size_t>(step);
    size_t K = (idx < params.K_schedule.size()) ? params.K_schedule[idx] : params.K_schedule.back();
    K = std::max(K, static_cast<size_t>(params.min_active_dets));
    return std::min(K, N);
}

template<typename StorageType>
DavidsonResult sparse_update_solve(
    const MatvecContext<StorageType>& ctx,
    std::function<void(const double*, double*, size_t)> full_matvec,
    size_t N,
    const SparseUpdateParams& params,
    const std::vector<std::vector<double>>& initial_guess)
{
    using Clock = std::chrono::high_resolution_clock;
    const auto& diag = ctx.diag;
    const int M = params.max_subspace;
    const bool use_K_schedule = !params.K_schedule.empty();

    DavidsonResult result;
    result.eigenvalues.resize(1, 0.0);
    result.eigenvectors.resize(1, std::vector<double>(N, 0.0));

    // Compute K for cruise mode (sparse phase after warm-up)
    // (used only when K_schedule is empty — backward compatible)
    size_t K_cruise = std::max(static_cast<size_t>(params.min_active_dets),
                               static_cast<size_t>(N * params.truncation_ratio));
    if (params.max_active_dets > 0)
        K_cruise = std::min(K_cruise, static_cast<size_t>(params.max_active_dets));
    K_cruise = std::min(K_cruise, N);

    // K for warm phase: larger ratio for global coverage, or full matvec
    size_t K_warm = (params.warm_ratio > 0.0)
        ? std::max(K_cruise, std::min(N, static_cast<size_t>(N * params.warm_ratio)))
        : N;  // warm_ratio=0 → full matvec (K_warm=N signals full)

    // K for random sampling supplement
    size_t K_rand = (params.random_sample_ratio > 0.0)
        ? std::max(static_cast<size_t>(1), static_cast<size_t>(N * params.random_sample_ratio))
        : 0;

    size_t K = use_K_schedule ? get_K_for_step(params, 1, N) : K_cruise;

    if (params.verbose >= 1) {
        if (use_K_schedule) {
            std::cout << "  Sparse-Davidson: N=" << N << ", K_schedule=[";
            for (size_t i = 0; i < std::min(params.K_schedule.size(), size_t(6)); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << params.K_schedule[i];
            }
            if (params.K_schedule.size() > 6) std::cout << ", ...";
            std::cout << "] (last repeats)";
        } else {
            std::cout << "  Sparse-Davidson: N=" << N
                      << ", K=" << K_cruise
                      << " (ratio=" << std::fixed << std::setprecision(2)
                      << params.truncation_ratio
                      << ", warm=" << params.n_warm_iters;
            if (params.warm_ratio > 0.0)
                std::cout << " @" << std::setprecision(2) << params.warm_ratio;
            if (K_rand > 0)
                std::cout << ", rand=" << K_rand;
            std::cout << ")";
        }
        if (params.ucb_kappa > 0.0)
            std::cout << ", ucb_k=" << std::setprecision(2) << params.ucb_kappa;
        std::cout << std::endl;
    }

    // ---- Pre-allocate all work buffers (avoid per-iter allocation) ----
    std::vector<double> ritz_vec(N, 0.0);
    std::vector<double> ritz_Hvec(N, 0.0);
    std::vector<double> r(N);
    std::vector<double> t(N);
    std::vector<double> Ht_work(N);    // for sparse_matvec result or full_matvec
    std::vector<double> pt2_score;     // for truncation_mode == 2: |r_j * t_j|
    if (params.truncation_mode == 2) pt2_score.resize(N);
    std::vector<double> ucb_score;    // for UCB exploration bonus
    std::vector<uint16_t> n_active_count;  // per-det active set membership count
    if (params.ucb_kappa > 0.0) {
        ucb_score.resize(N);
        n_active_count.assign(N, 0);
    }
    std::vector<double> proj_coeffs(M, 0.0);
    std::vector<double> ritz_prev;  // for momentum: previous Ritz vector
    if (params.momentum_beta > 0.0) ritz_prev.resize(N, 0.0);

    // RNG for random sampling
    std::mt19937 rng(42);

    // Persistent H_small buffer (M × M), incrementally updated
    std::vector<double> H_small(M * M, 0.0);

    // ---- Subspace initialization ----
    std::vector<std::vector<double>> V;
    std::vector<std::vector<double>> HV;
    V.reserve(M);
    HV.reserve(M);

    std::vector<double> c0(N, 0.0);
    if (!initial_guess.empty()) {
        c0 = initial_guess[0];
    } else {
        size_t min_idx = std::distance(diag.begin(),
            std::min_element(diag.begin(), diag.end()));
        c0[min_idx] = 1.0;
    }
    normalize(c0.data(), N);
    V.push_back(c0);

    // Initial matvec for V[0]: full or sparse depending on K_schedule
    int n_full_mv = 0, n_sparse_mv = 0;
    {
        size_t K_init = get_K_for_step(params, 0, N);
        auto t0 = Clock::now();
        std::vector<double> Hc0(N, 0.0);
        if (K_init == 0 || K_init >= N) {
            // Full matvec (original behavior, or K_schedule says full)
            full_matvec(c0.data(), Hc0.data(), N);
            n_full_mv = 1;
        } else {
            // Sparse initial matvec: select top-K_init by |c0_j|
            auto active_init = select_top_K(c0.data(), N, K_init);
            sparse_matvec(ctx, active_init.data(), active_init.size(),
                         c0.data(), Hc0.data(), N);
            n_sparse_mv = 1;
        }
        HV.push_back(std::move(Hc0));
        double t_init = std::chrono::duration<double>(Clock::now() - t0).count();
        if (params.verbose >= 2) {
            std::cout << "  Initial " << (n_full_mv ? "full" : "sparse") << " matvec";
            if (n_sparse_mv) std::cout << " (K=" << K_init << ")";
            std::cout << ": " << std::fixed << std::setprecision(2) << t_init << "s" << std::endl;
        }
    }

    // Initialize H_small[0,0]
    H_small[0] = dot(V[0].data(), HV[0].data(), N);

    int k = 1;  // current subspace size
    double prev_E = 0.0;
    int energy_conv_count = 0;
    const int energy_conv_required = 3;

    // Timing accumulators
    double time_proj = 0, time_eig = 0, time_ritz = 0;
    double time_resid = 0, time_trunc = 0, time_mv = 0;
    double time_ortho = 0, time_htcorr = 0;

    for (int iter = 0; iter < params.max_iter; ++iter) {

        // ---- Jacobi eigenvalue decomposition of H_small[0:k, 0:k] ----
        auto t_eig_start = Clock::now();
        std::vector<double> evals(k);
        std::vector<double> evecs(k * k, 0.0);
        for (int i = 0; i < k; ++i) evecs[i * k + i] = 1.0;
        {
            // Copy the active k×k block from H_small (which uses stride M)
            std::vector<double> A(k * k);
            for (int i = 0; i < k; ++i)
                for (int j = 0; j < k; ++j)
                    A[i * k + j] = H_small[i * M + j];

            const int max_sweeps = 100;
            for (int sweep = 0; sweep < max_sweeps; ++sweep) {
                double off_diag = 0.0;
                for (int i = 0; i < k; ++i)
                    for (int j = i + 1; j < k; ++j)
                        off_diag += A[i * k + j] * A[i * k + j];
                if (off_diag < 1e-30) break;

                for (int p = 0; p < k; ++p) {
                    for (int q = p + 1; q < k; ++q) {
                        double apq = A[p * k + q];
                        if (std::abs(apq) < 1e-15) continue;
                        double d = A[q * k + q] - A[p * k + p];
                        double tau_val;
                        if (std::abs(d) < 1e-15) {
                            tau_val = 1.0;
                        } else {
                            double tau = d / (2.0 * apq);
                            tau_val = 1.0 / (std::abs(tau) + std::sqrt(1.0 + tau * tau));
                            if (tau < 0) tau_val = -tau_val;
                        }
                        double cos_t = 1.0 / std::sqrt(1.0 + tau_val * tau_val);
                        double sin_t = tau_val * cos_t;

                        double app = A[p * k + p], aqq = A[q * k + q];
                        A[p * k + p] = app - tau_val * apq;
                        A[q * k + q] = aqq + tau_val * apq;
                        A[p * k + q] = 0.0;
                        A[q * k + p] = 0.0;
                        for (int rr = 0; rr < k; ++rr) {
                            if (rr == p || rr == q) continue;
                            double arp = A[rr * k + p], arq = A[rr * k + q];
                            A[rr * k + p] = cos_t * arp - sin_t * arq;
                            A[p * k + rr] = A[rr * k + p];
                            A[rr * k + q] = sin_t * arp + cos_t * arq;
                            A[q * k + rr] = A[rr * k + q];
                        }
                        for (int rr = 0; rr < k; ++rr) {
                            double erp = evecs[rr * k + p], erq = evecs[rr * k + q];
                            evecs[rr * k + p] = cos_t * erp - sin_t * erq;
                            evecs[rr * k + q] = sin_t * erp + cos_t * erq;
                        }
                    }
                }
            }
            for (int i = 0; i < k; ++i) evals[i] = A[i * k + i];
        }
        time_eig += std::chrono::duration<double>(Clock::now() - t_eig_start).count();

        // ---- Find smallest eigenvalue ----
        int min_idx = 0;
        for (int i = 1; i < k; ++i) {
            if (evals[i] < evals[min_idx]) min_idx = i;
        }
        double theta = evals[min_idx];
        result.eigenvalues[0] = theta;

        // ---- Ritz vector: c = V*y, Hc = HV*y ----
        auto t_ritz_start = Clock::now();
        std::fill(ritz_vec.begin(), ritz_vec.end(), 0.0);
        std::fill(ritz_Hvec.begin(), ritz_Hvec.end(), 0.0);
        for (int j = 0; j < k; ++j) {
            double coeff = evecs[j * k + min_idx];
            axpy(coeff, V[j].data(), ritz_vec.data(), N);
            axpy(coeff, HV[j].data(), ritz_Hvec.data(), N);
        }
        time_ritz += std::chrono::duration<double>(Clock::now() - t_ritz_start).count();

        // ---- Residual: r = Hc - θ*c ----
        auto t_resid_start = Clock::now();
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < static_cast<int>(N); ++i) {
            r[i] = ritz_Hvec[i] - theta * ritz_vec[i];
        }
        double rnorm = vec_norm(r.data(), N);
        result.residual_norm = rnorm;
        result.n_iters = iter + 1;
        time_resid += std::chrono::duration<double>(Clock::now() - t_resid_start).count();

        // ---- Convergence check ----
        double dE = (iter > 0) ? std::abs(theta - prev_E) : 0.0;

        if (params.verbose >= 2) {
            std::cout << "  SD iter " << std::setw(3) << iter + 1
                      << "  E = " << std::fixed << std::setprecision(10) << theta
                      << "  |r| = " << std::scientific << std::setprecision(2) << rnorm
                      << "  k = " << k;
            if (iter > 0)
                std::cout << "  |dE| = " << std::scientific << std::setprecision(2) << dE;
            std::cout << std::endl;
        }

        bool resid_conv = (rnorm < params.residual_tol);
        bool energy_conv = false;
        if (iter > 0) {
            if (params.energy_tol > 0.0 && dE < params.energy_tol) {
                energy_conv_count++;
            } else {
                energy_conv_count = 0;
            }
            energy_conv = (energy_conv_count >= energy_conv_required);
        }
        prev_E = theta;

        if (resid_conv || energy_conv) {
            result.converged = true;
            result.eigenvectors[0] = ritz_vec;

            // Final full matvec: recompute exact variational energy E = c†·H·c
            if (params.final_full_matvec && n_sparse_mv > 0) {
                auto t_ffm = Clock::now();
                std::vector<double> Hc_full(N, 0.0);
                full_matvec(ritz_vec.data(), Hc_full.data(), N);
                double E_full = dot(ritz_vec.data(), Hc_full.data(), N);
                double E_ritz = theta;
                result.eigenvalues[0] = E_full;
                n_full_mv++;
                double t_ffm_s = std::chrono::duration<double>(Clock::now() - t_ffm).count();
                result.ffm_wall_time = t_ffm_s;
                if (params.verbose >= 1) {
                    std::cout << "  Final full matvec: E_ritz=" << std::fixed << std::setprecision(10) << E_ritz
                              << "  E_full=" << E_full
                              << "  correction=" << std::scientific << std::setprecision(2) << (E_full - E_ritz)
                              << "  time=" << std::fixed << std::setprecision(2) << t_ffm_s << "s" << std::endl;
                }
            }

            if (params.verbose >= 1) {
                std::cout << "  Sparse-Davidson converged in " << iter + 1
                          << " iterations  E = " << std::fixed << std::setprecision(10) << result.eigenvalues[0]
                          << "  |r| = " << std::scientific << std::setprecision(2) << rnorm;
                if (energy_conv && !resid_conv)
                    std::cout << "  (energy_tol: |dE|=" << dE
                              << " for " << energy_conv_required << " consecutive iters)";
                std::cout << "  [" << n_full_mv << " full + "
                          << n_sparse_mv << " sparse matvecs]" << std::endl;
            }
            if (params.verbose >= 2) {
                std::cout << "  Timers: eig=" << std::fixed << std::setprecision(3) << time_eig
                          << "  ritz=" << time_ritz << "  resid=" << time_resid
                          << "  trunc=" << time_trunc << "  mv=" << time_mv
                          << "  ortho=" << time_ortho << "  htcorr=" << time_htcorr
                          << "  proj=" << time_proj << std::endl;
            }
            return result;
        }

        // ---- Subspace restart if full ----
        if (k >= M) {
            int n_keep = std::min(params.thick_restart_size, k);
            if (n_keep <= 1) {
                // Original: keep only lowest Ritz vector
                V.resize(1);
                HV.resize(1);
                V[0] = ritz_vec;
                HV[0] = ritz_Hvec;
                H_small[0] = dot(V[0].data(), HV[0].data(), N);
                k = 1;
            } else {
                // Thick restart: keep n_keep lowest Ritz pairs
                std::vector<int> sorted_idx(k);
                std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
                std::partial_sort(sorted_idx.begin(), sorted_idx.begin() + n_keep,
                                  sorted_idx.end(),
                                  [&](int a, int b) { return evals[a] < evals[b]; });

                std::vector<std::vector<double>> V_new(n_keep, std::vector<double>(N, 0.0));
                std::vector<std::vector<double>> HV_new(n_keep, std::vector<double>(N, 0.0));
                for (int m = 0; m < n_keep; ++m) {
                    int idx = sorted_idx[m];
                    for (int j = 0; j < k; ++j) {
                        double coeff = evecs[j * k + idx];
                        axpy(coeff, V[j].data(), V_new[m].data(), N);
                        axpy(coeff, HV[j].data(), HV_new[m].data(), N);
                    }
                }
                V = std::move(V_new);
                HV = std::move(HV_new);
                // Rebuild H_small for kept vectors
                for (int i = 0; i < n_keep; ++i) {
                    for (int j = 0; j <= i; ++j) {
                        double val = dot(V[i].data(), HV[j].data(), N);
                        H_small[i * M + j] = val;
                        H_small[j * M + i] = val;
                    }
                }
                k = n_keep;
            }
            energy_conv_count = 0;
            if (params.verbose >= 2) {
                std::cout << "  Sparse-Davidson restart at iter " << iter + 1
                          << " (kept " << k << " vectors)" << std::endl;
            }
            continue;
        }

        // ---- Preconditioned correction: t = -r / (D - θ) ----
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < static_cast<int>(N); ++i) {
            double denom = diag[i] - theta;
            if (std::abs(denom) < 1e-4) {
                denom = (denom >= 0) ? 1e-4 : -1e-4;
            }
            t[i] = -r[i] / denom;
        }

        // ---- Olsen correction: t += ε·(D-θ)^{-1}·x ----
        // Rank-1 correction that ensures t ⊥ x in preconditioned sense.
        // ε = -x^T·t_jacobi / x^T·(D-θ)^{-1}·x
        if (params.olsen_correction) {
            double xTt = 0.0, xTq = 0.0;
            #pragma omp parallel for schedule(static) reduction(+:xTt,xTq)
            for (int i = 0; i < static_cast<int>(N); ++i) {
                double denom = diag[i] - theta;
                if (std::abs(denom) < 1e-4)
                    denom = (denom >= 0) ? 1e-4 : -1e-4;
                xTt += ritz_vec[i] * t[i];
                xTq += ritz_vec[i] * ritz_vec[i] / denom;
            }
            if (std::abs(xTq) > 1e-30) {
                double eps = -xTt / xTq;
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < static_cast<int>(N); ++i) {
                    double denom = diag[i] - theta;
                    if (std::abs(denom) < 1e-4)
                        denom = (denom >= 0) ? 1e-4 : -1e-4;
                    t[i] += eps * ritz_vec[i] / denom;
                }
            }
        }

        // ---- Momentum: t += β·(ritz_curr - ritz_prev) ----
        if (params.momentum_beta > 0.0 && iter > 0) {
            double beta = params.momentum_beta;
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < static_cast<int>(N); ++i) {
                t[i] += beta * (ritz_vec[i] - ritz_prev[i]);
            }
        }
        if (params.momentum_beta > 0.0) {
            std::copy(ritz_vec.begin(), ritz_vec.end(), ritz_prev.begin());
        }

        // ---- Decide: full or sparse, and set K for this iteration ----
        bool use_sparse_this_iter;
        size_t K_sched = get_K_for_step(params, iter + 1, N);  // +1: step 0 is initial
        if (K_sched > 0) {
            // K_schedule mode: per-iteration K from schedule
            K = K_sched;
            use_sparse_this_iter = (K < N);
        } else {
            // Original warm/cruise logic (backward compatible)
            if (iter < params.n_warm_iters) {
                if (params.warm_ratio > 0.0) {
                    use_sparse_this_iter = true;
                    K = K_warm;
                } else {
                    use_sparse_this_iter = false;
                }
            } else {
                use_sparse_this_iter = true;
                K = K_cruise;
            }
        }

        // ---- BLOCK SPARSE PATH (block_size > 1, sparse phase) ----
        // Algorithm: truncate → block sparse matvec → per-vector MGS + analytical H correction
        //
        // Key invariant: V[0..k-1] must remain orthonormal (same as single-vector path).
        // The trick: do matvec FIRST on truncated vectors, then orthogonalize analytically
        // (identical to single-vector path) so that HV[k] = H * V[k]_orthonormal is correct.
        // Amortization: all B truncated vectors share one sparse_block_matvec call,
        // with H_ij computed once per source det × connection (across all B vectors).
        if (params.block_size > 1 && use_sparse_this_iter && K < N) {
            int max_new = std::min(params.block_size, M - k);
            if (max_new <= 0) continue;

            // Spectral shift range for multi-shift Jacobi
            double diag_mean = 0.0;
            for (size_t i = 0; i < N; ++i) diag_mean += diag[i];
            diag_mean /= static_cast<double>(N);
            double gap = diag_mean - theta;
            if (gap < 0.1) gap = 0.1;
            double shift_step = (max_new > 1) ? gap / (2.0 * max_new) : 0.0;

            // ---- Step 1: Generate and truncate trial vectors ----
            auto t_block_trunc = Clock::now();
            std::vector<std::vector<double>> t_batch;   // truncated (sparse) trial vectors
            std::vector<std::vector<double>> Ht_batch;  // H * t_batch[b] (from block matvec)
            std::vector<uint32_t> active_union;

            for (int b = 0; b < max_new; ++b) {
                double shift = b * shift_step;
                std::vector<double> t_b(N);
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < static_cast<int>(N); ++i) {
                    double denom = diag[i] - theta - shift;
                    if (std::abs(denom) < 1e-4)
                        denom = (denom >= 0) ? 1e-4 : -1e-4;
                    t_b[i] = -r[i] / denom;
                }
                // Truncate to top-K: select by mode 0:|t_j|, 1:|r_j|, 2:|r_j*t_j|
                const double* sel_b;
                std::vector<double> pt2_b;
                if (params.truncation_mode == 2) {
                    pt2_b.resize(N);
                    #pragma omp parallel for schedule(static)
                    for (int i = 0; i < static_cast<int>(N); ++i)
                        pt2_b[i] = std::abs(r[i]) * std::abs(t_b[i]);
                    sel_b = pt2_b.data();
                } else {
                    sel_b = (params.truncation_mode == 1) ? r.data() : t_b.data();
                }
                auto active_b = select_top_K(sel_b, N, K, nullptr);
                {
                    std::vector<double> t_vals(K);
                    for (size_t i = 0; i < K; ++i) t_vals[i] = t_b[active_b[i]];
                    std::memset(t_b.data(), 0, N * sizeof(double));
                    for (size_t i = 0; i < K; ++i) t_b[active_b[i]] = t_vals[i];
                }
                for (uint32_t idx : active_b) active_union.push_back(idx);
                t_batch.push_back(std::move(t_b));
                Ht_batch.push_back(std::vector<double>(N, 0.0));
            }

            int n_cands = static_cast<int>(t_batch.size());
            // Deduplicate active_union → K_union ≤ n_cands * K
            std::sort(active_union.begin(), active_union.end());
            active_union.erase(std::unique(active_union.begin(), active_union.end()),
                               active_union.end());
            time_trunc += std::chrono::duration<double>(Clock::now() - t_block_trunc).count();

            // ---- Step 2: Block sparse matvec (H_ij shared across all candidates) ----
            auto t_mv_block = Clock::now();
            {
                std::vector<const double*> dc_ptrs(n_cands);
                std::vector<double*> dsigma_ptrs(n_cands);
                for (int b = 0; b < n_cands; ++b) {
                    dc_ptrs[b] = t_batch[b].data();
                    dsigma_ptrs[b] = Ht_batch[b].data();
                }
                sparse_block_matvec(ctx, active_union.data(), active_union.size(),
                                    dc_ptrs.data(), dsigma_ptrs.data(), N, n_cands);
            }
            double mv_elapsed = std::chrono::duration<double>(Clock::now() - t_mv_block).count();
            n_sparse_mv++;
            time_mv += mv_elapsed;

            if (params.verbose >= 2) {
                std::cout << "    block_sparse K_union=" << active_union.size()
                          << " n_new=" << n_cands
                          << "  mv=" << std::fixed << std::setprecision(3) << mv_elapsed << "s"
                          << std::endl;
            }

            // ---- Step 3: Per-vector MGS + analytical H correction + subspace update ----
            // Mirrors single-vector path exactly: for each b, orthogonalize t_b against
            // the CURRENT subspace V (which includes vectors added from b=0,1,...,b-1),
            // then apply the analytical H correction:
            //   H * V[k] = (Ht_b - sum_j proj_b[j] * HV[j]) / tnorm
            // This correctly computes HV[k] = H * V[k]_orthonormal from the sparse Ht_b.
            auto t_proj_block = Clock::now();
            std::vector<double> proj_b(M, 0.0);
            int n_added = 0;

            for (int b = 0; b < n_cands && k < M; ++b) {
                auto& t_b = t_batch[b];
                auto& Ht_b = Ht_batch[b];

                // 2-pass MGS against V[0..k-1] (k grows as b=0,1,... get added)
                std::fill(proj_b.begin(), proj_b.begin() + k, 0.0);
                for (int pass = 0; pass < 2; ++pass) {
                    for (int j = 0; j < k; ++j) {
                        double proj = dot(t_b.data(), V[j].data(), N);
                        axpy(-proj, V[j].data(), t_b.data(), N);
                        proj_b[j] += proj;
                    }
                }
                double tnorm = normalize(t_b.data(), N);
                if (tnorm < 1e-14) continue;  // linearly dependent, skip

                // Analytical H correction (same as single-vector path):
                //   H * t_orth = (H * t_sparse - sum_j proj_b[j] * H * V[j]) / tnorm
                double inv_tnorm = 1.0 / tnorm;
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < static_cast<int>(N); ++i) Ht_b[i] *= inv_tnorm;
                for (int j = 0; j < k; ++j) {
                    axpy(-proj_b[j] * inv_tnorm, HV[j].data(), Ht_b.data(), N);
                }

                // Add to subspace + H_small update
                V.push_back(std::move(t_b));
                HV.push_back(std::move(Ht_b));
                for (int j = 0; j <= k; ++j) {
                    double val = dot(V[k].data(), HV[j].data(), N);
                    H_small[k * M + j] = val;
                    H_small[j * M + k] = val;
                }
                k++;
                n_added++;
            }
            time_proj += std::chrono::duration<double>(Clock::now() - t_proj_block).count();

            if (n_added == 0) continue;
            continue;  // Skip single-vector path below
        }

        // ---- SINGLE-VECTOR PATH (original v2 code, unchanged) ----
        auto t_trunc_start = Clock::now();
        std::vector<uint32_t> active;
        double capture_ratio = 1.0;

        if (use_sparse_this_iter && K < N) {
            // Truncate to top-K: select by mode 0:|t_j|, 1:|r_j|, 2:|r_j*t_j|
            const double* select_vec;
            if (params.truncation_mode == 2) {
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < static_cast<int>(N); ++i)
                    pt2_score[i] = std::abs(r[i]) * std::abs(t[i]);
                select_vec = pt2_score.data();
            } else {
                select_vec = (params.truncation_mode == 1) ? r.data() : t.data();
            }

            // UCB exploration bonus: score_j = base_j + kappa * sqrt(ln(iter+1) / max(1, n_j))
            // Normalizes base scores to [0,1] range first so kappa is scale-independent.
            if (params.ucb_kappa > 0.0 && iter >= params.n_warm_iters) {
                double log_iter = std::log(static_cast<double>(iter + 1));
                // Find max of base score for normalization
                double base_max = 0.0;
                for (size_t i = 0; i < N; ++i) {
                    double v = std::abs(select_vec[i]);
                    if (v > base_max) base_max = v;
                }
                double inv_base_max = (base_max > 1e-30) ? 1.0 / base_max : 0.0;
                double kappa = params.ucb_kappa;
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < static_cast<int>(N); ++i) {
                    double base = std::abs(select_vec[i]) * inv_base_max;
                    double bonus = std::sqrt(log_iter / std::max(1.0, static_cast<double>(n_active_count[i])));
                    ucb_score[i] = base + kappa * bonus;
                }
                active = select_top_K(ucb_score.data(), N, K, &capture_ratio);
                // Update active counts
                for (uint32_t idx : active) {
                    if (n_active_count[idx] < 65535) n_active_count[idx]++;
                }
            } else {
                active = select_top_K(select_vec, N, K, &capture_ratio);
                // Update active counts if UCB is enabled (warm phase still tracks)
                if (params.ucb_kappa > 0.0) {
                    for (uint32_t idx : active) {
                        if (n_active_count[idx] < 65535) n_active_count[idx]++;
                    }
                }
            }

            // Random sampling supplement: add K_rand importance-sampled dets
            // Uses precomputed CDF + binary search: O(N + K_rand * log N)
            size_t n_sampled = 0;
            if (K_rand > 0 && K < N) {
                std::vector<bool> in_active(N, false);
                for (uint32_t idx : active) in_active[idx] = true;

                // Build CDF over non-active dets: O(N)
                std::vector<uint32_t> pool_idx;
                std::vector<double> cdf;
                pool_idx.reserve(N - active.size());
                cdf.reserve(N - active.size() + 1);
                double cum = 0.0;
                cdf.push_back(0.0);
                for (size_t i = 0; i < N; ++i) {
                    if (!in_active[i]) {
                        cum += select_vec[i] * select_vec[i];
                        pool_idx.push_back(static_cast<uint32_t>(i));
                        cdf.push_back(cum);
                    }
                }
                if (cum > 1e-30) {
                    std::uniform_real_distribution<double> unif(0.0, cum);
                    for (size_t s = 0; s < K_rand * 2 && n_sampled < K_rand; ++s) {
                        double target = unif(rng);
                        // Binary search: O(log N)
                        auto it = std::upper_bound(cdf.begin(), cdf.end(), target);
                        size_t ci = static_cast<size_t>(it - cdf.begin());
                        if (ci > 0) ci--;
                        if (ci < pool_idx.size() && !in_active[pool_idx[ci]]) {
                            active.push_back(pool_idx[ci]);
                            in_active[pool_idx[ci]] = true;
                            n_sampled++;
                        }
                    }
                }
            }

            // Adaptive override: if capture is too low
            if (params.capture_threshold > 0.0 && capture_ratio < params.capture_threshold) {
                if (params.adaptive_K) {
                    // Increase K to meet capture target (stay sparse with larger K)
                    size_t K_new = static_cast<size_t>(
                        K * (params.capture_threshold / capture_ratio) * 1.5);
                    K_new = std::min(K_new, N);
                    if (K_new < N) {
                        double cap_new = 0.0;
                        active = select_top_K(select_vec, N, K_new, &cap_new);
                        if (params.verbose >= 2) {
                            std::cout << "    adaptive K: " << K << " -> " << K_new
                                      << "  cap=" << std::fixed << std::setprecision(3)
                                      << capture_ratio << " -> " << cap_new << std::endl;
                        }
                        capture_ratio = cap_new;
                        // Zero non-active components
                        std::vector<double> t_vals(K_new);
                        for (size_t i = 0; i < K_new; ++i) t_vals[i] = t[active[i]];
                        std::memset(t.data(), 0, N * sizeof(double));
                        for (size_t i = 0; i < K_new; ++i) t[active[i]] = t_vals[i];
                    } else {
                        use_sparse_this_iter = false;
                    }
                } else {
                    use_sparse_this_iter = false;
                    if (params.verbose >= 2) {
                        std::cout << "    capture=" << std::fixed << std::setprecision(3)
                                  << capture_ratio << " < " << params.capture_threshold
                                  << " -> full matvec" << std::endl;
                    }
                }
            } else {
                // Normal: zero non-active, keep active (including random samples)
                size_t K_total = active.size();
                std::vector<double> t_vals(K_total);
                for (size_t i = 0; i < K_total; ++i) t_vals[i] = t[active[i]];
                std::memset(t.data(), 0, N * sizeof(double));
                for (size_t i = 0; i < K_total; ++i) t[active[i]] = t_vals[i];
            }
        }
        time_trunc += std::chrono::duration<double>(Clock::now() - t_trunc_start).count();

        // ---- Matvec (full or sparse) ----
        auto t_mv_start = Clock::now();
        if (!use_sparse_this_iter || K >= N) {
            // Full matvec — no truncation needed, t is the full preconditioned residual
            full_matvec(t.data(), Ht_work.data(), N);
            n_full_mv++;
        } else {
            sparse_matvec(ctx, active.data(), active.size(),
                         t.data(), Ht_work.data(), N);
            n_sparse_mv++;
        }
        time_mv += std::chrono::duration<double>(Clock::now() - t_mv_start).count();

        // ---- Orthogonalize t against V (2-pass Modified Gram-Schmidt) ----
        // Track projection coefficients for analytical H*t correction:
        //   H * t_orth = (H * t_pre_orth - Σ proj_j * HV_j) / ||t_orth||
        auto t_ortho_start = Clock::now();
        std::fill(proj_coeffs.begin(), proj_coeffs.begin() + k, 0.0);
        for (int pass = 0; pass < 2; ++pass) {
            for (int j = 0; j < k; ++j) {
                double proj = dot(t.data(), V[j].data(), N);
                axpy(-proj, V[j].data(), t.data(), N);
                proj_coeffs[j] += proj;
            }
        }
        double tnorm = normalize(t.data(), N);
        time_ortho += std::chrono::duration<double>(Clock::now() - t_ortho_start).count();

        if (tnorm < 1e-14) {
            if (params.verbose >= 2) {
                std::cout << "  Sparse-Davidson: trial vector linearly dependent, skipping"
                          << std::endl;
            }
            continue;
        }

        // ---- Compute H*t_orth analytically ----
        // H * t_orth = (Ht_work - Σ proj_j * HV_j) / tnorm
        auto t_htcorr_start = Clock::now();
        double inv_tnorm = 1.0 / tnorm;
        // Ht_work already contains H * t_pre_orth; scale and subtract projections
        // Reuse Ht_work in-place: Ht_work = Ht_work / tnorm - Σ (proj_j/tnorm) * HV_j
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < static_cast<int>(N); ++i) {
            Ht_work[i] *= inv_tnorm;
        }
        for (int j = 0; j < k; ++j) {
            double coeff = -proj_coeffs[j] * inv_tnorm;
            axpy(coeff, HV[j].data(), Ht_work.data(), N);
        }
        time_htcorr += std::chrono::duration<double>(Clock::now() - t_htcorr_start).count();

        if (params.verbose >= 2) {
            double mv_time = std::chrono::duration<double>(t_ortho_start - t_mv_start).count();
            std::cout << "    ";
            if (use_sparse_this_iter && K < N) {
                std::string mode_tag = (params.truncation_mode == 2 ? "(rt)" :
                                        params.truncation_mode == 1 ? "(r)" : "(t)");
                if (params.ucb_kappa > 0.0 && iter >= params.n_warm_iters)
                    mode_tag += "+ucb";
                std::cout << "K=" << active.size()
                          << "  cap=" << std::fixed << std::setprecision(3) << capture_ratio
                          << mode_tag;
            } else {
                std::cout << "full_mv";
            }
            std::cout << "  mv=" << std::fixed << std::setprecision(3) << mv_time << "s"
                      << std::endl;
        }

        // ---- Add to subspace ----
        // Move t into V, copy Ht_work into HV
        V.push_back(std::vector<double>(t.begin(), t.end()));
        HV.push_back(std::vector<double>(Ht_work.begin(), Ht_work.end()));

        // ---- Incremental H_small update: only new row/column ----
        auto t_proj_start = Clock::now();
        int new_k = k;  // index of the newly added vector
        for (int j = 0; j <= new_k; ++j) {
            double val = dot(V[new_k].data(), HV[j].data(), N);
            H_small[new_k * M + j] = val;
            H_small[j * M + new_k] = val;
        }
        time_proj += std::chrono::duration<double>(Clock::now() - t_proj_start).count();

        k++;
    }

    // Did not converge
    result.converged = false;
    result.eigenvectors[0] = ritz_vec;

    // Final full matvec even when not converged (energy correction still valuable)
    if (params.final_full_matvec && n_sparse_mv > 0) {
        auto t_ffm = Clock::now();
        std::vector<double> Hc_full(N, 0.0);
        full_matvec(ritz_vec.data(), Hc_full.data(), N);
        double E_full = dot(ritz_vec.data(), Hc_full.data(), N);
        double E_ritz = result.eigenvalues[0];
        result.eigenvalues[0] = E_full;
        n_full_mv++;
        double t_ffm_s = std::chrono::duration<double>(Clock::now() - t_ffm).count();
        result.ffm_wall_time = t_ffm_s;
        if (params.verbose >= 1) {
            std::cout << "  Final full matvec: E_ritz=" << std::fixed << std::setprecision(10) << E_ritz
                      << "  E_full=" << E_full
                      << "  correction=" << std::scientific << std::setprecision(2) << (E_full - E_ritz)
                      << "  time=" << std::fixed << std::setprecision(2) << t_ffm_s << "s" << std::endl;
        }
    }

    if (params.verbose >= 1) {
        std::cout << "  Sparse-Davidson did NOT converge after " << params.max_iter
                  << " iterations  |r| = " << std::scientific << std::setprecision(2)
                  << result.residual_norm
                  << "  [" << n_full_mv << " full + "
                  << n_sparse_mv << " sparse matvecs]" << std::endl;
    }
    if (params.verbose >= 2) {
        std::cout << "  Timers: eig=" << std::fixed << std::setprecision(3) << time_eig
                  << "  ritz=" << time_ritz << "  resid=" << time_resid
                  << "  trunc=" << time_trunc << "  mv=" << time_mv
                  << "  ortho=" << time_ortho << "  htcorr=" << time_htcorr
                  << "  proj=" << time_proj << std::endl;
    }
    return result;
}

// Explicit template instantiation
template DavidsonResult sparse_update_solve<uint64_t>(
    const MatvecContext<uint64_t>& ctx,
    std::function<void(const double*, double*, size_t)> full_matvec,
    size_t N,
    const SparseUpdateParams& params,
    const std::vector<std::vector<double>>& initial_guess);

template void sparse_matvec<uint64_t>(
    const MatvecContext<uint64_t>& ctx,
    const uint32_t* active_ids, size_t K,
    const double* delta_c,
    double* delta_sigma,
    size_t N);

template void sparse_block_matvec<uint64_t>(
    const MatvecContext<uint64_t>& ctx,
    const uint32_t* active_union, size_t K_union,
    const double* const* delta_c_batch,
    double** delta_sigma_batch,
    size_t N,
    int n_vecs);

// 128-bit instantiations
using Bit128 = std::array<uint64_t, 2>;

template DavidsonResult sparse_update_solve<Bit128>(
    const MatvecContext<Bit128>& ctx,
    std::function<void(const double*, double*, size_t)> full_matvec,
    size_t N,
    const SparseUpdateParams& params,
    const std::vector<std::vector<double>>& initial_guess);

template void sparse_matvec<Bit128>(
    const MatvecContext<Bit128>& ctx,
    const uint32_t* active_ids, size_t K,
    const double* delta_c,
    double* delta_sigma,
    size_t N);

template void sparse_block_matvec<Bit128>(
    const MatvecContext<Bit128>& ctx,
    const uint32_t* active_union, size_t K_union,
    const double* const* delta_c_batch,
    double** delta_sigma_batch,
    size_t N,
    int n_vecs);

}  // namespace fe
}  // namespace trimci_core
