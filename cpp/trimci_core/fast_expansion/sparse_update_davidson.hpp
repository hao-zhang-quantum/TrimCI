#pragma once
/**
 * Sparse-Update Davidson solver.
 *
 * Maintains the invariant sigma = H * c exactly throughout iterations.
 * Trial vectors are truncated to top-K components before computing H*t,
 * reducing per-iteration cost from O(N*C) to O(K*C) where K << N.
 *
 * Because HV columns are exact (H applied to the actual trial vector),
 * the Ritz value E = c^T H c / c^T c is always strictly variational.
 *
 * Algorithm:
 *   1. First iteration: full matvec to initialize sigma = H*c
 *   2. Subsequent iterations: truncate preconditioned residual t to top-K,
 *      compute H*t_sparse via sparse column-scatter, update subspace
 *   3. Standard Davidson convergence (residual + energy tolerance)
 */

#include <vector>
#include <functional>
#include "detspace_davidson.hpp"
#include "detspace_matvec.hpp"

namespace trimci_core {
namespace fe {

struct SparseUpdateParams {
    int max_subspace = 40;
    int max_iter = 500;
    double residual_tol = 1e-7;
    double energy_tol = 1e-7;
    int verbose = 1;

    // Sparse update controls
    double truncation_ratio = 0.1;   // Keep top fraction of |t_i| components
    int min_active_dets = 1000;      // Lower bound on K
    int max_active_dets = 0;         // 0 = determined by truncation_ratio

    // Hybrid warm/sparse: first N iterations use warm_ratio (large K) to seed
    // global subspace info, then switch to truncation_ratio (small K).
    // warm_ratio = 0: warm iters use full matvec (original behavior).
    // warm_ratio > 0: warm iters use sparse matvec with this larger ratio.
    //   E.g., warm_ratio=0.15 → first N iters at 15%, then cruise at 2%.
    //   Cost: N × 0.15 × C vs N × 1.0 × C for full matvec. ~7× cheaper warm phase.
    int n_warm_iters = 5;            // Number of warm iterations (large-K or full)
    double warm_ratio = 0.0;         // 0 = full matvec during warm; >0 = sparse with this ratio

    // Random sampling: supplement deterministic top-K with importance-sampled dets.
    // Each sparse iteration, K_rand = random_sample_ratio * N additional dets are
    // sampled with probability p_j ∝ |select_vec_j|², merged into active set.
    // Improves coverage: different dets enter each iteration, breaking the
    // "always the same top-K" dead zone. 0 = disabled (default).
    double random_sample_ratio = 0.0;
    double capture_threshold = 0.0;  // Minimum capture ratio threshold (0 = disabled).
                                     // Without adaptive_K: fall back to full matvec when
                                     //   capture < threshold (original behavior).
                                     // With adaptive_K: increase K to meet threshold.
    bool adaptive_K = false;         // When true + capture_threshold > 0, dynamically increase
                                     // K to achieve capture_threshold instead of falling back
                                     // to full matvec. Keeps sparse mode with larger K.

    // Truncation mode: how to select the top-K active determinants.
    // 0: |t_j| — preconditioned correction magnitude (default, original behavior)
    // 1: |r_j| — residual magnitude (avoids Jacobi denominator diffusion at high spectral density)
    // 2: |r_j * t_j| — PT2 energy contribution (Epstein-Nesbet ΔE_j^(2) = r_j²/|H_jj-θ|).
    //     Geometric mean of modes 0 and 1. Penalizes near-resonant dets as 1/|gap|
    //     instead of 1/gap² (mode 0), while still using spectral gap info (unlike mode 1).
    int truncation_mode = 0;

    // UCB exploration: when ucb_kappa > 0, adds an exploration bonus to the
    // truncation score, inspired by UCB1 from the bandit/active learning literature.
    // Each det j tracks n_j = number of times it has been in the active set.
    // The selection score becomes:  base_score_j + kappa * sqrt(ln(iter) / max(1, n_j))
    // where base_score is determined by truncation_mode (0/1/2).
    // This breaks the "coverage dead zone" — dets never selected get infinite bonus.
    // kappa controls exploration strength. 0 = disabled (default).
    double ucb_kappa = 0.0;

    // Block sparse Davidson: generate block_size trial vectors per outer iteration
    // via multi-shift Jacobi, sharing H_ij computation across all vectors.
    // block_size = 1: standard single-vector path (default, backward compatible)
    // block_size > 1: sparse_block_matvec shares connection traversal + H_ij
    int block_size = 1;

    // Olsen preconditioner: rank-1 correction to Jacobi that accounts for the
    // current Ritz vector. t_olsen = t_jacobi + ε·(D-θ)^{-1}x where
    // ε = -x^T·t_jacobi / x^T·(D-θ)^{-1}x. Reduces iteration count by 10-25%.
    bool olsen_correction = false;

    // Thick restart: when subspace reaches max_subspace, keep n Ritz vectors
    // instead of just the lowest one. Preserves subspace information across restarts.
    // 1 = original behavior (keep only lowest Ritz vector).
    int thick_restart_size = 1;

    // Momentum: after preconditioner, add β·(ritz_curr - ritz_prev) to the
    // trial vector. Biases correction toward eigenvector movement direction.
    // 0 = disabled (default).
    double momentum_beta = 0.0;

    // K schedule: per-step K values. When non-empty, overrides truncation_ratio/n_warm_iters/warm_ratio.
    // K_schedule[0] → initial H*c0 (K >= N means full matvec)
    // K_schedule[i] for i >= 1 → iteration (i-1) of Davidson loop
    // When i >= K_schedule.size(): repeat K_schedule.back()
    std::vector<size_t> K_schedule;

    // Final full matvec: after sparse Davidson converges, recompute energy via one exact
    // full matvec: E = c†·H·c. Eliminates Ritz value bias from all-sparse subspace.
    // Cost: one full matvec (~120s @32M). Only applies when sparse matvecs were used.
    bool final_full_matvec = false;
};

/// Sparse-update Davidson solver.
///
/// @param ctx          MatvecContext with dets, ab_index, integrals, diag
/// @param full_matvec  Full matvec function (used only for first iteration)
/// @param N            Problem dimension
/// @param params       Solver + sparsity parameters
/// @param initial_guess  Optional warm-start vectors
template<typename StorageType>
DavidsonResult sparse_update_solve(
    const MatvecContext<StorageType>& ctx,
    std::function<void(const double*, double*, size_t)> full_matvec,
    size_t N,
    const SparseUpdateParams& params,
    const std::vector<std::vector<double>>& initial_guess = {});

}  // namespace fe
}  // namespace trimci_core
