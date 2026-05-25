#pragma once
/**
 * Fast Expansion: Standard eigenvalue Davidson solver.
 *
 * Solves H x = λ x for the smallest eigenvalue(s), given only a matvec
 * function and diagonal preconditioner.
 *
 * Distinct from the existing matfree_davidson_gep which solves the
 * generalized eigenvalue problem H c = E S c in the MR-LVCC exc space.
 *
 * Features:
 *   - Warm start from previous eigenvector (critical for expansion loop)
 *   - Jacobi preconditioner with denominator protection
 *   - Two-pass Gram-Schmidt for numerical stability
 *   - Subspace restart at max_subspace
 */

#include <vector>
#include <functional>

namespace trimci_core {
namespace fe {

struct DavidsonParams {
    int max_subspace = 40;     // Restart threshold (40 balances memory vs convergence)
    int max_iter = 500;        // Maximum iterations
    double residual_tol = 1e-7; // Residual convergence: ||r|| < residual_tol
    double energy_tol = 1e-7;   // Energy convergence: |dE| < energy_tol (0 = disabled)
    int n_states = 1;          // Number of eigenstates to compute
    int verbose = 1;           // 0=silent, 1=summary, 2=per-iter
    int block_size = 1;        // Trial vectors per iteration (>1 = block Davidson)
};

struct DavidsonResult {
    std::vector<double> eigenvalues;                // [n_states]
    std::vector<std::vector<double>> eigenvectors;  // [n_states][N]
    int n_iters = 0;
    bool converged = false;
    double residual_norm = 0.0;
    double ffm_wall_time = 0.0;  // Wall time of final full matvec (0 if not used)
};

/// Block matvec function: compute sigma_b = H * v_b for b = 0..n_vecs-1.
using BlockMatvecFunc = std::function<void(const double* const* v_batch,
                                           double** sigma_batch,
                                           size_t N, int n_vecs)>;

/// Solve standard eigenvalue problem H x = λ x.
///
/// @param matvec_func   Computes sigma = H * v. Signature: (const double* v, double* sigma, size_t N).
/// @param diag          Diagonal of H, used for Jacobi preconditioner. [N]
/// @param N             Dimension of the problem.
/// @param params        Solver parameters.
/// @param initial_guess Optional warm-start vectors. [n_states][N], or empty.
/// @param block_matvec  Optional block matvec (shared H_ij). Used when block_size > 1.
DavidsonResult davidson_solve(
    std::function<void(const double* v, double* sigma, size_t N)> matvec_func,
    const std::vector<double>& diag,
    size_t N,
    const DavidsonParams& params = {},
    const std::vector<std::vector<double>>& initial_guess = {},
    BlockMatvecFunc block_matvec = nullptr);

}  // namespace fe
}  // namespace trimci_core
