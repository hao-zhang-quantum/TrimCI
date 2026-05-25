#pragma once

/**
 * @file davidson_gep.hpp
 * @brief Davidson iterative solver for Generalized Eigenvalue Problem (GEP).
 *
 * Solves Hc = E Sc for the lowest eigenvalue, where H is symmetric
 * and S is symmetric positive definite (or nearly so).
 *
 * Complexity: O(n^2) per iteration (matrix-vector products) vs O(n^3) for
 * dense diagonalization. Typically converges in 20-100 iterations.
 *
 * @date 2026-02-06
 */

#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <Eigen/Dense>

namespace trimci_core {

struct DavidsonGEPResult {
    double eigenvalue;
    Eigen::VectorXd eigenvector;
    bool converged;
    int iterations;
    double residual_norm;
};

/**
 * @brief Davidson solver for generalized eigenvalue problem Hc = ESc.
 *
 * @param H  Symmetric matrix (n x n)
 * @param S  Symmetric positive definite matrix (n x n)
 * @param max_iter     Maximum number of iterations (default 200)
 * @param tol          Convergence tolerance on energy (default 1e-8)
 * @param max_subspace Maximum subspace size before restart (default 40)
 * @param verbose      Verbosity level (0=silent, 1=summary, 2=per-iteration)
 * @return DavidsonGEPResult with lowest eigenvalue and eigenvector
 *
 * Recommended parameter presets (benchmarked on random SPD matrices):
 *
 *   Preset      max_iter  tol    max_subspace  Typical dE     Speed
 *   ─────────   ────────  ─────  ────────────  ─────────────  ──────
 *   Fast        50        1e-4   20            ~1e-4          Fastest
 *   Default     100       1e-6   30            ~1e-6          Good
 *   Accurate    200       1e-8   40            ~1e-8 - 1e-12  Slower
 *   Tight       500       1e-10  60            ~1e-12+        Slowest
 *
 * For quantum chemistry, "Default" (1e-6 Ha) is well below chemical
 * accuracy (1e-3 Ha). Use "Fast" for exploratory runs.
 */
DavidsonGEPResult davidson_gep(
    const Eigen::MatrixXd& H,
    const Eigen::MatrixXd& S,
    int max_iter = 200,
    double tol = 1e-8,
    int max_subspace = 40,
    int verbose = 0,
    const Eigen::VectorXd& v_init = Eigen::VectorXd());

} // namespace trimci_core
