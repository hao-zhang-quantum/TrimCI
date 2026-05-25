#pragma once
/**
 * Matrix-free Davidson GEP solver for MR-LVCC.
 *
 * Instead of building the full n_basis² H and S matrices,
 * this solver computes H*v and S*v products on-the-fly using
 * the detspace inverse map + Slater-Condon rules.
 *
 * Key advantages over explicit matrix build:
 *   - Memory: O(n_basis × n_subspace) vs O(n_basis²)
 *   - Cache: writes to O(n_basis) vector (L1) vs O(n_basis²) matrix
 *   - Inner loop: O(k) gather-scatter vs O(k²) outer product
 *   - Threading: O(n_basis)/thread vs O(n_basis²)/thread
 *
 * Complexity per matvec: O(N_det × N_orb² × k)
 * Total for Davidson:    O(n_iter × N_det × N_orb² × k)
 */

#include <cstdint>
#include <Eigen/Dense>

namespace trimci_core {

struct MatfreeDavidsonResult {
    double eigenvalue;
    Eigen::VectorXd eigenvector;
    bool converged;
    int iterations;
    double residual_norm;
    double h_diag_0;  // H[0,0] for computing E_ref
};

/**
 * Solve the LVCC generalized eigenvalue problem H c = E S c
 * using matrix-free Davidson iteration.
 *
 * @param ref_alpha   Alpha bitstrings for reference determinants (n_ref)
 * @param ref_beta    Beta bitstrings for reference determinants (n_ref)
 * @param ref_coeffs  Reference CI coefficients (n_ref)
 * @param n_ref       Number of reference determinants
 * @param exc_types   Excitation type IDs (n_exc)
 * @param exc_indices Flattened excitation indices (n_exc × 4)
 * @param n_exc       Number of excitations
 * @param n_basis     Total basis size (1 + n_exc)
 * @param h1          One-body integrals (n_orb × n_orb, row-major)
 * @param eri         Two-body integrals (n_orb^4, row-major chemist notation)
 * @param n_orb       Number of spatial orbitals
 * @param max_iter    Maximum Davidson iterations (default 100)
 * @param tol         Convergence tolerance on residual norm (default 1e-6)
 * @param max_subspace Maximum subspace dimension before restart (default 30)
 * @param verbose     Verbosity level (0=silent, 1=summary, 2=per-iter)
 * @return            MatfreeDavidsonResult with eigenvalue, eigenvector, convergence info
 */
MatfreeDavidsonResult matfree_davidson_gep(
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
    int max_iter = 100,
    double tol = 1e-6,
    int max_subspace = 30,
    int verbose = 0);

} // namespace trimci_core
