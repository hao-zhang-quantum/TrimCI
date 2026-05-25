#pragma once
/**
 * Direct S-matrix builder using determinant expansion with fermionic phase.
 *
 * Implements the batch inverse-map algorithm:
 *   1. For each excitation μ, apply τ̂_μ to each ref det with phase → det maps
 *   2. Build inverse map: det → [(matrix_row, accumulated_coeff)]
 *   3. Accumulate S via outer products per determinant
 *
 * Complexity: O(n_ref × n_exc + Σ_d k_d²) ≈ O(n_ref × n_exc)
 * compared to O(n_ref × n_exc²) for element-by-element.
 */

#include <cstdint>

namespace trimci_core {

/**
 * Build S matrix using direct determinant expansion with fermionic phase.
 *
 * @param ref_alpha  Alpha bitstrings for reference determinants (n_ref)
 * @param ref_beta   Beta bitstrings for reference determinants (n_ref)
 * @param ref_coeffs Reference CI coefficients (n_ref)
 * @param n_ref      Number of reference determinants
 * @param exc_types  Excitation type IDs (n_exc):
 *                   0=S_alpha, 1=S_beta, 2=D_aa, 3=D_ab, 4=D_bb
 * @param exc_indices Flattened excitation indices (n_exc × 4):
 *                   Singles [i,a,0,0], Doubles [i,j,a,b]
 * @param n_exc      Number of excitations
 * @param n_basis    Total basis size (1 + n_exc)
 * @param S_out      Pre-allocated output buffer (n_basis × n_basis, zeroed)
 */
void build_S_direct(
    const uint64_t* ref_alpha,
    const uint64_t* ref_beta,
    const double* ref_coeffs,
    int n_ref,
    const int* exc_types,
    const int* exc_indices,
    int n_exc,
    int n_basis,
    double* S_out);

} // namespace trimci_core
