#pragma once
/**
 * Det-space H-matrix builder using inverse map + Slater-Condon connectivity.
 *
 * Algorithm:
 *   1. Build inverse map: det → [(matrix_row, accumulated_coeff)]
 *   2. For each unique det, enumerate all single/double excitations,
 *      look up connected dets in the map.
 *   3. For each connected pair (d, d'), compute ⟨d|H|d'⟩ directly
 *      and accumulate outer product into H.
 *
 * Complexity: O(N_det × N_orb⁴ + N_conn × k²)
 * compared to O(N_exc² × N_ref²) for the excitation-space algorithm.
 *
 * Advantageous when N_exc × N_ref > N_orb⁴ (large reference spaces).
 */

#include <cstdint>

namespace trimci_core {

/**
 * Build H matrix using det-space algorithm with inverse map.
 *
 * Same interface as build_H_direct_excspace() for drop-in replacement.
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
 * @param h1         One-body integrals (n_orb × n_orb, row-major)
 * @param eri        Two-body integrals (n_orb^4, row-major chemist notation)
 * @param n_orb      Number of spatial orbitals
 * @param H_out      Pre-allocated output buffer (n_basis × n_basis, zeroed)
 */
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
    double* H_out);

} // namespace trimci_core
