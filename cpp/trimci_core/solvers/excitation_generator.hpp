#pragma once
/**
 * Heat-bath excitation generator for MR-LVCC.
 *
 * Replaces the slow Python _generate_excitations() with a C++ implementation.
 *
 * Algorithm:
 *   For each reference determinant I with coefficient c_I:
 *     1. Find occ/vir orbitals from bitstrings
 *     2. Enumerate all single and double excitation operators
 *     3. Accumulate |c_I|^2 weight per unique excitation key
 *   Then rank by heat-bath score = |H_coupling| × weight
 *   and return the top max_excitations.
 *
 * Returns: (exc_types, exc_indices) arrays ready for build_H/S_detspace.
 *
 * Complexity: O(n_ref × n_occ² × n_vir²) with hash map merging.
 * For 9177 dets, 27 occ, 9 vir: ~5.4 billion ops → ~1-3 seconds in C++.
 */

#include <cstdint>
#include <vector>

namespace trimci_core {

/**
 * Generate excitations with heat-bath selection.
 *
 * @param ref_alpha   Alpha bitstrings for reference determinants (n_ref)
 * @param ref_beta    Beta bitstrings for reference determinants (n_ref)
 * @param ref_coeffs  Reference CI coefficients (n_ref)
 * @param n_ref       Number of reference determinants
 * @param n_orb       Number of spatial orbitals
 * @param F_aa        Alpha Fock matrix (n_orb × n_orb, row-major)
 * @param F_bb        Beta Fock matrix (n_orb × n_orb, row-major)
 * @param eri         Two-body integrals (n_orb^4, row-major, chemist notation)
 * @param max_excitations  Maximum number of excitations to return (-1 = no limit)
 * @param out_types   Output: excitation type IDs (0=S_α, 1=S_β, 2=D_αα, 3=D_αβ, 4=D_ββ)
 * @param out_indices Output: flattened indices (n_exc × 4)
 * @param scoring_mode 0=heat_bath (|coupling|×weight), 1=enpt2 ((coupling×weight)²/Δε)
 * @return            Number of excitations generated
 */
int generate_excitations_heatbath(
    const uint64_t* ref_alpha,
    const uint64_t* ref_beta,
    const double* ref_coeffs,
    int n_ref,
    int n_orb,
    const double* F_aa,
    const double* F_bb,
    const double* eri,
    int max_excitations,
    std::vector<int>& out_types,
    std::vector<int>& out_indices,
    int scoring_mode = 0);

} // namespace trimci_core
