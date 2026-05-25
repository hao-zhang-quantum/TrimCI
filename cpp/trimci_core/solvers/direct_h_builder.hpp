#pragma once
/**
 * Excitation-space H-matrix builder using determinant expansion with Slater-Condon rules.
 *
 * NOTE: This is the legacy O(K² × M²) excitation-space algorithm.
 * For most use cases, prefer build_H_detspace() in direct_h_detspace.hpp
 * which has O(N_det × N_orb⁴) complexity and is faster for large references.
 *
 * Implements the direct computation of Hamiltonian matrix elements:
 *   H[μ,ν] = Σ_{I,J} c_I c_J φ^μ_I φ^ν_J ⟨D^μ_I|H|D^ν_J⟩
 *
 * where |D^μ_I⟩ = τ̂_μ|D_I⟩ is the determinant obtained by applying
 * excitation operator τ̂_μ to reference determinant |D_I⟩.
 *
 * Slater-Condon rules:
 *   - 0 orbital differences: diagonal energy
 *   - 1 orbital difference:  F_ia + exchange sum
 *   - 2 orbital differences: (ia|jb) - (ib|ja) or (ia|jb) for mixed spin
 *   - >2 differences: 0
 *
 * Complexity: O(K² × M² × N_e) where
 *   K = number of excitations
 *   M = number of reference determinants
 *   N_e = number of electrons
 */

#include <cstdint>

namespace trimci_core {

/**
 * Build H matrix using excitation-space algorithm (legacy).
 *
 * Prefer build_H_detspace() for better performance.
 */
void build_H_direct_excspace(
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
