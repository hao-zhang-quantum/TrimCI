#pragma once

#include <vector>
#include <tuple>
#include <utility>
#include "determinant.hpp"

namespace trimci_core {

/**
 * RDM computation for LVCC.
 * 
 * Provides exact 1/2-RDM and cumulant approximations for 3/4-RDM.
 * 
 * Note: Currently only supports uint64_t determinants (n_orb <= 64).
 */

// ============================================================================
// 1-RDM and 2-RDM: Exact computation
// ============================================================================

/**
 * Compute exact 1-RDM: γ[p,q] = ⟨Ψ|a†_p a_q|Ψ⟩
 * 
 * Complexity: O(N_det² × N²)
 * 
 * @param dets List of determinants
 * @param coeffs CI coefficients
 * @param n_orb Number of spatial orbitals
 * @return gamma: 1-RDM as flattened vector (n_orb²)
 */
std::vector<double> compute_1rdm(
    const std::vector<Determinant>& dets,
    const std::vector<double>& coeffs,
    int n_orb
);

/**
 * Compute exact 2-RDM: Γ[p,q,r,s] = ⟨Ψ|a†_p a†_q a_s a_r|Ψ⟩
 * 
 * Index ordering: physicist convention (pq|rs)
 * Complexity: O(N_det² × N⁴)
 * 
 * @return Gamma2: 2-RDM as flattened vector (n_orb⁴)
 */
std::vector<double> compute_2rdm(
    const std::vector<Determinant>& dets,
    const std::vector<double>& coeffs,
    int n_orb
);

/**
 * Compute spin-resolved 1-RDM.
 * 
 * Returns (γ^αα, γ^ββ) where:
 *   γ^αα[p,q] = ⟨Ψ|a†_{pα} a_{qα}|Ψ⟩
 *   γ^ββ[p,q] = ⟨Ψ|a†_{pβ} a_{qβ}|Ψ⟩
 * 
 * @return pair of (gamma_aa, gamma_bb), each n_orb²
 */
std::pair<std::vector<double>, std::vector<double>> compute_1rdm_spin_resolved(
    const std::vector<Determinant>& dets,
    const std::vector<double>& coeffs,
    int n_orb
);

/**
 * Compute spin-resolved 2-RDM.
 * 
 * Returns (Γ^αα, Γ^αβ, Γ^ββ) where:
 *   Γ^αα[p,q,r,s] = ⟨Ψ|a†_{pα} a†_{qα} a_{sα} a_{rα}|Ψ⟩  (antisymmetric)
 *   Γ^αβ[p,q,r,s] = ⟨Ψ|a†_{pα} a†_{qβ} a_{sβ} a_{rα}|Ψ⟩  (no antisymmetry between spins)
 *   Γ^ββ[p,q,r,s] = ⟨Ψ|a†_{pβ} a†_{qβ} a_{sβ} a_{rβ}|Ψ⟩  (antisymmetric)
 * 
 * Energy formula:
 *   E = Σ_{pq} h[p,q] (γ^αα[p,q] + γ^ββ[p,q])
 *     + 0.5 Σ_{pqrs} (pq|rs) (Γ^αα[p,q,r,s] + Γ^ββ[p,q,r,s])
 *     + Σ_{pqrs} (pr|qs) Γ^αβ[p,q,r,s]
 * 
 * @return tuple of (Gamma_aa, Gamma_ab, Gamma_bb), each n_orb⁴
 */
std::tuple<std::vector<double>, std::vector<double>, std::vector<double>> 
compute_2rdm_spin_resolved(
    const std::vector<Determinant>& dets,
    const std::vector<double>& coeffs,
    int n_orb
);

// ============================================================================
// Transition RDMs: For computing ⟨Ψ_I|Ĥ|Ψ_J⟩
// ============================================================================

/**
 * Compute transition 1-RDM: γ^{IJ}[p,q] = ⟨Ψ_I|a†_p a_q|Ψ_J⟩
 * 
 * @param dets_I Determinants for state I
 * @param coeffs_I Coefficients for state I
 * @param dets_J Determinants for state J
 * @param coeffs_J Coefficients for state J
 * @param n_orb Number of spatial orbitals
 * @return pair of (gamma_aa, gamma_bb)
 */
std::pair<std::vector<double>, std::vector<double>> compute_transition_1rdm_spin_resolved(
    const std::vector<Determinant>& dets_I,
    const std::vector<double>& coeffs_I,
    const std::vector<Determinant>& dets_J,
    const std::vector<double>& coeffs_J,
    int n_orb
);

/**
 * Compute transition 2-RDM: Γ^{IJ}[p,q,r,s] = ⟨Ψ_I|a†_p a†_q a_s a_r|Ψ_J⟩
 * 
 * @return tuple of (Gamma_aa, Gamma_ab, Gamma_bb)
 */
std::tuple<std::vector<double>, std::vector<double>, std::vector<double>> 
compute_transition_2rdm_spin_resolved(
    const std::vector<Determinant>& dets_I,
    const std::vector<double>& coeffs_I,
    const std::vector<Determinant>& dets_J,
    const std::vector<double>& coeffs_J,
    int n_orb
);

// ============================================================================
// 4-RDM: Cumulant contraction (on-the-fly)
// ============================================================================

/**
 * Contract 4-RDM (via cumulant) with amplitude tensor on-the-fly.
 * 
 * Computes: result = Σ_{tuvw} Γ⁴[p,q,r,s,t,u,v,w] * c[t,u,v,w]
 * 
 * Much more efficient than storing full 4-RDM: O(N⁴) per contraction.
 */
double contract_4rdm_cumulant(
    const std::vector<double>& Gamma2,
    const std::vector<double>& c_ijab,
    int p, int q, int r, int s,
    int n_orb
);

// ============================================================================
// Energy from RDMs
// ============================================================================

/**
 * Compute energy from RDMs.
 * 
 * E = tr(h1 @ γ) + 0.5 * tr(eri @ Γ²) + E_nuc
 */
double energy_from_rdm(
    const std::vector<double>& gamma,
    const std::vector<double>& Gamma2,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    double E_nuc,
    int n_orb
);

// ============================================================================
// Exact 3-RDM and 4-RDM (for small systems debug only)
// ============================================================================

/**
 * Compute spin-resolved 3-RDM from CI vector (EXACT).
 * 
 * == TrimCI Convention (creation-first) ==
 * Indices [p,q,r,s,t,u] correspond to:
 *   Γ³[p,q,r,s,t,u] = ⟨Ψ| a†_p a†_q a†_r a_u a_t a_s |Ψ⟩
 * 
 * First 3 indices (p,q,r) = creation operators (left to right)
 * Last 3 indices (s,t,u) = annihilation operators (reverse order: u,t,s)
 * 
 * Returns tuple of 4 tensors (each n_orb^6):
 * - Γ³_aaa[p,q,r,s,t,u] = ⟨a†_pα a†_qα a†_rα a_uα a_tα a_sα⟩
 * - Γ³_aab[p,q,r,s,t,u] = ⟨a†_pα a†_qα a†_rβ a_uβ a_tα a_sα⟩
 * - Γ³_abb[p,q,r,s,t,u] = ⟨a†_pα a†_qβ a†_rβ a_uβ a_tβ a_sα⟩
 * - Γ³_bbb[p,q,r,s,t,u] = ⟨a†_pβ a†_qβ a†_rβ a_uβ a_tβ a_sβ⟩
 * 
 * Antisymmetry: Same-spin indices are stored antisymmetrically.
 * - aaa: fully antisymmetric in (p,q,r) and (s,t,u)
 * - aab: antisymmetric in (p,q) and (s,t) only (alpha pairs)
 * - abb: antisymmetric in (q,r) and (t,u) only (beta pairs)
 * - bbb: fully antisymmetric in (p,q,r) and (s,t,u)
 * 
 * == PySCF Convention (alternating) ==
 * PySCF uses rdm3[i,j,k,l,m,n] = ⟨i† j k† l m† n⟩ (alternating create-annihilate).
 * PySCF's reorder=True also subtracts lower-order RDM contributions (cumulant).
 * 
 * To convert TrimCI → PySCF ordering (without cumulant subtraction):
 *   pyscf[i,j,k,l,m,n] = trimci[i,k,m, j,l,n] (axes permutation: 0,3,1,4,2,5)
 * 
 * Warning: 4x O(n^6) storage. Only use for n_orb <= 8.
 */
std::tuple<std::vector<double>, std::vector<double>, 
           std::vector<double>, std::vector<double>>
compute_3rdm_spin_resolved(
    const std::vector<Determinant>& dets,
    const std::vector<double>& coeffs,
    int n_orb
);

/**
 * Compute spin-resolved 3-RDM via cumulant approximation.
 *
 * Approximates 3-RDM by ignoring the 3-body cumulant:
 *   Γ³ ≈ A_S[Γ² ⊗ γ] − 2·det(γ)
 *
 * Uses spin-resolved 1-RDM and 2-RDM as input.
 * 
 * @param gamma_aa Alpha 1-RDM (n_orb²)
 * @param gamma_bb Beta 1-RDM (n_orb²)
 * @param Gamma_aa Alpha-alpha 2-RDM (n_orb⁴)
 * @param Gamma_ab Alpha-beta 2-RDM (n_orb⁴)
 * @param Gamma_bb Beta-beta 2-RDM (n_orb⁴)
 * @param n_orb Number of orbitals
 * @return Tuple of (Γ³_aaa, Γ³_aab, Γ³_abb, Γ³_bbb)
 */
std::tuple<std::vector<double>, std::vector<double>, 
           std::vector<double>, std::vector<double>>
compute_3rdm_cumulant_spin_resolved(
    const std::vector<double>& gamma_aa,
    const std::vector<double>& gamma_bb,
    const std::vector<double>& Gamma_aa,
    const std::vector<double>& Gamma_ab,
    const std::vector<double>& Gamma_bb,
    int n_orb
);

// ============================================================================
// Spin-Resolved 4-RDM (for small systems only)
// ============================================================================

/**
 * Compute spin-resolved 4-RDM from CI vector (EXACT).
 * 
 * Returns tuple of 5 tensors (each n_orb^8):
 * - Γ⁴_aaaa, Γ⁴_aaab, Γ⁴_aabb, Γ⁴_abbb, Γ⁴_bbbb
 * 
 * Warning: 5x O(n^8) storage. Only use for n_orb <= 5.
 */
std::tuple<std::vector<double>, std::vector<double>, std::vector<double>,
           std::vector<double>, std::vector<double>>
compute_4rdm_spin_resolved(
    const std::vector<Determinant>& dets,
    const std::vector<double>& coeffs,
    int n_orb
);

/**
 * Compute spin-resolved 4-RDM via cumulant approximation.
 * 
 * Approximates 4-RDM by ignoring the 4-body cumulant:
 *   Γ⁴ ≈ A[Γ³ ⊗ γ] + A[Γ² ⊗ Γ²] + A[Γ² ⊗ γ ⊗ γ] + A[γ ⊗ γ ⊗ γ ⊗ γ]
 * 
 * If 3-RDM is also approximated via cumulant, this becomes pure 1-RDM/2-RDM based.
 * 
 * @return Tuple of (Γ⁴_aaaa, Γ⁴_aaab, Γ⁴_aabb, Γ⁴_abbb, Γ⁴_bbbb)
 */
std::tuple<std::vector<double>, std::vector<double>, std::vector<double>,
           std::vector<double>, std::vector<double>>
compute_4rdm_cumulant_from_2rdm_spin_resolved(
    const std::vector<double>& gamma_aa,
    const std::vector<double>& gamma_bb,
    const std::vector<double>& Gamma_aa,
    const std::vector<double>& Gamma_ab,
    const std::vector<double>& Gamma_bb,
    int n_orb
);

/**
 * Compute spin-resolved 4-RDM cumulant approximation using exact 3-RDM.
 *
 * Sets only λ⁴ = 0 (4-body cumulant), keeping exact λ³ from 3-RDM.
 * Closed-form: Γ⁴ = A_S[Γ³ ⊗ γ] + A_S[λ² ⊗ λ²] − 3·det₄(γ)
 * where λ² = Γ² − det₂(γ).
 *
 * More accurate than compute_4rdm_cumulant_from_2rdm_spin_resolved which also sets λ³=0.
 *
 * @return Tuple of (Γ⁴_aaaa, Γ⁴_aaab, Γ⁴_aabb, Γ⁴_abbb, Γ⁴_bbbb)
 */
std::tuple<std::vector<double>, std::vector<double>, std::vector<double>,
           std::vector<double>, std::vector<double>>
compute_4rdm_cumulant_from_3rdm_spin_resolved(
    const std::vector<double>& gamma_aa,
    const std::vector<double>& gamma_bb,
    const std::vector<double>& Gamma_aa,
    const std::vector<double>& Gamma_ab,
    const std::vector<double>& Gamma_bb,
    const std::vector<double>& Gamma3_aaa,
    const std::vector<double>& Gamma3_aab,
    const std::vector<double>& Gamma3_abb,
    const std::vector<double>& Gamma3_bbb,
    int n_orb
);

// ============================================================================
// Per-element Cumulant-2 Reconstruction (for any n-RDM order)
// ============================================================================

/**
 * Compute a single n-RDM element using cumulant-2 approximation.
 *
 * Recursively decomposes into 1-body (gamma) and 2-body cumulant (lambda2)
 * contractions. All cumulants of order >= 3 are set to zero.
 *
 * Exact for single-determinant states; approximate for multi-reference.
 * Works for any n (typically n=5,6 where full tensor storage is infeasible).
 *
 * @param creators Vector of (orbital, spin) pairs. spin: 0=alpha, 1=beta
 * @param annihilators Vector of (orbital, spin) pairs
 * @param gamma_aa Alpha 1-RDM (n_orb^2)
 * @param gamma_bb Beta 1-RDM (n_orb^2)
 * @param Gamma_aa Alpha-alpha 2-RDM (n_orb^4)
 * @param Gamma_ab Alpha-beta 2-RDM (n_orb^4)
 * @param Gamma_bb Beta-beta 2-RDM (n_orb^4)
 * @param n_orb Number of spatial orbitals
 * @return n-RDM element value
 */
double compute_rdm_element_cumulant2(
    const std::vector<std::pair<int,int>>& creators,
    const std::vector<std::pair<int,int>>& annihilators,
    const std::vector<double>& gamma_aa,
    const std::vector<double>& gamma_bb,
    const std::vector<double>& Gamma_aa,
    const std::vector<double>& Gamma_ab,
    const std::vector<double>& Gamma_bb,
    int n_orb
);

// ============================================================================
// Spin-Resolved 5-RDM (COMPLETE - all 6 spin components)
// ============================================================================

/**
 * Compute spin-resolved 5-RDM from CI vector (EXACT).
 * 
 * Returns 6 tensors: aaaaa, aaaab, aaabb, aabbb, abbbb, bbbbb
 * 
 * Warning: O(n^10) storage. Only use for n_orb <= 5.
 */
std::tuple<std::vector<double>, std::vector<double>, std::vector<double>,
           std::vector<double>, std::vector<double>, std::vector<double>> 
compute_5rdm_spin_resolved(
    const std::vector<Determinant>& dets,
    const std::vector<double>& coeffs,
    int n_orb
);

// ============================================================================
// Spin-Resolved 6-RDM (COMPLETE - all 7 spin components)
// ============================================================================

/**
 * Compute spin-resolved 6-RDM from CI vector (EXACT).
 * 
 * Returns 7 tensors: aaaaaa, aaaaab, aaaabb, aaabbb, aabbbb, abbbbb, bbbbbb
 * 
 * Warning: O(n^12) storage. Only use for n_orb <= 4.
 */
std::tuple<std::vector<double>, std::vector<double>, std::vector<double>,
           std::vector<double>, std::vector<double>, std::vector<double>,
           std::vector<double>> 
compute_6rdm_spin_resolved(
    const std::vector<Determinant>& dets,
    const std::vector<double>& coeffs,
    int n_orb
);

} // namespace trimci_core

