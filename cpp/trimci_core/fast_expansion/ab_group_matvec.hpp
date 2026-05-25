#pragma once
/**
 * Alpha-Beta group-level matvec for distributed Davidson.
 *
 * Three standalone functions, one per channel:
 *   Ch1: Same-α (Sβ + Dββ) — process one alpha group
 *   Ch3: Same-β (Sα + Dαα) — process one beta group
 *   Ch2: Mixed Dαβ — process one (self, partner) alpha group pair
 *
 * No ABIndex required. Works on raw sorted bitstring arrays.
 * Uses SIMD popcount scanning for connection enumeration.
 */

#include <cstdint>
#include <cstddef>
#include <vector>

namespace trimci_core {
namespace fe {

/// Channel 1: Same-α (Sβ + Dββ) within one alpha group.
/// For each (α, βi), finds all βj in the group differing by 1 or 2 orbitals,
/// computes H_ij, and accumulates sigma_group[i] += H_ij * v_group[j].
/// sigma_group is accumulated (caller must zero it first).
void matvec_same_alpha_group(
    uint64_t alpha,
    const uint64_t* sorted_betas,
    size_t group_size,
    const double* v_group,
    double* sigma_group,
    const std::vector<std::vector<double>>& h1,
    const double* eri_data,
    int n_orb);

/// Channel 3: Same-β (Sα + Dαα) within one beta group.
void matvec_same_beta_group(
    uint64_t beta,
    const uint64_t* sorted_alphas,
    size_t group_size,
    const double* v_group,
    double* sigma_group,
    const std::vector<std::vector<double>>& h1,
    const double* eri_data,
    int n_orb);

/// Channel 2: Mixed Dαβ — one alpha group pair.
/// For each βi in self group, finds βj in partner group with popcount(βi⊕βj)==2,
/// computes H_mixed(α,βi | α',βj), accumulates sigma_self[i] += H * v_partner[j].
void matvec_mixed_pair(
    uint64_t alpha_self,
    const uint64_t* betas_self,
    size_t self_size,
    double* sigma_self,
    uint64_t alpha_partner,
    const uint64_t* betas_partner,
    size_t partner_size,
    const double* v_partner,
    const std::vector<std::vector<double>>& h1,
    const double* eri_data,
    int n_orb);

}  // namespace fe
}  // namespace trimci_core
