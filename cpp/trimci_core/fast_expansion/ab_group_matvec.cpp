#include "ab_group_matvec.hpp"
#include "simd_popcount.hpp"
#include "detspace_matvec.hpp"
#include "hamiltonian.hpp"
#include <cstring>
#include "omp_compat.hpp"
namespace trimci_core {
namespace fe {

// ============================================================================
// J-table precomputation: J[m*N+p] = Σ_{n∈occ(fixed_string)} eri[m,p,n,n]
// Fits L1 cache (~10 KB for n_orb=36). Reusable within a group.
// ============================================================================
static void precompute_J_table(
    uint64_t fixed_string,
    const double* eri_data,
    int n_orb,
    double* J_table)
{
    const size_t N = n_orb;
    const size_t N2 = N * N;
    const size_t N3 = N2 * N;

    std::memset(J_table, 0, N * N * sizeof(double));

    uint64_t tmp = fixed_string;
    while (tmp) {
        int n = __builtin_ctzll(tmp);
        tmp &= tmp - 1;
        const size_t n_stride = static_cast<size_t>(n) * N + n;
        for (int m = 0; m < n_orb; ++m) {
            const size_t base_m = static_cast<size_t>(m) * N3;
            for (int p = 0; p < n_orb; ++p) {
                J_table[m * N + p] += eri_data[base_m
                    + static_cast<size_t>(p) * N2 + n_stride];
            }
        }
    }
}

// ============================================================================
// Same-spin single excitation with J-table optimization.
// var_i → var_j is the single excitation in the varying spin;
// J_fixed provides cross-spin Coulomb from the fixed spin.
// ============================================================================
static double h_same_spin_single_dressed(
    uint64_t var_i, uint64_t var_j,
    const std::vector<std::vector<double>>& h1,
    const double* eri_data,
    const double* J_fixed,
    int n_orb)
{
    const size_t N = n_orb;
    const size_t N2 = N * N;
    const size_t N3 = N2 * N;

    // Extract hole (m) and particle (p) from single excitation
    int m = __builtin_ctzll(var_i & ~var_j);
    int p = __builtin_ctzll(var_j & ~var_i);
    int phase = detail::cre_des_sign_t<uint64_t>(m, p, var_j);

    // One-electron + cross-spin Coulomb (table lookup)
    double Hij = h1[m][p] + J_fixed[m * N + p];

    // Same-spin Coulomb-exchange: Σ_{n∈occ(var_i), n≠m}
    const size_t base_mp = static_cast<size_t>(m) * N3
                         + static_cast<size_t>(p) * N2;
    const size_t base_m  = static_cast<size_t>(m) * N3;
    const size_t Np1  = N + 1;
    const size_t N2pN = N2 + N;

    uint64_t occ = var_i;
    while (occ) {
        int n = __builtin_ctzll(occ);
        occ &= occ - 1;
        if (n != m) {
            Hij += eri_data[base_mp + n * Np1]
                 - eri_data[base_m + n * N2pN + p];
        }
    }

    return Hij * phase;
}

// ============================================================================
// Channel 1: Same-α (Sβ + Dββ)
// ============================================================================
void matvec_same_alpha_group(
    uint64_t alpha,
    const uint64_t* sorted_betas,
    size_t group_size,
    const double* v_group,
    double* sigma_group,
    const std::vector<std::vector<double>>& h1,
    const double* eri_data,
    int n_orb)
{
    std::vector<double> J_alpha(n_orb * n_orb);
    precompute_J_table(alpha, eri_data, n_orb, J_alpha.data());
    const double* J_alpha_ptr = J_alpha.data();

    #pragma omp parallel for schedule(dynamic, 16) if(group_size >= 64)
    for (int i = 0; i < static_cast<int>(group_size); ++i) {
        uint64_t bi = sorted_betas[i];

        simd::scan_popcount_sd(sorted_betas, group_size, bi,
            [&](size_t j) {  // beta single (popcount == 2)
                if (j == i) return;
                double hij = h_same_spin_single_dressed(
                    bi, sorted_betas[j], h1, eri_data,
                    J_alpha_ptr, n_orb);
                sigma_group[i] += hij * v_group[j];
            },
            [&](size_t j) {  // beta double (popcount == 4)
                double hij = compute_H_ij_strings<uint64_t>(
                    alpha, bi, alpha, sorted_betas[j],
                    ExcType::SAME_ALPHA_D, h1, eri_data, n_orb);
                sigma_group[i] += hij * v_group[j];
            }
        );
    }
}

// ============================================================================
// Channel 3: Same-β (Sα + Dαα)
// ============================================================================
void matvec_same_beta_group(
    uint64_t beta,
    const uint64_t* sorted_alphas,
    size_t group_size,
    const double* v_group,
    double* sigma_group,
    const std::vector<std::vector<double>>& h1,
    const double* eri_data,
    int n_orb)
{
    std::vector<double> J_beta(n_orb * n_orb);
    precompute_J_table(beta, eri_data, n_orb, J_beta.data());
    const double* J_beta_ptr = J_beta.data();

    #pragma omp parallel for schedule(dynamic, 16) if(group_size >= 64)
    for (int i = 0; i < static_cast<int>(group_size); ++i) {
        uint64_t ai = sorted_alphas[i];

        simd::scan_popcount_sd(sorted_alphas, group_size, ai,
            [&](size_t j) {  // alpha single (popcount == 2)
                if (j == i) return;
                double hij = h_same_spin_single_dressed(
                    ai, sorted_alphas[j], h1, eri_data,
                    J_beta_ptr, n_orb);
                sigma_group[i] += hij * v_group[j];
            },
            [&](size_t j) {  // alpha double (popcount == 4)
                double hij = compute_H_ij_strings<uint64_t>(
                    ai, beta, sorted_alphas[j], beta,
                    ExcType::SAME_BETA_D, h1, eri_data, n_orb);
                sigma_group[i] += hij * v_group[j];
            }
        );
    }
}

// ============================================================================
// Channel 2: Mixed Dαβ — one (self, partner) alpha group pair
//
// Alpha excitation (m_a → p_a) is constant for the pair, precomputed once.
// For each βi in self, SIMD-scans partner betas for single-β matches.
// ============================================================================
void matvec_mixed_pair(
    uint64_t alpha_self,
    const uint64_t* betas_self,
    size_t self_size,
    double* sigma_self,
    uint64_t alpha_partner,
    const uint64_t* betas_partner,
    size_t partner_size,
    const double* v_partner,
    const std::vector<std::vector<double>>& /* h1 */,
    const double* eri_data,
    int n_orb)
{
    const size_t N = n_orb;
    const size_t N2 = N * N;
    const size_t N3 = N2 * N;

    // Precompute alpha excitation (constant for all pairs in this call)
    int m_a = __builtin_ctzll(alpha_self & ~alpha_partner);
    int p_a = __builtin_ctzll(alpha_partner & ~alpha_self);
    int phase_a = detail::cre_des_sign_t<uint64_t>(m_a, p_a, alpha_partner);

    const size_t eri_base = static_cast<size_t>(m_a) * N3
                          + static_cast<size_t>(p_a) * N2;

    #pragma omp parallel for schedule(dynamic, 16) if(self_size >= 64)
    for (int i = 0; i < static_cast<int>(self_size); ++i) {
        uint64_t bi = betas_self[i];

        simd::scan_popcount_eq(betas_partner, partner_size, bi, 2,
            [&](size_t j) {
                uint64_t bj = betas_partner[j];
                int m_b = __builtin_ctzll(bi & ~bj);
                int p_b = __builtin_ctzll(bj & ~bi);
                int phase_b = detail::cre_des_sign_t<uint64_t>(
                    m_b, p_b, bj);

                double hij = phase_a * phase_b
                    * eri_data[eri_base
                        + static_cast<size_t>(m_b) * N + p_b];
                sigma_self[i] += hij * v_partner[j];
            }
        );
    }
}

}  // namespace fe
}  // namespace trimci_core
