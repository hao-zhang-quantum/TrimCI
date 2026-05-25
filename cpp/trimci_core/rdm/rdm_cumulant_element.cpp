/**
 * Per-element Cumulant-2 Reconstruction for n-RDM.
 *
 * Computes individual n-RDM elements using the cumulant-2 approximation:
 * all cumulants of order >= 3 are set to zero, so the n-RDM is reconstructed
 * from 1-RDM (gamma) and 2-body cumulant (lambda2 = Gamma2 - A{gamma x gamma}).
 *
 * Algorithm: Recursive expansion processing creators left-to-right.
 * Each creator either:
 *   1. Singles with an annihilator -> contributes gamma[c, a]
 *   2. Pairs with another creator via lambda2 -> contributes lambda2[c1,c2,a1,a2]
 *
 * Exact for single-determinant states. Approximate for multi-reference.
 * Cost: O(n!! * n!) per element, fast for n <= 6.
 */

#include "rdm_full.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

namespace trimci_core {

namespace {

// Index helpers
inline int idx2(int p, int q, int N) {
    return p * N + q;
}

inline int idx4(int p, int q, int r, int s, int N) {
    return p * N * N * N + q * N * N + r * N + s;
}

/**
 * Get gamma[c, a] with spin awareness.
 * Returns 0 if spins don't match.
 */
inline double get_gamma(
    int c_orb, int c_spin,
    int a_orb, int a_spin,
    const double* gamma_aa, const double* gamma_bb,
    int N)
{
    if (c_spin != a_spin) return 0.0;
    if (c_spin == 0) return gamma_aa[idx2(c_orb, a_orb, N)];
    return gamma_bb[idx2(c_orb, a_orb, N)];
}

/**
 * Get lambda2[c1, c2, a1, a2] = Gamma2[...] - gamma*gamma (+/- exchange).
 *
 * The 2-body cumulant lambda2 = Gamma2 - A{gamma x gamma}.
 * Spin sectors:
 *   alpha-alpha: Gamma_aa[c1,c2,a1,a2] - gamma_a[c1,a1]*gamma_a[c2,a2]
 *                                        + gamma_a[c1,a2]*gamma_a[c2,a1]
 *   alpha-beta:  Gamma_ab[c1,c2,a1,a2] - gamma_a[c1,a1]*gamma_b[c2,a2]
 *   beta-alpha:  -(Gamma_ab[c2,c1,a2,a1] - gamma_b[c1,a1]*gamma_a[c2,a2])
 *                = swap to canonical alpha-beta form
 *   beta-beta:   Gamma_bb[c1,c2,a1,a2] - gamma_b[c1,a1]*gamma_b[c2,a2]
 *                                        + gamma_b[c1,a2]*gamma_b[c2,a1]
 *
 * Returns 0 if spins are incompatible (e.g., 3 alpha + 1 beta creators).
 *
 * Convention: Gamma2[p,q,r,s] = <a^dag_p a^dag_q a_s a_r> (LMRM)
 * So Gamma2_aa[c1,c2,a1,a2] uses indices (c1, c2, a1, a2) directly.
 */
inline double get_lambda2(
    int c1_orb, int c1_spin,
    int c2_orb, int c2_spin,
    int a1_orb, int a1_spin,
    int a2_orb, int a2_spin,
    const double* gamma_aa, const double* gamma_bb,
    const double* Gamma_aa, const double* Gamma_ab, const double* Gamma_bb,
    int N)
{
    // Count alpha/beta among creators and annihilators
    int c_alpha = (c1_spin == 0) + (c2_spin == 0);
    int a_alpha = (a1_spin == 0) + (a2_spin == 0);

    // Spin conservation: must have same number of alpha/beta
    if (c_alpha != a_alpha) return 0.0;

    if (c_alpha == 2) {
        // alpha-alpha sector
        // Both creators alpha, both annihilators alpha
        double G2 = Gamma_aa[idx4(c1_orb, c2_orb, a1_orb, a2_orb, N)];
        double g1 = gamma_aa[idx2(c1_orb, a1_orb, N)] * gamma_aa[idx2(c2_orb, a2_orb, N)];
        double g2 = gamma_aa[idx2(c1_orb, a2_orb, N)] * gamma_aa[idx2(c2_orb, a1_orb, N)];
        return G2 - g1 + g2;
    }
    else if (c_alpha == 0) {
        // beta-beta sector
        double G2 = Gamma_bb[idx4(c1_orb, c2_orb, a1_orb, a2_orb, N)];
        double g1 = gamma_bb[idx2(c1_orb, a1_orb, N)] * gamma_bb[idx2(c2_orb, a2_orb, N)];
        double g2 = gamma_bb[idx2(c1_orb, a2_orb, N)] * gamma_bb[idx2(c2_orb, a1_orb, N)];
        return G2 - g1 + g2;
    }
    else {
        // Mixed: 1 alpha, 1 beta among creators; 1 alpha, 1 beta among annihilators
        // Need to identify which is alpha and which is beta
        int ca_orb, cb_orb;  // alpha creator, beta creator
        int aa_orb, ab_orb;  // alpha annihilator, beta annihilator
        int sign = 1;

        // Sort creators: alpha first
        if (c1_spin == 0) {
            ca_orb = c1_orb; cb_orb = c2_orb;
            // No swap needed for creators
        } else {
            ca_orb = c2_orb; cb_orb = c1_orb;
            sign *= -1;  // Swap c1 <-> c2 in fermionic sense
        }

        // Sort annihilators: alpha first
        if (a1_spin == 0) {
            aa_orb = a1_orb; ab_orb = a2_orb;
            // No swap needed for annihilators
        } else {
            aa_orb = a2_orb; ab_orb = a1_orb;
            sign *= -1;  // Swap a1 <-> a2 in fermionic sense
        }

        // Gamma_ab[alpha_c, beta_c, alpha_a, beta_a]
        double G2 = Gamma_ab[idx4(ca_orb, cb_orb, aa_orb, ab_orb, N)];
        // Disconnected: gamma_a[alpha_c, alpha_a] * gamma_b[beta_c, beta_a]
        double g1 = gamma_aa[idx2(ca_orb, aa_orb, N)] * gamma_bb[idx2(cb_orb, ab_orb, N)];
        // No exchange term for alpha-beta (different spins can't exchange)
        return sign * (G2 - g1);
    }
}

/**
 * Recursive cumulant-2 expansion.
 *
 * Processes the first available creator, either as a singleton (paired with
 * one annihilator via gamma) or as part of a pair (with another creator and
 * two annihilators via lambda2).
 *
 * @param c_avail Available creator indices (into the original arrays)
 * @param a_avail Available annihilator indices (into the original arrays)
 * @param creators/annihilators The full creator/annihilator arrays
 * @param gamma/Gamma RDM tensors
 * @param N n_orb
 * @return The cumulant-2 approximation for this sub-problem
 */
double expand(
    std::vector<int>& c_avail,
    std::vector<int>& a_avail,
    const std::vector<std::pair<int,int>>& creators,
    const std::vector<std::pair<int,int>>& annihilators,
    const double* gamma_aa, const double* gamma_bb,
    const double* Gamma_aa, const double* Gamma_ab, const double* Gamma_bb,
    int N)
{
    int n = static_cast<int>(c_avail.size());
    if (n == 0) return 1.0;

    // Take the first available creator
    int c0_idx = c_avail[0];
    int c0_orb = creators[c0_idx].first;
    int c0_spin = creators[c0_idx].second;

    double result = 0.0;

    // === Option 1: Singleton — pair c0 with one annihilator ===
    for (int j = 0; j < n; ++j) {
        int aj_idx = a_avail[j];
        int aj_orb = annihilators[aj_idx].first;
        int aj_spin = annihilators[aj_idx].second;

        double g = get_gamma(c0_orb, c0_spin, aj_orb, aj_spin,
                             gamma_aa, gamma_bb, N);
        if (std::abs(g) < 1e-15) continue;

        // Sign: (-1)^j from removing annihilator at position j
        double sign = (j % 2 == 0) ? 1.0 : -1.0;

        // Remove c0 from c_avail and aj from a_avail
        // (modify in-place, restore after recursion)
        c_avail.erase(c_avail.begin());
        a_avail.erase(a_avail.begin() + j);

        result += sign * g * expand(c_avail, a_avail, creators, annihilators,
                                     gamma_aa, gamma_bb,
                                     Gamma_aa, Gamma_ab, Gamma_bb, N);

        // Restore
        c_avail.insert(c_avail.begin(), c0_idx);
        a_avail.insert(a_avail.begin() + j, aj_idx);
    }

    // === Option 2: Pair — pair c0 with another creator ci, using two annihilators ===
    for (int i = 1; i < n; ++i) {
        int ci_idx = c_avail[i];
        int ci_orb = creators[ci_idx].first;
        int ci_spin = creators[ci_idx].second;

        // Sign from moving creator ci past (i-1) creators to pair with c0
        double sign_c = ((i - 1) % 2 == 0) ? 1.0 : -1.0;

        for (int j = 0; j < n; ++j) {
            for (int k = j + 1; k < n; ++k) {
                int aj_idx = a_avail[j];
                int ak_idx = a_avail[k];

                double l2 = get_lambda2(
                    c0_orb, c0_spin,
                    ci_orb, ci_spin,
                    annihilators[aj_idx].first, annihilators[aj_idx].second,
                    annihilators[ak_idx].first, annihilators[ak_idx].second,
                    gamma_aa, gamma_bb,
                    Gamma_aa, Gamma_ab, Gamma_bb, N);
                if (std::abs(l2) < 1e-15) continue;

                // Sign from removing annihilators at positions j, k:
                // Remove j first: (-1)^j
                // Then remove k (now at position k-1): (-1)^(k-1)
                // Total: (-1)^(j+k-1)
                double sign_a = ((j + k - 1) % 2 == 0) ? 1.0 : -1.0;
                double sign = sign_c * sign_a;

                // Remove c0 (pos 0) and ci (pos i) from c_avail
                // Remove aj (pos j) and ak (pos k) from a_avail
                c_avail.erase(c_avail.begin() + i);
                c_avail.erase(c_avail.begin());
                a_avail.erase(a_avail.begin() + k);
                a_avail.erase(a_avail.begin() + j);

                result += sign * l2 * expand(c_avail, a_avail, creators, annihilators,
                                              gamma_aa, gamma_bb,
                                              Gamma_aa, Gamma_ab, Gamma_bb, N);

                // Restore (reverse order of removal)
                a_avail.insert(a_avail.begin() + j, aj_idx);
                a_avail.insert(a_avail.begin() + k, ak_idx);
                c_avail.insert(c_avail.begin(), c0_idx);
                c_avail.insert(c_avail.begin() + i, ci_idx);
            }
        }
    }

    return result;
}

} // anonymous namespace


double compute_rdm_element_cumulant2(
    const std::vector<std::pair<int,int>>& creators,
    const std::vector<std::pair<int,int>>& annihilators,
    const std::vector<double>& gamma_aa,
    const std::vector<double>& gamma_bb,
    const std::vector<double>& Gamma_aa,
    const std::vector<double>& Gamma_ab,
    const std::vector<double>& Gamma_bb,
    int n_orb)
{
    int n = static_cast<int>(creators.size());
    if (n != static_cast<int>(annihilators.size())) {
        return 0.0;  // Mismatched sizes
    }

    if (n == 0) return 1.0;

    if (n == 1) {
        return get_gamma(
            creators[0].first, creators[0].second,
            annihilators[0].first, annihilators[0].second,
            gamma_aa.data(), gamma_bb.data(), n_orb);
    }

    // For n == 2, use exact Gamma2 (no cumulant approximation needed)
    if (n == 2) {
        int c1_orb = creators[0].first, c1_spin = creators[0].second;
        int c2_orb = creators[1].first, c2_spin = creators[1].second;
        int a1_orb = annihilators[0].first, a1_spin = annihilators[0].second;
        int a2_orb = annihilators[1].first, a2_spin = annihilators[1].second;

        // Gamma2 + disconnected = full 2-RDM
        // lambda2 + A{gamma^2} = Gamma2
        // So directly return Gamma2 element
        // But Gamma2 storage depends on spin sector. Let's use lambda2 + A{gamma^2}.

        double g1 = get_gamma(c1_orb, c1_spin, a1_orb, a1_spin,
                              gamma_aa.data(), gamma_bb.data(), n_orb);
        double g2 = get_gamma(c2_orb, c2_spin, a2_orb, a2_spin,
                              gamma_aa.data(), gamma_bb.data(), n_orb);
        double g3 = get_gamma(c1_orb, c1_spin, a2_orb, a2_spin,
                              gamma_aa.data(), gamma_bb.data(), n_orb);
        double g4 = get_gamma(c2_orb, c2_spin, a1_orb, a1_spin,
                              gamma_aa.data(), gamma_bb.data(), n_orb);

        double l2 = get_lambda2(
            c1_orb, c1_spin, c2_orb, c2_spin,
            a1_orb, a1_spin, a2_orb, a2_spin,
            gamma_aa.data(), gamma_bb.data(),
            Gamma_aa.data(), Gamma_ab.data(), Gamma_bb.data(),
            n_orb);

        // Gamma_2 = lambda2 + A{gamma x gamma}
        // A{gamma x gamma}[c1c2,a1a2] = gamma[c1,a1]*gamma[c2,a2] - gamma[c1,a2]*gamma[c2,a1]
        return l2 + g1 * g2 - g3 * g4;
    }

    // For n >= 3, use recursive cumulant-2 expansion
    std::vector<int> c_avail(n), a_avail(n);
    for (int i = 0; i < n; ++i) {
        c_avail[i] = i;
        a_avail[i] = i;
    }

    return expand(c_avail, a_avail, creators, annihilators,
                  gamma_aa.data(), gamma_bb.data(),
                  Gamma_aa.data(), Gamma_ab.data(), Gamma_bb.data(),
                  n_orb);
}

} // namespace trimci_core
