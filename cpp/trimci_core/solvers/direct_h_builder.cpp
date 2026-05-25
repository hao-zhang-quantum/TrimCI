#include "direct_h_builder.hpp"

#include <vector>
#include <utility>
#include <algorithm>
#include <cstring>
#include <cstddef>

#include "bit_compat.hpp"

#ifdef _OPENMP
#include "omp_compat.hpp"
#endif

namespace trimci_core {

// ── Excitation type IDs (matching Python convention) ──
enum ExcType : int {
    EXC_S_ALPHA = 0,
    EXC_S_BETA  = 1,
    EXC_D_AA    = 2,
    EXC_D_AB    = 3,
    EXC_D_BB    = 4
};

// ── Determinant representation ──
struct DetBits {
    uint64_t alpha;
    uint64_t beta;
};

// ── Phase computation ──

/**
 * Count occupied orbitals strictly between positions pos1 and pos2
 * (excluding both endpoints) in a bitstring.
 */
inline int count_between(uint64_t bits, int pos1, int pos2) {
    int lo = (pos1 < pos2) ? pos1 : pos2;
    int hi = (pos1 < pos2) ? pos2 : pos1;
    if (hi - lo <= 1) return 0;
    uint64_t mask = ((1ULL << hi) - 1) & ~((1ULL << (lo + 1)) - 1);
    return popcount64(bits & mask);
}

// ── Excitation application with phase ──

struct ExcResult {
    uint64_t alpha;
    uint64_t beta;
    int phase;  // +1 or -1, or 0 for invalid
};

/**
 * Apply excitation operator τ̂_μ to a determinant and compute fermionic phase.
 */
inline ExcResult apply_excitation(uint64_t alpha, uint64_t beta,
                                  int exc_type, const int* idx) {
    ExcResult r;
    r.phase = 0;

    switch (exc_type) {
    case EXC_S_ALPHA: {
        int i = idx[0], a = idx[1];
        uint64_t mi = 1ULL << i, ma = 1ULL << a;
        if (!(alpha & mi) || (alpha & ma)) return r;
        r.alpha = (alpha ^ mi) | ma;
        r.beta = beta;
        r.phase = (count_between(alpha, i, a) & 1) ? -1 : 1;
        return r;
    }
    case EXC_S_BETA: {
        int i = idx[0], a = idx[1];
        uint64_t mi = 1ULL << i, ma = 1ULL << a;
        if (!(beta & mi) || (beta & ma)) return r;
        r.alpha = alpha;
        r.beta = (beta ^ mi) | ma;
        r.phase = (count_between(beta, i, a) & 1) ? -1 : 1;
        return r;
    }
    case EXC_D_AA: {
        int i = idx[0], j = idx[1], a = idx[2], b = idx[3];
        uint64_t mi = 1ULL << i, mj = 1ULL << j;
        uint64_t ma = 1ULL << a, mb = 1ULL << b;
        if (!(alpha & mi) || !(alpha & mj)) return r;
        if ((alpha & ma) || (alpha & mb)) return r;
        r.alpha = (alpha ^ mi ^ mj) | ma | mb;
        r.beta = beta;
        int cnt1 = count_between(alpha, i, a);
        uint64_t modified = (alpha ^ mi) | ma;
        int cnt2 = count_between(modified, j, b);
        r.phase = ((cnt1 + cnt2) & 1) ? -1 : 1;
        return r;
    }
    case EXC_D_BB: {
        int i = idx[0], j = idx[1], a = idx[2], b = idx[3];
        uint64_t mi = 1ULL << i, mj = 1ULL << j;
        uint64_t ma = 1ULL << a, mb = 1ULL << b;
        if (!(beta & mi) || !(beta & mj)) return r;
        if ((beta & ma) || (beta & mb)) return r;
        r.alpha = alpha;
        r.beta = (beta ^ mi ^ mj) | ma | mb;
        int cnt1 = count_between(beta, i, a);
        uint64_t modified = (beta ^ mi) | ma;
        int cnt2 = count_between(modified, j, b);
        r.phase = ((cnt1 + cnt2) & 1) ? -1 : 1;
        return r;
    }
    case EXC_D_AB: {
        int i = idx[0], j = idx[1], a = idx[2], b = idx[3];
        uint64_t mi = 1ULL << i, mj = 1ULL << j;
        uint64_t ma = 1ULL << a, mb = 1ULL << b;
        if (!(alpha & mi) || !(beta & mj)) return r;
        if ((alpha & ma) || (beta & mb)) return r;
        r.alpha = (alpha ^ mi) | ma;
        r.beta = (beta ^ mj) | mb;
        int cnt_a = count_between(alpha, i, a);
        int cnt_b = count_between(beta, j, b);
        r.phase = ((cnt_a + cnt_b) & 1) ? -1 : 1;
        return r;
    }
    default:
        return r;
    }
}

// ── ERI access helper ──
inline double get_eri(const double* eri, int n, int i, int j, int k, int l) {
    size_t idx = (((static_cast<size_t>(i) * n + j) * n + k) * n + l);
    return eri[idx];
}

// ── h1 access helper ──
inline double get_h1(const double* h1, int n, int p, int q) {
    return h1[static_cast<size_t>(p) * n + q];
}

// ── Extract occupied orbital indices from bitstring ──
inline void get_occupied(uint64_t bits, std::vector<int>& occ) {
    occ.clear();
    uint64_t temp = bits;
    while (temp) {
        int p = ctz64(temp);
        occ.push_back(p);
        temp &= temp - 1;  // Clear lowest set bit
    }
}

// ── Slater-Condon rules implementation ──

/**
 * Compute ⟨bra|H|ket⟩ using Slater-Condon rules.
 *
 * @param bra_alpha, bra_beta: Bra determinant bitstrings
 * @param ket_alpha, ket_beta: Ket determinant bitstrings
 * @param h1: One-body integrals (n_orb × n_orb)
 * @param eri: Two-body integrals (n_orb^4)
 * @param n_orb: Number of orbitals
 * @return: Matrix element value
 */
double slater_condon_element(
    uint64_t bra_alpha, uint64_t bra_beta,
    uint64_t ket_alpha, uint64_t ket_beta,
    const double* h1, const double* eri, int n_orb)
{
    // Count orbital differences
    uint64_t alpha_diff = bra_alpha ^ ket_alpha;
    uint64_t beta_diff = bra_beta ^ ket_beta;
    int n_alpha_diff = popcount64(alpha_diff) / 2;  // Each difference counted twice (in and out)
    int n_beta_diff = popcount64(beta_diff) / 2;
    int n_total_diff = n_alpha_diff + n_beta_diff;

    if (n_total_diff > 2) {
        return 0.0;
    }

    // Helper vectors for occupied orbitals
    thread_local std::vector<int> occ_a_bra, occ_b_bra, occ_a_ket, occ_b_ket;

    if (n_total_diff == 0) {
        // ── Diagonal: ⟨D|H|D⟩ ──
        get_occupied(bra_alpha, occ_a_bra);
        get_occupied(bra_beta, occ_b_bra);

        double E = 0.0;
        // One-electron: Σ_i h_ii
        for (int i : occ_a_bra) E += get_h1(h1, n_orb, i, i);
        for (int i : occ_b_bra) E += get_h1(h1, n_orb, i, i);

        // Two-electron α-α: Σ_{i<j} [(ii|jj) - (ij|ji)]
        for (size_t ii = 0; ii < occ_a_bra.size(); ++ii) {
            int i = occ_a_bra[ii];
            for (size_t jj = ii + 1; jj < occ_a_bra.size(); ++jj) {
                int j = occ_a_bra[jj];
                E += get_eri(eri, n_orb, i, i, j, j) - get_eri(eri, n_orb, i, j, j, i);
            }
        }

        // Two-electron β-β: Σ_{i<j} [(ii|jj) - (ij|ji)]
        for (size_t ii = 0; ii < occ_b_bra.size(); ++ii) {
            int i = occ_b_bra[ii];
            for (size_t jj = ii + 1; jj < occ_b_bra.size(); ++jj) {
                int j = occ_b_bra[jj];
                E += get_eri(eri, n_orb, i, i, j, j) - get_eri(eri, n_orb, i, j, j, i);
            }
        }

        // Two-electron α-β: Σ_{ij} (ii|jj) [Coulomb only, no exchange]
        for (int i : occ_a_bra) {
            for (int j : occ_b_bra) {
                E += get_eri(eri, n_orb, i, i, j, j);
            }
        }

        return E;
    }
    else if (n_total_diff == 1) {
        // ── Single excitation: i → a ──
        get_occupied(ket_alpha, occ_a_ket);
        get_occupied(ket_beta, occ_b_ket);

        if (n_alpha_diff == 1) {
            // Alpha single excitation
            uint64_t in_ket_only = ket_alpha & ~bra_alpha;  // orbital i (occupied in ket, not in bra)
            uint64_t in_bra_only = bra_alpha & ~ket_alpha;  // orbital a (occupied in bra, not in ket)
            int i = ctz64(in_ket_only);
            int a = ctz64(in_bra_only);

            // Phase: count α electrons between i and a in ket
            int phase = (count_between(ket_alpha, i, a) & 1) ? -1 : 1;

            // One-electron: h_ai
            double elem = get_h1(h1, n_orb, a, i);

            // Two-electron with other α: Σ_k [(ai|kk) - (ak|ki)] for k ∈ occ_α, k ≠ i
            for (int k : occ_a_ket) {
                if (k != i) {
                    elem += get_eri(eri, n_orb, a, i, k, k) - get_eri(eri, n_orb, a, k, k, i);
                }
            }

            // Two-electron with β: Σ_k (ai|kk) for k ∈ occ_β [Coulomb only]
            for (int k : occ_b_ket) {
                elem += get_eri(eri, n_orb, a, i, k, k);
            }

            return phase * elem;
        }
        else {
            // Beta single excitation
            uint64_t in_ket_only = ket_beta & ~bra_beta;
            uint64_t in_bra_only = bra_beta & ~ket_beta;
            int i = ctz64(in_ket_only);
            int a = ctz64(in_bra_only);

            int phase = (count_between(ket_beta, i, a) & 1) ? -1 : 1;

            double elem = get_h1(h1, n_orb, a, i);

            // Two-electron with other β: Σ_k [(ai|kk) - (ak|ki)]
            for (int k : occ_b_ket) {
                if (k != i) {
                    elem += get_eri(eri, n_orb, a, i, k, k) - get_eri(eri, n_orb, a, k, k, i);
                }
            }

            // Two-electron with α: Σ_k (ai|kk) [Coulomb only]
            for (int k : occ_a_ket) {
                elem += get_eri(eri, n_orb, a, i, k, k);
            }

            return phase * elem;
        }
    }
    else {  // n_total_diff == 2
        // ── Double excitation: (i,j) → (a,b) ──

        if (n_alpha_diff == 2) {
            // Double α-α
            uint64_t in_ket_only = ket_alpha & ~bra_alpha;
            uint64_t in_bra_only = bra_alpha & ~ket_alpha;

            // Extract i, j (from ket, not in bra) and a, b (from bra, not in ket)
            int i = ctz64(in_ket_only);
            in_ket_only &= in_ket_only - 1;
            int j = ctz64(in_ket_only);

            int a = ctz64(in_bra_only);
            in_bra_only &= in_bra_only - 1;
            int b = ctz64(in_bra_only);

            // Ensure canonical ordering
            if (i > j) std::swap(i, j);
            if (a > b) std::swap(a, b);

            // Phase calculation: sequential application i→a, then j→b
            int cnt1 = count_between(ket_alpha, i, a);
            uint64_t modified = (ket_alpha & ~(1ULL << i)) | (1ULL << a);
            int cnt2 = count_between(modified, j, b);
            int phase = ((cnt1 + cnt2) & 1) ? -1 : 1;

            // (ai|bj) - (aj|bi)
            double elem = get_eri(eri, n_orb, a, i, b, j) - get_eri(eri, n_orb, a, j, b, i);
            return phase * elem;
        }
        else if (n_beta_diff == 2) {
            // Double β-β
            uint64_t in_ket_only = ket_beta & ~bra_beta;
            uint64_t in_bra_only = bra_beta & ~ket_beta;

            int i = ctz64(in_ket_only);
            in_ket_only &= in_ket_only - 1;
            int j = ctz64(in_ket_only);

            int a = ctz64(in_bra_only);
            in_bra_only &= in_bra_only - 1;
            int b = ctz64(in_bra_only);

            if (i > j) std::swap(i, j);
            if (a > b) std::swap(a, b);

            int cnt1 = count_between(ket_beta, i, a);
            uint64_t modified = (ket_beta & ~(1ULL << i)) | (1ULL << a);
            int cnt2 = count_between(modified, j, b);
            int phase = ((cnt1 + cnt2) & 1) ? -1 : 1;

            double elem = get_eri(eri, n_orb, a, i, b, j) - get_eri(eri, n_orb, a, j, b, i);
            return phase * elem;
        }
        else {
            // Double α-β (n_alpha_diff == 1 && n_beta_diff == 1)
            uint64_t in_ket_a = ket_alpha & ~bra_alpha;
            uint64_t in_bra_a = bra_alpha & ~ket_alpha;
            uint64_t in_ket_b = ket_beta & ~bra_beta;
            uint64_t in_bra_b = bra_beta & ~ket_beta;

            int i_a = ctz64(in_ket_a);  // α orbital destroyed
            int a_a = ctz64(in_bra_a);  // α orbital created
            int i_b = ctz64(in_ket_b);  // β orbital destroyed
            int a_b = ctz64(in_bra_b);  // β orbital created

            // Phase: independent α and β
            int cnt_a = count_between(ket_alpha, i_a, a_a);
            int cnt_b = count_between(ket_beta, i_b, a_b);
            int phase = ((cnt_a + cnt_b) & 1) ? -1 : 1;

            // α-β: only Coulomb term (a_α i_α | a_β i_β)
            double elem = get_eri(eri, n_orb, a_a, i_a, a_b, i_b);
            return phase * elem;
        }
    }
}

// ── Main build function ──

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
    double* H_out)
{
    // ── H[0,0]: Reference energy ──
    // H[0,0] = Σ_{I,J} c_I c_J ⟨D_I|H|D_J⟩
    double H00 = 0.0;
    for (int I = 0; I < n_ref; ++I) {
        for (int J = 0; J < n_ref; ++J) {
            double h_IJ = slater_condon_element(
                ref_alpha[I], ref_beta[I],
                ref_alpha[J], ref_beta[J],
                h1, eri, n_orb);
            H00 += ref_coeffs[I] * ref_coeffs[J] * h_IJ;
        }
    }
    H_out[0] = H00;

    // ── Precompute all excited determinant lists ──
    // Cost: O(n_exc × n_ref) once. Reused by H[0,μ] and H[μ,ν].
    std::vector<std::vector<std::pair<DetBits, double>>> all_exc_dets(n_exc);
    for (int mu = 0; mu < n_exc; ++mu) {
        const int* idx_mu = &exc_indices[mu * 4];
        int etype_mu = exc_types[mu];
        all_exc_dets[mu].reserve(n_ref);
        for (int I = 0; I < n_ref; ++I) {
            ExcResult res = apply_excitation(
                ref_alpha[I], ref_beta[I], etype_mu, idx_mu);
            if (res.phase != 0) {
                DetBits d = {res.alpha, res.beta};
                double coeff = ref_coeffs[I] * res.phase;
                all_exc_dets[mu].push_back({d, coeff});
            }
        }
    }

    // ── H[0,μ] and H[μ,0]: Reference-excitation coupling ──
    // H[0,μ] = Σ_I c_I Σ_{(det_mu, c_mu)} ⟨D_I|H|det_mu⟩ * c_mu
    for (int mu = 0; mu < n_exc; ++mu) {
        int row = mu + 1;
        const auto& mu_dets = all_exc_dets[mu];

        double H_0mu = 0.0;
        for (int I = 0; I < n_ref; ++I) {
            for (const auto& [det_mu, c_mu] : mu_dets) {
                double h_elem = slater_condon_element(
                    ref_alpha[I], ref_beta[I],
                    det_mu.alpha, det_mu.beta,
                    h1, eri, n_orb);
                H_0mu += ref_coeffs[I] * c_mu * h_elem;
            }
        }

        H_out[row] = H_0mu;                          // H[0, mu+1]
        H_out[row * n_basis] = H_0mu;                // H[mu+1, 0] (symmetric)
    }

    // ── H[μ,ν]: Excitation-excitation matrix ──
    // H[μ,ν] = Σ_{I,J} c_I c_J φ^μ_I φ^ν_J ⟨D^μ_I|H|D^ν_J⟩
    // Use OpenMP for parallelization over μ rows

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int mu = 0; mu < n_exc; ++mu) {
        int row_mu = mu + 1;
        const auto& mu_dets = all_exc_dets[mu];
        if (mu_dets.empty()) continue;

        // Compute H[μ,ν] for ν >= μ (use symmetry)
        for (int nu = mu; nu < n_exc; ++nu) {
            int row_nu = nu + 1;
            const auto& nu_dets = all_exc_dets[nu];
            if (nu_dets.empty()) continue;

            double H_munu = 0.0;

            for (const auto& [det_mu, c_mu] : mu_dets) {
                for (const auto& [det_nu, c_nu] : nu_dets) {
                    double h_elem = slater_condon_element(
                        det_mu.alpha, det_mu.beta,
                        det_nu.alpha, det_nu.beta,
                        h1, eri, n_orb);
                    H_munu += c_mu * c_nu * h_elem;
                }
            }

            H_out[row_mu * n_basis + row_nu] = H_munu;
            if (mu != nu) {
                H_out[row_nu * n_basis + row_mu] = H_munu;  // Symmetric
            }
        }
    }
}

} // namespace trimci_core
