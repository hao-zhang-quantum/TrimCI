#pragma once
/**
 * Fast Expansion: shared diagonal element helpers.
 *
 * Used by streaming_pt2.cpp and enpt3.cpp.
 *   - compute_diagonal_element: full O(N_occ²) computation
 *   - compute_diagonal_incremental: O(N_occ) given parent diagonal
 */

#include <vector>
#include <cstdint>
#include <array>
#include "determinant.hpp"
#include "hamiltonian.hpp"
#include "bit_compat.hpp"

namespace trimci_core {
namespace fe {

// ============================================================================
// Full diagonal H_aa = <a|H|a> from scratch (O(N_occ²))
// ============================================================================
template<typename StorageType>
inline double compute_diagonal_element(
    const DeterminantT<StorageType>& det,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    int n_orb)
{
    auto occ_a = det.getOccupiedAlpha();
    auto occ_b = det.getOccupiedBeta();

    auto eri_idx = [&](int i, int j, int k, int l) -> double {
        return eri[((size_t(i) * n_orb + j) * n_orb + k) * n_orb + l];
    };

    double E = 0.0;

    // One-body
    for (int p : occ_a) E += h1[p][p];
    for (int p : occ_b) E += h1[p][p];

    // α-α Coulomb-exchange
    for (size_t i = 0; i < occ_a.size(); ++i)
        for (size_t j = i + 1; j < occ_a.size(); ++j) {
            int p = occ_a[i], q = occ_a[j];
            E += eri_idx(p, p, q, q) - eri_idx(p, q, q, p);
        }

    // β-β Coulomb-exchange
    for (size_t i = 0; i < occ_b.size(); ++i)
        for (size_t j = i + 1; j < occ_b.size(); ++j) {
            int p = occ_b[i], q = occ_b[j];
            E += eri_idx(p, p, q, q) - eri_idx(p, q, q, p);
        }

    // α-β Coulomb only
    for (int p : occ_a)
        for (int q : occ_b)
            E += eri_idx(p, p, q, q);

    return E;
}

// ============================================================================
// Incremental diagonal: H_aa = H_ii + ΔH  (O(N_occ) per excitation)
//
// Given parent det_i with known H_ii, compute H_aa for child det_a obtained
// via single/double excitation. Uses XOR to detect excitation type.
// ============================================================================
template<typename StorageType>
inline double compute_diagonal_incremental(
    const DeterminantT<StorageType>& det_i,
    const DeterminantT<StorageType>& det_a,
    double H_ii,
    const std::vector<int>& occ_a_i,
    const std::vector<int>& occ_b_i,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    int n_orb)
{
    using BitOpsDetail = detail::HamiltonianBitOps<StorageType>;

    auto eri_idx = [&](int i, int j, int k, int l) -> double {
        return eri[((size_t(i) * n_orb + j) * n_orb + k) * n_orb + l];
    };

    auto xor_storage = [](const StorageType& a, const StorageType& b) -> StorageType {
        if constexpr (std::is_same_v<StorageType, uint64_t>) {
            return a ^ b;
        } else {
            StorageType res;
            for (size_t i = 0; i < a.size(); ++i) res[i] = a[i] ^ b[i];
            return res;
        }
    };

    auto popcount_storage = [](const StorageType& s) -> int {
        if constexpr (std::is_same_v<StorageType, uint64_t>) {
            return __builtin_popcountll(s);
        } else {
            int count = 0;
            for (size_t i = 0; i < s.size(); ++i) count += __builtin_popcountll(s[i]);
            return count;
        }
    };

    StorageType diff_a = xor_storage(det_i.alpha, det_a.alpha);
    StorageType diff_b = xor_storage(det_i.beta, det_a.beta);
    int n_diff_a = popcount_storage(diff_a);
    int n_diff_b = popcount_storage(diff_b);

    // Extract up to 2 set-bit indices from a StorageType mask
    auto extract2 = [](const StorageType& mask_in, int& o1, int& o2) -> int {
        if constexpr (std::is_same_v<StorageType, uint64_t>) {
            uint64_t mask = mask_in;
            if (mask == 0) return 0;
            o1 = __builtin_ctzll(mask);
            mask &= mask - 1;
            if (mask == 0) return 1;
            o2 = __builtin_ctzll(mask);
            return 2;
        } else {
            int count = 0;
            for (size_t i = 0; i < mask_in.size(); ++i) {
                uint64_t w = mask_in[i];
                while (w) {
                    int bit = __builtin_ctzll(w) + static_cast<int>(i * 64);
                    if (count == 0) { o1 = bit; count = 1; }
                    else { o2 = bit; return 2; }
                    w &= w - 1;
                }
            }
            return count;
        }
    };

    int des_a1 = 0, des_a2 = 0, cre_a1 = 0, cre_a2 = 0;
    int des_b1 = 0, des_b2 = 0, cre_b1 = 0, cre_b2 = 0;
    extract2(BitOpsDetail::and_not(det_i.alpha, det_a.alpha), des_a1, des_a2);
    extract2(BitOpsDetail::and_not(det_a.alpha, det_i.alpha), cre_a1, cre_a2);
    extract2(BitOpsDetail::and_not(det_i.beta, det_a.beta), des_b1, des_b2);
    extract2(BitOpsDetail::and_not(det_a.beta, det_i.beta), cre_b1, cre_b2);

    double delta = 0.0;

    if (n_diff_a == 2 && n_diff_b == 0) {
        // α single: p→r
        int p = des_a1, r = cre_a1;
        delta = h1[r][r] - h1[p][p];
        for (int q : occ_a_i) {
            if (q == p) continue;
            delta += (eri_idx(r,r,q,q) - eri_idx(r,q,q,r))
                   - (eri_idx(p,p,q,q) - eri_idx(p,q,q,p));
        }
        for (int q : occ_b_i) {
            delta += eri_idx(r,r,q,q) - eri_idx(p,p,q,q);
        }
    }
    else if (n_diff_a == 0 && n_diff_b == 2) {
        // β single: p→r
        int p = des_b1, r = cre_b1;
        delta = h1[r][r] - h1[p][p];
        for (int q : occ_b_i) {
            if (q == p) continue;
            delta += (eri_idx(r,r,q,q) - eri_idx(r,q,q,r))
                   - (eri_idx(p,p,q,q) - eri_idx(p,q,q,p));
        }
        for (int q : occ_a_i) {
            delta += eri_idx(r,r,q,q) - eri_idx(p,p,q,q);
        }
    }
    else if (n_diff_a == 4 && n_diff_b == 0) {
        // αα double: p,q→r,s
        int p = des_a1, q = des_a2, r = cre_a1, s = cre_a2;
        delta = h1[r][r] + h1[s][s] - h1[p][p] - h1[q][q];
        for (int t : occ_a_i) {
            if (t == p || t == q) continue;
            delta += (eri_idx(r,r,t,t) - eri_idx(r,t,t,r))
                   + (eri_idx(s,s,t,t) - eri_idx(s,t,t,s))
                   - (eri_idx(p,p,t,t) - eri_idx(p,t,t,p))
                   - (eri_idx(q,q,t,t) - eri_idx(q,t,t,q));
        }
        delta += (eri_idx(r,r,s,s) - eri_idx(r,s,s,r))
               - (eri_idx(p,p,q,q) - eri_idx(p,q,q,p));
        for (int t : occ_b_i) {
            delta += eri_idx(r,r,t,t) + eri_idx(s,s,t,t)
                   - eri_idx(p,p,t,t) - eri_idx(q,q,t,t);
        }
    }
    else if (n_diff_a == 0 && n_diff_b == 4) {
        // ββ double: p,q→r,s
        int p = des_b1, q = des_b2, r = cre_b1, s = cre_b2;
        delta = h1[r][r] + h1[s][s] - h1[p][p] - h1[q][q];
        for (int t : occ_b_i) {
            if (t == p || t == q) continue;
            delta += (eri_idx(r,r,t,t) - eri_idx(r,t,t,r))
                   + (eri_idx(s,s,t,t) - eri_idx(s,t,t,s))
                   - (eri_idx(p,p,t,t) - eri_idx(p,t,t,p))
                   - (eri_idx(q,q,t,t) - eri_idx(q,t,t,q));
        }
        delta += (eri_idx(r,r,s,s) - eri_idx(r,s,s,r))
               - (eri_idx(p,p,q,q) - eri_idx(p,q,q,p));
        for (int t : occ_a_i) {
            delta += eri_idx(r,r,t,t) + eri_idx(s,s,t,t)
                   - eri_idx(p,p,t,t) - eri_idx(q,q,t,t);
        }
    }
    else if (n_diff_a == 2 && n_diff_b == 2) {
        // αβ double: p→r in α, q→s in β
        int p = des_a1, r = cre_a1;
        int q = des_b1, s = cre_b1;
        delta = h1[r][r] - h1[p][p] + h1[s][s] - h1[q][q];
        for (int t : occ_a_i) {
            if (t == p) continue;
            delta += (eri_idx(r,r,t,t) - eri_idx(r,t,t,r))
                   - (eri_idx(p,p,t,t) - eri_idx(p,t,t,p));
        }
        for (int t : occ_b_i) {
            if (t == q) continue;
            delta += (eri_idx(s,s,t,t) - eri_idx(s,t,t,s))
                   - (eri_idx(q,q,t,t) - eri_idx(q,t,t,q));
        }
        for (int a : occ_a_i) {
            if (a == p) continue;
            delta += eri_idx(a,a,s,s) - eri_idx(a,a,q,q);
        }
        for (int b : occ_b_i) {
            if (b == q) continue;
            delta += eri_idx(r,r,b,b) - eri_idx(p,p,b,b);
        }
        delta += eri_idx(r,r,s,s) - eri_idx(p,p,q,q);
    }

    return H_ii + delta;
}

// ============================================================================
// Merged excitation enumeration + H_ai + incremental diagonal.
//
// Combines for_each_excitation_t, compute_H_ij_t, and
// compute_diagonal_incremental into a single pass.  Orbital indices are known
// at enumeration time, so XOR/popcount re-detection is eliminated entirely.
//
// cb(det_a, H_ai_signed, H_aa) is called for every single/double excitation.
// ============================================================================
template<typename StorageType, typename Callback>
void for_each_excitation_H_diag(
    const DeterminantT<StorageType>& det,
    double H_ii,
    const std::vector<int>& occ_alpha,
    const std::vector<int>& virt_alpha,
    const std::vector<int>& occ_beta,
    const std::vector<int>& virt_beta,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    int n_orb,
    Callback&& cb,
    double integral_threshold = 0.0)
{
    using BitOpsType = BitOps<StorageType>;

    auto eri_idx = [&](int i, int j, int k, int l) -> double {
        return eri[((size_t(i) * n_orb + j) * n_orb + k) * n_orb + l];
    };

    const size_t na = occ_alpha.size();
    const size_t nb = occ_beta.size();
    const bool do_screen = (integral_threshold > 0.0);

    // ---- α single: m → p ----
    for (int m : occ_alpha)
    for (int p : virt_alpha) {
        // H_ai
        double H = h1[m][p];
        for (int n : occ_alpha)
            if (n != m) H += eri_idx(m, p, n, n) - eri_idx(m, n, n, p);
        for (int n : occ_beta)
            H += eri_idx(m, p, n, n);

        // Screen singles by |H_ai| (after full Fock element computed)
        if (do_screen && std::abs(H) < integral_threshold) continue;

        // phase
        StorageType na_s = det.alpha;
        BitOpsType::flip_bit(na_s, m);
        BitOpsType::flip_bit(na_s, p);
        int phase = detail::cre_des_sign_t(m, p, det.alpha);
        double H_ai = phase * H;

        // diagonal increment
        double delta = h1[p][p] - h1[m][m];
        for (int q : occ_alpha) {
            if (q == m) continue;
            delta += (eri_idx(p,p,q,q) - eri_idx(p,q,q,p))
                   - (eri_idx(m,m,q,q) - eri_idx(m,q,q,m));
        }
        for (int q : occ_beta) {
            delta += eri_idx(p,p,q,q) - eri_idx(m,m,q,q);
        }

        cb(DeterminantT<StorageType>(na_s, det.beta), H_ai, H_ii + delta);
    }

    // ---- β single: m → p ----
    for (int m : occ_beta)
    for (int p : virt_beta) {
        double H = h1[m][p];
        for (int n : occ_beta)
            if (n != m) H += eri_idx(m, p, n, n) - eri_idx(m, n, n, p);
        for (int n : occ_alpha)
            H += eri_idx(m, p, n, n);

        if (do_screen && std::abs(H) < integral_threshold) continue;

        StorageType nb_s = det.beta;
        BitOpsType::flip_bit(nb_s, m);
        BitOpsType::flip_bit(nb_s, p);
        int phase = detail::cre_des_sign_t(m, p, det.beta);
        double H_ai = phase * H;

        double delta = h1[p][p] - h1[m][m];
        for (int q : occ_beta) {
            if (q == m) continue;
            delta += (eri_idx(p,p,q,q) - eri_idx(p,q,q,p))
                   - (eri_idx(m,m,q,q) - eri_idx(m,q,q,m));
        }
        for (int q : occ_alpha) {
            delta += eri_idx(p,p,q,q) - eri_idx(m,m,q,q);
        }

        cb(DeterminantT<StorageType>(det.alpha, nb_s), H_ai, H_ii + delta);
    }

    // ---- αα double: (m, n) → (p, q) with m<n, p<q ----
    for (size_t ii = 0; ii < na; ++ii)
    for (size_t jj = ii + 1; jj < na; ++jj) {
        int m = occ_alpha[ii], n = occ_alpha[jj];
        for (int p : virt_alpha)
        for (int q : virt_alpha) {
            if (q <= p) continue;

            // Pre-screen: check driving integrals before expensive work
            if (do_screen) {
                double eri1 = std::abs(eri_idx(m, p, n, q));
                double eri2 = std::abs(eri_idx(m, q, n, p));
                if (std::max(eri1, eri2) < integral_threshold) continue;
            }

            double H_raw = eri_idx(m, p, n, q) - eri_idx(m, q, n, p);

            StorageType na_s = det.alpha;
            BitOpsType::flip_bit(na_s, m);
            BitOpsType::flip_bit(na_s, n);
            BitOpsType::flip_bit(na_s, p);
            BitOpsType::flip_bit(na_s, q);
            // Two-step phase: m→p, then n→q
            int ph1 = detail::cre_des_sign_t(p, m, det.alpha);
            StorageType tmp_ph = det.alpha;
            BitOpsType::clear_bit(tmp_ph, m);
            BitOpsType::set_bit(tmp_ph, p);
            int phase = ph1 * detail::cre_des_sign_t(q, n, tmp_ph);
            double H_ai = phase * H_raw;

            double delta = h1[p][p] + h1[q][q] - h1[m][m] - h1[n][n];
            for (int t : occ_alpha) {
                if (t == m || t == n) continue;
                delta += (eri_idx(p,p,t,t) - eri_idx(p,t,t,p))
                       + (eri_idx(q,q,t,t) - eri_idx(q,t,t,q))
                       - (eri_idx(m,m,t,t) - eri_idx(m,t,t,m))
                       - (eri_idx(n,n,t,t) - eri_idx(n,t,t,n));
            }
            delta += (eri_idx(p,p,q,q) - eri_idx(p,q,q,p))
                   - (eri_idx(m,m,n,n) - eri_idx(m,n,n,m));
            for (int t : occ_beta) {
                delta += eri_idx(p,p,t,t) + eri_idx(q,q,t,t)
                       - eri_idx(m,m,t,t) - eri_idx(n,n,t,t);
            }

            cb(DeterminantT<StorageType>(na_s, det.beta), H_ai, H_ii + delta);
        }
    }

    // ---- ββ double: (m, n) → (p, q) with m<n, p<q ----
    for (size_t ii = 0; ii < nb; ++ii)
    for (size_t jj = ii + 1; jj < nb; ++jj) {
        int m = occ_beta[ii], n = occ_beta[jj];
        for (int p : virt_beta)
        for (int q : virt_beta) {
            if (q <= p) continue;

            // Pre-screen: check driving integrals before expensive work
            if (do_screen) {
                double eri1 = std::abs(eri_idx(m, p, n, q));
                double eri2 = std::abs(eri_idx(m, q, n, p));
                if (std::max(eri1, eri2) < integral_threshold) continue;
            }

            double H_raw = eri_idx(m, p, n, q) - eri_idx(m, q, n, p);

            StorageType nb_s = det.beta;
            BitOpsType::flip_bit(nb_s, m);
            BitOpsType::flip_bit(nb_s, n);
            BitOpsType::flip_bit(nb_s, p);
            BitOpsType::flip_bit(nb_s, q);
            int ph1 = detail::cre_des_sign_t(p, m, det.beta);
            StorageType tmp_ph = det.beta;
            BitOpsType::clear_bit(tmp_ph, m);
            BitOpsType::set_bit(tmp_ph, p);
            int phase = ph1 * detail::cre_des_sign_t(q, n, tmp_ph);
            double H_ai = phase * H_raw;

            double delta = h1[p][p] + h1[q][q] - h1[m][m] - h1[n][n];
            for (int t : occ_beta) {
                if (t == m || t == n) continue;
                delta += (eri_idx(p,p,t,t) - eri_idx(p,t,t,p))
                       + (eri_idx(q,q,t,t) - eri_idx(q,t,t,q))
                       - (eri_idx(m,m,t,t) - eri_idx(m,t,t,m))
                       - (eri_idx(n,n,t,t) - eri_idx(n,t,t,n));
            }
            delta += (eri_idx(p,p,q,q) - eri_idx(p,q,q,p))
                   - (eri_idx(m,m,n,n) - eri_idx(m,n,n,m));
            for (int t : occ_alpha) {
                delta += eri_idx(p,p,t,t) + eri_idx(q,q,t,t)
                       - eri_idx(m,m,t,t) - eri_idx(n,n,t,t);
            }

            cb(DeterminantT<StorageType>(det.alpha, nb_s), H_ai, H_ii + delta);
        }
    }

    // ---- αβ mixed double: m_α→p_α, m_β→p_β ----
    for (int ma : occ_alpha)
    for (int pa : virt_alpha) {
        StorageType am = det.alpha;
        BitOpsType::flip_bit(am, ma);
        BitOpsType::flip_bit(am, pa);
        int phase_a = detail::cre_des_sign_t(ma, pa, det.alpha);

        for (int mb : occ_beta)
        for (int pb : virt_beta) {
            // Pre-screen: check driving integral (single ERI for αβ)
            if (do_screen && std::abs(eri_idx(ma, pa, mb, pb)) < integral_threshold) continue;

            double H_raw = eri_idx(ma, pa, mb, pb);

            StorageType bm = det.beta;
            BitOpsType::flip_bit(bm, mb);
            BitOpsType::flip_bit(bm, pb);
            int phase_b = detail::cre_des_sign_t(mb, pb, det.beta);
            double H_ai = phase_a * phase_b * H_raw;

            // Diagonal increment for αβ double: (ma→pa in α, mb→pb in β)
            double delta = h1[pa][pa] - h1[ma][ma] + h1[pb][pb] - h1[mb][mb];
            for (int t : occ_alpha) {
                if (t == ma) continue;
                delta += (eri_idx(pa,pa,t,t) - eri_idx(pa,t,t,pa))
                       - (eri_idx(ma,ma,t,t) - eri_idx(ma,t,t,ma));
            }
            for (int t : occ_beta) {
                if (t == mb) continue;
                delta += (eri_idx(pb,pb,t,t) - eri_idx(pb,t,t,pb))
                       - (eri_idx(mb,mb,t,t) - eri_idx(mb,t,t,mb));
            }
            // Cross terms: new α with all β (except replaced), new β with all α (except replaced)
            for (int a : occ_alpha) {
                if (a == ma) continue;
                delta += eri_idx(a,a,pb,pb) - eri_idx(a,a,mb,mb);
            }
            for (int b : occ_beta) {
                if (b == mb) continue;
                delta += eri_idx(pa,pa,b,b) - eri_idx(ma,ma,b,b);
            }
            delta += eri_idx(pa,pa,pb,pb) - eri_idx(ma,ma,mb,mb);

            cb(DeterminantT<StorageType>(am, bm), H_ai, H_ii + delta);
        }
    }
}

// ============================================================================
// Precomputed Coulomb-Exchange column sums for O(1) diagonal increments.
// ============================================================================
struct OrbitalSums {
    std::vector<double> JK_aa;  // JK_aa[p] = Σ_{q∈occ_α} [eri(p,p,q,q) - eri(p,q,q,p)]
    std::vector<double> JK_bb;  // JK_bb[p] = Σ_{q∈occ_β} [eri(p,p,q,q) - eri(p,q,q,p)]
    std::vector<double> J_ab;   // J_ab[p]  = Σ_{q∈occ_β} eri(p,p,q,q)  (cross-spin Coulomb)
    std::vector<double> J_ba;   // J_ba[p]  = Σ_{q∈occ_α} eri(p,p,q,q)  (cross-spin Coulomb)
};

inline OrbitalSums precompute_orbital_sums(
    const std::vector<int>& occ_alpha,
    const std::vector<int>& occ_beta,
    const std::vector<double>& eri,
    int n_orb)
{
    auto eri_idx = [&](int i, int j, int k, int l) -> double {
        return eri[((size_t(i) * n_orb + j) * n_orb + k) * n_orb + l];
    };

    OrbitalSums s;
    s.JK_aa.resize(n_orb, 0.0);
    s.JK_bb.resize(n_orb, 0.0);
    s.J_ab.resize(n_orb, 0.0);
    s.J_ba.resize(n_orb, 0.0);

    for (int p = 0; p < n_orb; ++p) {
        for (int q : occ_alpha) {
            s.JK_aa[p] += eri_idx(p,p,q,q) - eri_idx(p,q,q,p);
            s.J_ba[p]  += eri_idx(p,p,q,q);
        }
        for (int q : occ_beta) {
            s.JK_bb[p] += eri_idx(p,p,q,q) - eri_idx(p,q,q,p);
            s.J_ab[p]  += eri_idx(p,p,q,q);
        }
    }
    return s;
}

// ============================================================================
// Optimized version with precomputed column sums: O(1) diagonal per excitation.
// ============================================================================
template<typename StorageType, typename Callback>
void for_each_excitation_H_diag_fast(
    const DeterminantT<StorageType>& det,
    double H_ii,
    const std::vector<int>& occ_alpha,
    const std::vector<int>& virt_alpha,
    const std::vector<int>& occ_beta,
    const std::vector<int>& virt_beta,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    int n_orb,
    const OrbitalSums& sums,
    Callback&& cb,
    double integral_threshold = 0.0,
    size_t* n_screened_out = nullptr)
{
    using BitOpsType = BitOps<StorageType>;

    auto eri_idx = [&](int i, int j, int k, int l) -> double {
        return eri[((size_t(i) * n_orb + j) * n_orb + k) * n_orb + l];
    };

    // Shorthand for Coulomb-Exchange between two orbitals
    auto CX = [&](int a, int b) -> double {
        return eri_idx(a,a,b,b) - eri_idx(a,b,b,a);
    };

    const bool do_screen = (integral_threshold > 0.0);

    // ---- α single: m → p ----
    for (int m : occ_alpha)
    for (int p : virt_alpha) {
        // H_ai (still O(N_occ) — would need full Fock matrix to optimize)
        double H = h1[m][p];
        for (int n : occ_alpha)
            if (n != m) H += eri_idx(m, p, n, n) - eri_idx(m, n, n, p);
        for (int n : occ_beta)
            H += eri_idx(m, p, n, n);

        if (do_screen && std::abs(H) < integral_threshold) {
            if (n_screened_out) ++(*n_screened_out);
            continue;
        }

        StorageType na_s = det.alpha;
        BitOpsType::flip_bit(na_s, m);
        BitOpsType::flip_bit(na_s, p);
        int phase = detail::cre_des_sign_t(m, p, det.alpha);
        double H_ai = phase * H;

        // O(1) diagonal: delta = h1 + (JK_aa[p]-JK_aa[m]) - CX(p,m) + (J_ab[p]-J_ab[m])
        double delta = h1[p][p] - h1[m][m]
            + (sums.JK_aa[p] - sums.JK_aa[m])
            - CX(p, m)          // remove m's contribution from p's sum
            + (sums.J_ab[p] - sums.J_ab[m]);

        cb(DeterminantT<StorageType>(na_s, det.beta), H_ai, H_ii + delta);
    }

    // ---- β single: m → p ----
    for (int m : occ_beta)
    for (int p : virt_beta) {
        double H = h1[m][p];
        for (int n : occ_beta)
            if (n != m) H += eri_idx(m, p, n, n) - eri_idx(m, n, n, p);
        for (int n : occ_alpha)
            H += eri_idx(m, p, n, n);

        if (do_screen && std::abs(H) < integral_threshold) {
            if (n_screened_out) ++(*n_screened_out);
            continue;
        }

        StorageType nb_s = det.beta;
        BitOpsType::flip_bit(nb_s, m);
        BitOpsType::flip_bit(nb_s, p);
        int phase = detail::cre_des_sign_t(m, p, det.beta);
        double H_ai = phase * H;

        double delta = h1[p][p] - h1[m][m]
            + (sums.JK_bb[p] - sums.JK_bb[m])
            - CX(p, m)
            + (sums.J_ba[p] - sums.J_ba[m]);

        cb(DeterminantT<StorageType>(det.alpha, nb_s), H_ai, H_ii + delta);
    }

    // ---- αα double: (m, n) → (p, q) with m<n, p<q ----
    for (size_t ii = 0; ii < occ_alpha.size(); ++ii)
    for (size_t jj = ii + 1; jj < occ_alpha.size(); ++jj) {
        int m = occ_alpha[ii], n = occ_alpha[jj];
        for (int p : virt_alpha)
        for (int q : virt_alpha) {
            if (q <= p) continue;

            if (do_screen) {
                double eri1 = std::abs(eri_idx(m, p, n, q));
                double eri2 = std::abs(eri_idx(m, q, n, p));
                if (std::max(eri1, eri2) < integral_threshold) {
                    if (n_screened_out) ++(*n_screened_out);
                    continue;
                }
            }

            double H_raw = eri_idx(m, p, n, q) - eri_idx(m, q, n, p);

            StorageType na_s = det.alpha;
            BitOpsType::flip_bit(na_s, m);
            BitOpsType::flip_bit(na_s, n);
            BitOpsType::flip_bit(na_s, p);
            BitOpsType::flip_bit(na_s, q);
            int ph1 = detail::cre_des_sign_t(p, m, det.alpha);
            StorageType tmp_ph = det.alpha;
            BitOpsType::clear_bit(tmp_ph, m);
            BitOpsType::set_bit(tmp_ph, p);
            int phase = ph1 * detail::cre_des_sign_t(q, n, tmp_ph);
            double H_ai = phase * H_raw;

            // O(1) diagonal for αα double
            double delta = h1[p][p] + h1[q][q] - h1[m][m] - h1[n][n]
                + (sums.JK_aa[p] + sums.JK_aa[q] - sums.JK_aa[m] - sums.JK_aa[n])
                - CX(p,m) - CX(p,n) - CX(q,m) - CX(q,n)
                + CX(m,n) + CX(p,q)
                + (sums.J_ab[p] + sums.J_ab[q] - sums.J_ab[m] - sums.J_ab[n]);

            cb(DeterminantT<StorageType>(na_s, det.beta), H_ai, H_ii + delta);
        }
    }

    // ---- ββ double: (m, n) → (p, q) with m<n, p<q ----
    for (size_t ii = 0; ii < occ_beta.size(); ++ii)
    for (size_t jj = ii + 1; jj < occ_beta.size(); ++jj) {
        int m = occ_beta[ii], n = occ_beta[jj];
        for (int p : virt_beta)
        for (int q : virt_beta) {
            if (q <= p) continue;

            if (do_screen) {
                double eri1 = std::abs(eri_idx(m, p, n, q));
                double eri2 = std::abs(eri_idx(m, q, n, p));
                if (std::max(eri1, eri2) < integral_threshold) {
                    if (n_screened_out) ++(*n_screened_out);
                    continue;
                }
            }

            double H_raw = eri_idx(m, p, n, q) - eri_idx(m, q, n, p);

            StorageType nb_s = det.beta;
            BitOpsType::flip_bit(nb_s, m);
            BitOpsType::flip_bit(nb_s, n);
            BitOpsType::flip_bit(nb_s, p);
            BitOpsType::flip_bit(nb_s, q);
            int ph1 = detail::cre_des_sign_t(p, m, det.beta);
            StorageType tmp_ph = det.beta;
            BitOpsType::clear_bit(tmp_ph, m);
            BitOpsType::set_bit(tmp_ph, p);
            int phase = ph1 * detail::cre_des_sign_t(q, n, tmp_ph);
            double H_ai = phase * H_raw;

            double delta = h1[p][p] + h1[q][q] - h1[m][m] - h1[n][n]
                + (sums.JK_bb[p] + sums.JK_bb[q] - sums.JK_bb[m] - sums.JK_bb[n])
                - CX(p,m) - CX(p,n) - CX(q,m) - CX(q,n)
                + CX(m,n) + CX(p,q)
                + (sums.J_ba[p] + sums.J_ba[q] - sums.J_ba[m] - sums.J_ba[n]);

            cb(DeterminantT<StorageType>(det.alpha, nb_s), H_ai, H_ii + delta);
        }
    }

    // ---- αβ mixed double: m_α→p_α, m_β→p_β ----
    for (int ma : occ_alpha)
    for (int pa : virt_alpha) {
        StorageType am = det.alpha;
        BitOpsType::flip_bit(am, ma);
        BitOpsType::flip_bit(am, pa);
        int phase_a = detail::cre_des_sign_t(ma, pa, det.alpha);

        for (int mb : occ_beta)
        for (int pb : virt_beta) {
            if (do_screen && std::abs(eri_idx(ma, pa, mb, pb)) < integral_threshold) {
                if (n_screened_out) ++(*n_screened_out);
                continue;
            }

            double H_raw = eri_idx(ma, pa, mb, pb);

            StorageType bm = det.beta;
            BitOpsType::flip_bit(bm, mb);
            BitOpsType::flip_bit(bm, pb);
            int phase_b = detail::cre_des_sign_t(mb, pb, det.beta);
            double H_ai = phase_a * phase_b * H_raw;

            // O(1) diagonal for αβ double
            // α part: same-spin Coulomb-Exchange for pa vs ma
            // β part: same-spin Coulomb-Exchange for pb vs mb
            // Cross: α↔β Coulomb changes
            double delta = h1[pa][pa] - h1[ma][ma] + h1[pb][pb] - h1[mb][mb]
                // α same-spin: remove ma self-contribution
                + (sums.JK_aa[pa] - sums.JK_aa[ma]) - CX(pa, ma)
                // β same-spin: remove mb self-contribution
                + (sums.JK_bb[pb] - sums.JK_bb[mb]) - CX(pb, mb)
                // Cross: α orbitals with new/old β
                + (sums.J_ba[pb] - sums.J_ba[mb])       // all α with new/old β
                - (eri_idx(ma,ma,pb,pb) - eri_idx(ma,ma,mb,mb))  // remove ma's contribution
                // Cross: β orbitals with new/old α
                + (sums.J_ab[pa] - sums.J_ab[ma])       // all β with new/old α
                - (eri_idx(pa,pa,mb,mb) - eri_idx(ma,ma,mb,mb))  // remove mb's contribution
                // New cross pair pa↔pb minus old pair ma↔mb
                + eri_idx(pa,pa,pb,pb) - eri_idx(ma,ma,mb,mb);

            cb(DeterminantT<StorageType>(am, bm), H_ai, H_ii + delta);
        }
    }
}

}  // namespace fe
}  // namespace trimci_core
