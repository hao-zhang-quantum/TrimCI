/**
 * RDM computation: 3-body (exact and cumulant)
 *
 * Split from rdm_full.cpp for maintainability.
 */

#include "rdm_full.hpp"
#include "rdm_helpers.hpp"
#include <cmath>
#include "omp_compat.hpp"
using namespace trimci_core::rdm_detail;

namespace trimci_core {

// ============================================================================
// Spin-Resolved 3-RDM
// ============================================================================
// 
// TrimCI Convention (creation-first):
//   Γ³[p,q,r,s,t,u] = ⟨Ψ| a†_p a†_q a†_r a_u a_t a_s |Ψ⟩
//
// Index layout:
//   [p, q, r]     = creation operator indices (left to right in bra-ket)
//   [s, t, u]     = annihilation operator indices (reverse order: u, t, s applied)
//
// For spin sectors:
//   aaa: all alpha, fully antisymmetric in (p,q,r) and (s,t,u)
//   aab: 2 alpha + 1 beta, antisymmetric in alpha pairs (p,q) and (s,t)
//   abb: 1 alpha + 2 beta, antisymmetric in beta pairs (q,r) and (t,u)
//   bbb: all beta, fully antisymmetric in (p,q,r) and (s,t,u)
//
// PySCF uses alternating convention: rdm3[i,j,k,l,m,n] = ⟨i† j k† l m† n⟩
// To convert: pyscf = np.transpose(trimci, (0,3,1,4,2,5))
// ============================================================================

std::tuple<std::vector<double>, std::vector<double>, 
           std::vector<double>, std::vector<double>>
compute_3rdm_spin_resolved(
    const std::vector<Determinant>& dets,
    const std::vector<double>& coeffs,
    int n_orb
) {
    size_t N = static_cast<size_t>(n_orb);
    size_t N2 = N * N;
    size_t N3 = N2 * N;
    size_t N4 = N3 * N;
    size_t N5 = N4 * N;
    size_t N6 = N5 * N;
    
    std::vector<double> Gamma3_aaa(N6, 0.0);
    std::vector<double> Gamma3_aab(N6, 0.0);
    std::vector<double> Gamma3_abb(N6, 0.0);
    std::vector<double> Gamma3_bbb(N6, 0.0);
    
    int n_det = static_cast<int>(dets.size());

    // Memory guard: each thread needs 4 × N^6 doubles of local storage.
    // Cap thread count to stay within ~400 GB total.
    size_t mem_per_thread = 4 * N6 * sizeof(double);
    int max_threads = omp_get_max_threads();
    constexpr size_t MAX_TOTAL_MEM = 400ULL * 1024ULL * 1024ULL * 1024ULL;
    if (static_cast<size_t>(max_threads) * mem_per_thread > MAX_TOTAL_MEM) {
        max_threads = static_cast<int>(MAX_TOTAL_MEM / mem_per_thread);
        if (max_threads < 1) max_threads = 1;
    }

    // Helper: full antisymmetrization for same-spin triples (aaa or bbb).
    // Generates all S₃(creators) × S₃(annihilators) = 36 entries.
    // G[σ(p,q,r), τ(s,t,u)] += sgn(σ) × sgn(τ) × val.
    auto add_antisym_3 = [&](std::vector<double>& G, int p, int q, int r, int s, int t, int u, double val) {
        int c[3] = {p, q, r};
        int a[3] = {s, t, u};
        int perms[6][3] = {{0,1,2}, {0,2,1}, {1,0,2}, {1,2,0}, {2,0,1}, {2,1,0}};
        int signs[6] = {1, -1, -1, 1, 1, -1};
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 6; ++j) {
                G[c[perms[i][0]]*N5 + c[perms[i][1]]*N4 + c[perms[i][2]]*N3
                + a[perms[j][0]]*N2 + a[perms[j][1]]*N  + a[perms[j][2]]] += signs[i] * signs[j] * val;
            }
        }
    };

    // Helper: full antisymmetrization for aab (2α+1β).
    // S₂(α-creators p,q) × S₂(α-annihilators s,t) = 4 entries.
    // β indices (r, u) are untouched.
    auto add_antisym_2 = [&](std::vector<double>& G, int p, int q, int r, int s, int t, int u, double val) {
        G[p*N5 + q*N4 + r*N3 + s*N2 + t*N + u] += val;    // (id, id): +1
        G[q*N5 + p*N4 + r*N3 + s*N2 + t*N + u] -= val;    // (swap, id): -1
        G[p*N5 + q*N4 + r*N3 + t*N2 + s*N + u] -= val;    // (id, swap): -1
        G[q*N5 + p*N4 + r*N3 + t*N2 + s*N + u] += val;    // (swap, swap): +1
    };

    // Helper: full antisymmetrization for abb (1α+2β).
    // S₂(β-creators q,r) × S₂(β-annihilators t,u) = 4 entries.
    // α indices (p, s) are untouched.
    auto add_antisym_2b = [&](std::vector<double>& G, int p, int q, int r, int s, int t, int u, double val) {
        G[p*N5 + q*N4 + r*N3 + s*N2 + t*N + u] += val;    // (id, id): +1
        G[p*N5 + r*N4 + q*N3 + s*N2 + t*N + u] -= val;    // (swap, id): -1
        G[p*N5 + q*N4 + r*N3 + s*N2 + u*N + t] -= val;    // (id, swap): -1
        G[p*N5 + r*N4 + q*N3 + s*N2 + u*N + t] += val;    // (swap, swap): +1
    };
    
    #pragma omp parallel num_threads(max_threads)
    {
        std::vector<double> aaa_local(N6, 0.0);
        std::vector<double> aab_local(N6, 0.0);
        std::vector<double> abb_local(N6, 0.0);
        std::vector<double> bbb_local(N6, 0.0);
        
        #pragma omp for schedule(dynamic)
        for (int I = 0; I < n_det; ++I) {
            for (int J = 0; J < n_det; ++J) {
                double c_IJ = coeffs[I] * coeffs[J];
                if (std::abs(c_IJ) < 1e-15) continue;
                
                const auto& det_I = dets[I];
                const auto& det_J = dets[J];
                
                auto occ_a_I = get_set_bits(det_I.alpha);
                auto occ_b_I = get_set_bits(det_I.beta);
                auto occ_a_J = get_set_bits(det_J.alpha);
                auto occ_b_J = get_set_bits(det_J.beta);
                
                uint64_t diff_a_in_I = det_I.alpha & ~det_J.alpha;
                uint64_t diff_a_in_J = det_J.alpha & ~det_I.alpha;
                uint64_t diff_b_in_I = det_I.beta & ~det_J.beta;
                uint64_t diff_b_in_J = det_J.beta & ~det_I.beta;
                
                int n_alpha_exc = __builtin_popcountll(diff_a_in_I);
                int n_beta_exc = __builtin_popcountll(diff_b_in_I);
                int n_total_exc = n_alpha_exc + n_beta_exc;
                
                if (n_total_exc > 3) continue;
                
                // ======== DIAGONAL ========
                if (n_total_exc == 0) {
                    // Pure αααalpha triples
                    for (size_t ip = 0; ip < occ_a_I.size(); ++ip) {
                        for (size_t iq = ip + 1; iq < occ_a_I.size(); ++iq) {
                            for (size_t ir = iq + 1; ir < occ_a_I.size(); ++ir) {
                                int p = occ_a_I[ip], q = occ_a_I[iq], r = occ_a_I[ir];
                                add_antisym_3(aaa_local, p, q, r, p, q, r, c_IJ);
                            }
                        }
                    }
                    // Pure βββ triples
                    for (size_t ip = 0; ip < occ_b_I.size(); ++ip) {
                        for (size_t iq = ip + 1; iq < occ_b_I.size(); ++iq) {
                            for (size_t ir = iq + 1; ir < occ_b_I.size(); ++ir) {
                                int p = occ_b_I[ip], q = occ_b_I[iq], r = occ_b_I[ir];
                                add_antisym_3(bbb_local, p, q, r, p, q, r, c_IJ);
                            }
                        }
                    }
                    // ααβ: two alpha + one beta
                    for (size_t ip = 0; ip < occ_a_I.size(); ++ip) {
                        for (size_t iq = ip + 1; iq < occ_a_I.size(); ++iq) {
                            int p = occ_a_I[ip], q = occ_a_I[iq];
                            for (int r : occ_b_I) {
                                add_antisym_2(aab_local, p, q, r, p, q, r, c_IJ);
                            }
                        }
                    }
                    // αββ: one alpha + two beta
                    for (int p : occ_a_I) {
                        for (size_t iq = 0; iq < occ_b_I.size(); ++iq) {
                            for (size_t ir = iq + 1; ir < occ_b_I.size(); ++ir) {
                                int q = occ_b_I[iq], r = occ_b_I[ir];
                                add_antisym_2b(abb_local, p, q, r, p, q, r, c_IJ);
                            }
                        }
                    }
                }
                // ======== SINGLE EXCITATION ========
                else if (n_total_exc == 1) {
                    if (n_alpha_exc == 1) {
                        int p_new = __builtin_ctzll(diff_a_in_I);
                        int p_old = __builtin_ctzll(diff_a_in_J);
                        int phase = single_phase_alpha(det_J, p_old, p_new);
                        double val = c_IJ * phase;
                        
                        // αα spectators -> aaa
                        for (int k1 : occ_a_J) {
                            if (k1 == p_old) continue;
                            for (int k2 : occ_a_J) {
                                if (k2 == p_old || k2 <= k1) continue;
                                add_antisym_3(aaa_local, p_new, k1, k2, p_old, k1, k2, val);
                            }
                        }
                        // α spectator + β spectator -> aab
                        for (int k_a : occ_a_J) {
                            if (k_a == p_old) continue;
                            for (int k_b : occ_b_J) {
                                add_antisym_2(aab_local, p_new, k_a, k_b, p_old, k_a, k_b, val);
                            }
                        }
                        // ββ spectators -> abb
                        for (size_t ik1 = 0; ik1 < occ_b_J.size(); ++ik1) {
                            for (size_t ik2 = ik1 + 1; ik2 < occ_b_J.size(); ++ik2) {
                                int k1 = occ_b_J[ik1], k2 = occ_b_J[ik2];
                                add_antisym_2b(abb_local, p_new, k1, k2, p_old, k1, k2, val);
                            }
                        }
                    } else {  // n_beta_exc == 1
                        int p_new = __builtin_ctzll(diff_b_in_I);
                        int p_old = __builtin_ctzll(diff_b_in_J);
                        int phase = single_phase_beta(det_J, p_old, p_new);
                        double val = c_IJ * phase;
                        
                        // ββ spectators -> bbb
                        for (int k1 : occ_b_J) {
                            if (k1 == p_old) continue;
                            for (int k2 : occ_b_J) {
                                if (k2 == p_old || k2 <= k1) continue;
                                add_antisym_3(bbb_local, p_new, k1, k2, p_old, k1, k2, val);
                            }
                        }
                        // β spectator + α spectator -> abb
                        for (int k_b : occ_b_J) {
                            if (k_b == p_old) continue;
                            for (int k_a : occ_a_J) {
                                add_antisym_2b(abb_local, k_a, p_new, k_b, k_a, p_old, k_b, val);
                            }
                        }
                        // αα spectators -> aab
                        for (size_t ik1 = 0; ik1 < occ_a_J.size(); ++ik1) {
                            for (size_t ik2 = ik1 + 1; ik2 < occ_a_J.size(); ++ik2) {
                                int k1 = occ_a_J[ik1], k2 = occ_a_J[ik2];
                                add_antisym_2(aab_local, k1, k2, p_new, k1, k2, p_old, val);
                            }
                        }
                    }
                }
                // ======== DOUBLE EXCITATION ========
                else if (n_total_exc == 2) {
                    if (n_alpha_exc == 2) {
                        auto p_list = get_set_bits(diff_a_in_I);
                        auto q_list = get_set_bits(diff_a_in_J);
                        std::sort(p_list.begin(), p_list.end());
                        std::sort(q_list.begin(), q_list.end());
                        int p1 = p_list[0], p2 = p_list[1];
                        int q1 = q_list[0], q2 = q_list[1];
                        int phase = double_phase_alpha(det_J.alpha, q1, q2, p1, p2);
                        double val = c_IJ * phase;
                        
                        // α spectators -> aaa
                        for (int k : occ_a_J) {
                            if (k == q1 || k == q2) continue;
                            add_antisym_3(aaa_local, p1, p2, k, q1, q2, k, val);
                        }
                        // β spectators -> aab
                        for (int k : occ_b_J) {
                            add_antisym_2(aab_local, p1, p2, k, q1, q2, k, val);
                        }
                    }
                    else if (n_beta_exc == 2) {
                        auto p_list = get_set_bits(diff_b_in_I);
                        auto q_list = get_set_bits(diff_b_in_J);
                        std::sort(p_list.begin(), p_list.end());
                        std::sort(q_list.begin(), q_list.end());
                        int p1 = p_list[0], p2 = p_list[1];
                        int q1 = q_list[0], q2 = q_list[1];
                        int phase = double_phase_beta(det_J.beta, q1, q2, p1, p2);
                        double val = c_IJ * phase;
                        
                        // β spectators -> bbb
                        for (int k : occ_b_J) {
                            if (k == q1 || k == q2) continue;
                            add_antisym_3(bbb_local, p1, p2, k, q1, q2, k, val);
                        }
                        // α spectators -> abb
                        for (int k : occ_a_J) {
                            add_antisym_2b(abb_local, k, p1, p2, k, q1, q2, val);
                        }
                    }
                    else {  // 1α + 1β
                        int p_a = __builtin_ctzll(diff_a_in_I);
                        int q_a = __builtin_ctzll(diff_a_in_J);
                        int p_b = __builtin_ctzll(diff_b_in_I);
                        int q_b = __builtin_ctzll(diff_b_in_J);
                        int phase = single_phase_alpha(det_J, q_a, p_a) * single_phase_beta(det_J, q_b, p_b);
                        double val = c_IJ * phase;
                        
                        // α spectator -> aab
                        for (int k : occ_a_J) {
                            if (k == q_a) continue;
                            add_antisym_2(aab_local, p_a, k, p_b, q_a, k, q_b, val);
                        }
                        // β spectator -> abb
                        for (int k : occ_b_J) {
                            if (k == q_b) continue;
                            add_antisym_2b(abb_local, p_a, p_b, k, q_a, q_b, k, val);
                        }
                    }
                }
                // ======== TRIPLE EXCITATION ========
                else if (n_total_exc == 3) {
                    if (n_alpha_exc == 3) {
                        auto p_list = get_set_bits(diff_a_in_I);
                        auto q_list = get_set_bits(diff_a_in_J);
                        std::sort(p_list.begin(), p_list.end());
                        std::sort(q_list.begin(), q_list.end());
                        int p1 = p_list[0], p2 = p_list[1], p3 = p_list[2];
                        int q1 = q_list[0], q2 = q_list[1], q3 = q_list[2];
                        int phase = triple_phase_alpha(det_J.alpha, q1, q2, q3, p1, p2, p3);
                        add_antisym_3(aaa_local, p1, p2, p3, q1, q2, q3, c_IJ * phase);
                    }
                    else if (n_beta_exc == 3) {
                        auto p_list = get_set_bits(diff_b_in_I);
                        auto q_list = get_set_bits(diff_b_in_J);
                        std::sort(p_list.begin(), p_list.end());
                        std::sort(q_list.begin(), q_list.end());
                        int p1 = p_list[0], p2 = p_list[1], p3 = p_list[2];
                        int q1 = q_list[0], q2 = q_list[1], q3 = q_list[2];
                        int phase = triple_phase_beta(det_J.beta, q1, q2, q3, p1, p2, p3);
                        add_antisym_3(bbb_local, p1, p2, p3, q1, q2, q3, c_IJ * phase);
                    }
                    else if (n_alpha_exc == 2) {  // 2α + 1β
                        auto p_a_list = get_set_bits(diff_a_in_I);
                        auto q_a_list = get_set_bits(diff_a_in_J);
                        std::sort(p_a_list.begin(), p_a_list.end());
                        std::sort(q_a_list.begin(), q_a_list.end());
                        int p1 = p_a_list[0], p2 = p_a_list[1];
                        int q1 = q_a_list[0], q2 = q_a_list[1];
                        int p_b = __builtin_ctzll(diff_b_in_I);
                        int q_b = __builtin_ctzll(diff_b_in_J);
                        int phase = double_phase_alpha(det_J.alpha, q1, q2, p1, p2) *
                                   single_phase_beta(det_J, q_b, p_b);
                        add_antisym_2(aab_local, p1, p2, p_b, q1, q2, q_b, c_IJ * phase);
                    }
                    else {  // 1α + 2β
                        auto p_b_list = get_set_bits(diff_b_in_I);
                        auto q_b_list = get_set_bits(diff_b_in_J);
                        std::sort(p_b_list.begin(), p_b_list.end());
                        std::sort(q_b_list.begin(), q_b_list.end());
                        int p1 = p_b_list[0], p2 = p_b_list[1];
                        int q1 = q_b_list[0], q2 = q_b_list[1];
                        int p_a = __builtin_ctzll(diff_a_in_I);
                        int q_a = __builtin_ctzll(diff_a_in_J);
                        int phase = double_phase_beta(det_J.beta, q1, q2, p1, p2) *
                                   single_phase_alpha(det_J, q_a, p_a);
                        add_antisym_2b(abb_local, p_a, p1, p2, q_a, q1, q2, c_IJ * phase);
                    }
                }
            }
        }
        
        #pragma omp critical
        {
            for (size_t k = 0; k < N6; ++k) {
                Gamma3_aaa[k] += aaa_local[k];
                Gamma3_aab[k] += aab_local[k];
                Gamma3_abb[k] += abb_local[k];
                Gamma3_bbb[k] += bbb_local[k];
            }
        }
    }
    
    return std::make_tuple(std::move(Gamma3_aaa), std::move(Gamma3_aab), 
                          std::move(Gamma3_abb), std::move(Gamma3_bbb));
}

// ============================================================================
// Spin-Resolved 3-RDM Cumulant Approximation
// ============================================================================
// Γ³ ≈ A_S[Γ² ⊗ γ] − 2·det(γ)   (ignoring 3-body cumulant λ³)
//
// For same-spin sectors (aaa/bbb): 9 A_S terms + 6-term determinant
// For mixed-spin sectors (aab/abb): 5 A_S terms + mixed determinant

std::tuple<std::vector<double>, std::vector<double>, 
           std::vector<double>, std::vector<double>>
compute_3rdm_cumulant_spin_resolved(
    const std::vector<double>& gamma_aa,
    const std::vector<double>& gamma_bb,
    const std::vector<double>& Gamma_aa,
    const std::vector<double>& Gamma_ab,
    const std::vector<double>& Gamma_bb,
    int n_orb
) {
    size_t N = static_cast<size_t>(n_orb);
    size_t N2 = N * N;
    size_t N3 = N2 * N;
    size_t N4 = N3 * N;
    size_t N5 = N4 * N;
    size_t N6 = N5 * N;
    
    std::vector<double> Gamma3_aaa(N6, 0.0);
    std::vector<double> Gamma3_aab(N6, 0.0);
    std::vector<double> Gamma3_abb(N6, 0.0);
    std::vector<double> Gamma3_bbb(N6, 0.0);
    
    // Helper for index
    auto idx6 = [N, N2, N3, N4, N5](int p, int q, int r, int s, int t, int u) {
        return p*N5 + q*N4 + r*N3 + s*N2 + t*N + u;
    };
    auto idx4 = [N, N2, N3](int p, int q, int r, int s) {
        return p*N3 + q*N2 + r*N + s;
    };
    auto idx2 = [N](int p, int q) {
        return p*N + q;
    };
    
    // ===================== aaa sector =====================
    // Γ³_aaa[p,q,r,s,t,u] = A_S[Γ²_aa ⊗ γ_a] − 2·det(γ_a)
    // Full antisymmetry in (p,q,r) and (s,t,u)
    // A_S has 9 terms (3 ways to pick 2-from-3 creators × 3 ways for annihilators)
    #pragma omp parallel for collapse(3)
    for (int p = 0; p < static_cast<int>(N); ++p) {
        for (size_t q = 0; q < N; ++q) {
            for (size_t r = 0; r < N; ++r) {
                for (size_t s = 0; s < N; ++s) {
                    for (size_t t = 0; t < N; ++t) {
                        for (size_t u = 0; u < N; ++u) {
                            double val = 0.0;
                            // --- 9 A_S[Γ² ⊗ γ] terms ---
                            // creators (p,q), annihilators (s,t), remainder r↔u
                            val += Gamma_aa[idx4(p,q,s,t)] * gamma_aa[idx2(r,u)];
                            // creators (p,q), annihilators (s,u), remainder r↔t
                            val -= Gamma_aa[idx4(p,q,s,u)] * gamma_aa[idx2(r,t)];
                            // creators (p,q), annihilators (t,u), remainder r↔s
                            val += Gamma_aa[idx4(p,q,t,u)] * gamma_aa[idx2(r,s)];
                            // creators (p,r), annihilators (s,t), remainder q↔u
                            val -= Gamma_aa[idx4(p,r,s,t)] * gamma_aa[idx2(q,u)];
                            // creators (p,r), annihilators (s,u), remainder q↔t
                            val += Gamma_aa[idx4(p,r,s,u)] * gamma_aa[idx2(q,t)];
                            // creators (p,r), annihilators (t,u), remainder q↔s
                            val -= Gamma_aa[idx4(p,r,t,u)] * gamma_aa[idx2(q,s)];
                            // creators (q,r), annihilators (s,t), remainder p↔u
                            val += Gamma_aa[idx4(q,r,s,t)] * gamma_aa[idx2(p,u)];
                            // creators (q,r), annihilators (s,u), remainder p↔t
                            val -= Gamma_aa[idx4(q,r,s,u)] * gamma_aa[idx2(p,t)];
                            // creators (q,r), annihilators (t,u), remainder p↔s
                            val += Gamma_aa[idx4(q,r,t,u)] * gamma_aa[idx2(p,s)];

                            // --- −2·det(γ_a) ---
                            double det_val =
                                  gamma_aa[idx2(p,s)] * gamma_aa[idx2(q,t)] * gamma_aa[idx2(r,u)]
                                - gamma_aa[idx2(p,s)] * gamma_aa[idx2(q,u)] * gamma_aa[idx2(r,t)]
                                - gamma_aa[idx2(p,t)] * gamma_aa[idx2(q,s)] * gamma_aa[idx2(r,u)]
                                + gamma_aa[idx2(p,t)] * gamma_aa[idx2(q,u)] * gamma_aa[idx2(r,s)]
                                + gamma_aa[idx2(p,u)] * gamma_aa[idx2(q,s)] * gamma_aa[idx2(r,t)]
                                - gamma_aa[idx2(p,u)] * gamma_aa[idx2(q,t)] * gamma_aa[idx2(r,s)];
                            val -= 2.0 * det_val;

                            Gamma3_aaa[idx6(p,q,r,s,t,u)] = val;
                        }
                    }
                }
            }
        }
    }
    
    // ===================== bbb sector =====================
    // Γ³_bbb[p,q,r,s,t,u] = A_S[Γ²_bb ⊗ γ_b] − 2·det(γ_b)
    #pragma omp parallel for collapse(3)
    for (int p = 0; p < static_cast<int>(N); ++p) {
        for (size_t q = 0; q < N; ++q) {
            for (size_t r = 0; r < N; ++r) {
                for (size_t s = 0; s < N; ++s) {
                    for (size_t t = 0; t < N; ++t) {
                        for (size_t u = 0; u < N; ++u) {
                            double val = 0.0;
                            // --- 9 A_S[Γ² ⊗ γ] terms ---
                            val += Gamma_bb[idx4(p,q,s,t)] * gamma_bb[idx2(r,u)];
                            val -= Gamma_bb[idx4(p,q,s,u)] * gamma_bb[idx2(r,t)];
                            val += Gamma_bb[idx4(p,q,t,u)] * gamma_bb[idx2(r,s)];
                            val -= Gamma_bb[idx4(p,r,s,t)] * gamma_bb[idx2(q,u)];
                            val += Gamma_bb[idx4(p,r,s,u)] * gamma_bb[idx2(q,t)];
                            val -= Gamma_bb[idx4(p,r,t,u)] * gamma_bb[idx2(q,s)];
                            val += Gamma_bb[idx4(q,r,s,t)] * gamma_bb[idx2(p,u)];
                            val -= Gamma_bb[idx4(q,r,s,u)] * gamma_bb[idx2(p,t)];
                            val += Gamma_bb[idx4(q,r,t,u)] * gamma_bb[idx2(p,s)];

                            // --- −2·det(γ_b) ---
                            double det_val =
                                  gamma_bb[idx2(p,s)] * gamma_bb[idx2(q,t)] * gamma_bb[idx2(r,u)]
                                - gamma_bb[idx2(p,s)] * gamma_bb[idx2(q,u)] * gamma_bb[idx2(r,t)]
                                - gamma_bb[idx2(p,t)] * gamma_bb[idx2(q,s)] * gamma_bb[idx2(r,u)]
                                + gamma_bb[idx2(p,t)] * gamma_bb[idx2(q,u)] * gamma_bb[idx2(r,s)]
                                + gamma_bb[idx2(p,u)] * gamma_bb[idx2(q,s)] * gamma_bb[idx2(r,t)]
                                - gamma_bb[idx2(p,u)] * gamma_bb[idx2(q,t)] * gamma_bb[idx2(r,s)];
                            val -= 2.0 * det_val;

                            Gamma3_bbb[idx6(p,q,r,s,t,u)] = val;
                        }
                    }
                }
            }
        }
    }
    
    // ===================== aab sector =====================
    // 2 alpha (p,q) + 1 beta (r); annihilate (s,t,u) with s,t alpha, u beta
    // Antisymmetric in (p,q) and (s,t)
    // A_S has 5 terms: 1 from Γ²_aa⊗γ_b + 4 from antisymmetrizing Γ²_ab⊗γ_a
    #pragma omp parallel for collapse(3)
    for (int p = 0; p < static_cast<int>(N); ++p) {
        for (size_t q = 0; q < N; ++q) {
            for (size_t r = 0; r < N; ++r) {
                for (size_t s = 0; s < N; ++s) {
                    for (size_t t = 0; t < N; ++t) {
                        for (size_t u = 0; u < N; ++u) {
                            double val = 0.0;
                            // --- 5 A_S[Γ² ⊗ γ] terms ---
                            // Partition A: Γ²_aa[p,q,s,t] · γ_b[r,u]
                            val += Gamma_aa[idx4(p,q,s,t)] * gamma_bb[idx2(r,u)];
                            // Partition B antisymmetrized in (p,q) and (s,t):
                            val += Gamma_ab[idx4(p,r,s,u)] * gamma_aa[idx2(q,t)];
                            val -= Gamma_ab[idx4(q,r,s,u)] * gamma_aa[idx2(p,t)];
                            val -= Gamma_ab[idx4(p,r,t,u)] * gamma_aa[idx2(q,s)];
                            val += Gamma_ab[idx4(q,r,t,u)] * gamma_aa[idx2(p,s)];

                            // --- −2·det_aab(γ) ---
                            // det_aab = (γ_a[p,s]·γ_a[q,t] − γ_a[p,t]·γ_a[q,s]) · γ_b[r,u]
                            double det_val = (gamma_aa[idx2(p,s)] * gamma_aa[idx2(q,t)]
                                            - gamma_aa[idx2(p,t)] * gamma_aa[idx2(q,s)])
                                           * gamma_bb[idx2(r,u)];
                            val -= 2.0 * det_val;

                            Gamma3_aab[idx6(p,q,r,s,t,u)] = val;
                        }
                    }
                }
            }
        }
    }
    
    // ===================== abb sector =====================
    // 1 alpha (p) + 2 beta (q,r); annihilate (s,t,u) with s alpha, t,u beta
    // Antisymmetric in (q,r) and (t,u)
    // A_S has 5 terms: 1 from Γ²_bb⊗γ_a + 4 from antisymmetrizing Γ²_ab⊗γ_b
    #pragma omp parallel for collapse(3)
    for (int p = 0; p < static_cast<int>(N); ++p) {
        for (size_t q = 0; q < N; ++q) {
            for (size_t r = 0; r < N; ++r) {
                for (size_t s = 0; s < N; ++s) {
                    for (size_t t = 0; t < N; ++t) {
                        for (size_t u = 0; u < N; ++u) {
                            double val = 0.0;
                            // --- 5 A_S[Γ² ⊗ γ] terms ---
                            // Partition A: Γ²_bb[q,r,t,u] · γ_a[p,s]
                            val += Gamma_bb[idx4(q,r,t,u)] * gamma_aa[idx2(p,s)];
                            // Partition B antisymmetrized in (q,r) and (t,u):
                            val += Gamma_ab[idx4(p,q,s,t)] * gamma_bb[idx2(r,u)];
                            val -= Gamma_ab[idx4(p,r,s,t)] * gamma_bb[idx2(q,u)];
                            val -= Gamma_ab[idx4(p,q,s,u)] * gamma_bb[idx2(r,t)];
                            val += Gamma_ab[idx4(p,r,s,u)] * gamma_bb[idx2(q,t)];

                            // --- −2·det_abb(γ) ---
                            // det_abb = γ_a[p,s] · (γ_b[q,t]·γ_b[r,u] − γ_b[q,u]·γ_b[r,t])
                            double det_val = gamma_aa[idx2(p,s)]
                                           * (gamma_bb[idx2(q,t)] * gamma_bb[idx2(r,u)]
                                            - gamma_bb[idx2(q,u)] * gamma_bb[idx2(r,t)]);
                            val -= 2.0 * det_val;

                            Gamma3_abb[idx6(p,q,r,s,t,u)] = val;
                        }
                    }
                }
            }
        }
    }
    
    return std::make_tuple(std::move(Gamma3_aaa), std::move(Gamma3_aab),
                          std::move(Gamma3_abb), std::move(Gamma3_bbb));
}


} // namespace trimci_core
