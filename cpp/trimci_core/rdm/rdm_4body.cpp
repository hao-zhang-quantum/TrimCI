/**
 * RDM computation: 4-body (exact and cumulant)
 *
 * Split from rdm_full.cpp for maintainability.
 */

#include "rdm_full.hpp"
#include "rdm_helpers.hpp"
#include <cmath>
#include <stdexcept>
#include <string>
#include "omp_compat.hpp"
using namespace trimci_core::rdm_detail;

namespace trimci_core {

// ============================================================================
// 4-RDM Cumulant Approximation (on-the-fly contraction)
// ============================================================================

double contract_4rdm_cumulant(
    const std::vector<double>& Gamma2,
    const std::vector<double>& c_ijab,
    int p, int q, int r, int s,
    int n_orb
) {
    size_t N = static_cast<size_t>(n_orb);
    size_t N2 = N * N;
    size_t N3 = N * N * N;
    
    double result = 0.0;
    
    // Γ⁴[p,q,r,s,t,u,v,w] ≈ A[Γ²[p,q,t,u] × Γ²[r,s,v,w]]
    // Contraction: Σ_{tuvw} Γ⁴[p,q,r,s,t,u,v,w] * c[t,u,v,w]
    
    for (int t = 0; t < n_orb; ++t) {
        for (int u = 0; u < n_orb; ++u) {
            for (int v = 0; v < n_orb; ++v) {
                for (int w = 0; w < n_orb; ++w) {
                    double c_val = c_ijab[t*N3 + u*N2 + v*N + w];
                    if (std::abs(c_val) < 1e-15) continue;
                    
                    // Leading term: Γ²[p,q,t,u] × Γ²[r,s,v,w]
                    double G2_pqtu = Gamma2[p*N3 + q*N2 + t*N + u];
                    double G2_rsvw = Gamma2[r*N3 + s*N2 + v*N + w];
                    
                    result += G2_pqtu * G2_rsvw * c_val;
                    
                    // Exchange terms (6 key permutations)
                    double G2_pqtv = Gamma2[p*N3 + q*N2 + t*N + v];
                    double G2_rsuw = Gamma2[r*N3 + s*N2 + u*N + w];
                    result -= G2_pqtv * G2_rsuw * c_val;
                    
                    double G2_pqtw = Gamma2[p*N3 + q*N2 + t*N + w];
                    double G2_rsuv = Gamma2[r*N3 + s*N2 + u*N + v];
                    result += G2_pqtw * G2_rsuv * c_val;
                }
            }
        }
    }
    
    return result;
}

// ============================================================================
// Spin-Resolved 4-RDM (EXACT)
// ============================================================================
// Returns 5 tensors: Γ⁴_aaaa, Γ⁴_aaab, Γ⁴_aabb, Γ⁴_abbb, Γ⁴_bbbb
//
// Index convention (creation-first):
//   Γ⁴[p,q,r,s,t,u,v,w] = ⟨Ψ| a†_p a†_q a†_r a†_s a_w a_v a_u a_t |Ψ⟩
//
// For spin sectors:
//   aaaa: all alpha, fully antisymmetric
//   aaab: 3 alpha + 1 beta, antisymmetric in alpha (p,q,r) and (t,u,v)
//   aabb: 2 alpha + 2 beta, antisymmetric in (p,q)/(t,u) and (r,s)/(v,w) separately
//   abbb: 1 alpha + 3 beta, antisymmetric in beta (q,r,s) and (u,v,w)
//   bbbb: all beta, fully antisymmetric

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>,
           std::vector<double>, std::vector<double>>
compute_4rdm_spin_resolved(
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
    size_t N7 = N6 * N;
    size_t N8 = N7 * N;
    
    // Warning: N8 can be huge! Limit to n_orb <= 14 (with 500GB RAM available)
    // n_orb=10: ~3.7GB, n_orb=12: ~16GB, n_orb=14: ~55GB total
    if (n_orb > 14) {
        size_t storage_mb = (N8 * 8 * 5) / (1024 * 1024);  // 5 spin types
        throw std::runtime_error("compute_4rdm_spin_resolved: n_orb > 14 requires ~" + 
                                 std::to_string(storage_mb) + "MB storage, aborting");
    }
    
    std::vector<double> Gamma4_aaaa(N8, 0.0);
    std::vector<double> Gamma4_aaab(N8, 0.0);
    std::vector<double> Gamma4_aabb(N8, 0.0);
    std::vector<double> Gamma4_abbb(N8, 0.0);
    std::vector<double> Gamma4_bbbb(N8, 0.0);
    
    int n_det = static_cast<int>(dets.size());
    
    // Helper for index calculation
    auto idx8 = [N, N2, N3, N4, N5, N6, N7](int p, int q, int r, int s, int t, int u, int v, int w) {
        return p*N7 + q*N6 + r*N5 + s*N4 + t*N3 + u*N2 + v*N + w;
    };
    
    // Helper: full antisymmetrization for same-spin quartets (aaaa or bbbb).
    // Generates all S₄(creators) × S₄(annihilators) = 24 × 24 = 576 entries.
    // G[σ(creators), τ(annihilators)] = sgn(σ) × sgn(τ) × val
    auto add_antisym_4 = [&](std::vector<double>& G, int p, int q, int r, int s,
                             int t, int u, int v, int w, double val) {
        int c[4] = {p, q, r, s};
        int a[4] = {t, u, v, w};
        static const int perms[24][4] = {
            {0,1,2,3}, {0,1,3,2}, {0,2,1,3}, {0,2,3,1}, {0,3,1,2}, {0,3,2,1},
            {1,0,2,3}, {1,0,3,2}, {1,2,0,3}, {1,2,3,0}, {1,3,0,2}, {1,3,2,0},
            {2,0,1,3}, {2,0,3,1}, {2,1,0,3}, {2,1,3,0}, {2,3,0,1}, {2,3,1,0},
            {3,0,1,2}, {3,0,2,1}, {3,1,0,2}, {3,1,2,0}, {3,2,0,1}, {3,2,1,0}
        };
        static const int signs[24] = {
            +1, -1, -1, +1, +1, -1,
            -1, +1, +1, -1, -1, +1,
            +1, -1, -1, +1, +1, -1,
            -1, +1, +1, -1, -1, +1
        };
        for (int i = 0; i < 24; ++i) {
            for (int j = 0; j < 24; ++j) {
                G[idx8(c[perms[i][0]], c[perms[i][1]], c[perms[i][2]], c[perms[i][3]],
                       a[perms[j][0]], a[perms[j][1]], a[perms[j][2]], a[perms[j][3]])]
                    += signs[i] * signs[j] * val;
            }
        }
    };
    
    // Helper: full antisymmetrization for aaab/abbb (3 same-spin + 1 other).
    // Generates all S₃(same-spin creators) × S₃(same-spin annihilators) = 6 × 6 = 36 entries.
    // (p,q,r) are same-spin creators, s is fixed; (t,u,v) are same-spin annihilators, w is fixed.
    auto add_antisym_3_1 = [&](std::vector<double>& G, int p, int q, int r, int s,
                               int t, int u, int v, int w, double val) {
        int c[3] = {p, q, r};
        int a[3] = {t, u, v};
        static const int perms[6][3] = {
            {0,1,2}, {0,2,1}, {1,0,2}, {1,2,0}, {2,0,1}, {2,1,0}
        };
        static const int signs[6] = {1, -1, -1, 1, 1, -1};
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 6; ++j) {
                G[idx8(c[perms[i][0]], c[perms[i][1]], c[perms[i][2]], s,
                       a[perms[j][0]], a[perms[j][1]], a[perms[j][2]], w)]
                    += signs[i] * signs[j] * val;
            }
        }
    };

    // Helper: full antisymmetrization for aabb (2α + 2β).
    // Generates S₂(α-creators) × S₂(β-creators) × S₂(α-annihilators) × S₂(β-annihilators)
    // = 2 × 2 × 2 × 2 = 16 entries.
    // (p,q) = α-creators, (r,s) = β-creators, (t,u) = α-annihilators, (v,w) = β-annihilators.
    auto add_antisym_2_2 = [&](std::vector<double>& G, int p, int q, int r, int s,
                               int t, int u, int v, int w, double val) {
        int ca[2] = {p, q};  // α creators
        int cb[2] = {r, s};  // β creators
        int aa[2] = {t, u};  // α annihilators
        int ab[2] = {v, w};  // β annihilators
        for (int sca = 0; sca <= 1; ++sca) {       // swap α creators
            for (int scb = 0; scb <= 1; ++scb) {   // swap β creators
                for (int saa = 0; saa <= 1; ++saa) {   // swap α annihilators
                    for (int sab = 0; sab <= 1; ++sab) {   // swap β annihilators
                        int sign = ((sca + scb + saa + sab) % 2 == 0) ? 1 : -1;
                        G[idx8(ca[sca], ca[1-sca], cb[scb], cb[1-scb],
                               aa[saa], aa[1-saa], ab[sab], ab[1-sab])]
                            += sign * val;
                    }
                }
            }
        }
    };
    
    // Memory-efficient parallel implementation:
    // Instead of each thread having local N^8 arrays (which requires N_threads * 5 * N^8 memory),
    // we use atomic updates directly to the shared arrays.
    // This trades some performance for much lower memory usage.
    // 
    // Memory comparison for N=12, 32 threads:
    //   Old: 32 * 5 * 12^8 * 8 bytes ≈ 544 GB
    //   New: 5 * 12^8 * 8 bytes ≈ 17 GB
    
    // Helper for atomic add (using relaxed memory order for performance)
    auto atomic_add = [](std::vector<double>& G, size_t idx, double val) {
        #pragma omp atomic
        G[idx] += val;
    };
    
    // Atomic version: full antisymmetrization for same-spin quartets (aaaa or bbbb).
    // S₄(creators) × S₄(annihilators) = 576 entries with atomic updates.
    auto add_antisym_4_atomic = [&](std::vector<double>& G, int p, int q, int r, int s,
                                     int t, int u, int v, int w, double val) {
        int c[4] = {p, q, r, s};
        int a[4] = {t, u, v, w};
        static const int perms[24][4] = {
            {0,1,2,3}, {0,1,3,2}, {0,2,1,3}, {0,2,3,1}, {0,3,1,2}, {0,3,2,1},
            {1,0,2,3}, {1,0,3,2}, {1,2,0,3}, {1,2,3,0}, {1,3,0,2}, {1,3,2,0},
            {2,0,1,3}, {2,0,3,1}, {2,1,0,3}, {2,1,3,0}, {2,3,0,1}, {2,3,1,0},
            {3,0,1,2}, {3,0,2,1}, {3,1,0,2}, {3,1,2,0}, {3,2,0,1}, {3,2,1,0}
        };
        static const int signs[24] = {
            +1, -1, -1, +1, +1, -1,
            -1, +1, +1, -1, -1, +1,
            +1, -1, -1, +1, +1, -1,
            -1, +1, +1, -1, -1, +1
        };
        for (int i = 0; i < 24; ++i) {
            for (int j = 0; j < 24; ++j) {
                atomic_add(G,
                    idx8(c[perms[i][0]], c[perms[i][1]], c[perms[i][2]], c[perms[i][3]],
                         a[perms[j][0]], a[perms[j][1]], a[perms[j][2]], a[perms[j][3]]),
                    signs[i] * signs[j] * val);
            }
        }
    };
    
    // Atomic version: full antisymmetrization for aaab/abbb (3 same-spin + 1 other).
    // S₃ × S₃ = 36 entries with atomic updates.
    auto add_antisym_3_1_atomic = [&](std::vector<double>& G, int p, int q, int r, int s,
                                       int t, int u, int v, int w, double val) {
        int c[3] = {p, q, r};
        int a[3] = {t, u, v};
        static const int perms[6][3] = {
            {0,1,2}, {0,2,1}, {1,0,2}, {1,2,0}, {2,0,1}, {2,1,0}
        };
        static const int signs[6] = {1, -1, -1, 1, 1, -1};
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 6; ++j) {
                atomic_add(G,
                    idx8(c[perms[i][0]], c[perms[i][1]], c[perms[i][2]], s,
                         a[perms[j][0]], a[perms[j][1]], a[perms[j][2]], w),
                    signs[i] * signs[j] * val);
            }
        }
    };

    // Atomic version: abbb sector (1α + 3β) - position 0 fixed, positions 1,2,3 antisymmetrized.
    // Layout: G[p_α, q_β, r_β, s_β, t_α, u_β, v_β, w_β]
    // S₃(β-creators at 1,2,3) × S₃(β-annihilators at 5,6,7) = 36 entries.
    auto add_antisym_1_3_atomic = [&](std::vector<double>& G, int p, int q, int r, int s,
                                       int t, int u, int v, int w, double val) {
        int c[3] = {q, r, s};  // β creators at positions 1,2,3
        int a[3] = {u, v, w};  // β annihilators at positions 5,6,7
        static const int perms[6][3] = {
            {0,1,2}, {0,2,1}, {1,0,2}, {1,2,0}, {2,0,1}, {2,1,0}
        };
        static const int signs[6] = {1, -1, -1, 1, 1, -1};
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 6; ++j) {
                atomic_add(G,
                    idx8(p, c[perms[i][0]], c[perms[i][1]], c[perms[i][2]],
                         t, a[perms[j][0]], a[perms[j][1]], a[perms[j][2]]),
                    signs[i] * signs[j] * val);
            }
        }
    };

    // Atomic version: full antisymmetrization for aabb (2α + 2β).
    // S₂⁴ = 16 entries with atomic updates.
    auto add_antisym_2_2_atomic = [&](std::vector<double>& G, int p, int q, int r, int s,
                                       int t, int u, int v, int w, double val) {
        int ca[2] = {p, q};  // α creators
        int cb[2] = {r, s};  // β creators
        int aa[2] = {t, u};  // α annihilators
        int ab[2] = {v, w};  // β annihilators
        for (int sca = 0; sca <= 1; ++sca) {
            for (int scb = 0; scb <= 1; ++scb) {
                for (int saa = 0; saa <= 1; ++saa) {
                    for (int sab = 0; sab <= 1; ++sab) {
                        int sign = ((sca + scb + saa + sab) % 2 == 0) ? 1 : -1;
                        atomic_add(G,
                            idx8(ca[sca], ca[1-sca], cb[scb], cb[1-scb],
                                 aa[saa], aa[1-saa], ab[sab], ab[1-sab]),
                            sign * val);
                    }
                }
            }
        }
    };
    
    
    #pragma omp parallel for schedule(dynamic)
        for (int I = 0; I < n_det; ++I) {
            for (int J = 0; J < n_det; ++J) {
                double c_IJ = coeffs[I] * coeffs[J];
                if (std::abs(c_IJ) < 1e-15) continue;
                
                const auto& det_I = dets[I];
                const auto& det_J = dets[J];
                
                // Check excitation level
                uint64_t diff_a_in_I = det_I.alpha & ~det_J.alpha;
                uint64_t diff_b_in_I = det_I.beta & ~det_J.beta;
                uint64_t diff_a_in_J = det_J.alpha & ~det_I.alpha;
                uint64_t diff_b_in_J = det_J.beta & ~det_I.beta;
                int n_alpha_exc = __builtin_popcountll(diff_a_in_I);
                int n_beta_exc = __builtin_popcountll(diff_b_in_I);
                int n_excite = n_alpha_exc + n_beta_exc;
                if (n_excite > 4) continue;
                
                auto occ_a_I = get_set_bits(det_I.alpha);
                auto occ_b_I = get_set_bits(det_I.beta);
                auto occ_a_J = get_set_bits(det_J.alpha);
                auto occ_b_J = get_set_bits(det_J.beta);

                
                // Diagonal case: det_I == det_J
                if (det_I.alpha == det_J.alpha && det_I.beta == det_J.beta) {
                    // aaaa: 4 alpha
                    for (size_t ip = 0; ip < occ_a_I.size(); ++ip) {
                        for (size_t iq = ip + 1; iq < occ_a_I.size(); ++iq) {
                            for (size_t ir = iq + 1; ir < occ_a_I.size(); ++ir) {
                                for (size_t is = ir + 1; is < occ_a_I.size(); ++is) {
                                    int p = occ_a_I[ip], q = occ_a_I[iq];
                                    int r = occ_a_I[ir], s = occ_a_I[is];
                                    add_antisym_4_atomic(Gamma4_aaaa, p, q, r, s, p, q, r, s, c_IJ);
                                }
                            }
                        }
                    }
                    
                    // bbbb: 4 beta
                    for (size_t ip = 0; ip < occ_b_I.size(); ++ip) {
                        for (size_t iq = ip + 1; iq < occ_b_I.size(); ++iq) {
                            for (size_t ir = iq + 1; ir < occ_b_I.size(); ++ir) {
                                for (size_t is = ir + 1; is < occ_b_I.size(); ++is) {
                                    int p = occ_b_I[ip], q = occ_b_I[iq];
                                    int r = occ_b_I[ir], s = occ_b_I[is];
                                    add_antisym_4_atomic(Gamma4_bbbb, p, q, r, s, p, q, r, s, c_IJ);
                                }
                            }
                        }
                    }
                    
                    // aaab: 3 alpha + 1 beta
                    for (size_t ip = 0; ip < occ_a_I.size(); ++ip) {
                        for (size_t iq = ip + 1; iq < occ_a_I.size(); ++iq) {
                            for (size_t ir = iq + 1; ir < occ_a_I.size(); ++ir) {
                                int p = occ_a_I[ip], q = occ_a_I[iq], r = occ_a_I[ir];
                                for (int s : occ_b_I) {
                                    add_antisym_3_1_atomic(Gamma4_aaab, p, q, r, s, p, q, r, s, c_IJ);
                                }
                            }
                        }
                    }
                    
                    // abbb: 1 alpha + 3 beta
                    for (int p : occ_a_I) {
                        for (size_t iq = 0; iq < occ_b_I.size(); ++iq) {
                            for (size_t ir = iq + 1; ir < occ_b_I.size(); ++ir) {
                                for (size_t is = ir + 1; is < occ_b_I.size(); ++is) {
                                    int q = occ_b_I[iq], r = occ_b_I[ir], s = occ_b_I[is];
                                    add_antisym_1_3_atomic(Gamma4_abbb, p, q, r, s, p, q, r, s, c_IJ);
                                }
                            }
                        }
                    }
                    
                    // aabb: 2 alpha + 2 beta
                    for (size_t ip = 0; ip < occ_a_I.size(); ++ip) {
                        for (size_t iq = ip + 1; iq < occ_a_I.size(); ++iq) {
                            int p = occ_a_I[ip], q = occ_a_I[iq];
                            for (size_t ir = 0; ir < occ_b_I.size(); ++ir) {
                                for (size_t is = ir + 1; is < occ_b_I.size(); ++is) {
                                    int r = occ_b_I[ir], s = occ_b_I[is];
                                    add_antisym_2_2_atomic(Gamma4_aabb, p, q, r, s, p, q, r, s, c_IJ);
                                }
                            }
                        }
                    }
                }
                // ======== SINGLE EXCITATION ========
                else if (n_excite == 1) {
                    if (n_alpha_exc == 1) {
                        int p_new = __builtin_ctzll(diff_a_in_I);
                        int p_old = __builtin_ctzll(diff_a_in_J);
                        int phase = single_phase_alpha(det_J, p_old, p_new);
                        double val = c_IJ * phase;
                        
                        // 3 alpha spectators -> aaaa
                        for (size_t ik1 = 0; ik1 < occ_a_J.size(); ++ik1) {
                            int k1 = occ_a_J[ik1];
                            if (k1 == p_old) continue;
                            for (size_t ik2 = ik1 + 1; ik2 < occ_a_J.size(); ++ik2) {
                                int k2 = occ_a_J[ik2];
                                if (k2 == p_old) continue;
                                for (size_t ik3 = ik2 + 1; ik3 < occ_a_J.size(); ++ik3) {
                                    int k3 = occ_a_J[ik3];
                                    if (k3 == p_old) continue;
                                    add_antisym_4_atomic(Gamma4_aaaa, p_new, k1, k2, k3, p_old, k1, k2, k3, val);
                                }
                            }
                        }
                        
                        // 2 alpha + 1 beta spectators -> aaab
                        for (size_t ik1 = 0; ik1 < occ_a_J.size(); ++ik1) {
                            int k1 = occ_a_J[ik1];
                            if (k1 == p_old) continue;
                            for (size_t ik2 = ik1 + 1; ik2 < occ_a_J.size(); ++ik2) {
                                int k2 = occ_a_J[ik2];
                                if (k2 == p_old) continue;
                                for (int k3 : occ_b_J) {
                                    add_antisym_3_1_atomic(Gamma4_aaab, p_new, k1, k2, k3, p_old, k1, k2, k3, val);
                                }
                            }
                        }
                        
                        // 1 alpha + 2 beta spectators -> aabb
                        for (int k1 : occ_a_J) {
                            if (k1 == p_old) continue;
                            for (size_t ik2 = 0; ik2 < occ_b_J.size(); ++ik2) {
                                for (size_t ik3 = ik2 + 1; ik3 < occ_b_J.size(); ++ik3) {
                                    int k2 = occ_b_J[ik2], k3 = occ_b_J[ik3];
                                    add_antisym_2_2_atomic(Gamma4_aabb, p_new, k1, k2, k3, p_old, k1, k2, k3, val);
                                }
                            }
                        }
                        
                        // 3 beta spectators -> abbb
                        for (size_t ik1 = 0; ik1 < occ_b_J.size(); ++ik1) {
                            for (size_t ik2 = ik1 + 1; ik2 < occ_b_J.size(); ++ik2) {
                                for (size_t ik3 = ik2 + 1; ik3 < occ_b_J.size(); ++ik3) {
                                    int k1 = occ_b_J[ik1], k2 = occ_b_J[ik2], k3 = occ_b_J[ik3];
                                    add_antisym_1_3_atomic(Gamma4_abbb, p_new, k1, k2, k3, p_old, k1, k2, k3, val);
                                }
                            }
                        }
                    }
                    else {  // beta excitation
                        int p_new = __builtin_ctzll(diff_b_in_I);
                        int p_old = __builtin_ctzll(diff_b_in_J);
                        int phase = single_phase_beta(det_J, p_old, p_new);
                        double val = c_IJ * phase;
                        
                        // 3 beta spectators -> bbbb
                        for (size_t ik1 = 0; ik1 < occ_b_J.size(); ++ik1) {
                            int k1 = occ_b_J[ik1];
                            if (k1 == p_old) continue;
                            for (size_t ik2 = ik1 + 1; ik2 < occ_b_J.size(); ++ik2) {
                                int k2 = occ_b_J[ik2];
                                if (k2 == p_old) continue;
                                for (size_t ik3 = ik2 + 1; ik3 < occ_b_J.size(); ++ik3) {
                                    int k3 = occ_b_J[ik3];
                                    if (k3 == p_old) continue;
                                    add_antisym_4_atomic(Gamma4_bbbb, p_new, k1, k2, k3, p_old, k1, k2, k3, val);
                                }
                            }
                        }
                        
                        // 2 beta + 1 alpha spectators -> abbb
                        for (size_t ik1 = 0; ik1 < occ_b_J.size(); ++ik1) {
                            int k1 = occ_b_J[ik1];
                            if (k1 == p_old) continue;
                            for (size_t ik2 = ik1 + 1; ik2 < occ_b_J.size(); ++ik2) {
                                int k2 = occ_b_J[ik2];
                                if (k2 == p_old) continue;
                                for (int k3 : occ_a_J) {
                                    add_antisym_1_3_atomic(Gamma4_abbb, k3, p_new, k1, k2, k3, p_old, k1, k2, val);
                                }
                            }
                        }
                        
                        // 1 beta + 2 alpha spectators -> aabb
                        for (int k1 : occ_b_J) {
                            if (k1 == p_old) continue;
                            for (size_t ik2 = 0; ik2 < occ_a_J.size(); ++ik2) {
                                for (size_t ik3 = ik2 + 1; ik3 < occ_a_J.size(); ++ik3) {
                                    int k2 = occ_a_J[ik2], k3 = occ_a_J[ik3];
                                    add_antisym_2_2_atomic(Gamma4_aabb, k2, k3, p_new, k1, k2, k3, p_old, k1, val);
                                }
                            }
                        }
                        
                        // 3 alpha spectators -> aaab
                        for (size_t ik1 = 0; ik1 < occ_a_J.size(); ++ik1) {
                            for (size_t ik2 = ik1 + 1; ik2 < occ_a_J.size(); ++ik2) {
                                for (size_t ik3 = ik2 + 1; ik3 < occ_a_J.size(); ++ik3) {
                                    int k1 = occ_a_J[ik1], k2 = occ_a_J[ik2], k3 = occ_a_J[ik3];
                                    add_antisym_3_1_atomic(Gamma4_aaab, k1, k2, k3, p_new, k1, k2, k3, p_old, val);
                                }
                            }
                        }
                    }
                }
                // ======== DOUBLE EXCITATION ========
                else if (n_excite == 2) {
                    if (n_alpha_exc == 2) {
                        auto p_list = get_set_bits(diff_a_in_I);
                        auto q_list = get_set_bits(diff_a_in_J);
                        std::sort(p_list.begin(), p_list.end());
                        std::sort(q_list.begin(), q_list.end());
                        int p1 = p_list[0], p2 = p_list[1];
                        int q1 = q_list[0], q2 = q_list[1];
                        int phase = double_phase_alpha(det_J.alpha, q1, q2, p1, p2);
                        double val = c_IJ * phase;
                        
                        // 2 alpha spectators -> aaaa
                        for (size_t ik1 = 0; ik1 < occ_a_J.size(); ++ik1) {
                            int k1 = occ_a_J[ik1];
                            if (k1 == q1 || k1 == q2) continue;
                            for (size_t ik2 = ik1 + 1; ik2 < occ_a_J.size(); ++ik2) {
                                int k2 = occ_a_J[ik2];
                                if (k2 == q1 || k2 == q2) continue;
                                add_antisym_4_atomic(Gamma4_aaaa, p1, p2, k1, k2, q1, q2, k1, k2, val);
                            }
                        }
                        
                        // 1 alpha + 1 beta spectators -> aaab
                        for (int k1 : occ_a_J) {
                            if (k1 == q1 || k1 == q2) continue;
                            for (int k2 : occ_b_J) {
                                add_antisym_3_1_atomic(Gamma4_aaab, p1, p2, k1, k2, q1, q2, k1, k2, val);
                            }
                        }
                        
                        // 2 beta spectators -> aabb
                        for (size_t ik1 = 0; ik1 < occ_b_J.size(); ++ik1) {
                            for (size_t ik2 = ik1 + 1; ik2 < occ_b_J.size(); ++ik2) {
                                int k1 = occ_b_J[ik1], k2 = occ_b_J[ik2];
                                add_antisym_2_2_atomic(Gamma4_aabb, p1, p2, k1, k2, q1, q2, k1, k2, val);
                            }
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
                        
                        // 2 beta spectators -> bbbb
                        for (size_t ik1 = 0; ik1 < occ_b_J.size(); ++ik1) {
                            int k1 = occ_b_J[ik1];
                            if (k1 == q1 || k1 == q2) continue;
                            for (size_t ik2 = ik1 + 1; ik2 < occ_b_J.size(); ++ik2) {
                                int k2 = occ_b_J[ik2];
                                if (k2 == q1 || k2 == q2) continue;
                                add_antisym_4_atomic(Gamma4_bbbb, p1, p2, k1, k2, q1, q2, k1, k2, val);
                            }
                        }
                        
                        // 1 beta + 1 alpha spectators -> abbb
                        for (int k1 : occ_b_J) {
                            if (k1 == q1 || k1 == q2) continue;
                            for (int k2 : occ_a_J) {
                                add_antisym_1_3_atomic(Gamma4_abbb, k2, p1, p2, k1, k2, q1, q2, k1, val);
                            }
                        }
                        
                        // 2 alpha spectators -> aabb
                        for (size_t ik1 = 0; ik1 < occ_a_J.size(); ++ik1) {
                            for (size_t ik2 = ik1 + 1; ik2 < occ_a_J.size(); ++ik2) {
                                int k1 = occ_a_J[ik1], k2 = occ_a_J[ik2];
                                add_antisym_2_2_atomic(Gamma4_aabb, k1, k2, p1, p2, k1, k2, q1, q2, val);
                            }
                        }
                    }
                    else {  // 1α + 1β
                        int p_a = __builtin_ctzll(diff_a_in_I);
                        int q_a = __builtin_ctzll(diff_a_in_J);
                        int p_b = __builtin_ctzll(diff_b_in_I);
                        int q_b = __builtin_ctzll(diff_b_in_J);
                        int phase = single_phase_alpha(det_J, q_a, p_a) * single_phase_beta(det_J, q_b, p_b);
                        double val = c_IJ * phase;
                        
                        // 2 alpha + 1 beta spectators -> aaab
                        // Need 2 α spectators from common, 0 β spectators (excitation provides the β)
                        {
                            uint64_t common_a = det_I.alpha & det_J.alpha;
                            auto common_a_list = get_set_bits(common_a);
                            for (size_t ik1 = 0; ik1 < common_a_list.size(); ++ik1) {
                                int k1 = common_a_list[ik1];
                                for (size_t ik2 = ik1 + 1; ik2 < common_a_list.size(); ++ik2) {
                                    int k2 = common_a_list[ik2];
                                    // aaab: creators=(p_a, k1, k2)α + (p_b)β, annihilators=(q_a, k1, k2)α + (q_b)β
                                    add_antisym_3_1_atomic(Gamma4_aaab, p_a, k1, k2, p_b, q_a, k1, k2, q_b, val);
                                }
                            }
                        }
                        
                        // 1 alpha + 2 beta spectators -> abbb
                        // Need 0 α spectators (excitation provides α), 2 β spectators from common
                        {
                            uint64_t common_b = det_I.beta & det_J.beta;
                            auto common_b_list = get_set_bits(common_b);
                            for (size_t ik1 = 0; ik1 < common_b_list.size(); ++ik1) {
                                int k1 = common_b_list[ik1];
                                for (size_t ik2 = ik1 + 1; ik2 < common_b_list.size(); ++ik2) {
                                    int k2 = common_b_list[ik2];
                                    // Use add_antisym_1_3_atomic for correct 36-term antisymmetrization
                                    add_antisym_1_3_atomic(Gamma4_abbb, p_a, p_b, k1, k2, q_a, q_b, k1, k2, val);
                                }
                            }
                        }
                        
                        // aabb with common spectators (1 common α + 1 common β)
                        {
                            uint64_t common_a = det_I.alpha & det_J.alpha;
                            uint64_t common_b = det_I.beta & det_J.beta;
                            auto common_a_list = get_set_bits(common_a);
                            auto common_b_list = get_set_bits(common_b);
                            
                            for (int spec_a : common_a_list) {
                                for (int spec_b : common_b_list) {
                                    add_antisym_2_2_atomic(Gamma4_aabb, 
                                        p_a, spec_a, p_b, spec_b,
                                        q_a, spec_a, q_b, spec_b,
                                        val);
                                }
                            }
                        }
                    }
                }
                // ======== TRIPLE EXCITATION ========
                else if (n_excite == 3) {
                    // 4-RDM with 3-excitation needs 1 spectator from common orbitals
                    
                    if (n_alpha_exc == 3 && n_beta_exc == 0) {  // 3α excitation
                        auto p_list = get_set_bits(diff_a_in_I);
                        auto q_list = get_set_bits(diff_a_in_J);
                        std::sort(p_list.begin(), p_list.end());
                        std::sort(q_list.begin(), q_list.end());
                        int p1 = p_list[0], p2 = p_list[1], p3 = p_list[2];
                        int q1 = q_list[0], q2 = q_list[1], q3 = q_list[2];
                        int phase = triple_phase_alpha(det_J.alpha, q1, q2, q3, p1, p2, p3);
                        double val = c_IJ * phase;
                        
                        // α spectator → aaaa
                        uint64_t common_a = det_I.alpha & det_J.alpha;
                        for (int k : get_set_bits(common_a)) {
                            add_antisym_4_atomic(Gamma4_aaaa, p1, p2, p3, k, q1, q2, q3, k, val);
                        }
                        
                        // β spectator → aaab
                        uint64_t common_b = det_I.beta & det_J.beta;
                        for (int k : get_set_bits(common_b)) {
                            add_antisym_3_1_atomic(Gamma4_aaab, p1, p2, p3, k, q1, q2, q3, k, val);
                        }
                    }
                    else if (n_alpha_exc == 0 && n_beta_exc == 3) {  // 3β excitation
                        auto p_list = get_set_bits(diff_b_in_I);
                        auto q_list = get_set_bits(diff_b_in_J);
                        std::sort(p_list.begin(), p_list.end());
                        std::sort(q_list.begin(), q_list.end());
                        int p1 = p_list[0], p2 = p_list[1], p3 = p_list[2];
                        int q1 = q_list[0], q2 = q_list[1], q3 = q_list[2];
                        int phase = triple_phase_beta(det_J.beta, q1, q2, q3, p1, p2, p3);
                        double val = c_IJ * phase;
                        
                        // β spectator → bbbb
                        uint64_t common_b = det_I.beta & det_J.beta;
                        for (int k : get_set_bits(common_b)) {
                            add_antisym_4_atomic(Gamma4_bbbb, p1, p2, p3, k, q1, q2, q3, k, val);
                        }
                        
                        // α spectator → abbb
                        uint64_t common_a = det_I.alpha & det_J.alpha;
                        for (int k : get_set_bits(common_a)) {
                            add_antisym_1_3_atomic(Gamma4_abbb, k, p1, p2, p3, k, q1, q2, q3, val);
                        }
                    }
                    else if (n_alpha_exc == 2 && n_beta_exc == 1) {  // 2α + 1β excitation
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
                        double val = c_IJ * phase;
                        
                        // α spectator → aaab
                        uint64_t common_a = det_I.alpha & det_J.alpha;
                        for (int k : get_set_bits(common_a)) {
                            add_antisym_3_1_atomic(Gamma4_aaab, p1, p2, k, p_b, q1, q2, k, q_b, val);
                        }
                        
                        // β spectator → aabb
                        uint64_t common_b = det_I.beta & det_J.beta;
                        for (int k : get_set_bits(common_b)) {
                            add_antisym_2_2_atomic(Gamma4_aabb, p1, p2, p_b, k, q1, q2, q_b, k, val);
                        }
                    }
                    else if (n_alpha_exc == 1 && n_beta_exc == 2) {  // 1α + 2β excitation
                        auto p_b_list = get_set_bits(diff_b_in_I);
                        auto q_b_list = get_set_bits(diff_b_in_J);
                        std::sort(p_b_list.begin(), p_b_list.end());
                        std::sort(q_b_list.begin(), q_b_list.end());
                        int p1 = p_b_list[0], p2 = p_b_list[1];
                        int q1 = q_b_list[0], q2 = q_b_list[1];
                        int p_a = __builtin_ctzll(diff_a_in_I);
                        int q_a = __builtin_ctzll(diff_a_in_J);
                        int phase = single_phase_alpha(det_J, q_a, p_a) *
                                   double_phase_beta(det_J.beta, q1, q2, p1, p2);
                        double val = c_IJ * phase;
                        
                        // β spectator → abbb
                        uint64_t common_b = det_I.beta & det_J.beta;
                        for (int k : get_set_bits(common_b)) {
                            add_antisym_1_3_atomic(Gamma4_abbb, p_a, p1, p2, k, q_a, q1, q2, k, val);
                        }
                        
                        // α spectator → aabb
                        uint64_t common_a = det_I.alpha & det_J.alpha;
                        for (int k : get_set_bits(common_a)) {
                            add_antisym_2_2_atomic(Gamma4_aabb, p_a, k, p1, p2, q_a, k, q1, q2, val);
                        }
                    }
                }
                // ======== QUADRUPLE EXCITATION ========
                else if (n_excite == 4) {
                    // 4-RDM with 4-excitation has no spectators
                    if (n_alpha_exc == 4) {
                        auto p_list = get_set_bits(diff_a_in_I);
                        auto q_list = get_set_bits(diff_a_in_J);
                        std::sort(p_list.begin(), p_list.end());
                        std::sort(q_list.begin(), q_list.end());
                        int p1 = p_list[0], p2 = p_list[1], p3 = p_list[2], p4 = p_list[3];
                        int q1 = q_list[0], q2 = q_list[1], q3 = q_list[2], q4 = q_list[3];
                        int phase = quadruple_phase_alpha(det_J.alpha, q1, q2, q3, q4, p1, p2, p3, p4);
                        add_antisym_4_atomic(Gamma4_aaaa, p1, p2, p3, p4, q1, q2, q3, q4, c_IJ * phase);
                    }
                    else if (n_beta_exc == 4) {
                        auto p_list = get_set_bits(diff_b_in_I);
                        auto q_list = get_set_bits(diff_b_in_J);
                        std::sort(p_list.begin(), p_list.end());
                        std::sort(q_list.begin(), q_list.end());
                        int p1 = p_list[0], p2 = p_list[1], p3 = p_list[2], p4 = p_list[3];
                        int q1 = q_list[0], q2 = q_list[1], q3 = q_list[2], q4 = q_list[3];
                        int phase = quadruple_phase_beta(det_J.beta, q1, q2, q3, q4, p1, p2, p3, p4);
                        add_antisym_4_atomic(Gamma4_bbbb, p1, p2, p3, p4, q1, q2, q3, q4, c_IJ * phase);
                    }
                    else if (n_alpha_exc == 3) {  // 3α + 1β
                        auto p_a_list = get_set_bits(diff_a_in_I);
                        auto q_a_list = get_set_bits(diff_a_in_J);
                        std::sort(p_a_list.begin(), p_a_list.end());
                        std::sort(q_a_list.begin(), q_a_list.end());
                        int p1 = p_a_list[0], p2 = p_a_list[1], p3 = p_a_list[2];
                        int q1 = q_a_list[0], q2 = q_a_list[1], q3 = q_a_list[2];
                        int p_b = __builtin_ctzll(diff_b_in_I);
                        int q_b = __builtin_ctzll(diff_b_in_J);
                        int phase = triple_phase_alpha(det_J.alpha, q1, q2, q3, p1, p2, p3) *
                                   single_phase_beta(det_J, q_b, p_b);
                        add_antisym_3_1_atomic(Gamma4_aaab, p1, p2, p3, p_b, q1, q2, q3, q_b, c_IJ * phase);
                    }
                    else if (n_beta_exc == 3) {  // 1α + 3β
                        auto p_b_list = get_set_bits(diff_b_in_I);
                        auto q_b_list = get_set_bits(diff_b_in_J);
                        std::sort(p_b_list.begin(), p_b_list.end());
                        std::sort(q_b_list.begin(), q_b_list.end());
                        int p1 = p_b_list[0], p2 = p_b_list[1], p3 = p_b_list[2];
                        int q1 = q_b_list[0], q2 = q_b_list[1], q3 = q_b_list[2];
                        int p_a = __builtin_ctzll(diff_a_in_I);
                        int q_a = __builtin_ctzll(diff_a_in_J);
                        int phase = single_phase_alpha(det_J, q_a, p_a) *
                                   triple_phase_beta(det_J.beta, q1, q2, q3, p1, p2, p3);
                        add_antisym_1_3_atomic(Gamma4_abbb, p_a, p1, p2, p3, q_a, q1, q2, q3, c_IJ * phase);
                    }
                    else {  // 2α + 2β
                        auto p_a_list = get_set_bits(diff_a_in_I);
                        auto q_a_list = get_set_bits(diff_a_in_J);
                        auto p_b_list = get_set_bits(diff_b_in_I);
                        auto q_b_list = get_set_bits(diff_b_in_J);
                        std::sort(p_a_list.begin(), p_a_list.end());
                        std::sort(q_a_list.begin(), q_a_list.end());
                        std::sort(p_b_list.begin(), p_b_list.end());
                        std::sort(q_b_list.begin(), q_b_list.end());
                        int p1_a = p_a_list[0], p2_a = p_a_list[1];
                        int q1_a = q_a_list[0], q2_a = q_a_list[1];
                        int p1_b = p_b_list[0], p2_b = p_b_list[1];
                        int q1_b = q_b_list[0], q2_b = q_b_list[1];
                        int phase = double_phase_alpha(det_J.alpha, q1_a, q2_a, p1_a, p2_a) *
                                   double_phase_beta(det_J.beta, q1_b, q2_b, p1_b, p2_b);
                        add_antisym_2_2_atomic(Gamma4_aabb, p1_a, p2_a, p1_b, p2_b, q1_a, q2_a, q1_b, q2_b, c_IJ * phase);
                    }
                }
            }
        }
    
    return std::make_tuple(std::move(Gamma4_aaaa), std::move(Gamma4_aaab),
                          std::move(Gamma4_aabb), std::move(Gamma4_abbb),
                          std::move(Gamma4_bbbb));
}

// ============================================================================
// Spin-Resolved 4-RDM Cumulant Approximation (2-body cumulant only)
// ============================================================================
// Setting λ³ = λ⁴ = 0, the closed-form reconstruction is:
//   Γ⁴ = A_S[Γ² ⊗ Γ²] − 2·det₄(γ)
//
// This is the exact analogue of the 3-RDM formula:
//   Γ³ = A_S[Γ² ⊗ γ] − 2·det₃(γ)
//
// Derivation: Starting from Γ⁴ = A[λ²⊗λ²] + A[λ²⊗γ⊗γ] + det₄(γ),
// substituting λ² = Γ² − det₂(γ), the cross terms simplify:
//   A[λ²⊗γ⊗γ] = A[Γ²⊗det₂(γ)] − A[det₂(γ)⊗det₂(γ)]
// and the identity A[det₂⊗det₂] = 3·det₄ (for same-spin) yields:
//   Γ⁴ = (1/2)·S₃₆[Γ²,Γ²] − 2·det₄(γ)
//
// Per-sector term counts:
//   aaaa/bbbb: 18 (Γ²⊗Γ²) + det₄         (3 creator partitions × 6 ann. assignments)
//   aaab/abbb:  9 (Γ²⊗Γ²) + det₃·γ       (3 creator partitions × 3 ann. assignments)
//   aabb:       9 (Γ²⊗Γ²) + Δ²_aa·Δ²_bb  (1 aa·bb + 4+4 ab·ab)

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>,
           std::vector<double>, std::vector<double>>
compute_4rdm_cumulant_from_2rdm_spin_resolved(
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
    size_t N7 = N6 * N;
    size_t N8 = N7 * N;

    if (n_orb > 14) {
        size_t storage_mb = (N8 * 8 * 5) / (1024 * 1024);
        throw std::runtime_error("compute_4rdm_cumulant_from_2rdm_spin_resolved: n_orb > 14 requires ~" +
                                 std::to_string(storage_mb) + "MB storage, aborting");
    }

    std::vector<double> Gamma4_aaaa(N8, 0.0);
    std::vector<double> Gamma4_aaab(N8, 0.0);
    std::vector<double> Gamma4_aabb(N8, 0.0);
    std::vector<double> Gamma4_abbb(N8, 0.0);
    std::vector<double> Gamma4_bbbb(N8, 0.0);

    auto idx8 = [N, N2, N3, N4, N5, N6, N7](int p, int q, int r, int s,
                                              int t, int u, int v, int w) {
        return p*N7 + q*N6 + r*N5 + s*N4 + t*N3 + u*N2 + v*N + w;
    };
    auto idx4 = [N, N2, N3](int p, int q, int r, int s) {
        return p*N3 + q*N2 + r*N + s;
    };
    auto idx2 = [N](int p, int q) {
        return p*N + q;
    };

    // ===================== aaaa sector =====================
    // Γ⁴_aaaa = A_S[Γ²_aa ⊗ Γ²_aa] − 2·det₄(γ_a)
    // 18 A_S terms: 3 creator partitions {pq|rs},{pr|qs},{ps|qr}
    //             × 6 annihilator assignments each (signs: +−+,+−+)
    #pragma omp parallel for collapse(4)
    for (int p = 0; p < static_cast<int>(N); ++p) {
        for (size_t q = 0; q < N; ++q) {
            for (size_t r = 0; r < N; ++r) {
                for (size_t s = 0; s < N; ++s) {
                    for (size_t t = 0; t < N; ++t) {
                        for (size_t u = 0; u < N; ++u) {
                            for (size_t v = 0; v < N; ++v) {
                                for (size_t w = 0; w < N; ++w) {
                                    double val = 0.0;
                                    // Creator {pq|rs}, sign_c = +1
                                    val += Gamma_aa[idx4(p,q,t,u)] * Gamma_aa[idx4(r,s,v,w)];
                                    val -= Gamma_aa[idx4(p,q,t,v)] * Gamma_aa[idx4(r,s,u,w)];
                                    val += Gamma_aa[idx4(p,q,t,w)] * Gamma_aa[idx4(r,s,u,v)];
                                    val += Gamma_aa[idx4(p,q,u,v)] * Gamma_aa[idx4(r,s,t,w)];
                                    val -= Gamma_aa[idx4(p,q,u,w)] * Gamma_aa[idx4(r,s,t,v)];
                                    val += Gamma_aa[idx4(p,q,v,w)] * Gamma_aa[idx4(r,s,t,u)];
                                    // Creator {pr|qs}, sign_c = −1
                                    val -= Gamma_aa[idx4(p,r,t,u)] * Gamma_aa[idx4(q,s,v,w)];
                                    val += Gamma_aa[idx4(p,r,t,v)] * Gamma_aa[idx4(q,s,u,w)];
                                    val -= Gamma_aa[idx4(p,r,t,w)] * Gamma_aa[idx4(q,s,u,v)];
                                    val -= Gamma_aa[idx4(p,r,u,v)] * Gamma_aa[idx4(q,s,t,w)];
                                    val += Gamma_aa[idx4(p,r,u,w)] * Gamma_aa[idx4(q,s,t,v)];
                                    val -= Gamma_aa[idx4(p,r,v,w)] * Gamma_aa[idx4(q,s,t,u)];
                                    // Creator {ps|qr}, sign_c = +1
                                    val += Gamma_aa[idx4(p,s,t,u)] * Gamma_aa[idx4(q,r,v,w)];
                                    val -= Gamma_aa[idx4(p,s,t,v)] * Gamma_aa[idx4(q,r,u,w)];
                                    val += Gamma_aa[idx4(p,s,t,w)] * Gamma_aa[idx4(q,r,u,v)];
                                    val += Gamma_aa[idx4(p,s,u,v)] * Gamma_aa[idx4(q,r,t,w)];
                                    val -= Gamma_aa[idx4(p,s,u,w)] * Gamma_aa[idx4(q,r,t,v)];
                                    val += Gamma_aa[idx4(p,s,v,w)] * Gamma_aa[idx4(q,r,t,u)];

                                    // −2·det₄(γ_a) via Laplace expansion along rows {pq}
                                    double d_pq_tu = gamma_aa[idx2(p,t)]*gamma_aa[idx2(q,u)] - gamma_aa[idx2(p,u)]*gamma_aa[idx2(q,t)];
                                    double d_pq_tv = gamma_aa[idx2(p,t)]*gamma_aa[idx2(q,v)] - gamma_aa[idx2(p,v)]*gamma_aa[idx2(q,t)];
                                    double d_pq_tw = gamma_aa[idx2(p,t)]*gamma_aa[idx2(q,w)] - gamma_aa[idx2(p,w)]*gamma_aa[idx2(q,t)];
                                    double d_pq_uv = gamma_aa[idx2(p,u)]*gamma_aa[idx2(q,v)] - gamma_aa[idx2(p,v)]*gamma_aa[idx2(q,u)];
                                    double d_pq_uw = gamma_aa[idx2(p,u)]*gamma_aa[idx2(q,w)] - gamma_aa[idx2(p,w)]*gamma_aa[idx2(q,u)];
                                    double d_pq_vw = gamma_aa[idx2(p,v)]*gamma_aa[idx2(q,w)] - gamma_aa[idx2(p,w)]*gamma_aa[idx2(q,v)];
                                    double d_rs_tu = gamma_aa[idx2(r,t)]*gamma_aa[idx2(s,u)] - gamma_aa[idx2(r,u)]*gamma_aa[idx2(s,t)];
                                    double d_rs_tv = gamma_aa[idx2(r,t)]*gamma_aa[idx2(s,v)] - gamma_aa[idx2(r,v)]*gamma_aa[idx2(s,t)];
                                    double d_rs_tw = gamma_aa[idx2(r,t)]*gamma_aa[idx2(s,w)] - gamma_aa[idx2(r,w)]*gamma_aa[idx2(s,t)];
                                    double d_rs_uv = gamma_aa[idx2(r,u)]*gamma_aa[idx2(s,v)] - gamma_aa[idx2(r,v)]*gamma_aa[idx2(s,u)];
                                    double d_rs_uw = gamma_aa[idx2(r,u)]*gamma_aa[idx2(s,w)] - gamma_aa[idx2(r,w)]*gamma_aa[idx2(s,u)];
                                    double d_rs_vw = gamma_aa[idx2(r,v)]*gamma_aa[idx2(s,w)] - gamma_aa[idx2(r,w)]*gamma_aa[idx2(s,v)];
                                    double det4 = d_pq_tu*d_rs_vw - d_pq_tv*d_rs_uw + d_pq_tw*d_rs_uv
                                                + d_pq_uv*d_rs_tw - d_pq_uw*d_rs_tv + d_pq_vw*d_rs_tu;
                                    val -= 2.0 * det4;

                                    Gamma4_aaaa[idx8(p,q,r,s,t,u,v,w)] = val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // ===================== bbbb sector =====================
    // Same structure as aaaa with β
    #pragma omp parallel for collapse(4)
    for (int p = 0; p < static_cast<int>(N); ++p) {
        for (size_t q = 0; q < N; ++q) {
            for (size_t r = 0; r < N; ++r) {
                for (size_t s = 0; s < N; ++s) {
                    for (size_t t = 0; t < N; ++t) {
                        for (size_t u = 0; u < N; ++u) {
                            for (size_t v = 0; v < N; ++v) {
                                for (size_t w = 0; w < N; ++w) {
                                    double val = 0.0;
                                    val += Gamma_bb[idx4(p,q,t,u)] * Gamma_bb[idx4(r,s,v,w)];
                                    val -= Gamma_bb[idx4(p,q,t,v)] * Gamma_bb[idx4(r,s,u,w)];
                                    val += Gamma_bb[idx4(p,q,t,w)] * Gamma_bb[idx4(r,s,u,v)];
                                    val += Gamma_bb[idx4(p,q,u,v)] * Gamma_bb[idx4(r,s,t,w)];
                                    val -= Gamma_bb[idx4(p,q,u,w)] * Gamma_bb[idx4(r,s,t,v)];
                                    val += Gamma_bb[idx4(p,q,v,w)] * Gamma_bb[idx4(r,s,t,u)];
                                    val -= Gamma_bb[idx4(p,r,t,u)] * Gamma_bb[idx4(q,s,v,w)];
                                    val += Gamma_bb[idx4(p,r,t,v)] * Gamma_bb[idx4(q,s,u,w)];
                                    val -= Gamma_bb[idx4(p,r,t,w)] * Gamma_bb[idx4(q,s,u,v)];
                                    val -= Gamma_bb[idx4(p,r,u,v)] * Gamma_bb[idx4(q,s,t,w)];
                                    val += Gamma_bb[idx4(p,r,u,w)] * Gamma_bb[idx4(q,s,t,v)];
                                    val -= Gamma_bb[idx4(p,r,v,w)] * Gamma_bb[idx4(q,s,t,u)];
                                    val += Gamma_bb[idx4(p,s,t,u)] * Gamma_bb[idx4(q,r,v,w)];
                                    val -= Gamma_bb[idx4(p,s,t,v)] * Gamma_bb[idx4(q,r,u,w)];
                                    val += Gamma_bb[idx4(p,s,t,w)] * Gamma_bb[idx4(q,r,u,v)];
                                    val += Gamma_bb[idx4(p,s,u,v)] * Gamma_bb[idx4(q,r,t,w)];
                                    val -= Gamma_bb[idx4(p,s,u,w)] * Gamma_bb[idx4(q,r,t,v)];
                                    val += Gamma_bb[idx4(p,s,v,w)] * Gamma_bb[idx4(q,r,t,u)];

                                    double d_pq_tu = gamma_bb[idx2(p,t)]*gamma_bb[idx2(q,u)] - gamma_bb[idx2(p,u)]*gamma_bb[idx2(q,t)];
                                    double d_pq_tv = gamma_bb[idx2(p,t)]*gamma_bb[idx2(q,v)] - gamma_bb[idx2(p,v)]*gamma_bb[idx2(q,t)];
                                    double d_pq_tw = gamma_bb[idx2(p,t)]*gamma_bb[idx2(q,w)] - gamma_bb[idx2(p,w)]*gamma_bb[idx2(q,t)];
                                    double d_pq_uv = gamma_bb[idx2(p,u)]*gamma_bb[idx2(q,v)] - gamma_bb[idx2(p,v)]*gamma_bb[idx2(q,u)];
                                    double d_pq_uw = gamma_bb[idx2(p,u)]*gamma_bb[idx2(q,w)] - gamma_bb[idx2(p,w)]*gamma_bb[idx2(q,u)];
                                    double d_pq_vw = gamma_bb[idx2(p,v)]*gamma_bb[idx2(q,w)] - gamma_bb[idx2(p,w)]*gamma_bb[idx2(q,v)];
                                    double d_rs_tu = gamma_bb[idx2(r,t)]*gamma_bb[idx2(s,u)] - gamma_bb[idx2(r,u)]*gamma_bb[idx2(s,t)];
                                    double d_rs_tv = gamma_bb[idx2(r,t)]*gamma_bb[idx2(s,v)] - gamma_bb[idx2(r,v)]*gamma_bb[idx2(s,t)];
                                    double d_rs_tw = gamma_bb[idx2(r,t)]*gamma_bb[idx2(s,w)] - gamma_bb[idx2(r,w)]*gamma_bb[idx2(s,t)];
                                    double d_rs_uv = gamma_bb[idx2(r,u)]*gamma_bb[idx2(s,v)] - gamma_bb[idx2(r,v)]*gamma_bb[idx2(s,u)];
                                    double d_rs_uw = gamma_bb[idx2(r,u)]*gamma_bb[idx2(s,w)] - gamma_bb[idx2(r,w)]*gamma_bb[idx2(s,u)];
                                    double d_rs_vw = gamma_bb[idx2(r,v)]*gamma_bb[idx2(s,w)] - gamma_bb[idx2(r,w)]*gamma_bb[idx2(s,v)];
                                    double det4 = d_pq_tu*d_rs_vw - d_pq_tv*d_rs_uw + d_pq_tw*d_rs_uv
                                                + d_pq_uv*d_rs_tw - d_pq_uw*d_rs_tv + d_pq_vw*d_rs_tu;
                                    val -= 2.0 * det4;

                                    Gamma4_bbbb[idx8(p,q,r,s,t,u,v,w)] = val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // ===================== aabb sector =====================
    // Γ⁴_aabb = A_S[Γ² ⊗ Γ²] − 2·Δ²_aa(pq;tu)·Δ²_bb(rs;vw)
    // (p,q)=α creators, (r,s)=β creators, (t,u)=α ann., (v,w)=β ann.
    // 9 A_S terms: 1 from {αα|ββ} + 4 from {pr_αβ|qs_αβ} + 4 from {ps_αβ|qr_αβ}
    #pragma omp parallel for collapse(4)
    for (int p = 0; p < static_cast<int>(N); ++p) {
        for (size_t q = 0; q < N; ++q) {
            for (size_t r = 0; r < N; ++r) {
                for (size_t s = 0; s < N; ++s) {
                    for (size_t t = 0; t < N; ++t) {
                        for (size_t u = 0; u < N; ++u) {
                            for (size_t v = 0; v < N; ++v) {
                                for (size_t w = 0; w < N; ++w) {
                                    double val = 0.0;
                                    // Creator {pq_αα|rs_ββ}
                                    val += Gamma_aa[idx4(p,q,t,u)] * Gamma_bb[idx4(r,s,v,w)];
                                    // Creator {pr_αβ|qs_αβ}, sign_c = −1
                                    val += Gamma_ab[idx4(p,r,t,v)] * Gamma_ab[idx4(q,s,u,w)];
                                    val -= Gamma_ab[idx4(p,r,t,w)] * Gamma_ab[idx4(q,s,u,v)];
                                    val -= Gamma_ab[idx4(p,r,u,v)] * Gamma_ab[idx4(q,s,t,w)];
                                    val += Gamma_ab[idx4(p,r,u,w)] * Gamma_ab[idx4(q,s,t,v)];
                                    // Creator {ps_αβ|qr_αβ}, sign_c = +1
                                    val -= Gamma_ab[idx4(p,s,t,v)] * Gamma_ab[idx4(q,r,u,w)];
                                    val += Gamma_ab[idx4(p,s,t,w)] * Gamma_ab[idx4(q,r,u,v)];
                                    val += Gamma_ab[idx4(p,s,u,v)] * Gamma_ab[idx4(q,r,t,w)];
                                    val -= Gamma_ab[idx4(p,s,u,w)] * Gamma_ab[idx4(q,r,t,v)];

                                    // −2·Δ²_aa(pq;tu)·Δ²_bb(rs;vw)
                                    double d_aa = gamma_aa[idx2(p,t)]*gamma_aa[idx2(q,u)]
                                                - gamma_aa[idx2(p,u)]*gamma_aa[idx2(q,t)];
                                    double d_bb = gamma_bb[idx2(r,v)]*gamma_bb[idx2(s,w)]
                                                - gamma_bb[idx2(r,w)]*gamma_bb[idx2(s,v)];
                                    val -= 2.0 * d_aa * d_bb;

                                    Gamma4_aabb[idx8(p,q,r,s,t,u,v,w)] = val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // ===================== aaab sector =====================
    // Γ⁴_aaab = A_S[Γ²_aa ⊗ Γ²_ab] − 2·det₃(γ_a)(pqr;tuv)·γ_b(s,w)
    // (p,q,r)=α creators, (s)=β creator, (t,u,v)=α ann., (w)=β ann.
    // 9 A_S terms: 3 creator partitions {pq|rs},{pr|qs},{qr|ps}
    //            × 3 annihilator assignments {tu,vw},{tv,uw},{uv,tw}
    #pragma omp parallel for collapse(4)
    for (int p = 0; p < static_cast<int>(N); ++p) {
        for (size_t q = 0; q < N; ++q) {
            for (size_t r = 0; r < N; ++r) {
                for (size_t s = 0; s < N; ++s) {
                    for (size_t t = 0; t < N; ++t) {
                        for (size_t u = 0; u < N; ++u) {
                            for (size_t v = 0; v < N; ++v) {
                                for (size_t w = 0; w < N; ++w) {
                                    double val = 0.0;
                                    // Creator {pq_αα|rs_αβ}, sign_c = +1
                                    val += Gamma_aa[idx4(p,q,t,u)] * Gamma_ab[idx4(r,s,v,w)];
                                    val -= Gamma_aa[idx4(p,q,t,v)] * Gamma_ab[idx4(r,s,u,w)];
                                    val += Gamma_aa[idx4(p,q,u,v)] * Gamma_ab[idx4(r,s,t,w)];
                                    // Creator {pr_αα|qs_αβ}, sign_c = −1
                                    val -= Gamma_aa[idx4(p,r,t,u)] * Gamma_ab[idx4(q,s,v,w)];
                                    val += Gamma_aa[idx4(p,r,t,v)] * Gamma_ab[idx4(q,s,u,w)];
                                    val -= Gamma_aa[idx4(p,r,u,v)] * Gamma_ab[idx4(q,s,t,w)];
                                    // Creator {qr_αα|ps_αβ}, sign_c = +1
                                    val += Gamma_aa[idx4(q,r,t,u)] * Gamma_ab[idx4(p,s,v,w)];
                                    val -= Gamma_aa[idx4(q,r,t,v)] * Gamma_ab[idx4(p,s,u,w)];
                                    val += Gamma_aa[idx4(q,r,u,v)] * Gamma_ab[idx4(p,s,t,w)];

                                    // −2·det₃(γ_a)(pqr;tuv)·γ_b(s,w)
                                    double det3 =
                                          gamma_aa[idx2(p,t)] * gamma_aa[idx2(q,u)] * gamma_aa[idx2(r,v)]
                                        - gamma_aa[idx2(p,t)] * gamma_aa[idx2(q,v)] * gamma_aa[idx2(r,u)]
                                        - gamma_aa[idx2(p,u)] * gamma_aa[idx2(q,t)] * gamma_aa[idx2(r,v)]
                                        + gamma_aa[idx2(p,u)] * gamma_aa[idx2(q,v)] * gamma_aa[idx2(r,t)]
                                        + gamma_aa[idx2(p,v)] * gamma_aa[idx2(q,t)] * gamma_aa[idx2(r,u)]
                                        - gamma_aa[idx2(p,v)] * gamma_aa[idx2(q,u)] * gamma_aa[idx2(r,t)];
                                    val -= 2.0 * det3 * gamma_bb[idx2(s,w)];

                                    Gamma4_aaab[idx8(p,q,r,s,t,u,v,w)] = val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // ===================== abbb sector =====================
    // Γ⁴_abbb = A_S[Γ²_ab ⊗ Γ²_bb] − 2·γ_a(p,t)·det₃(γ_b)(qrs;uvw)
    // (p)=α creator, (q,r,s)=β creators, (t)=α ann., (u,v,w)=β ann.
    // 9 A_S terms: 3 creator partitions {pq|rs},{pr|qs},{ps|qr}
    //            × 3 annihilator assignments {tu,vw},{tv,uw},{tw,uv}
    #pragma omp parallel for collapse(4)
    for (int p = 0; p < static_cast<int>(N); ++p) {
        for (size_t q = 0; q < N; ++q) {
            for (size_t r = 0; r < N; ++r) {
                for (size_t s = 0; s < N; ++s) {
                    for (size_t t = 0; t < N; ++t) {
                        for (size_t u = 0; u < N; ++u) {
                            for (size_t v = 0; v < N; ++v) {
                                for (size_t w = 0; w < N; ++w) {
                                    double val = 0.0;
                                    // Creator {pq_αβ|rs_ββ}, sign_c = +1
                                    val += Gamma_ab[idx4(p,q,t,u)] * Gamma_bb[idx4(r,s,v,w)];
                                    val -= Gamma_ab[idx4(p,q,t,v)] * Gamma_bb[idx4(r,s,u,w)];
                                    val += Gamma_ab[idx4(p,q,t,w)] * Gamma_bb[idx4(r,s,u,v)];
                                    // Creator {pr_αβ|qs_ββ}, sign_c = −1
                                    val -= Gamma_ab[idx4(p,r,t,u)] * Gamma_bb[idx4(q,s,v,w)];
                                    val += Gamma_ab[idx4(p,r,t,v)] * Gamma_bb[idx4(q,s,u,w)];
                                    val -= Gamma_ab[idx4(p,r,t,w)] * Gamma_bb[idx4(q,s,u,v)];
                                    // Creator {ps_αβ|qr_ββ}, sign_c = +1
                                    val += Gamma_ab[idx4(p,s,t,u)] * Gamma_bb[idx4(q,r,v,w)];
                                    val -= Gamma_ab[idx4(p,s,t,v)] * Gamma_bb[idx4(q,r,u,w)];
                                    val += Gamma_ab[idx4(p,s,t,w)] * Gamma_bb[idx4(q,r,u,v)];

                                    // −2·γ_a(p,t)·det₃(γ_b)(qrs;uvw)
                                    double det3 =
                                          gamma_bb[idx2(q,u)] * gamma_bb[idx2(r,v)] * gamma_bb[idx2(s,w)]
                                        - gamma_bb[idx2(q,u)] * gamma_bb[idx2(r,w)] * gamma_bb[idx2(s,v)]
                                        - gamma_bb[idx2(q,v)] * gamma_bb[idx2(r,u)] * gamma_bb[idx2(s,w)]
                                        + gamma_bb[idx2(q,v)] * gamma_bb[idx2(r,w)] * gamma_bb[idx2(s,u)]
                                        + gamma_bb[idx2(q,w)] * gamma_bb[idx2(r,u)] * gamma_bb[idx2(s,v)]
                                        - gamma_bb[idx2(q,w)] * gamma_bb[idx2(r,v)] * gamma_bb[idx2(s,u)];
                                    val -= 2.0 * gamma_aa[idx2(p,t)] * det3;

                                    Gamma4_abbb[idx8(p,q,r,s,t,u,v,w)] = val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return std::make_tuple(std::move(Gamma4_aaaa), std::move(Gamma4_aaab),
                           std::move(Gamma4_aabb), std::move(Gamma4_abbb),
                          std::move(Gamma4_bbbb));
}


} // namespace trimci_core
