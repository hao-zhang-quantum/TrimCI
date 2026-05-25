/**
 * RDM computation: 6-body (exact)
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
// Spin-Resolved 6-RDM (COMPLETE - all 7 spin components)
// ============================================================================
// Returns 7 tensors: Γ⁶_aaaaaa, Γ⁶_aaaaab, Γ⁶_aaaabb, Γ⁶_aaabbb, 
//                    Γ⁶_aabbbb, Γ⁶_abbbbb, Γ⁶_bbbbbb
//
// Index convention (creation-first):
//   Γ⁶[p,q,r,s,t,u,v,w,x,y,z,w'] = ⟨Ψ| a†_p...a†_u a_w'...a_v |Ψ⟩
//
// NOTE: Due to massive storage (N^12), limited to n_orb <= 4 (~200MB)
// ============================================================================

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>,
           std::vector<double>, std::vector<double>, std::vector<double>,
           std::vector<double>>
compute_6rdm_spin_resolved(
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
    size_t N9 = N8 * N;
    size_t N10 = N9 * N;
    size_t N11 = N10 * N;
    size_t N12 = N11 * N;
    
    // Warning: N12 is huge! Limit to n_orb <= 4
    if (n_orb > 4) {
        size_t storage_mb = (N12 * 8 * 7) / (1024 * 1024);  // 7 spin types
        throw std::runtime_error("compute_6rdm_spin_resolved: n_orb > 4 requires ~" + 
                                 std::to_string(storage_mb) + "MB storage, aborting");
    }
    
    std::vector<double> Gamma6_aaaaaa(N12, 0.0);
    std::vector<double> Gamma6_aaaaab(N12, 0.0);
    std::vector<double> Gamma6_aaaabb(N12, 0.0);
    std::vector<double> Gamma6_aaabbb(N12, 0.0);
    std::vector<double> Gamma6_aabbbb(N12, 0.0);
    std::vector<double> Gamma6_abbbbb(N12, 0.0);
    std::vector<double> Gamma6_bbbbbb(N12, 0.0);
    
    int n_det = static_cast<int>(dets.size());
    
    // Helper for 12-index calculation
    auto idx12 = [N, N2, N3, N4, N5, N6, N7, N8, N9, N10, N11](
                    int p, int q, int r, int s, int t, int u,
                    int v, int w, int x, int y, int z, int wp) {
        return p*N11 + q*N10 + r*N9 + s*N8 + t*N7 + u*N6 + 
               v*N5 + w*N4 + x*N3 + y*N2 + z*N + wp;
    };
    
    #pragma omp parallel
    {
        std::vector<double> local_aaaaaa(N12, 0.0);
        std::vector<double> local_aaaaab(N12, 0.0);
        std::vector<double> local_aaaabb(N12, 0.0);
        std::vector<double> local_aaabbb(N12, 0.0);
        std::vector<double> local_aabbbb(N12, 0.0);
        std::vector<double> local_abbbbb(N12, 0.0);
        std::vector<double> local_bbbbbb(N12, 0.0);
        
        #pragma omp for schedule(dynamic)
        for (int I = 0; I < n_det; ++I) {
            for (int J = 0; J < n_det; ++J) {
                double c_IJ = coeffs[I] * coeffs[J];
                if (std::abs(c_IJ) < 1e-15) continue;
                
                const auto& det_I = dets[I];
                const auto& det_J = dets[J];
                
                // Check excitation level
                uint64_t diff_a_in_I = det_I.alpha & ~det_J.alpha;
                uint64_t diff_b_in_I = det_I.beta & ~det_J.beta;
                int n_alpha_exc = __builtin_popcountll(diff_a_in_I);
                int n_beta_exc = __builtin_popcountll(diff_b_in_I);
                int n_total_exc = n_alpha_exc + n_beta_exc;
                if (n_total_exc > 6) continue;
                
                auto occ_a_I = get_set_bits(det_I.alpha);
                auto occ_b_I = get_set_bits(det_I.beta);
                
                // Diagonal case only (det_I == det_J)
                if (det_I.alpha == det_J.alpha && det_I.beta == det_J.beta) {
                    // aaaaaa: 6 alpha
                    if (occ_a_I.size() >= 6) {
                        for (size_t i0 = 0; i0 < occ_a_I.size(); ++i0) {
                            for (size_t i1 = i0+1; i1 < occ_a_I.size(); ++i1) {
                                for (size_t i2 = i1+1; i2 < occ_a_I.size(); ++i2) {
                                    for (size_t i3 = i2+1; i3 < occ_a_I.size(); ++i3) {
                                        for (size_t i4 = i3+1; i4 < occ_a_I.size(); ++i4) {
                                            for (size_t i5 = i4+1; i5 < occ_a_I.size(); ++i5) {
                                                int p = occ_a_I[i0], q = occ_a_I[i1], r = occ_a_I[i2];
                                                int s = occ_a_I[i3], t = occ_a_I[i4], u = occ_a_I[i5];
                                                local_aaaaaa[idx12(p,q,r,s,t,u,p,q,r,s,t,u)] += c_IJ;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    
                    // bbbbbb: 6 beta
                    if (occ_b_I.size() >= 6) {
                        for (size_t i0 = 0; i0 < occ_b_I.size(); ++i0) {
                            for (size_t i1 = i0+1; i1 < occ_b_I.size(); ++i1) {
                                for (size_t i2 = i1+1; i2 < occ_b_I.size(); ++i2) {
                                    for (size_t i3 = i2+1; i3 < occ_b_I.size(); ++i3) {
                                        for (size_t i4 = i3+1; i4 < occ_b_I.size(); ++i4) {
                                            for (size_t i5 = i4+1; i5 < occ_b_I.size(); ++i5) {
                                                int p = occ_b_I[i0], q = occ_b_I[i1], r = occ_b_I[i2];
                                                int s = occ_b_I[i3], t = occ_b_I[i4], u = occ_b_I[i5];
                                                local_bbbbbb[idx12(p,q,r,s,t,u,p,q,r,s,t,u)] += c_IJ;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    
                    // aaaaab: 5 alpha + 1 beta
                    if (occ_a_I.size() >= 5 && occ_b_I.size() >= 1) {
                        for (size_t i0 = 0; i0 < occ_a_I.size(); ++i0) {
                            for (size_t i1 = i0+1; i1 < occ_a_I.size(); ++i1) {
                                for (size_t i2 = i1+1; i2 < occ_a_I.size(); ++i2) {
                                    for (size_t i3 = i2+1; i3 < occ_a_I.size(); ++i3) {
                                        for (size_t i4 = i3+1; i4 < occ_a_I.size(); ++i4) {
                                            int p = occ_a_I[i0], q = occ_a_I[i1], r = occ_a_I[i2];
                                            int s = occ_a_I[i3], t = occ_a_I[i4];
                                            for (int u : occ_b_I) {
                                                local_aaaaab[idx12(p,q,r,s,t,u,p,q,r,s,t,u)] += c_IJ;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    
                    // abbbbb: 1 alpha + 5 beta
                    if (occ_a_I.size() >= 1 && occ_b_I.size() >= 5) {
                        for (int p : occ_a_I) {
                            for (size_t i1 = 0; i1 < occ_b_I.size(); ++i1) {
                                for (size_t i2 = i1+1; i2 < occ_b_I.size(); ++i2) {
                                    for (size_t i3 = i2+1; i3 < occ_b_I.size(); ++i3) {
                                        for (size_t i4 = i3+1; i4 < occ_b_I.size(); ++i4) {
                                            for (size_t i5 = i4+1; i5 < occ_b_I.size(); ++i5) {
                                                int q = occ_b_I[i1], r = occ_b_I[i2], s = occ_b_I[i3];
                                                int t = occ_b_I[i4], u = occ_b_I[i5];
                                                local_abbbbb[idx12(p,q,r,s,t,u,p,q,r,s,t,u)] += c_IJ;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    
                    // aaaabb: 4 alpha + 2 beta
                    if (occ_a_I.size() >= 4 && occ_b_I.size() >= 2) {
                        for (size_t i0 = 0; i0 < occ_a_I.size(); ++i0) {
                            for (size_t i1 = i0+1; i1 < occ_a_I.size(); ++i1) {
                                for (size_t i2 = i1+1; i2 < occ_a_I.size(); ++i2) {
                                    for (size_t i3 = i2+1; i3 < occ_a_I.size(); ++i3) {
                                        int p = occ_a_I[i0], q = occ_a_I[i1];
                                        int r = occ_a_I[i2], s = occ_a_I[i3];
                                        for (size_t i4 = 0; i4 < occ_b_I.size(); ++i4) {
                                            for (size_t i5 = i4+1; i5 < occ_b_I.size(); ++i5) {
                                                int t = occ_b_I[i4], u = occ_b_I[i5];
                                                local_aaaabb[idx12(p,q,r,s,t,u,p,q,r,s,t,u)] += c_IJ;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    
                    // aabbbb: 2 alpha + 4 beta
                    if (occ_a_I.size() >= 2 && occ_b_I.size() >= 4) {
                        for (size_t i0 = 0; i0 < occ_a_I.size(); ++i0) {
                            for (size_t i1 = i0+1; i1 < occ_a_I.size(); ++i1) {
                                int p = occ_a_I[i0], q = occ_a_I[i1];
                                for (size_t i2 = 0; i2 < occ_b_I.size(); ++i2) {
                                    for (size_t i3 = i2+1; i3 < occ_b_I.size(); ++i3) {
                                        for (size_t i4 = i3+1; i4 < occ_b_I.size(); ++i4) {
                                            for (size_t i5 = i4+1; i5 < occ_b_I.size(); ++i5) {
                                                int r = occ_b_I[i2], s = occ_b_I[i3];
                                                int t = occ_b_I[i4], u = occ_b_I[i5];
                                                local_aabbbb[idx12(p,q,r,s,t,u,p,q,r,s,t,u)] += c_IJ;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    
                    // aaabbb: 3 alpha + 3 beta
                    if (occ_a_I.size() >= 3 && occ_b_I.size() >= 3) {
                        for (size_t i0 = 0; i0 < occ_a_I.size(); ++i0) {
                            for (size_t i1 = i0+1; i1 < occ_a_I.size(); ++i1) {
                                for (size_t i2 = i1+1; i2 < occ_a_I.size(); ++i2) {
                                    int p = occ_a_I[i0], q = occ_a_I[i1], r = occ_a_I[i2];
                                    for (size_t i3 = 0; i3 < occ_b_I.size(); ++i3) {
                                        for (size_t i4 = i3+1; i4 < occ_b_I.size(); ++i4) {
                                            for (size_t i5 = i4+1; i5 < occ_b_I.size(); ++i5) {
                                                int s = occ_b_I[i3], t = occ_b_I[i4], u = occ_b_I[i5];
                                                local_aaabbb[idx12(p,q,r,s,t,u,p,q,r,s,t,u)] += c_IJ;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Reduce to global
        #pragma omp critical
        {
            for (size_t i = 0; i < N12; ++i) {
                Gamma6_aaaaaa[i] += local_aaaaaa[i];
                Gamma6_aaaaab[i] += local_aaaaab[i];
                Gamma6_aaaabb[i] += local_aaaabb[i];
                Gamma6_aaabbb[i] += local_aaabbb[i];
                Gamma6_aabbbb[i] += local_aabbbb[i];
                Gamma6_abbbbb[i] += local_abbbbb[i];
                Gamma6_bbbbbb[i] += local_bbbbbb[i];
            }
        }
    }
    
    return std::make_tuple(std::move(Gamma6_aaaaaa), std::move(Gamma6_aaaaab),
                           std::move(Gamma6_aaaabb), std::move(Gamma6_aaabbb),
                           std::move(Gamma6_aabbbb), std::move(Gamma6_abbbbb),
                           std::move(Gamma6_bbbbbb));
}


} // namespace trimci_core
