#include "rdm.hpp"
#include "hamiltonian.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include "omp_compat.hpp"
#include "bit_compat.hpp"

namespace trimci_core {

/**
 * Compute orbital gradient for CI wavefunction (FIXED CI coefficients).
 * 
 * This function computes the gradient of the CI energy with respect to orbital
 * rotation parameters, assuming FIXED CI coefficients.
 * 
 * Mathematical Formula:
 *   g[p,q] = 2 * (F[q,p] - F[p,q])
 * 
 * where F is the generalized Fock matrix:
 *   F[p,q] = F^(1)[p,q] + F^(2)[p,q]
 *          = sum_r h[p,r] * Gamma1[r,q] - sum_{rst} (pr|st) * Gamma2_eff[q,r,s,t]
 * 
 * WARNING: This gradient assumes FIXED CI coefficients. When used in an
 * optimization that re-diagonalizes at each step, the response of the CI
 * coefficients to orbital rotation is not captured. For such cases, use
 * finite difference on the objective function.
 * 
 * @param dets      List of determinants
 * @param coeffs    CI coefficients
 * @param h1        One-electron integrals h[p][q]
 * @param eri       Two-electron integrals (chemist notation) flattened: (pq|rs) = eri[p*N^3 + q*N^2 + r*N + s]
 * @param n_orb     Number of spatial orbitals
 * @param n_elec    Number of electrons (currently unused but kept for API consistency)
 * @return          {gradient matrix, Fock diagonal}
 */
template<typename StorageType>
std::tuple<std::vector<std::vector<double>>, std::vector<double>>
compute_orbital_gradient_t(
    const std::vector<DeterminantT<StorageType>>& dets,
    const std::vector<double>& coeffs,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    int n_orb,
    int n_elec,
    const std::vector<int>& attentive_orbitals
) {
    int dim = (int)dets.size();
    if (dim == 0) return {{}, {}};

    int N = n_orb;
    size_t N2 = static_cast<size_t>(N) * N;
    size_t N3 = static_cast<size_t>(N) * N * N;
    size_t N4 = static_cast<size_t>(N) * N * N * N;
    
    // Attentive mode: only compute for specified orbitals
    bool attentive_mode = !attentive_orbitals.empty();
    std::vector<bool> is_attentive(N, false);
    if (attentive_mode) {
        for (int orb : attentive_orbitals) {
            if (orb >= 0 && orb < N) is_attentive[orb] = true;
        }
    }
    
    // Global accumulators (will be reduced from thread-local copies)
    std::vector<double> Gamma1(N2, 0.0);
    std::vector<double> G2_uuuu(N4, 0.0);
    std::vector<double> G2_dddd(N4, 0.0);
    std::vector<double> G2_udud(N4, 0.0);
    
    auto get_eri = [&](int p, int q, int r, int s) {
        size_t idx = static_cast<size_t>(p)*N3 + static_cast<size_t>(q)*N2 + static_cast<size_t>(r)*N + s;
        return eri[idx];
    };
    
    // ========== PHASE 1: RDM Accumulation with Thread-Local Reduction ==========
    // Compute the 1-RDM (Gamma1) and spin-resolved 2-RDMs (G2_uuuu, G2_dddd, G2_udud)
    // by looping over all pairs of determinants and accumulating contributions
    // based on excitation level (0, 1, or 2).
    #pragma omp parallel
    {
        // Thread-local copies
        std::vector<double> Gamma1_local(N2, 0.0);
        std::vector<double> G2_uuuu_local(N4, 0.0);
        std::vector<double> G2_dddd_local(N4, 0.0);
        std::vector<double> G2_udud_local(N4, 0.0);
        
        // Direct update for mixed-spin terms (no antisymmetry enforced)
        auto add_g2_direct = [&](std::vector<double>& vec, int i, int j, int k, int l, double v) {
            size_t idx = static_cast<size_t>(i)*N3 + static_cast<size_t>(j)*N2 + static_cast<size_t>(k)*N + l;
            vec[idx] += v;
        };

        // Antisymmetric update for same-spin terms
        auto add_g2_antisym = [&](std::vector<double>& vec, int i, int j, int k, int l, double v) {
            size_t idx1 = static_cast<size_t>(i)*N3 + static_cast<size_t>(j)*N2 + static_cast<size_t>(k)*N + l;
            size_t idx2 = static_cast<size_t>(j)*N3 + static_cast<size_t>(i)*N2 + static_cast<size_t>(k)*N + l;
            size_t idx3 = static_cast<size_t>(i)*N3 + static_cast<size_t>(j)*N2 + static_cast<size_t>(l)*N + k;
            size_t idx4 = static_cast<size_t>(j)*N3 + static_cast<size_t>(i)*N2 + static_cast<size_t>(l)*N + k;
            
            vec[idx1] += v;
            vec[idx2] -= v;
            vec[idx3] -= v;
            vec[idx4] += v;
        };
        
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < dim; ++i) {
            for (int j = 0; j < dim; ++j) {
                double w = coeffs[i] * coeffs[j];
                if (std::abs(w) < 1e-15) continue;

                const auto& det_i = dets[i];
                const auto& det_j = dets[j];

                std::vector<int> da_rem, da_add, db_rem, db_add;
                
                auto diff_a_in_i = detail::HamiltonianBitOps<StorageType>::and_not(det_i.alpha, det_j.alpha);
                auto diff_a_in_j = detail::HamiltonianBitOps<StorageType>::and_not(det_j.alpha, det_i.alpha);
                int buf[64]; 
                int c = detail::HamiltonianBitOps<StorageType>::storage_to_indices_inline(diff_a_in_i, buf, 64);
                for(int k=0;k<c;++k) da_rem.push_back(buf[k]);
                c = detail::HamiltonianBitOps<StorageType>::storage_to_indices_inline(diff_a_in_j, buf, 64);
                for(int k=0;k<c;++k) da_add.push_back(buf[k]);

                auto diff_b_in_i = detail::HamiltonianBitOps<StorageType>::and_not(det_i.beta, det_j.beta);
                auto diff_b_in_j = detail::HamiltonianBitOps<StorageType>::and_not(det_j.beta, det_i.beta);
                c = detail::HamiltonianBitOps<StorageType>::storage_to_indices_inline(diff_b_in_i, buf, 64);
                for(int k=0;k<c;++k) db_rem.push_back(buf[k]);
                c = detail::HamiltonianBitOps<StorageType>::storage_to_indices_inline(diff_b_in_j, buf, 64);
                for(int k=0;k<c;++k) db_add.push_back(buf[k]);

                int da = (int)da_rem.size();
                int db = (int)db_rem.size();
                int ex_level = da + db;

                if (ex_level > 2) continue;

                // --- CASE 0: Diagonal ---
                if (ex_level == 0) {
                    auto occ_a = det_i.getOccupiedAlpha();
                    auto occ_b = det_i.getOccupiedBeta();
                    
                    // 1-RDM (Diagonal) 
                    for(int k : occ_a) { 
                        Gamma1_local[k*N+k] += w; 
                    }
                    for(int k : occ_b) { 
                        Gamma1_local[k*N+k] += w; 
                    }

                    // 2-RDM Diagonal
                    // AA: Use antisym update for unique pairs (idx1 < idx2)
                    for(size_t idx1=0; idx1<occ_a.size(); ++idx1) {
                        for(size_t idx2=idx1+1; idx2<occ_a.size(); ++idx2) {
                            add_g2_antisym(G2_uuuu_local, occ_a[idx1], occ_a[idx2], occ_a[idx1], occ_a[idx2], w);
                        }
                    }
                    // BB: Use antisym update for unique pairs
                    for(size_t idx1=0; idx1<occ_b.size(); ++idx1) {
                        for(size_t idx2=idx1+1; idx2<occ_b.size(); ++idx2) {
                            add_g2_antisym(G2_dddd_local, occ_b[idx1], occ_b[idx2], occ_b[idx1], occ_b[idx2], w);
                        }
                    }
                    // Mixed: No antisymmetry between spins, iterate all pairs
                    for(int k : occ_a) {
                        for(int l : occ_b) {
                            add_g2_direct(G2_udud_local, k, l, k, l, w);
                        }
                    }
                }
                // --- CASE 1: Single Excitation ---
                else if (ex_level == 1) {
                    bool is_alpha = (da == 1);
                    int m, p_idx;
                    int phase = 0;
                    
                    if (is_alpha) {
                        m = da_rem[0]; p_idx = da_add[0];
                        phase = detail::cre_des_sign_t<StorageType>(p_idx, m, det_i.alpha);
                    } else {
                        m = db_rem[0]; p_idx = db_add[0];
                        phase = detail::cre_des_sign_t<StorageType>(p_idx, m, det_i.beta);
                    }
                    double val = w * phase;
                    
                    Gamma1_local[p_idx*N + m] += val;

                    // Same spin spectators
                    const auto& occ_same = is_alpha ? det_j.getOccupiedAlpha() : det_j.getOccupiedBeta();
                    for(int k : occ_same) {
                        if (k == m || k == p_idx) continue;
                        if (is_alpha) add_g2_antisym(G2_uuuu_local, p_idx, k, m, k, val);
                        else          add_g2_antisym(G2_dddd_local, p_idx, k, m, k, val);
                    }
                    
                    // Opposite spin spectators
                    const auto& occ_opp = is_alpha ? det_j.getOccupiedBeta() : det_j.getOccupiedAlpha();
                    for(int k : occ_opp) {
                        if (is_alpha) {
                            // Alpha exc p<-m, beta spectator k
                            add_g2_direct(G2_udud_local, p_idx, k, m, k, val);
                        } else {
                            // Beta exc p<-m, alpha spectator k
                            add_g2_direct(G2_udud_local, k, p_idx, k, m, val);
                        }
                    }
                }
                // --- CASE 2: Double Excitation ---
                else if (ex_level == 2) {
                    if (da == 2) { // AA
                        int m = da_rem[0]; int n = da_rem[1];
                        int p = da_add[0]; int q = da_add[1];
                        // Ensure canonical ordering for phase consistency
                        if(m > n) std::swap(m, n);
                        if(p > q) std::swap(p, q);
                        
                        auto temp = det_i.alpha;
                        int ph1 = detail::cre_des_sign_t<StorageType>(p, m, temp);
                        BitOps<StorageType>::set_bit(temp, p); BitOps<StorageType>::clear_bit(temp, m);
                        int ph2 = detail::cre_des_sign_t<StorageType>(q, n, temp);
                        
                        double val = w * ph1 * ph2;
                        add_g2_antisym(G2_uuuu_local, p, q, m, n, val);
                        
                    } else if (db == 2) { // BB
                        int m = db_rem[0]; int n = db_rem[1];
                        int p = db_add[0]; int q = db_add[1];
                        if(m > n) std::swap(m, n);
                        if(p > q) std::swap(p, q);
                        
                        auto temp = det_i.beta;
                        int ph1 = detail::cre_des_sign_t<StorageType>(p, m, temp);
                        BitOps<StorageType>::set_bit(temp, p); BitOps<StorageType>::clear_bit(temp, m);
                        int ph2 = detail::cre_des_sign_t<StorageType>(q, n, temp);
                        
                        double val = w * ph1 * ph2;
                        add_g2_antisym(G2_dddd_local, p, q, m, n, val);
                        
                    } else { // Mixed
                        int m = da_rem[0]; int p_idx = da_add[0];
                        int n = db_rem[0]; int q_idx = db_add[0];
                        
                        int ph_a = detail::cre_des_sign_t<StorageType>(p_idx, m, det_i.alpha);
                        int ph_b = detail::cre_des_sign_t<StorageType>(q_idx, n, det_i.beta);
                        double val = w * ph_a * ph_b;
                        
                        add_g2_direct(G2_udud_local, p_idx, q_idx, m, n, val);
                    }
                }
            }
        }
        
        // Reduce thread-local to global
        #pragma omp critical
        {
            for(int k = 0; k < N2; ++k) Gamma1[k] += Gamma1_local[k];
            for(int k = 0; k < N4; ++k) {
                G2_uuuu[k] += G2_uuuu_local[k];
                G2_dddd[k] += G2_dddd_local[k];
                G2_udud[k] += G2_udud_local[k];
            }
        }
    } // end parallel region


    // ========== PHASE 2: Build Gamma2_eff (Parallelized) ==========
    // Construct the effective 2-RDM by applying index permutations to the spin-resolved 2-RDMs.
    // 
    // The permutations used are:
    //   P_{0,2,1,3}: [i0,i1,i2,i3] -> [i0,i2,i1,i3]  (for same/mixed spin contributions)
    //   P_{1,3,0,2}: [i0,i1,i2,i3] -> [i1,i3,i0,i2]  (for cross-spin exchange term)
    //
    // Formula: Gamma2_eff = P_{0,2,1,3}[G_uu + G_dd + G_ud] + P_{1,3,0,2}[G_ud]
    //
    // ATTENTIVE MODE: Only compute Gamma2_eff[q,*,*,*] for q in attentive_orbitals.
    // This reduces O(N^4) to O(k*N^3).
    std::vector<double> Gamma2_eff(N4, 0.0);
    
    if (!attentive_mode) {
        // Full computation
        #pragma omp parallel for collapse(2) schedule(static)
        for(int i0=0; i0<N; ++i0) {
            for(int i1=0; i1<N; ++i1) {
                for(int i2=0; i2<N; ++i2) {
                    for(int i3=0; i3<N; ++i3) {
                        size_t old_idx = static_cast<size_t>(i0)*N3 + static_cast<size_t>(i1)*N2 + static_cast<size_t>(i2)*N + i3;
                        
                        double g_same = G2_uuuu[old_idx] + G2_dddd[old_idx] + G2_udud[old_idx];
                        double g_cross = G2_udud[old_idx];
                        
                        // P_same = (0,2,1,3): [p,q,r,s] -> [p,r,q,s]
                        size_t new_idx_same = static_cast<size_t>(i0)*N3 + static_cast<size_t>(i2)*N2 + static_cast<size_t>(i1)*N + i3;
                        // P_cross = (1,3,0,2): [p,q,r,s] -> [q,s,p,r]
                        size_t new_idx_cross = static_cast<size_t>(i1)*N3 + static_cast<size_t>(i3)*N2 + static_cast<size_t>(i0)*N + i2;
                        
                        // Use atomic because different (i0, i1) can yield the same (p, r, q, s) index
                        #pragma omp atomic
                        Gamma2_eff[new_idx_same] += g_same;
                        #pragma omp atomic
                        Gamma2_eff[new_idx_cross] += g_cross;
                    }
                }
            }
        }
    } else {
        // ATTENTIVE MODE: Only compute slices needed for F[p,q] where p,q in attentive
        // F^(2)[p,q] uses Gamma2_eff[q,r,s,t], so we only need q in attentive
        // 
        // For each source index (i0,i1,i2,i3):
        //   P_same: Gamma2_eff[i0,i2,i1,i3] -> only if i0 in attentive
        //   P_cross: Gamma2_eff[i1,i3,i0,i2] -> only if i1 in attentive
        
        #pragma omp parallel for collapse(2) schedule(static)
        for(int i0=0; i0<N; ++i0) {
            for(int i1=0; i1<N; ++i1) {
                bool i0_att = is_attentive[i0];
                bool i1_att = is_attentive[i1];
                
                // Skip if neither contributes to attentive slices
                if (!i0_att && !i1_att) continue;
                
                for(int i2=0; i2<N; ++i2) {
                    for(int i3=0; i3<N; ++i3) {
                        size_t old_idx = static_cast<size_t>(i0)*N3 + static_cast<size_t>(i1)*N2 + static_cast<size_t>(i2)*N + i3;
                        
                        // P_same contribution: Gamma2_eff[i0,i2,i1,i3]
                        if (i0_att) {
                            double g_same = G2_uuuu[old_idx] + G2_dddd[old_idx] + G2_udud[old_idx];
                            size_t new_idx_same = static_cast<size_t>(i0)*N3 + static_cast<size_t>(i2)*N2 + static_cast<size_t>(i1)*N + i3;
                            #pragma omp atomic
                            Gamma2_eff[new_idx_same] += g_same;
                        }
                        
                        // P_cross contribution: Gamma2_eff[i1,i3,i0,i2]
                        if (i1_att) {
                            double g_cross = G2_udud[old_idx];
                            size_t new_idx_cross = static_cast<size_t>(i1)*N3 + static_cast<size_t>(i3)*N2 + static_cast<size_t>(i0)*N + i2;
                            #pragma omp atomic
                            Gamma2_eff[new_idx_cross] += g_cross;
                        }
                    }
                }
            }
        }
    }

    // ========== PHASE 3: Compute Fock Matrix (Parallelized) ==========
    // Generalized Fock matrix: F[p,q] = F^(1)[p,q] + F^(2)[p,q]
    //
    // 1-body: F^(1)[p,q] = sum_r h[p,r] * Gamma1[r,q]
    // 2-body: F^(2)[p,q] = -sum_{rst} (pr|st) * Gamma2_eff[q,r,s,t]
    //
    // NOTE: The -1.0 factor in the 2-body part is required for the correct
    // gradient (this comes from the specific 2-RDM ordering convention).
    //
    // ATTENTIVE MODE: Only compute F[p,q] where p,q are in attentive_orbitals.
    // This reduces the O(N^5) bottleneck to O(k^2 * N^3).
    std::vector<double> F(N2, 0.0);
    
    if (!attentive_mode) {
        // Full computation: all (p,q) pairs
        
        // 1-body part: F^(1) = h @ Gamma1  (parallelized, O(N^3))
        #pragma omp parallel for collapse(2) schedule(static)
        for(int p=0; p<N; ++p) {
            for(int q=0; q<N; ++q) {
                double val = 0.0;
                for(int r=0; r<N; ++r) val += h1[p][r] * Gamma1[r*N + q];
                F[p*N + q] = val;
            }
        }
        
        // 2-body part: F^(2)[p,q] = -sum_{rst} (pr|st) * Gamma2_eff[q,r,s,t]
        // This is O(N^5) and is the major computational bottleneck.
        #pragma omp parallel for collapse(2) schedule(static)
        for(int p=0; p<N; ++p) {
            for(int q=0; q<N; ++q) {
                double val = 0.0;
                for(int r=0; r<N; ++r) {
                    for(int s=0; s<N; ++s) {
                        for(int t=0; t<N; ++t) {
                            size_t gamma_idx = static_cast<size_t>(q)*N3 + static_cast<size_t>(r)*N2 + static_cast<size_t>(s)*N + t;
                            val += get_eri(p, r, s, t) * Gamma2_eff[gamma_idx];
                        }
                    }
                }
                F[static_cast<size_t>(p)*N + q] += -1.0 * val;
            }
        }
    } else {
        // ATTENTIVE MODE: Only compute for attentive orbital pairs
        // This is O(k^2 * N^3) instead of O(N^5)
        
        // 1-body part: only for attentive orbitals
        #pragma omp parallel for schedule(dynamic)
        for(int i=0; i<static_cast<int>(attentive_orbitals.size()); ++i) {
            int p = attentive_orbitals[i];
            for(size_t j=0; j<attentive_orbitals.size(); ++j) {
                int q = attentive_orbitals[j];
                double val = 0.0;
                for(int r=0; r<N; ++r) val += h1[p][r] * Gamma1[r*N + q];
                F[p*N + q] = val;
            }
        }
        
        // 2-body part: only for attentive orbitals
        #pragma omp parallel for schedule(dynamic)
        for(int i=0; i<static_cast<int>(attentive_orbitals.size()); ++i) {
            int p = attentive_orbitals[i];
            for(size_t j=0; j<attentive_orbitals.size(); ++j) {
                int q = attentive_orbitals[j];
                double val = 0.0;
                for(int r=0; r<N; ++r) {
                    for(int s=0; s<N; ++s) {
                        for(int t=0; t<N; ++t) {
                            size_t gamma_idx = static_cast<size_t>(q)*N3 + static_cast<size_t>(r)*N2 + static_cast<size_t>(s)*N + t;
                            val += get_eri(p, r, s, t) * Gamma2_eff[gamma_idx];
                        }
                    }
                }
                F[static_cast<size_t>(p)*N + q] += -1.0 * val;
            }
        }
    }
    
    // ========== PHASE 4: Compute Gradient ==========
    // Gradient formula: g[p,q] = 2 * (F[q,p] - F[p,q])
    // This is ANTI-SYMMETRIC: g[p,q] = -g[q,p]
    // At a stationary point (e.g., HF orbitals), the occupied-virtual block is zero.
    //
    // In attentive mode, only compute for attentive pairs (other entries stay zero).
    std::vector<std::vector<double>> grad(N, std::vector<double>(N, 0.0));
    
    if (!attentive_mode) {
        // Full gradient
        for(int p=0; p<N; ++p) {
            for(int q=0; q<N; ++q) {
                grad[p][q] = 2.0 * (F[q*N + p] - F[p*N + q]);
            }
        }
    } else {
        // Attentive gradient: only for pairs within attentive set
        for(size_t i=0; i<attentive_orbitals.size(); ++i) {
            int p = attentive_orbitals[i];
            for(size_t j=0; j<attentive_orbitals.size(); ++j) {
                int q = attentive_orbitals[j];
                grad[p][q] = 2.0 * (F[q*N + p] - F[p*N + q]);
            }
        }
    }
    
    std::vector<double> f_diag(N);
    for(int i=0; i<N; ++i) f_diag[i] = F[i*N + i];

    return {grad, f_diag};
}

std::tuple<std::vector<std::vector<double>>, std::vector<double>>
compute_orbital_gradient(
    const std::vector<Determinant>& dets,
    const std::vector<double>& coeffs,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    int n_orb,
    int n_elec,
    const std::vector<int>& attentive_orbitals
) {
    return compute_orbital_gradient_t<uint64_t>(dets, coeffs, h1, eri, n_orb, n_elec, attentive_orbitals);
}

} // namespace trimci_core

#include <array>
#define INSTANTIATE_GRAD(...) \
    template std::tuple<std::vector<std::vector<double>>, std::vector<double>> \
    trimci_core::compute_orbital_gradient_t<__VA_ARGS__>( \
        const std::vector<trimci_core::DeterminantT<__VA_ARGS__>>& dets, \
        const std::vector<double>& coeffs, \
        const std::vector<std::vector<double>>& h1, \
        const std::vector<double>& eri, \
        int n_orb, \
        int n_elec, \
        const std::vector<int>& attentive_orbitals \
    );

INSTANTIATE_GRAD(uint64_t)
INSTANTIATE_GRAD(std::array<uint64_t, 2>)
INSTANTIATE_GRAD(std::array<uint64_t, 3>)
INSTANTIATE_GRAD(std::array<uint64_t, 4>)
INSTANTIATE_GRAD(std::array<uint64_t, 5>)
INSTANTIATE_GRAD(std::array<uint64_t, 6>)
INSTANTIATE_GRAD(std::array<uint64_t, 7>)
INSTANTIATE_GRAD(std::array<uint64_t, 8>)

