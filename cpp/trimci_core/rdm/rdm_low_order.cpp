/**
 * RDM computation: 1-body and 2-body (exact)
 *
 * Split from rdm_full.cpp for maintainability.
 */

#include "rdm_full.hpp"
#include "rdm_helpers.hpp"
#include <cmath>
#include <omp.h>

using namespace trimci_core::rdm_detail;

namespace trimci_core {

// ============================================================================
// 1-RDM Implementation
// ============================================================================

std::vector<double> compute_1rdm(
    const std::vector<Determinant>& dets,
    const std::vector<double>& coeffs,
    int n_orb
) {
    auto [gamma_aa, gamma_bb] = compute_1rdm_spin_resolved(dets, coeffs, n_orb);

    size_t N2 = static_cast<size_t>(n_orb) * static_cast<size_t>(n_orb);
    for (size_t k = 0; k < N2; ++k) {
        gamma_aa[k] += gamma_bb[k];
    }
    return gamma_aa;  // now contains gamma_aa + gamma_bb
}

// ============================================================================
// 2-RDM Implementation
// ============================================================================

std::vector<double> compute_2rdm(
    const std::vector<Determinant>& dets,
    const std::vector<double>& coeffs,
    int n_orb
) {
    auto [Gamma_aa, Gamma_ab, Gamma_bb] = compute_2rdm_spin_resolved(dets, coeffs, n_orb);

    size_t N = static_cast<size_t>(n_orb);
    size_t N2 = N * N;
    size_t N3 = N2 * N;
    size_t N4 = N3 * N;

    // Gamma2 = Gamma_aa + Gamma_bb + Gamma_ab + Gamma_ba
    // where Gamma_ba[p,q,r,s] = Gamma_ab[q,p,s,r]
    // Reuse Gamma_aa as accumulator
    for (size_t k = 0; k < N4; ++k) {
        Gamma_aa[k] += Gamma_bb[k];
    }

    // Add Gamma_ab and Gamma_ba
    for (size_t p = 0; p < N; ++p) {
        for (size_t q = 0; q < N; ++q) {
            for (size_t r = 0; r < N; ++r) {
                for (size_t s = 0; s < N; ++s) {
                    size_t idx = p*N3 + q*N2 + r*N + s;
                    // Gamma_ab[p,q,r,s]
                    Gamma_aa[idx] += Gamma_ab[idx];
                    // Gamma_ba[p,q,r,s] = Gamma_ab[q,p,s,r]
                    Gamma_aa[idx] += Gamma_ab[q*N3 + p*N2 + s*N + r];
                }
            }
        }
    }

    return Gamma_aa;  // now contains full spin-free 2-RDM
}

// ============================================================================
// Energy from RDMs
// ============================================================================

double energy_from_rdm(
    const std::vector<double>& gamma,
    const std::vector<double>& Gamma2,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    double E_nuc,
    int n_orb
) {
    size_t N = static_cast<size_t>(n_orb);
    size_t N2 = N * N;
    size_t N3 = N * N * N;
    
    double E_1body = 0.0;
    double E_2body = 0.0;
    
    // 1-body: tr(h1 @ γ)
    #pragma omp parallel for reduction(+:E_1body) collapse(2)
    for (int p = 0; p < n_orb; ++p) {
        for (int q = 0; q < n_orb; ++q) {
            E_1body += h1[p][q] * gamma[p*N + q];
        }
    }
    
    // 2-body: 0.5 * tr(eri @ Γ²)
    #pragma omp parallel for reduction(+:E_2body) collapse(4)
    for (int p = 0; p < n_orb; ++p) {
        for (int q = 0; q < n_orb; ++q) {
            for (int r = 0; r < n_orb; ++r) {
                for (int s = 0; s < n_orb; ++s) {
                    size_t idx = p*N3 + q*N2 + r*N + s;
                    E_2body += eri[idx] * Gamma2[idx];
                }
            }
        }
    }
    
    return E_1body + 0.5 * E_2body + E_nuc;
}

// ============================================================================
// Spin-resolved 1-RDM
// ============================================================================

std::pair<std::vector<double>, std::vector<double>> compute_1rdm_spin_resolved(
    const std::vector<Determinant>& dets,
    const std::vector<double>& coeffs,
    int n_orb
) {
    size_t N = static_cast<size_t>(n_orb);
    size_t N2 = N * N;
    std::vector<double> gamma_aa(N2, 0.0);
    std::vector<double> gamma_bb(N2, 0.0);
    
    int n_det = static_cast<int>(dets.size());
    
    #pragma omp parallel
    {
        std::vector<double> gamma_aa_local(N2, 0.0);
        std::vector<double> gamma_bb_local(N2, 0.0);
        
        #pragma omp for schedule(dynamic)
        for (int I = 0; I < n_det; ++I) {
            for (int J = 0; J < n_det; ++J) {
                double c_IJ = coeffs[I] * coeffs[J];
                if (std::abs(c_IJ) < 1e-15) continue;
                
                const auto& det_I = dets[I];
                const auto& det_J = dets[J];
                
                uint64_t diff_a_in_I = det_I.alpha & ~det_J.alpha;
                uint64_t diff_a_in_J = det_J.alpha & ~det_I.alpha;
                uint64_t diff_b_in_I = det_I.beta & ~det_J.beta;
                uint64_t diff_b_in_J = det_J.beta & ~det_I.beta;
                
                int n_alpha_diff = __builtin_popcountll(diff_a_in_I);
                int n_beta_diff = __builtin_popcountll(diff_b_in_I);
                
                if (n_alpha_diff + n_beta_diff > 1) continue;
                
                // Diagonal
                if (n_alpha_diff == 0 && n_beta_diff == 0) {
                    auto occ_a = get_set_bits(det_I.alpha);
                    auto occ_b = get_set_bits(det_I.beta);
                    for (int p : occ_a) gamma_aa_local[p*N + p] += c_IJ;
                    for (int p : occ_b) gamma_bb_local[p*N + p] += c_IJ;
                }
                // Single alpha
                else if (n_alpha_diff == 1 && n_beta_diff == 0) {
                    int p = __builtin_ctzll(diff_a_in_I);
                    int q = __builtin_ctzll(diff_a_in_J);
                    int phase = single_phase_alpha(det_J, q, p);
                    gamma_aa_local[p*N + q] += c_IJ * phase;
                }
                // Single beta
                else if (n_alpha_diff == 0 && n_beta_diff == 1) {
                    int p = __builtin_ctzll(diff_b_in_I);
                    int q = __builtin_ctzll(diff_b_in_J);
                    int phase = single_phase_beta(det_J, q, p);
                    gamma_bb_local[p*N + q] += c_IJ * phase;
                }
            }
        }
        
        #pragma omp critical
        {
            for (size_t i = 0; i < N2; ++i) {
                gamma_aa[i] += gamma_aa_local[i];
                gamma_bb[i] += gamma_bb_local[i];
            }
        }
    }
    
    return {gamma_aa, gamma_bb};
}

// ============================================================================
// Spin-resolved 2-RDM
// ============================================================================

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>> 
compute_2rdm_spin_resolved(
    const std::vector<Determinant>& dets,
    const std::vector<double>& coeffs,
    int n_orb
) {
    size_t N = static_cast<size_t>(n_orb);
    size_t N2 = N * N;
    size_t N3 = N * N * N;
    size_t N4 = N * N * N * N;
    
    std::vector<double> Gamma_aa(N4, 0.0);  // α-α pairs
    std::vector<double> Gamma_ab(N4, 0.0);  // α-β pairs
    std::vector<double> Gamma_bb(N4, 0.0);  // β-β pairs
    
    int n_det = static_cast<int>(dets.size());
    
    // Antisymmetric addition for same-spin pairs
    auto add_antisym = [&](std::vector<double>& vec, int p, int q, int r, int s, double v) {
        vec[p*N3 + q*N2 + r*N + s] += v;
        vec[q*N3 + p*N2 + s*N + r] += v;
        vec[p*N3 + q*N2 + s*N + r] -= v;
        vec[q*N3 + p*N2 + r*N + s] -= v;
    };
    
    #pragma omp parallel
    {
        std::vector<double> Gamma_aa_local(N4, 0.0);
        std::vector<double> Gamma_ab_local(N4, 0.0);
        std::vector<double> Gamma_bb_local(N4, 0.0);
        
        auto add_antisym_local = [&](std::vector<double>& vec, int p, int q, int r, int s, double v) {
            vec[p*N3 + q*N2 + r*N + s] += v;
            vec[q*N3 + p*N2 + s*N + r] += v;
            vec[p*N3 + q*N2 + s*N + r] -= v;
            vec[q*N3 + p*N2 + r*N + s] -= v;
        };
        
        #pragma omp for schedule(dynamic)
        for (int I = 0; I < n_det; ++I) {
            for (int J = 0; J < n_det; ++J) {
                double c_IJ = coeffs[I] * coeffs[J];
                if (std::abs(c_IJ) < 1e-15) continue;
                
                const auto& det_I = dets[I];
                const auto& det_J = dets[J];
                
                uint64_t diff_a_in_I = det_I.alpha & ~det_J.alpha;
                uint64_t diff_a_in_J = det_J.alpha & ~det_I.alpha;
                uint64_t diff_b_in_I = det_I.beta & ~det_J.beta;
                uint64_t diff_b_in_J = det_J.beta & ~det_I.beta;
                
                int n_alpha_diff = __builtin_popcountll(diff_a_in_I);
                int n_beta_diff = __builtin_popcountll(diff_b_in_I);
                int n_total = n_alpha_diff + n_beta_diff;
                
                if (n_total > 2) continue;
                
                // Diagonal contribution
                if (n_total == 0) {
                    auto occ_a = get_set_bits(det_I.alpha);
                    auto occ_b = get_set_bits(det_I.beta);
                    
                    // α-α pairs
                    for (size_t i = 0; i < occ_a.size(); ++i) {
                        for (size_t j = i+1; j < occ_a.size(); ++j) {
                            add_antisym_local(Gamma_aa_local, occ_a[i], occ_a[j], occ_a[i], occ_a[j], c_IJ);
                        }
                    }
                    // β-β pairs
                    for (size_t i = 0; i < occ_b.size(); ++i) {
                        for (size_t j = i+1; j < occ_b.size(); ++j) {
                            add_antisym_local(Gamma_bb_local, occ_b[i], occ_b[j], occ_b[i], occ_b[j], c_IJ);
                        }
                    }
                    // α-β pairs (no antisymmetry between different spins)
                    for (int p : occ_a) {
                        for (int q : occ_b) {
                            // Γ^αβ[p,q,p,q] = 1 for occupied pair
                            Gamma_ab_local[p*N3 + q*N2 + p*N + q] += c_IJ;
                        }
                    }
                }
                // Single alpha excitation
                else if (n_total == 1 && n_alpha_diff == 1) {
                    int p = __builtin_ctzll(diff_a_in_I);
                    int q = __builtin_ctzll(diff_a_in_J);
                    int phase = single_phase_alpha(det_J, q, p);
                    double val = c_IJ * phase;
                    
                    // α spectators (α-α contribution)
                    auto occ_a = get_set_bits(det_J.alpha);
                    for (int k : occ_a) {
                        if (k == q) continue;
                        add_antisym_local(Gamma_aa_local, p, k, q, k, val);
                    }
                    // β spectators (α-β contribution)
                    auto occ_b = get_set_bits(det_J.beta);
                    for (int k : occ_b) {
                        // Γ^αβ[p,k,q,k]: α excitation p←q with β spectator k
                        Gamma_ab_local[p*N3 + k*N2 + q*N + k] += val;
                    }
                }
                // Single beta excitation
                else if (n_total == 1 && n_beta_diff == 1) {
                    int p = __builtin_ctzll(diff_b_in_I);
                    int q = __builtin_ctzll(diff_b_in_J);
                    int phase = single_phase_beta(det_J, q, p);
                    double val = c_IJ * phase;
                    
                    // β spectators (β-β contribution)
                    auto occ_b = get_set_bits(det_J.beta);
                    for (int k : occ_b) {
                        if (k == q) continue;
                        add_antisym_local(Gamma_bb_local, p, k, q, k, val);
                    }
                    // α spectators (α-β contribution)
                    auto occ_a = get_set_bits(det_J.alpha);
                    for (int k : occ_a) {
                        // Γ^αβ[k,p,k,q]: α spectator k with β excitation p←q
                        Gamma_ab_local[k*N3 + p*N2 + k*N + q] += val;
                    }
                }
                // Double α-α excitation
                else if (n_total == 2 && n_alpha_diff == 2) {
                    std::vector<int> holes, parts;
                    uint64_t temp = diff_a_in_J;
                    while (temp) {
                        holes.push_back(__builtin_ctzll(temp));
                        temp &= temp - 1;
                    }
                    temp = diff_a_in_I;
                    while (temp) {
                        parts.push_back(__builtin_ctzll(temp));
                        temp &= temp - 1;
                    }
                    
                    int i = holes[0], j = holes[1];
                    int a = parts[0], b = parts[1];
                    
                    // Phase calculation
                    int lo1 = std::min(i, a), hi1 = std::max(i, a);
                    int cnt1 = count_bits_between(det_J.alpha, lo1, hi1);
                    uint64_t mod_alpha = (det_J.alpha & ~(1ULL << i)) | (1ULL << a);
                    int lo2 = std::min(j, b), hi2 = std::max(j, b);
                    int cnt2 = count_bits_between(mod_alpha, lo2, hi2);
                    int phase = ((cnt1 + cnt2) % 2 == 0) ? 1 : -1;
                    
                    double val = c_IJ * phase;
                    add_antisym_local(Gamma_aa_local, a, b, i, j, val);
                }
                // Double β-β excitation
                else if (n_total == 2 && n_beta_diff == 2) {
                    std::vector<int> holes, parts;
                    uint64_t temp = diff_b_in_J;
                    while (temp) {
                        holes.push_back(__builtin_ctzll(temp));
                        temp &= temp - 1;
                    }
                    temp = diff_b_in_I;
                    while (temp) {
                        parts.push_back(__builtin_ctzll(temp));
                        temp &= temp - 1;
                    }
                    
                    int i = holes[0], j = holes[1];
                    int a = parts[0], b = parts[1];
                    
                    int lo1 = std::min(i, a), hi1 = std::max(i, a);
                    int cnt1 = count_bits_between(det_J.beta, lo1, hi1);
                    uint64_t mod_beta = (det_J.beta & ~(1ULL << i)) | (1ULL << a);
                    int lo2 = std::min(j, b), hi2 = std::max(j, b);
                    int cnt2 = count_bits_between(mod_beta, lo2, hi2);
                    int phase = ((cnt1 + cnt2) % 2 == 0) ? 1 : -1;
                    
                    double val = c_IJ * phase;
                    add_antisym_local(Gamma_bb_local, a, b, i, j, val);
                }
                // Double α-β excitation
                else if (n_total == 2 && n_alpha_diff == 1 && n_beta_diff == 1) {
                    int i_a = __builtin_ctzll(diff_a_in_J);
                    int a_a = __builtin_ctzll(diff_a_in_I);
                    int i_b = __builtin_ctzll(diff_b_in_J);
                    int a_b = __builtin_ctzll(diff_b_in_I);
                    
                    int phase_a = single_phase_alpha(det_J, i_a, a_a);
                    int phase_b = single_phase_beta(det_J, i_b, a_b);
                    double val = c_IJ * phase_a * phase_b;
                    
                    // Γ^αβ[a_a, a_b, i_a, i_b]
                    Gamma_ab_local[a_a*N3 + a_b*N2 + i_a*N + i_b] += val;
                }
            }
        }
        
        #pragma omp critical
        {
            for (size_t i = 0; i < N4; ++i) {
                Gamma_aa[i] += Gamma_aa_local[i];
                Gamma_ab[i] += Gamma_ab_local[i];
                Gamma_bb[i] += Gamma_bb_local[i];
            }
        }
    }
    
    return {Gamma_aa, Gamma_ab, Gamma_bb};
}


} // namespace trimci_core
