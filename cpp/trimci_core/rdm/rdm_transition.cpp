/**
 * RDM computation: transition RDMs between states
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
// Transition 1-RDM (spin-resolved)
// ============================================================================

std::pair<std::vector<double>, std::vector<double>> compute_transition_1rdm_spin_resolved(
    const std::vector<Determinant>& dets_I,
    const std::vector<double>& coeffs_I,
    const std::vector<Determinant>& dets_J,
    const std::vector<double>& coeffs_J,
    int n_orb
) {
    size_t N = static_cast<size_t>(n_orb);
    size_t N2 = N * N;
    std::vector<double> gamma_aa(N2, 0.0);
    std::vector<double> gamma_bb(N2, 0.0);
    
    int n_det_I = static_cast<int>(dets_I.size());
    int n_det_J = static_cast<int>(dets_J.size());
    
    #pragma omp parallel
    {
        std::vector<double> gamma_aa_local(N2, 0.0);
        std::vector<double> gamma_bb_local(N2, 0.0);
        
        #pragma omp for schedule(dynamic)
        for (int I = 0; I < n_det_I; ++I) {
            for (int J = 0; J < n_det_J; ++J) {
                double c_IJ = coeffs_I[I] * coeffs_J[J];
                if (std::abs(c_IJ) < 1e-15) continue;
                
                const auto& det_I = dets_I[I];
                const auto& det_J = dets_J[J];
                
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
// Transition 2-RDM (spin-resolved)
// ============================================================================

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>> 
compute_transition_2rdm_spin_resolved(
    const std::vector<Determinant>& dets_I,
    const std::vector<double>& coeffs_I,
    const std::vector<Determinant>& dets_J,
    const std::vector<double>& coeffs_J,
    int n_orb
) {
    size_t N = static_cast<size_t>(n_orb);
    size_t N2 = N * N;
    size_t N3 = N * N * N;
    size_t N4 = N * N * N * N;
    
    std::vector<double> Gamma_aa(N4, 0.0);
    std::vector<double> Gamma_ab(N4, 0.0);
    std::vector<double> Gamma_bb(N4, 0.0);
    
    int n_det_I = static_cast<int>(dets_I.size());
    int n_det_J = static_cast<int>(dets_J.size());
    
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
        for (int I = 0; I < n_det_I; ++I) {
            for (int J = 0; J < n_det_J; ++J) {
                double c_IJ = coeffs_I[I] * coeffs_J[J];
                if (std::abs(c_IJ) < 1e-15) continue;
                
                const auto& det_I = dets_I[I];
                const auto& det_J = dets_J[J];
                
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
                    
                    for (size_t i = 0; i < occ_a.size(); ++i) {
                        for (size_t j = i+1; j < occ_a.size(); ++j) {
                            add_antisym_local(Gamma_aa_local, occ_a[i], occ_a[j], occ_a[i], occ_a[j], c_IJ);
                        }
                    }
                    for (size_t i = 0; i < occ_b.size(); ++i) {
                        for (size_t j = i+1; j < occ_b.size(); ++j) {
                            add_antisym_local(Gamma_bb_local, occ_b[i], occ_b[j], occ_b[i], occ_b[j], c_IJ);
                        }
                    }
                    for (int p : occ_a) {
                        for (int q : occ_b) {
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
                    
                    auto occ_a = get_set_bits(det_J.alpha);
                    for (int k : occ_a) {
                        if (k == q) continue;
                        add_antisym_local(Gamma_aa_local, p, k, q, k, val);
                    }
                    auto occ_b = get_set_bits(det_J.beta);
                    for (int k : occ_b) {
                        Gamma_ab_local[p*N3 + k*N2 + q*N + k] += val;
                    }
                }
                // Single beta excitation
                else if (n_total == 1 && n_beta_diff == 1) {
                    int p = __builtin_ctzll(diff_b_in_I);
                    int q = __builtin_ctzll(diff_b_in_J);
                    int phase = single_phase_beta(det_J, q, p);
                    double val = c_IJ * phase;
                    
                    auto occ_b = get_set_bits(det_J.beta);
                    for (int k : occ_b) {
                        if (k == q) continue;
                        add_antisym_local(Gamma_bb_local, p, k, q, k, val);
                    }
                    auto occ_a = get_set_bits(det_J.alpha);
                    for (int k : occ_a) {
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
