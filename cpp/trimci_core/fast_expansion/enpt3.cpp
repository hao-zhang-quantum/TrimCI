#include "enpt3.hpp"
#include "fe_diagonal.hpp"
#include "determinant.hpp"
#include "hamiltonian.hpp"
#include "fe_types.hpp"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace trimci_core {
namespace fe {

// ============================================================================
// EN-PT3: Two-pass algorithm with formal RSPT-EN partitioning.
//
// STATUS: EXPERIMENTAL / INCOMPLETE
//   The formal RSPT-EN series diverges for strongly correlated systems.
//   See enpt3.hpp header comment for details.
//
// Uses E₀^(0) = Σ_i c_i² H_{ii} as the zeroth-order energy (NOT E_var).
// The perturbation series: E = E_var + E^(2) + E^(3) + ...
// where E^(2) and E^(3) use denominators D_a = E₀^(0) - H_{aa}.
//
// Also reports CIPSI-PT2 (with E_var - H_{aa} denominators) for comparison.
// ============================================================================
template<typename StorageType>
PT3Result<StorageType> compute_enpt3(
    const std::vector<DeterminantT<StorageType>>& core_dets,
    const std::vector<double>& core_coeffs,
    double energy_var,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    int n_orb,
    const PT3Config& config)
{
    using DET = DeterminantT<StorageType>;
    PT3Result<StorageType> result;
    const size_t n_core = core_dets.size();

    if (n_core == 0) return result;

    if (config.verbose >= 1) {
        std::cout << "[EN-PT3] Starting EN-PT3 computation" << std::endl;
        std::cout << "  Core dets: " << n_core << std::endl;
        std::cout << "  n_orb:     " << n_orb << std::endl;
    }

    // ================================================================
    // Pass 1: Enumerate external dets, compute σ_a and H_{aa}
    // ================================================================
    auto t1_start = std::chrono::high_resolution_clock::now();

    fe_set<DET, DetHash<StorageType>> core_set(core_dets.begin(),
                                                core_dets.end());

    // Precompute core diagonals and occupied orbitals
    std::vector<double> core_diag(n_core);
    std::vector<std::vector<int>> core_occ_a(n_core), core_occ_b(n_core);
    for (size_t i = 0; i < n_core; ++i) {
        core_diag[i] = compute_diagonal_element(core_dets[i], h1, eri, n_orb);
        core_occ_a[i] = core_dets[i].getOccupiedAlpha();
        core_occ_b[i] = core_dets[i].getOccupiedBeta();
    }

    // E₀^(0) = Σ_i c_i² H_{ii}  (EN zeroth-order energy)
    // E^(1) = E_var - E₀^(0)     (first-order correction)
    double E0 = 0.0;
    for (size_t i = 0; i < n_core; ++i) {
        E0 += core_coeffs[i] * core_coeffs[i] * core_diag[i];
    }
    double E1 = energy_var - E0;
    result.e_first_order = E1;

    if (config.verbose >= 1) {
        std::cout << "  E₀^(0) = " << std::fixed << std::setprecision(10)
                  << E0 << std::endl;
        std::cout << "  E^(1)  = " << E1 << std::endl;
    }

    // Accumulate sigma and store H_{aa} for each external det
    fe_map<DET, double, DetHash<StorageType>> sigma_map;
    fe_map<DET, double, DetHash<StorageType>> haa_map;  // store H_{aa} directly
    fe_set<DET, DetHash<StorageType>> intruder_set;

    size_t total_intruder = 0;

    for (size_t i = 0; i < n_core; ++i) {
        const auto& det_i = core_dets[i];
        const double c_i = core_coeffs[i];
        if (std::abs(c_i) < 1e-15) continue;

        const double H_ii = core_diag[i];
        const auto& occ_a_i = core_occ_a[i];
        const auto& occ_b_i = core_occ_b[i];

        for_each_excitation_t(det_i, n_orb,
            [&](const DET& det_a)
        {
            if (core_set.count(det_a)) return;
            if (intruder_set.count(det_a)) return;

            double H_ai = compute_H_ij_t(det_i, det_a, h1, eri);
            if (std::abs(H_ai) < 1e-15) return;

            auto [it_h, new_det] = haa_map.emplace(det_a, 0.0);
            if (new_det) {
                double H_aa = compute_diagonal_incremental(
                    det_i, det_a, H_ii, occ_a_i, occ_b_i, h1, eri, n_orb);
                // Check intruder with BOTH denominators
                double d_cipsi = energy_var - H_aa;
                double d_formal = E0 - H_aa;
                if (std::abs(d_cipsi) < config.intruder_threshold ||
                    std::abs(d_formal) < config.intruder_threshold) {
                    haa_map.erase(it_h);
                    intruder_set.insert(det_a);
                    ++total_intruder;
                    return;
                }
                it_h->second = H_aa;
            }

            sigma_map[det_a] += H_ai * c_i;
        });

        if (config.verbose >= 2 && (i + 1) % 1000 == 0) {
            std::cout << "\r  Pass 1: " << (i + 1) << "/" << n_core
                      << " (" << sigma_map.size() << " externals)"
                      << std::flush;
        }
    }

    if (config.verbose >= 2) std::cout << std::endl;

    // Compute both CIPSI-PT2 and formal-PT2, plus f_a for PT3
    double delta_pt2_cipsi = 0.0;  // Σ σ²/(E_var - H_aa)
    double delta_pt2_formal = 0.0; // Σ σ²/(E₀^(0) - H_aa)
    double sum_f_sq = 0.0;         // Σ f_a² with formal denominators

    std::vector<DET> ext_dets;
    std::vector<double> ext_f;  // f_a = σ_a / (E₀^(0) - H_{aa})
    ext_dets.reserve(sigma_map.size());
    ext_f.reserve(sigma_map.size());

    for (const auto& [det_a, sigma] : sigma_map) {
        double H_aa = haa_map[det_a];
        double d_cipsi = energy_var - H_aa;
        double d_formal = E0 - H_aa;

        delta_pt2_cipsi += sigma * sigma / d_cipsi;

        double f_a = sigma / d_formal;  // formal RSPT-EN denominator
        delta_pt2_formal += sigma * f_a;
        sum_f_sq += f_a * f_a;

        ext_dets.push_back(det_a);
        ext_f.push_back(f_a);
    }

    result.delta_pt2 = delta_pt2_cipsi;
    result.delta_pt2_formal = delta_pt2_formal;
    result.sum_f_sq = sum_f_sq;
    result.n_external = ext_dets.size();
    result.n_intruder_skipped = total_intruder;

    auto t1_end = std::chrono::high_resolution_clock::now();
    result.time_pass1 = std::chrono::duration<double>(t1_end - t1_start).count();

    if (config.verbose >= 1) {
        std::cout << "  Pass 1 done: " << ext_dets.size() << " external dets, "
                  << std::fixed << std::setprecision(2)
                  << result.time_pass1 << " s" << std::endl;
        std::cout << std::setprecision(10);
        std::cout << "  ΔE^(2)_CIPSI  = " << delta_pt2_cipsi
                  << "  (D_a = E_var - H_aa)" << std::endl;
        std::cout << "  ΔE^(2)_formal = " << delta_pt2_formal
                  << "  (D_a = E0 - H_aa)" << std::endl;
        std::cout << "  Σ f_a²        = " << sum_f_sq << std::endl;
        std::cout << "  Intruder:     " << total_intruder << std::endl;
    }

    // ================================================================
    // Pass 2: Cross-term Σ_{a≠b} f_a · H_{ab} · f_b  (formal denom)
    // ================================================================
    auto t2_start = std::chrono::high_resolution_clock::now();

    fe_map<DET, size_t, DetHash<StorageType>> ext_index;
    for (size_t i = 0; i < ext_dets.size(); ++i) {
        ext_index[ext_dets[i]] = i;
    }

    double cross_term = 0.0;
    size_t n_connected = 0;
    const size_t n_ext = ext_dets.size();

    #pragma omp parallel reduction(+:cross_term, n_connected)
    {
        #pragma omp for schedule(dynamic, 256)
        for (int ia = 0; ia < static_cast<int>(n_ext); ++ia) {
            const auto& det_a = ext_dets[ia];
            const double f_a = ext_f[ia];

            for_each_excitation_t(det_a, n_orb,
                [&](const DET& det_b)
            {
                auto it = ext_index.find(det_b);
                if (it == ext_index.end()) return;

                size_t ib = it->second;
                if (ib == ia) return;

                double f_b = ext_f[ib];
                double H_ab = compute_H_ij_t(det_a, det_b, h1, eri);
                if (std::abs(H_ab) < 1e-15) return;

                cross_term += f_a * H_ab * f_b;
                ++n_connected;
            });

            if (config.verbose >= 2 && (ia + 1) % 10000 == 0) {
                #pragma omp critical
                {
                    std::cout << "\r  Pass 2: " << (ia + 1) << "/" << n_ext
                              << " external dets" << std::flush;
                }
            }
        }
    }

    if (config.verbose >= 2) std::cout << std::endl;

    auto t2_end = std::chrono::high_resolution_clock::now();
    result.time_pass2 = std::chrono::duration<double>(t2_end - t2_start).count();

    // ================================================================
    // Assemble ΔE^(3) = cross_term - E^(1) · Σ f_a²
    // ================================================================
    result.cross_term = cross_term;
    result.diag_correction = E1 * sum_f_sq;
    result.delta_pt3 = cross_term - result.diag_correction;
    result.n_connected_pairs = n_connected;

    if (config.verbose >= 1) {
        std::cout << "  Pass 2 done: " << n_connected << " connected pairs, "
                  << std::fixed << std::setprecision(2)
                  << result.time_pass2 << " s" << std::endl;
        std::cout << std::setprecision(10);
        std::cout << "  Cross term:          " << cross_term << std::endl;
        std::cout << "  E^(1)·Σf²:          " << result.diag_correction
                  << std::endl;
        std::cout << "  ΔE^(3):              " << result.delta_pt3 << std::endl;
        std::cout << std::endl;
        std::cout << "  --- Summary (formal RSPT-EN series) ---" << std::endl;
        std::cout << "  E_var + E^(2) + E^(3) = "
                  << (energy_var + delta_pt2_formal + result.delta_pt3)
                  << std::endl;
        std::cout << "  (CIPSI-PT2 for ref:   "
                  << (energy_var + delta_pt2_cipsi) << ")" << std::endl;
    }

    return result;
}

// ============================================================================
// Explicit template instantiation
// ============================================================================
template PT3Result<uint64_t> compute_enpt3<uint64_t>(
    const std::vector<DeterminantT<uint64_t>>&,
    const std::vector<double>&, double,
    const std::vector<std::vector<double>>&,
    const std::vector<double>&, int,
    const PT3Config&);

// 128-bit instantiations
using Bit128 = std::array<uint64_t, 2>;
template PT3Result<Bit128> compute_enpt3<Bit128>(
    const std::vector<DeterminantT<Bit128>>&,
    const std::vector<double>&, double,
    const std::vector<std::vector<double>>&,
    const std::vector<double>&, int,
    const PT3Config&);

}  // namespace fe
}  // namespace trimci_core
