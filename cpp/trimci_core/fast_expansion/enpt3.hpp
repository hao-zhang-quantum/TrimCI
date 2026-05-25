#pragma once
/**
 * Fast Expansion: Epstein-Nesbet third-order perturbation theory (EN-PT3).
 *
 * STATUS: EXPERIMENTAL / INCOMPLETE
 *   Implementation is mathematically correct but the formal RSPT-EN series
 *   diverges for strongly correlated systems (e.g., Hubbard at half-filling).
 *   The -E^(1)·Σf² diagonal correction dominates and blows up the series.
 *   The cross-term (two-hop physics) is negligible for Hubbard (~0.002).
 *   Needs alternative partitioning (e.g., shifted-Bk or dressed CI) to be useful.
 *
 * Computes the EN-PT3 energy correction:
 *   ΔE^(3) = Σ_{a≠b∈ext} f_a · H_{ab} · f_b  -  E^(1) · Σ_a f_a²
 *
 * where f_a = σ_a / D_a (from PT2), and E^(1) is the EN first-order correction.
 *
 * Two-pass algorithm:
 *   Pass 1: Enumerate external dets, compute σ_a and D_a (same as exact PT2)
 *   Pass 2: For each ext_a, enumerate excitations → ext_b, accumulate cross-term
 *
 * Designed for systems where N_ext is moderate (e.g., Hubbard models where
 * only single excitations contribute to PT2, giving N_ext ~ 10^5).
 */

#include <vector>
#include <cstddef>
#include <cstdint>
#include "determinant.hpp"

namespace trimci_core {
namespace fe {

struct PT3Config {
    double intruder_threshold = 1e-6;   // Skip if |D_a| < this
    int verbose = 1;                    // 0=silent, 1=summary, 2=per-pass detail
};

template<typename StorageType>
struct PT3Result {
    double delta_pt2 = 0.0;            // ΔE^(2) CIPSI-style (E_var - H_aa denom)
    double delta_pt2_formal = 0.0;     // ΔE^(2) formal RSPT (E₀^(0) - H_aa denom)
    double delta_pt3 = 0.0;            // ΔE^(3) = cross_term - diag_correction
    double e_first_order = 0.0;        // E^(1) = E_var - E₀^(0)
    double cross_term = 0.0;           // Σ_{a≠b} f_a H_{ab} f_b
    double diag_correction = 0.0;      // E^(1) · Σ_a f_a²
    double sum_f_sq = 0.0;             // Σ_a f_a² (formal denom)
    size_t n_external = 0;             // Number of unique external dets
    size_t n_connected_pairs = 0;      // Connected (a,b) pairs in Pass 2
    size_t n_intruder_skipped = 0;     // Dets with |D_a| < threshold
    double time_pass1 = 0.0;           // Pass 1 time (seconds)
    double time_pass2 = 0.0;           // Pass 2 time (seconds)
};

template<typename StorageType>
PT3Result<StorageType> compute_enpt3(
    const std::vector<DeterminantT<StorageType>>& core_dets,
    const std::vector<double>& core_coeffs,
    double energy_var,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    int n_orb,
    const PT3Config& config = {});

}  // namespace fe
}  // namespace trimci_core
