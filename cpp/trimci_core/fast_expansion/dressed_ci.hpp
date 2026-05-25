#pragma once
/**
 * Fast Expansion: Sketch-based Dressed CI.
 *
 * Corrects core coefficients c_i by incorporating external-space feedback
 * via per-det Count Sketches.
 *
 * Theory:
 *   H_eff = H_CC + K   where K_{ij} = Σ_a H_{ia}·H_{ja} / D_a
 *   K = -⟨f_i, f_j⟩   (negative Gram matrix of per-det coupling vectors)
 *   f_i(a) = H_{ia} / √|D_a|  encoded in per-det sketch S_i
 *
 * Dressed matvec:
 *   (H_eff·v)_i = (H·v)_i - ⟨S_i, S_v⟩
 *   where S_v = Σ_j v_j · S_j  (linear combination of per-det sketches)
 *
 * Consistency: c^T · K · c = ΔE_PT2  (PT2 energy = sketch norm squared)
 */

#include <vector>
#include <cstddef>
#include <cstdint>
#include "determinant.hpp"
#include "detspace_davidson.hpp"

namespace trimci_core {
namespace fe {

struct DressedCIConfig {
    // Sketch parameters for per-det sketches
    size_t sketch_width = 10000;       // w: buckets per row
    size_t sketch_depth = 5;           // d: rows (median-of-d)
    uint64_t sketch_seed = 12345;

    double intruder_threshold = 1e-6;  // Skip if |D_a| < this
    int verbose = 1;                   // 0=silent, 1=summary, 2=detail

    // Davidson params for the dressed eigensolve
    DavidsonParams davidson_params;

    // Also compute exact K for validation (only feasible for small N_core)
    bool compute_exact_K = false;

    // Compute PT2 on dressed coefficients (reuses stored connections)
    bool compute_pt2_dressed = true;

    // Auto-tuning: if memory_budget_mb > 0, override sketch_width to fit
    // within budget.  Per-det sketch memory = N_core × d × w × 8 bytes.
    double memory_budget_mb = 0.0;        // 0 = disabled (use explicit sketch_width)
    size_t auto_tune_w_min = 1000;        // minimum sketch width
    size_t auto_tune_w_max = 1000000;     // maximum sketch width

    // Export per-external-det coupling data (H_ia, H_aa) for post-hoc analysis
    // e.g. frequency-resolved self-energy Σ(ω).  Implies need_source_map.
    bool export_external_data = false;
};

template<typename StorageType>
struct DressedCIResult {
    // Sketch diagnostics
    double cKc_sketch = 0.0;           // c^T K_sketch c (from sketch inner products)
    double delta_pt2 = 0.0;            // Exact PT2 (accumulated during sketch construction)
    double variance_ext = 0.0;         // σ²_ext = Σ_a σ_a² (external energy variance)
    size_t n_external = 0;             // Number of unique external dets
    size_t n_intruder_skipped = 0;
    double sketch_memory_mb = 0.0;     // N_core × d × w × 8

    // Exact K diagnostics (if compute_exact_K = true)
    double cKc_exact = 0.0;            // c^T K_exact c
    double K_sketch_rel_error = 0.0;   // |cKc_sketch - cKc_exact| / |cKc_exact|

    // Full K matrices (if compute_exact_K = true), row-major N_core × N_core
    std::vector<double> K_exact_matrix;
    std::vector<double> K_sketch_matrix;

    // Dressed CI results
    double energy_dressed = 0.0;       // Eigenvalue of H + K
    double energy_var = 0.0;           // Original variational energy
    double energy_improvement = 0.0;   // E_dressed - (E_var + delta_pt2)
    std::vector<double> coefficients_dressed;
    int davidson_iters = 0;
    bool davidson_converged = false;

    // PT2 on dressed coefficients
    double cKc_dressed = 0.0;          // <c_d|K|c_d> = Σ_a σ_a(c_d)² / (E_var - H_aa)
    double energy_var_dressed = 0.0;   // <c_d|H_CC|c_d> = E_dressed - cKc_dressed
    double delta_pt2_dressed = 0.0;    // PT2 with c_d and E_var_dressed reference

    // Per-external-det coupling data (if export_external_data = true)
    // Sparse coupling matrix in COO format: H_coupling[ext_idx, core_idx] = H_ia
    std::vector<double> external_H_aa;       // (N_ext,) diagonal of each external det
    std::vector<int32_t> coupling_ext_idx;   // COO row indices (external det index)
    std::vector<int32_t> coupling_core_idx;  // COO col indices (core det index)
    std::vector<double> coupling_H_ia;       // COO values

    // Timing
    double time_sketches = 0.0;        // Per-det sketch construction
    double time_exact_K = 0.0;         // Exact K computation (if requested)
    double time_davidson = 0.0;        // Dressed Davidson solve
    double time_pt2_dressed = 0.0;     // PT2 on dressed coefficients
    double time_total = 0.0;
};

/// Original per-det sketch approach (for validation / small systems).
/// NOTE: Assumes all D_a = E_var - H_aa have uniform (negative) sign.
/// Incorrect for random-integral systems (e.g. SYK) with mixed D_a signs.
/// Use compute_dressed_ci_hybrid() for such systems.
template<typename StorageType>
DressedCIResult<StorageType> compute_dressed_ci(
    const std::vector<DeterminantT<StorageType>>& core_dets,
    const std::vector<double>& core_coeffs,
    double energy_var,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    int n_orb,
    const DressedCIConfig& config = {});

// Forward declaration — PT2Config is defined in streaming_pt2.hpp.
// Including it here would create a circular dependency, so we forward-declare.
template<typename StorageType> struct PT2Config;

/// Hybrid dressed CI: on-the-fly dressed matvec using CountSketch.
/// No per-det sketch storage — builds a single global sketch_v per matvec call.
/// Uses PT2Config for sketch/screening params (shared configuration).
/// Cost: ~2× PT2 per Davidson iteration (build sketch_v + query).
template<typename StorageType>
DressedCIResult<StorageType> compute_dressed_ci_hybrid(
    const std::vector<DeterminantT<StorageType>>& core_dets,
    const std::vector<double>& core_coeffs,
    double energy_var,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    int n_orb,
    const PT2Config<StorageType>& pt2_config,
    const DavidsonParams& dav_params = {});

}  // namespace fe
}  // namespace trimci_core
