#pragma once
/**
 * Fast Expansion: Sketch PT2 — Count Sketch based PT2 energy correction.
 *
 * Computes Epstein-Nesbet PT2 correction via streaming:
 *   1. Build Bloom filter from core dets (membership check)
 *   2. For each core det i, enumerate all single/double excitations
 *   3. For each external det a: sketch.update(a, H_ai * c_i / √|d_a|)
 *   4. ΔE_PT2 = -sketch.estimate_l2_norm_sq()
 *
 * Key advantage: O(MB) memory regardless of number of external dets.
 * w=200K → ~8 MB sketch, ~0.3% error, single pass, any scale.
 *
 * Semistochastic mode:
 *   Top deterministic_fraction of dets (by |c_i|) processed exactly; rest via
 *   importance sampling.  Two independent samples → sketch inner product
 *   removes variance bias.
 *
 *   Tuning guide (from empirical sweep on N2 cc-pVDZ, N=5K/20K/50K):
 *     det_frac  stoch_frac  cost/N  |err|+2σ (N=50K)  note
 *     0.10      0.05        0.20    0.39%              minimal cost
 *     0.20      0.02        0.24    0.14%              *** recommended default
 *     0.30      0.05        0.40    0.14%              higher cost, same accuracy
 *     0.50      0.05        0.60    0.10%              conservative
 *
 *   Accuracy improves with N: sketch bias ~0.2% at N=5K, ~0.1% at N=50K.
 *   Bottleneck is Phase 1 cost ∝ det_frac*N, NOT sketch width or N_s.
 */

#include <vector>
#include <cstddef>
#include <cstdint>
#include <string>
#include <optional>

#include "determinant.hpp"

namespace trimci_core {
namespace fe {

template<typename StorageType>
struct PT2Config {
    // Sketch parameters
    size_t sketch_width = 200000;       // w (buckets per row)
    size_t sketch_depth = 5;            // d (number of rows for median)
    uint64_t sketch_seed = 42;          // random seed for hash functions

    // Bloom filter
    double bloom_fp_rate = 0.001;       // false positive rate

    // HyperLogLog cardinality estimator
    int hll_precision = 14;             // p: m=2^p registers (14 → 12 KB, ~0.8% error)

    // Numerical
    double intruder_threshold = 1e-6;   // skip if |d_a| < this

    // Semistochastic mode
    // deterministic_fraction=0.2 captures >99.5% of ||c||², making stochastic noise negligible.
    // stochastic_fraction=0.02 → N_s = 2% of N_core, Phase 2 cost ≈ 8% of Phase 1.
    // Rule of thumb: cost/N ≈ det_frac + 2*stoch_frac (Phase 1 dominates).
    //
    // Empirical results (N2 cc-pVDZ, N=5K/20K/50K, 5 seeds each):
    //   det_frac=0.20, sf=0.02: |err|+2σ = 0.69%/0.69%/0.14% at cost/N = 0.24
    //   det_frac=0.30, sf=0.05: |err|+2σ = 0.47%/0.32%/0.14% at cost/N = 0.40
    //   det_frac=0.50, sf=0.02: |err|+2σ = 0.26%/0.18%/0.15% at cost/N = 0.54
    // Accuracy improves with N (sketch bias ~0.2% at 5K, ~0.1% at 50K).
    size_t n_deterministic = 0;              // top-k dets by |c_i| (0 = use deterministic_fraction)
    double deterministic_fraction = 0.2;     // fraction of N for deterministic part (0 = all deterministic)
    double norm_target = 0.99;               // adaptive k: find smallest k s.t. Σ c²_{1:k} / Σ c² >= target
                                             // 0 = disabled, use deterministic_fraction. Overrides det_fraction when > 0.
    double stochastic_fraction = 0.02;       // N_s = stochastic_fraction * N_core (overrides n_stochastic_samples)
    size_t n_stochastic_samples = 1000;      // fallback when stochastic_fraction == 0
    uint64_t stochastic_seed = 12345;   // RNG seed for importance sampling

    // SHCI-style integral pre-screening threshold in Ha (0 = disabled)
    // Screens excitations where driving integral < eps / |c_i|
    // When screening_error_target > 0, this serves as the coarsest starting eps
    // for adaptive refinement.
    double eps_hc_filter = 1e-5;

    // Adaptive screening error control (default: 3%).
    // When > 0 and eps_hc_filter > 0, the caller (expansion_loop) progressively
    // refines eps (×0.3 each step) until |ΔE(eps_prev) - ΔE(eps)| / |ΔE(eps)| < target.
    //
    // Error model: screening error ~ A·eps^α (empirically α ≈ 1.26 for Fe4S4).
    // With step ×0.3, convergence criterion overestimates true remaining error by
    // ~2.9×, so target=3% → true error ≈ 1%.
    // ×0.3 chosen as balance: ×0.1 wastes compute (17× overestimate), ×0.5 has
    // insufficient safety margin (1.4× overestimate).
    //
    // When 0, single-shot PT2 with eps_hc_filter, no error estimation.
    double screening_error_target = 0.03;

    // SHCI-style integral screening threshold (0 = disabled)
    // When > 0, compute_pt2_shci_for_experiment uses eps/|c_i| as integral threshold
    double eps_pt_shci = 0.0;

    // Dressed CI sketch width: -1 = sketch_width (default), >0 = explicit override
    // Dressed CI requires larger w than PT2 because sketch noise in K*v causes
    // systematic eigenvalue collapse (second-order perturbation from collision noise).
    int64_t dressed_sketch_width = -1;

    // Dressed CI build-phase truncation: keep top dets covering this fraction of ||v||²
    // 1.0 = no truncation (default), 0.999 = skip tail 0.1% (~1mHa loss, faster for large systems)
    double dressed_build_norm_target = 1.0;

    // Dressed CI query-phase truncation: only compute (K·v)_i for top dets covering
    // this fraction of ||c₀||² (variational coefficients, precomputed once).
    // Error bound: ΔE ≈ ||c_tail||² × max|K_ii| (local, ~0.05 mHa at 0.99).
    // 0.99 = skip ~90% of 2M dets; 1.0 = disabled (query all dets)
    double dressed_query_norm_target = 1.0;

    // Dressed CI Davidson energy convergence tolerance.
    // Sketch-based matvec has O(0.1 mHa) noise; converging below this wastes iterations.
    // 0 = use davidson_params.energy_tol as-is; >0 = override with max(davidson_energy_tol, this)
    double dressed_davidson_energy_tol = 1e-4;

    // PSF (Precomputed Sketch Fingerprint) memory limit in GB.
    // If estimated fingerprint memory exceeds this, fall back to on-the-fly enumeration.
    // 0 = auto (use PSF when < 50% of estimated available memory)
    double psf_max_memory_gb = 32.0;

    // Output
    int verbose = 1;                    // 0=silent, 1=summary, 2=progress

    // Sketch persistence: save final sketches to disk (empty = don't save).
    // Saves sketch1 to {path}_1.bin, sketch2 to {path}_2.bin,
    // var_sketch1 to {path}_var1.bin, var_sketch2 to {path}_var2.bin.
    std::string sketch_save_path;
};

template<typename StorageType>
struct PT2Result {
    double energy_pt2 = 0.0;           // E_var + ΔE
    double delta_pt2 = 0.0;            // ΔE (negative for ground state)
    double l2_norm_sq = 0.0;           // ‖g‖² estimate from sketch
    double variance_ext = 0.0;         // σ²_ext = Σ_a σ_a² (external energy variance)
    double sum_pt1_sq = 0.0;           // Σ (σ_a/D_a)² — norm of PT1 wavefunction

    size_t n_pairs_processed = 0;      // total (core, external) pairs
    size_t n_bloom_filtered = 0;       // pairs filtered by Bloom
    size_t n_intruder_skipped = 0;     // pairs with |d_a| < threshold
    size_t n_integral_screened = 0;    // excitations screened by SHCI integral threshold
    double screening_error_estimate = 0.0; // double-eps estimate: 1.15 * |ΔE(eps) - ΔE(2eps)|

    double n_unique_external_est = 0.0;// HyperLogLog cardinality estimate
    double avg_overlap_k = 0.0;        // n_pairs / n_unique (avg connections per ext det)

    double sketch_memory_mb = 0.0;
    double bloom_memory_mb = 0.0;
    double time_seconds = 0.0;
};

/// Main PT2 function: semistochastic dual-sketch with importance sampling.
/// Top deterministic_fraction of dets processed exactly; rest via importance sampling.
/// When deterministic_fraction=0, all dets are processed deterministically.
/// Time: O((k + 2*N_sample) * C_avg), Memory: O(sketch), independent of N_core.
template<typename StorageType>
PT2Result<StorageType> compute_pt2(
    const std::vector<DeterminantT<StorageType>>& core_dets,
    const std::vector<double>& core_coeffs,
    double energy_var,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    int n_orb,
    const PT2Config<StorageType>& config = {});

// ---- Experimental / ablation functions (for validation and comparison) ----

/// Exact (hash-table) PT2 for validation. Single-threaded, O(N_external) memory.
template<typename StorageType>
PT2Result<StorageType> compute_pt2_exact_for_experiment(
    const std::vector<DeterminantT<StorageType>>& core_dets,
    const std::vector<double>& core_coeffs,
    double energy_var,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    int n_orb,
    const PT2Config<StorageType>& config = {});

/// SHCI-style PT2 with integral-level screening for comparison/ablation.
/// Uses exact hash-table accumulation but applies SHCI's integral screening:
/// for each core det i, excitations are skipped if driving integral < eps_pt_shci / |c_i|.
template<typename StorageType>
PT2Result<StorageType> compute_pt2_shci_for_experiment(
    const std::vector<DeterminantT<StorageType>>& core_dets,
    const std::vector<double>& core_coeffs,
    double energy_var,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    int n_orb,
    const PT2Config<StorageType>& config = {});

}  // namespace fe
}  // namespace trimci_core
