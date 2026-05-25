#pragma once
/**
 * Fast Expansion: Main expansion loop.
 *
 * Orchestrates: expand → index → diagonalize → repeat.
 * From TrimCI core (10^2 dets) to large variational space (10^8 dets).
 */

#include <vector>
#include <cstddef>
#include <string>
#include <optional>

#include "determinant.hpp"
#include "detspace_davidson.hpp"
#include "streaming_pt2.hpp"
#include "dressed_ci.hpp"

namespace trimci_core {
namespace fe {

template<typename StorageType>
struct ExpansionConfig {
    // Growth control
    double growth_factor = 5.0;
    size_t max_n_dets = 100000000;  // 10^8
    int max_expansion_rounds = 10000;
    bool strict_target_size = false; // truncate pool to exactly target_size

    // Screening
    std::string screening_mode = "hb";
    double threshold = 1e-3;        // starting threshold
    double threshold_decay = 0.5;   // FE outer loop decays threshold each round

    // Davidson
    DavidsonParams davidson_params;

    // Convergence
    double expansion_energy_tol = 1e-6;
    double dets_conv_ratio = 1.001;  // det growth < 0.1% → converged

    // Matvec mode
    bool use_connection_cache = true;  // false → on-the-fly matvec (no cache memory)
    bool symmetric_sigma = true;       // true: upper-triangle + per-thread buffers + reduce
                                       // false: full-matrix row-only (2x H_ij, but O(N) memory)
    int davidson_block_size = 1;       // >1: block Davidson with shared H_ij computation
    // Backend selection — mirrors Phase 0's args.backend field for API
    // consistency. Allowed values: "cpu" (default), "gpu", "auto".
    // GPU path offloads Davidson H*v to CUDA kernel_ch1/ch2/ch3 (uploads
    // integrals + dets + AB groups to GpuContext once per round, then each
    // Davidson iter does H2D v → kernel → D2H sigma). Pool build, PT2,
    // dressed CI, and orbopt always stay on CPU.
    //
    // Requires the build to include trimci_gpu (TRIMCI_HAS_GPU defined).
    // Mutually exclusive with use_connection_cache (GPU path ignores it),
    // sparse_update (CPU-only), and davidson_block_size > 1.
    //
    // "auto" is expected to be resolved by the Python wrapper before
    // reaching C++ — the C++ dispatcher treats unknown values as "cpu".
    std::string backend = "cpu";

    // Data layout
    bool use_morton_sort = false;      // Z-order (Morton code) det sorting — no benefit in benchmarks
    bool use_dual_order = false;       // Dual-order beta pass: gather/scatter for perfect same-beta locality
    bool use_sparse_update = false;    // Sparse-update Davidson: truncated trial vectors, O(K*C) matvec
    double sparse_truncation_ratio = 0.1;  // Fraction of trial vector components to keep
    int sparse_min_active = 1000;      // Minimum number of active dets (K lower bound)
    int sparse_n_warm_iters = 5;       // Full matvec for first N iters (hybrid approach)
    double sparse_capture_threshold = 0.0; // Minimum capture ratio (0 = disabled)
    bool sparse_adaptive_K = false;        // Increase K dynamically to meet capture threshold
    int sparse_truncation_mode = 0;        // 0: |t| (default), 1: |r| (residual), 2: |r*t| (PT2)
    double sparse_warm_ratio = 0.0;        // 0 = full matvec during warm; >0 = sparse with this ratio
    double sparse_random_sample_ratio = 0.0; // 0 = disabled; >0 = fraction of N to importance-sample
    double sparse_ucb_kappa = 0.0;          // UCB exploration bonus strength (0 = disabled)
    bool sparse_olsen_correction = false;   // Olsen preconditioner (rank-1 correction to Jacobi)
    int sparse_thick_restart_size = 1;      // Ritz vectors to keep at restart (1 = original)
    double sparse_momentum_beta = 0.0;      // Momentum coefficient (0 = disabled)
    std::vector<size_t> sparse_K_schedule;  // Per-step K values (empty = use ratio-based params)
    bool sparse_final_full_matvec = false;  // Recompute energy via full matvec after convergence

    // Perturbative warm-start: applied BEFORE any solver (both standard & sparse Davidson).
    // Enhances zero-coefficient dets (new dets from expansion) via 1st-order PT:
    //   c_j = (H*c₀)_j / (θ₀ - H_jj), cost: ONE extra full matvec.
    // Both solver paths start from the same enhanced guess → fair comparison.
    bool perturbative_warmstart = false;
    bool no_warm_start = false;           // Skip warm-start entirely (use random/unit guess)

    // Orbital optimization (BFGS)
    bool orbital_optimization = false;   // Run BFGS orbital opt after Davidson each round
    int orbital_opt_max_iter = 20;       // Max BFGS iterations per round
    double orbital_opt_grad_tol = 1e-5;  // Gradient convergence tolerance
    double orbital_opt_energy_tol = 1e-8; // Energy convergence tolerance

    // Per-round PT2 (optional: set to enable Sketch PT2 after each Davidson)
    std::optional<PT2Config<StorageType>> pt2_config;

    // Per-round Dressed CI (two modes, mutually exclusive):
    //   1. dressed_energy=true: hybrid on-the-fly mode, uses PT2 config (recommended)
    //   2. dressed_ci_config set: original per-det sketch mode (for validation)
    bool dressed_energy = false;

    // Self-consistent Brillouin-Wigner dressing: iterate dressed CI with
    // updated reference energy until convergence. Captures higher-order
    // virtual processes beyond 2nd order (important for strongly correlated systems).
    int dressed_sc_max_iter = 1;        // 1 = single-shot (default), >1 = self-consistent
    double dressed_sc_energy_tol = 1e-4; // convergence: |E_dressed^{n} - E_dressed^{n-1}| < tol
    std::optional<DressedCIConfig> dressed_ci_config;

    // Evaluate only: skip expansion, run Davidson+PT2+dressed on initial dets, 1 round
    bool evaluate_only = false;

    // Pool-build only: expand + sort + ABIndex + diag, skip Davidson/orbopt/PT2
    // Saves checkpoint with warm-start coefficients (old dets keep coeffs, new dets get 0)
    // Also saves ab_index.bin alongside the checkpoint for distributed Davidson use
    bool pool_build_only = false;

    // Checkpoint: save wavefunction (dets + coeffs) after each round
    std::string checkpoint_dir;  // empty = no checkpointing

    int checkpoint_round_offset = 0;  // added to internal round counter for checkpoint filenames

    // Output
    int verbose = 1;
    std::string log_file;  // if non-empty, tee all output to this file (real-time flush)
};

template<typename StorageType>
struct ExpansionResult {
    std::vector<DeterminantT<StorageType>> dets;
    std::vector<double> coefficients;
    double energy_var = 0.0;
    int n_rounds = 0;
    std::vector<double> energy_history;
    std::vector<size_t> ndets_history;
    std::vector<double> pt2_history;       // ΔE_PT2 per round (empty if PT2 not enabled)
    std::vector<double> variance_ext_history; // σ²_ext per round (empty if PT2 not enabled)
    std::vector<double> screening_error_history; // screening error estimate per round
    std::vector<double> dressed_energy_history;     // E_dressed per round
    std::vector<double> dressed_pt2_energy_history; // E_ref(dressed) + PT2(dressed) per round

    double ffm_wall_time = 0.0;  // Wall time of final full matvec (0 if not used)

    // Orbital optimization output (always populated; identity/copy if no orbopt)
    std::vector<double> U_total;                     // n_orb × n_orb, row-major
    std::vector<std::vector<double>> h1_optimized;   // [n_orb][n_orb]
    std::vector<double> eri_optimized;               // n_orb^4 flat
};

template<typename StorageType>
ExpansionResult<StorageType> run_expansion(
    const std::vector<DeterminantT<StorageType>>& initial_dets,
    const std::vector<double>& initial_coeffs,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    int n_orb,
    const ExpansionConfig<StorageType>& config);

// Multi-phase wrapper: runs phases sequentially, passing state between them.
// Each config is a fully resolved ExpansionConfig (Python-side JSON merge).
template<typename StorageType>
ExpansionResult<StorageType> run_expansion_phased(
    const std::vector<DeterminantT<StorageType>>& initial_dets,
    const std::vector<double>& initial_coeffs,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    int n_orb,
    const std::vector<ExpansionConfig<StorageType>>& phase_configs);

}  // namespace fe
}  // namespace trimci_core
