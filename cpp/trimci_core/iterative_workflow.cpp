#include "iterative_workflow.hpp"
#include "npy_save.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <random>
#include <numeric>
#include <filesystem>
#include <sstream>
#include <climits>

namespace trimci_core {

// ============================================================================
// Helper: Normalize coefficients
// ============================================================================
static void normalize_coeffs(std::vector<double>& coeffs) {
    double norm = 0.0;
    for (double c : coeffs) norm += c * c;
    norm = std::sqrt(norm);
    if (norm > 1e-15) {
        for (double& c : coeffs) c /= norm;
    }
}

// ============================================================================
// Helper: Sort by |coefficient| descending, return sorted indices
// ============================================================================
static std::vector<size_t> argsort_by_abs_descending(const std::vector<double>& coeffs) {
    std::vector<size_t> indices(coeffs.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&coeffs](size_t a, size_t b) {
        return std::abs(coeffs[a]) > std::abs(coeffs[b]);
    });
    return indices;
}

// ============================================================================
// Helper: Logging
// ============================================================================
static void log_msg(int verbosity, int min_level, const std::string& msg) {
    if (verbosity >= min_level) {
        std::cout << msg << std::endl;
    }
}

// ============================================================================
// Main Implementation
// ============================================================================
template<typename BitType>
IterativeWorkflowResult<BitType> iterative_workflow_t(
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    int n_alpha, int n_beta, int n_orb,
    const std::string& system_name,
    const std::vector<DeterminantT<BitType>>& initial_dets,
    const std::vector<double>& initial_coeffs,
    double nuclear_repulsion,
    const IterativeWorkflowParams& params
) {
    using DetType = DeterminantT<BitType>;
    using CacheType = HijCacheT<BitType>;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    IterativeWorkflowResult<BitType> result;
    
    // =========================================================================
    // Parameter Extraction
    // =========================================================================
    int max_iterations = params.max_iterations;
    double energy_threshold = params.energy_threshold;
    int max_final_dets = params.max_final_dets;
    
    std::vector<double> core_set_ratio = params.core_set_ratio;
    if (core_set_ratio.empty()) core_set_ratio = {2.0};
    
    int initial_pool_size = params.initial_pool_size;
    std::vector<int> core_set_schedule = params.core_set_schedule;
    bool use_schedule = !core_set_schedule.empty();
    
    int pool_core_ratio = params.pool_core_ratio;
    std::string pool_build_strategy = params.pool_build_strategy;
    double threshold = params.threshold;
    double threshold_decay = params.threshold_decay;
    int max_rounds = params.max_rounds;
    std::vector<int> attentive_orbitals = params.attentive_orbitals;
    int strategy_factor = params.strategy_factor;
    
    double noise_strength = params.noise_strength;
    
    int num_groups_base = params.num_groups;
    double num_groups_ratio = params.num_groups_ratio;
    double local_trim_keep_ratio = params.local_trim_keep_ratio;
    double keep_ratio = params.keep_ratio;
    int first_cycle_keep_size = params.first_cycle_keep_size;
    
    int verbosity = params.verbosity;
    
    // Saving parameters
    int save_period = params.save_period;
    bool save_pool_flag = params.save_pool;
    bool save_initial = params.save_initial;
    std::string output_dir = params.output_dir;
    bool saving_enabled = !output_dir.empty() && save_period > 0;
    
    // Handle max_iterations = -1
    if (max_iterations <= 0) {
        if (max_final_dets <= 0 && !use_schedule) {
            result.success = false;
            result.error_message = "max_iterations=-1 requires max_final_dets or core_set_schedule";
            return result;
        }
        max_iterations = 200000;  // Safety cap
    }
    
    if (use_schedule) {
        log_msg(verbosity, 1, "[C++ Workflow] Using core_set_schedule with " + 
                std::to_string(core_set_schedule.size()) + " stages");
    }
    
    // =========================================================================
    // STEP 0: Core Set Initialization
    // =========================================================================
    std::vector<DetType> current_core_set = initial_dets;
    std::vector<double> current_core_coeffs = initial_coeffs;
    
    // Normalize initial coefficients
    normalize_coeffs(current_core_coeffs);
    
    log_msg(verbosity, 1, "[C++ Workflow] Starting with " + 
            std::to_string(current_core_set.size()) + " initial determinants");
    
    // Calculate total possible configurations: C(n_orb, n_alpha) * C(n_orb, n_beta)
    auto binomial = [](int n, int k) -> size_t {
        if (k > n || k < 0) return 0;
        if (k == 0 || k == n) return 1;
        if (k > n - k) k = n - k;
        size_t result = 1;
        for (int i = 0; i < k; ++i) {
            result = result * (n - i) / (i + 1);
        }
        return result;
    };
    size_t n_total_configs = binomial(n_orb, n_alpha) * binomial(n_orb, n_beta);
    
    log_msg(verbosity, 1, "[C++ Workflow] Total possible configurations: " + std::to_string(n_total_configs));
    
    int pool_size = std::max(
        static_cast<int>(std::ceil(current_core_set.size() * pool_core_ratio)), 
        initial_pool_size
    );
    // Cap pool_size to not exceed total possible configurations
    pool_size = std::min(pool_size, static_cast<int>(std::min(n_total_configs, static_cast<size_t>(INT_MAX))));
    
    double previous_energy = 0.0;
    bool has_previous_energy = false;
    double current_energy = 0.0;
    
    std::vector<DetType> current_dets = current_core_set;
    
    // Empty cache (created per-system)
    CacheType cache;
    std::string cache_file = system_name + ".bin";
    
    // Random generator for noise injection
    std::mt19937 rng(42);
    std::normal_distribution<double> normal_dist(0.0, 1.0);
    
    // =========================================================================
    // Save initial state (before any iteration) - matches Python behavior
    // =========================================================================
    if (saving_enabled && save_initial) {
        try {
            std::filesystem::create_directories(output_dir);
            
            // Save with iteration = -1 (Python uses iter_-01.npz naming)
            std::string npz_path = output_dir + "/iter_-01.npz";
            npy::save_npz(npz_path, current_core_set, current_core_coeffs);
            
            log_msg(verbosity, 1, "[C++] 💾 Saved initial state (iter=-1) → " + npz_path);
        } catch (const std::exception& e) {
            log_msg(verbosity, 1, "[C++] ⚠️ Failed to save initial state: " + std::string(e.what()));
        }
    }
    
    // =========================================================================
    // Progress Bar Setup (verbosity == 0 mode: minimal output with progress bar)
    // =========================================================================
    bool show_progress_bar = (verbosity == 0 && max_final_dets > 0);
    int last_progress_bars = 0;  // Track how many bars we've printed
    const int total_bars = 200;  // 0.5% per bar
    if (show_progress_bar) {
        std::cout << "[TrimCI] " << std::flush;  // Start progress line
    }
    
    // =========================================================================
    // Iterative Loop
    // =========================================================================
    int stagnation_count = 0;  // Count consecutive iterations without core set growth
    size_t prev_core_size = current_core_set.size();  // Track previous core set size
    
    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        auto iter_start = std::chrono::high_resolution_clock::now();
        
        IterationInfo iter_info;
        iter_info.iteration = iteration + 1;
        iter_info.core_set_size_before = static_cast<int>(current_core_set.size());
        
        // Get current core_set_ratio (cycling through list)
        double current_ratio = core_set_ratio[iteration % core_set_ratio.size()];
        
        log_msg(verbosity, 1, "============================================================");
        log_msg(verbosity, 1, "[C++] Iteration " + std::to_string(iteration + 1) + 
                "/" + std::to_string(max_iterations));
        
        // Cap pool size
        pool_size = std::min(static_cast<size_t>(pool_size), n_total_configs);
        iter_info.target_pool_size = pool_size;
        
        // =====================================================================
        // STEP 1: Pool Construction via Heat-Bath Screening
        // =====================================================================
        auto pool_start = std::chrono::high_resolution_clock::now();
        
        log_msg(verbosity, 2, "[C++] Building pool with " + 
                std::to_string(current_core_set.size()) + " core determinants");
        
        // Prepare coefficients based on strategy
        std::vector<double> screening_coeffs = current_core_coeffs;
        
        if (pool_build_strategy == "normalized_uniform") {
            // Use uniform normalized coefficients: 1/sqrt(N) for each
            double uniform_coeff = 1.0 / std::sqrt(static_cast<double>(current_core_set.size()));
            screening_coeffs.assign(current_core_set.size(), uniform_coeff);
            log_msg(verbosity, 2, "[C++] Using normalized_uniform coeffs: " + 
                    std::to_string(uniform_coeff));
        } else if (pool_build_strategy == "uniform") {
            // Empty coeffs - screening will treat all equally
            screening_coeffs.clear();
        } else {
            // heat_bath, heat_bath_pt2, pt2 - use actual normalized coeffs
            normalize_coeffs(screening_coeffs);
        }
        
        // Build screening params
        PoolBuildParams pb_params;
        // For normalized_uniform, the actual mode is still heat_bath
        if (pool_build_strategy == "normalized_uniform" || pool_build_strategy == "uniform") {
            pb_params.screening_mode = "heat_bath";
        } else {
            pb_params.screening_mode = pool_build_strategy;
        }
        pb_params.e0 = params.e0;
        pb_params.max_rounds = max_rounds;
        pb_params.threshold_decay = threshold_decay;
        pb_params.strategy_factor = strategy_factor;
        pb_params.strict_target_size = params.pool_strict_target_size;
        pb_params.max_pool_size = params.max_pool_size;

        // First-cycle threshold override
        double effective_threshold = threshold;
        if (iteration == 0 && params.first_cycle_threshold > 0) {
            effective_threshold = params.first_cycle_threshold;
            log_msg(verbosity, 1, "[C++] Using first_cycle_threshold=" +
                    std::to_string(params.first_cycle_threshold) + " for iteration 1");
        }

        // Call pool_build (stays in C++, no Python conversion!)
        auto pool_result = pool_build_t<BitType>(
            current_core_set, screening_coeffs, n_orb, h1, eri,
            effective_threshold, pool_size, cache, cache_file,
            attentive_orbitals, verbosity, pb_params
        );
        
        std::vector<DetType> pool = pool_result.first;
        double final_threshold = pool_result.second;
        
        auto pool_end = std::chrono::high_resolution_clock::now();
        double pool_time = std::chrono::duration<double>(pool_end - pool_start).count();
        
        iter_info.actual_pool_size = static_cast<int>(pool.size());
        iter_info.final_threshold = final_threshold;
        iter_info.pool_building_time = pool_time;
        
        log_msg(verbosity, 2, "[C++] Pool built: " + std::to_string(pool.size()) + 
                " dets in " + std::to_string(pool_time) + "s");
        
        // Update threshold for next iteration
        if (iteration > 0) {
            threshold = final_threshold;
        }
        
        // =====================================================================
        // STEP 2: Local Trim Parameter Setup
        // =====================================================================
        int num_groups = num_groups_base;
        if (num_groups_ratio > 0) {
            num_groups = std::max(num_groups_base, 
                                  static_cast<int>(current_core_set.size() * num_groups_ratio));
        }
        
        int trim_k;
        if (num_groups >= 1) {
            if (local_trim_keep_ratio > 0) {
                int keep_pool_size = static_cast<int>(
                    std::ceil(current_core_set.size() * local_trim_keep_ratio));
                trim_k = static_cast<int>(std::ceil(static_cast<double>(keep_pool_size) / num_groups));
            } else {
                trim_k = static_cast<int>(std::ceil(static_cast<double>(pool_size) * keep_ratio / num_groups));
            }
        } else {
            // Fractional power case (rarely used)
            int m = static_cast<int>(std::pow(pool_size, num_groups_base));
            num_groups = m;
            if (local_trim_keep_ratio > 0) {
                int keep_pool_size = static_cast<int>(
                    std::ceil(current_core_set.size() * local_trim_keep_ratio));
                trim_k = static_cast<int>(std::ceil(static_cast<double>(keep_pool_size) / m));
            } else {
                trim_k = static_cast<int>(std::ceil(static_cast<double>(pool_size) * keep_ratio / m));
            }
        }
        
        iter_info.trim_m = num_groups;
        iter_info.trim_k = trim_k;
        
        std::vector<int> group_sizes = {num_groups};
        std::vector<int> keep_sizes = {trim_k};
        
        // =====================================================================
        // STEP 3: Subspace Diagonalization & Selection (TRIM Core)
        // =====================================================================
        log_msg(verbosity, 2, "[C++] Running TRIM with m=" + std::to_string(num_groups) + 
                ", k=" + std::to_string(trim_k));
        
        // run_trim stays entirely in C++!
        auto trim_result = run_trim_t<BitType>(
            pool, h1, eri, system_name, n_alpha + n_beta, n_orb,
            group_sizes, keep_sizes,
            false,  // quantization
            false,  // save_cache (we manage our own)
            current_core_set,  // external_core_dets
            1e-3,   // tol
            verbosity,
            params.davidson_init
        );
        
        current_energy = std::get<0>(trim_result);
        current_dets = std::get<1>(trim_result);
        std::vector<double> trim_coeffs = std::get<2>(trim_result);
        
        iter_info.raw_dets_count = static_cast<int>(current_dets.size());
        iter_info.raw_electronic_energy = current_energy;
        
        // Full coeffs for output
        std::vector<double> full_coeffs = trim_coeffs;
        current_core_coeffs = trim_coeffs;
        
        // =====================================================================
        // STEP 4: Energy Accounting & Statistics
        // =====================================================================
        double total_energy = current_energy + nuclear_repulsion;
        iter_info.raw_energy = total_energy;
        
        log_msg(verbosity, 1, "[C++] Iteration " + std::to_string(iteration + 1) + 
                " energy: " + std::to_string(total_energy));
        log_msg(verbosity, 1, "[C++] Core set: " + std::to_string(current_core_set.size()) + 
                ", Raw dets: " + std::to_string(current_dets.size()));
        
        if (has_previous_energy) {
            double energy_change = total_energy - previous_energy;
            iter_info.energy_change = energy_change;
            iter_info.converged = (std::abs(energy_change) < energy_threshold);
            log_msg(verbosity, 1, "[C++] ΔE = " + std::to_string(energy_change));
        } else {
            iter_info.converged = false;
        }
        previous_energy = total_energy;
        has_previous_energy = true;
        
        // =====================================================================
        // STEP 5: Core Set Growth & Preparation for Next Iteration
        // =====================================================================
        int old_size = static_cast<int>(current_core_set.size());
        
        // DMRG-style noise injection
        std::vector<double> abs_coeffs(full_coeffs.size());
        if (noise_strength > 0) {
            double max_coeff = 0.0;
            for (double c : full_coeffs) max_coeff = std::max(max_coeff, std::abs(c));
            double noise_scale = noise_strength * max_coeff;
            for (size_t i = 0; i < full_coeffs.size(); ++i) {
                abs_coeffs[i] = std::abs(full_coeffs[i]) + normal_dist(rng) * noise_scale;
            }
            log_msg(verbosity, 2, "[C++] Noise injection: strength=" + 
                    std::to_string(noise_strength));
        } else {
            for (size_t i = 0; i < full_coeffs.size(); ++i) {
                abs_coeffs[i] = std::abs(full_coeffs[i]);
            }
        }
        
        // Sort by |coefficient| descending
        auto sorted_idx = argsort_by_abs_descending(abs_coeffs);
        
        std::vector<DetType> sorted_dets(current_dets.size());
        std::vector<double> sorted_coeffs(full_coeffs.size());
        for (size_t i = 0; i < sorted_idx.size(); ++i) {
            sorted_dets[i] = current_dets[sorted_idx[i]];
            sorted_coeffs[i] = full_coeffs[sorted_idx[i]];
        }
        
        // Compute new core size
        int new_size;
        if (use_schedule && iteration < static_cast<int>(core_set_schedule.size())) {
            int scheduled_size = core_set_schedule[iteration];
            new_size = std::min(static_cast<int>(current_dets.size()), scheduled_size);
            log_msg(verbosity, 1, "[C++] Core set (scheduled): " + std::to_string(old_size) + 
                    " -> " + std::to_string(new_size) + 
                    " (schedule[" + std::to_string(iteration) + "]=" + 
                    std::to_string(scheduled_size) + ")");
        } else if (iteration == 0 && first_cycle_keep_size > 0) {
            new_size = std::min(static_cast<int>(current_dets.size()), first_cycle_keep_size);
            log_msg(verbosity, 1, "[C++] Core set: " + std::to_string(old_size) + 
                    " -> " + std::to_string(new_size) + " (first cycle)");
        } else {
            new_size = std::min(static_cast<int>(current_dets.size()), 
                               static_cast<int>(std::ceil(old_size * current_ratio)));
            if (current_ratio <= 0) new_size = 1;
            log_msg(verbosity, 1, "[C++] Core set: " + std::to_string(old_size) + 
                    " -> " + std::to_string(new_size));
        }
        
        // Truncate and normalize
        current_core_set.resize(new_size);
        current_core_coeffs.resize(new_size);
        for (int i = 0; i < new_size; ++i) {
            current_core_set[i] = sorted_dets[i];
            current_core_coeffs[i] = sorted_coeffs[i];
        }
        normalize_coeffs(current_core_coeffs);
        
        iter_info.core_set_size_after = new_size;
        
        // Progress bar update (verbosity == 0 mode)
        if (show_progress_bar) {
            int current_progress_pct = (new_size * 100) / max_final_dets;
            int current_bars = (current_progress_pct * total_bars) / 100;
            current_bars = std::min(current_bars, total_bars);  // Cap at 100%
            
            // Print new bars since last update
            for (int b = last_progress_bars; b < current_bars; ++b) {
                std::cout << "█" << std::flush;
            }
            last_progress_bars = current_bars;
        }
        
        pool_size = static_cast<int>(std::ceil(new_size * pool_core_ratio));
        iter_info.next_pool_size = pool_size;
        
        // =====================================================================
        // STEP 6: Termination Conditions
        // =====================================================================
        auto iter_end = std::chrono::high_resolution_clock::now();
        double iter_time = std::chrono::duration<double>(iter_end - iter_start).count();
        double cumulative_time = std::chrono::duration<double>(iter_end - start_time).count();
        
        iter_info.iteration_time = iter_time;
        iter_info.cumulative_time = cumulative_time;
        
        log_msg(verbosity, 1, "[C++] Iteration " + std::to_string(iteration + 1) + 
                " time: " + std::to_string(iter_time) + "s (Total: " + 
                std::to_string(cumulative_time) + "s)");
        
        result.iteration_history.push_back(iter_info);
        
        // Periodic save (save_initial is handled before the loop)
        if (saving_enabled && (iteration + 1) % save_period == 0) {
            try {
                std::filesystem::create_directories(output_dir);
                
                // Format iteration number with padding
                std::ostringstream oss;
                oss << std::setfill('0') << std::setw(3) << (iteration + 1);
                std::string iter_str = oss.str();
                
                // Save to NPZ file (dets + coeffs + optional pool)
                std::string npz_path = output_dir + "/iter_" + iter_str + ".npz";
                if (save_pool_flag && !pool.empty()) {
                    npy::save_npz(npz_path, current_core_set, current_core_coeffs, &pool);
                } else {
                    npy::save_npz(npz_path, current_core_set, current_core_coeffs);
                }
                
                log_msg(verbosity, 1, "[C++] 💾 Saved iteration " + std::to_string(iteration + 1) + 
                        " → " + npz_path);
            } catch (const std::exception& e) {
                log_msg(verbosity, 1, "[C++] ⚠️ Failed to save iteration: " + std::string(e.what()));
            }
        }
        
        // Check stopping criteria
        
        // 1. Schedule exhausted
        if (use_schedule && iteration >= static_cast<int>(core_set_schedule.size()) - 1) {
            log_msg(verbosity, 1, "[C++] Core set schedule exhausted, stopping.");
            iter_info.stop_reason = "schedule_exhausted";
            break;
        }
        
        // 2. Max iterations
        if (iteration >= max_iterations - 1) {
            log_msg(verbosity, 1, "[C++] Reached max_iterations, stopping.");
            iter_info.stop_reason = "max_iterations";
            break;
        }
        
        // 3. Max final dets (only if not using schedule)
        if (!use_schedule && max_final_dets > 0 && 
            static_cast<int>(current_core_set.size()) >= max_final_dets) {
            log_msg(verbosity, 1, "[C++] Reached max_final_dets, stopping.");
            iter_info.stop_reason = "max_final_dets";
            break;
        }
        
        // 4. Stagnation detection
        if (params.stagnation_limit > 0) {
            size_t cur_core_size = current_core_set.size();
            if (cur_core_size <= prev_core_size) {
                stagnation_count++;
                if (stagnation_count >= params.stagnation_limit) {
                    log_msg(verbosity, 1, "[C++] Core set stagnated for " + 
                            std::to_string(stagnation_count) + " iterations, stopping.");
                    iter_info.stop_reason = "stagnation";
                    break;
                }
            } else {
                stagnation_count = 0;  // Reset counter on growth
            }
            prev_core_size = cur_core_size;
        }
    }
    
    // =========================================================================
    // FINALIZATION
    // =========================================================================
    
    // Complete progress bar if it was shown
    if (show_progress_bar) {
        // Fill remaining bars to 100%
        for (int b = last_progress_bars; b < total_bars; ++b) {
            std::cout << "█";
        }
        std::cout << " 100%" << std::endl;
    }
    
    // Final Davidson diagonalization on core set (with warm start)
    auto final_diag = diagonalize_subspace_davidson_t<BitType>(
        current_core_set, h1, eri, cache,
        false,  // quantization
        500,    // max_iter
        1e-6,   // tol
        verbosity,
        n_orb,
        current_core_coeffs,  // warm start from previous coeffs
        nullptr,  // sparsity
        params.davidson_init
    );
    
    double final_core_energy = std::get<0>(final_diag) + nuclear_repulsion;
    std::vector<double> final_core_coeffs = std::get<1>(final_diag);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(end_time - start_time).count();
    
    log_msg(verbosity, 1, "[C++] Final energy: " + std::to_string(final_core_energy) + 
            ", Total time: " + std::to_string(total_time) + "s");
    
    // Populate result
    result.final_energy = final_core_energy;
    result.final_dets = current_core_set;
    result.final_coeffs = final_core_coeffs;
    result.total_time = total_time;
    result.total_iterations = static_cast<int>(result.iteration_history.size());
    result.success = true;
    
    return result;
}

// ============================================================================
// Explicit instantiation for 64-bit
// ============================================================================
template IterativeWorkflowResult<uint64_t> iterative_workflow_t<uint64_t>(
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    int n_alpha, int n_beta, int n_orb,
    const std::string& system_name,
    const std::vector<Determinant64>& initial_dets,
    const std::vector<double>& initial_coeffs,
    double nuclear_repulsion,
    const IterativeWorkflowParams& params
);

// Convenience wrapper
IterativeWorkflowResult64 iterative_workflow(
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    int n_alpha, int n_beta, int n_orb,
    const std::string& system_name,
    const std::vector<Determinant64>& initial_dets,
    const std::vector<double>& initial_coeffs,
    double nuclear_repulsion,
    const IterativeWorkflowParams& params
) {
    return iterative_workflow_t<uint64_t>(
        h1, eri, n_alpha, n_beta, n_orb,
        system_name, initial_dets, initial_coeffs,
        nuclear_repulsion, params
    );
}

// ============================================================================
// Explicit instantiation for scalable types (128-512 bit)
// ============================================================================
template IterativeWorkflowResult<std::array<uint64_t, 2>> iterative_workflow_t<std::array<uint64_t, 2>>(
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    int n_alpha, int n_beta, int n_orb,
    const std::string& system_name,
    const std::vector<DeterminantT<std::array<uint64_t, 2>>>& initial_dets,
    const std::vector<double>& initial_coeffs,
    double nuclear_repulsion,
    const IterativeWorkflowParams& params
);

template IterativeWorkflowResult<std::array<uint64_t, 3>> iterative_workflow_t<std::array<uint64_t, 3>>(
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    int n_alpha, int n_beta, int n_orb,
    const std::string& system_name,
    const std::vector<DeterminantT<std::array<uint64_t, 3>>>& initial_dets,
    const std::vector<double>& initial_coeffs,
    double nuclear_repulsion,
    const IterativeWorkflowParams& params
);

template IterativeWorkflowResult<std::array<uint64_t, 4>> iterative_workflow_t<std::array<uint64_t, 4>>(
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    int n_alpha, int n_beta, int n_orb,
    const std::string& system_name,
    const std::vector<DeterminantT<std::array<uint64_t, 4>>>& initial_dets,
    const std::vector<double>& initial_coeffs,
    double nuclear_repulsion,
    const IterativeWorkflowParams& params
);

template IterativeWorkflowResult<std::array<uint64_t, 5>> iterative_workflow_t<std::array<uint64_t, 5>>(
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    int n_alpha, int n_beta, int n_orb,
    const std::string& system_name,
    const std::vector<DeterminantT<std::array<uint64_t, 5>>>& initial_dets,
    const std::vector<double>& initial_coeffs,
    double nuclear_repulsion,
    const IterativeWorkflowParams& params
);

template IterativeWorkflowResult<std::array<uint64_t, 6>> iterative_workflow_t<std::array<uint64_t, 6>>(
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    int n_alpha, int n_beta, int n_orb,
    const std::string& system_name,
    const std::vector<DeterminantT<std::array<uint64_t, 6>>>& initial_dets,
    const std::vector<double>& initial_coeffs,
    double nuclear_repulsion,
    const IterativeWorkflowParams& params
);

template IterativeWorkflowResult<std::array<uint64_t, 7>> iterative_workflow_t<std::array<uint64_t, 7>>(
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    int n_alpha, int n_beta, int n_orb,
    const std::string& system_name,
    const std::vector<DeterminantT<std::array<uint64_t, 7>>>& initial_dets,
    const std::vector<double>& initial_coeffs,
    double nuclear_repulsion,
    const IterativeWorkflowParams& params
);

template IterativeWorkflowResult<std::array<uint64_t, 8>> iterative_workflow_t<std::array<uint64_t, 8>>(
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    int n_alpha, int n_beta, int n_orb,
    const std::string& system_name,
    const std::vector<DeterminantT<std::array<uint64_t, 8>>>& initial_dets,
    const std::vector<double>& initial_coeffs,
    double nuclear_repulsion,
    const IterativeWorkflowParams& params
);

} // namespace trimci_core

