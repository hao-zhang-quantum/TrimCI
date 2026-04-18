#pragma once

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <random>

#include "determinant.hpp"
#include "screening.hpp"
#include "trim.hpp"
#include "hamiltonian.hpp"

namespace trimci_core {

// ============================================================================
// Iterative Workflow Parameters
// ============================================================================
struct IterativeWorkflowParams {
    // --- Termination Conditions ---
    int max_iterations = 200000;
    double energy_threshold = 1e-12;
    int max_final_dets = -1;  // -1 = no limit
    
    // --- Core Set Growth ---
    std::vector<double> core_set_ratio = {2.0};  // Can be a list for cycling
    int initial_pool_size = 100;
    std::vector<int> core_set_schedule;  // Empty means use core_set_ratio
    int first_cycle_keep_size = 10;
    
    // --- Pool Building (STEP 1) ---
    int pool_core_ratio = 10;
    std::string pool_build_strategy = "heat_bath";
    double threshold = 0.01;
    double first_cycle_threshold = -1.0;  // If > 0, override threshold for iteration 1 only
    double threshold_decay = 0.9;
    int max_rounds = 1;
    std::vector<int> attentive_orbitals;
    int strategy_factor = -1;
    double e0 = 0.0;  // For PT2 modes
    bool pool_strict_target_size = false;  // If true, truncate pool to exactly target_size
    int max_pool_size = 0;  // Hard cap on pool size during merge (0=no limit)
    int stagnation_limit = 0;  // Stop if core set doesn't grow for N consecutive iterations (0=disabled)
    
    // --- DMRG-style noise ---
    double noise_strength = 0.0;
    
    // --- TRIM Parameters (STEP 2-3) ---
    int num_groups = 10;
    double num_groups_ratio = 0.0;  // If > 0, num_groups = max(num_groups, core_size * ratio)
    double local_trim_keep_ratio = 0.0;
    double keep_ratio = 0.1;
    
    // --- Davidson Initialization ---
    std::string davidson_init = "lowest_diag_noise";

    // --- Logging ---
    int verbosity = 1;  // 0=silent, 1=basic, 2=detailed
    
    // --- Saving ---
    int save_period = 1000000;       // Save every N iterations (default: effectively disabled)
    bool save_pool = false;          // Whether to save pool in periodic saves
    bool save_initial = false;       // Whether to save initial state
    std::string output_dir = "";     // Output directory for saves (empty = disabled)
    
    // Constructor with defaults
    IterativeWorkflowParams() = default;
};

// ============================================================================
// Iteration Info (per-iteration statistics)
// ============================================================================
struct IterationInfo {
    int iteration = 0;
    int core_set_size_before = 0;
    int target_pool_size = 0;
    int actual_pool_size = 0;
    double final_threshold = 0.0;
    double pool_building_time = 0.0;
    int trim_m = 0;
    int trim_k = 0;
    int raw_dets_count = 0;
    double raw_electronic_energy = 0.0;
    double raw_energy = 0.0;
    double energy_change = 0.0;
    bool converged = false;
    int core_set_size_after = 0;
    int next_pool_size = 0;
    double iteration_time = 0.0;
    double cumulative_time = 0.0;
    std::string stop_reason;
};

// ============================================================================
// Iterative Workflow Result
// ============================================================================
template<typename BitType>
struct IterativeWorkflowResult {
    double final_energy = 0.0;
    std::vector<DeterminantT<BitType>> final_dets;
    std::vector<double> final_coeffs;
    std::vector<IterationInfo> iteration_history;
    double total_time = 0.0;
    int total_iterations = 0;
    bool success = true;
    std::string error_message;
};

// ============================================================================
// Main Iterative Workflow Function (C++ implementation)
// ============================================================================
// 
// Strictly follows the Python version's logic:
//   STEP 0: Core Set Initialization
//   [BEGIN ITERATIVE LOOP]
//     STEP 1: Pool Construction via Heat-Bath Screening
//     STEP 2: Local Trim Parameter Setup
//     STEP 3: Subspace Diagonalization & Selection (TRIM Core)
//     STEP 4: Energy Accounting & Statistics
//     STEP 5: Core Set Growth & Preparation for Next Iteration
//     STEP 6: Termination Conditions
//   [END ITERATIVE LOOP]
//   FINALIZATION: Post-Processing & Result Assembly
//
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
);

// Convenience alias for 64-bit determinants
using IterativeWorkflowResult64 = IterativeWorkflowResult<uint64_t>;

IterativeWorkflowResult64 iterative_workflow(
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    int n_alpha, int n_beta, int n_orb,
    const std::string& system_name,
    const std::vector<Determinant64>& initial_dets,
    const std::vector<double>& initial_coeffs,
    double nuclear_repulsion,
    const IterativeWorkflowParams& params
);

} // namespace trimci_core
