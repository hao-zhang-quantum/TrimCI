#include "streaming_pt2.hpp"
#include "count_sketch.hpp"
#include "bloom_filter.hpp"
#include "fe_diagonal.hpp"
#include "determinant.hpp"
#include "hamiltonian.hpp"
#include "fe_types.hpp"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <atomic>

#ifdef _OPENMP
#include "omp_compat.hpp"
#endif

namespace trimci_core {
namespace fe {

// Main compute_pt2 (dual-sketch) implementation is at the end of this file.

#if 0  // DELETED: old single-sketch compute_pt2 (subsumed by dual-sketch)
template<typename StorageType>
PT2Result<StorageType> compute_pt2_DISABLED(
    const std::vector<DeterminantT<StorageType>>& core_dets,
    const std::vector<double>& core_coeffs,
    double energy_var,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    int n_orb,
    const PT2Config<StorageType>& config)
{
    PT2Result<StorageType> result;
    const size_t n_core = core_dets.size();

    if (n_core == 0) {
        result.energy_pt2 = energy_var;
        return result;
    }

    auto t_start = std::chrono::high_resolution_clock::now();

    if (config.verbose >= 1) {
        std::cout << "[SketchPT2] Starting Sketch PT2 computation" << std::endl;
        std::cout << "  Core dets:    " << n_core << std::endl;
        std::cout << "  n_orb:        " << n_orb << std::endl;
        std::cout << "  Sketch:       w=" << config.sketch_width
                  << ", d=" << config.sketch_depth << std::endl;
    }

    // ---- Step 1: Build Bloom filter ----
    BloomFilter<StorageType> bloom;
    bloom.init(n_core, config.bloom_fp_rate);
    for (const auto& det : core_dets) {
        bloom.add(det);
    }
    result.bloom_memory_mb = bloom.memory_bytes() / (1024.0 * 1024.0);

    if (config.verbose >= 1) {
        std::cout << "  Bloom filter: " << std::fixed << std::setprecision(1)
                  << result.bloom_memory_mb << " MB ("
                  << bloom.n_hash_functions() << " hash functions)" << std::endl;
    }

    // ---- Step 2: Initialize global sketch + variance sketch + HyperLogLog ----
    CountSketch<StorageType> sketch(config.sketch_width, config.sketch_depth,
                                    config.sketch_seed);
    CountSketch<StorageType> var_sketch(config.sketch_width, config.sketch_depth,
                                        config.sketch_seed + 7919);
    HyperLogLog<StorageType> hll(config.hll_precision);
    result.sketch_memory_mb = 2.0 * sketch.memory_bytes() / (1024.0 * 1024.0);

    if (config.verbose >= 1) {
        std::cout << "  Sketch mem:   " << std::fixed << std::setprecision(1)
                  << result.sketch_memory_mb << " MB" << std::endl;
        std::cout << "  HLL:          " << hll.n_registers()
                  << " registers (" << hll.memory_bytes()
                  << " B, ±" << std::setprecision(1)
                  << hll.relative_error() * 100 << "%)" << std::endl;
    }

    // ---- Step 2b: Precompute core diagonals and occupied orbitals ----
    std::vector<double> core_diag(n_core);
    std::vector<std::vector<int>> core_occ_a(n_core), core_occ_b(n_core);
    for (size_t i = 0; i < n_core; ++i) {
        core_diag[i] = compute_diagonal_element(core_dets[i], h1, eri, n_orb);
        core_occ_a[i] = core_dets[i].getOccupiedAlpha();
        core_occ_b[i] = core_dets[i].getOccupiedBeta();
    }

    // ---- Step 3: Process core dets (OpenMP parallel) ----
    size_t total_pairs = 0;
    size_t total_bloom_filtered = 0;
    size_t total_intruder = 0;

    // Shared progress counter for time-based reporting
    std::atomic<size_t> progress_counter{0};
    auto t_loop_start = std::chrono::high_resolution_clock::now();
    auto last_report_time = t_loop_start;
    const double report_interval_s = 30.0;  // Report every 30 seconds

    #pragma omp parallel
    {
        // Thread-local sketch, variance sketch, and HLL: separate accumulators
        CountSketch<StorageType> local_sketch(config.sketch_width,
                                               config.sketch_depth,
                                               config.sketch_seed);
        CountSketch<StorageType> local_var(config.sketch_width,
                                            config.sketch_depth,
                                            config.sketch_seed + 7919);
        HyperLogLog<StorageType> local_hll(config.hll_precision);

        size_t local_pairs = 0;
        size_t local_bloom = 0;
        size_t local_intruder = 0;

        #pragma omp for schedule(dynamic, 64)
        for (int i = 0; i < static_cast<int>(n_core); ++i) {
            const auto& det_i = core_dets[i];
            const double c_i = core_coeffs[i];

            // Skip negligible coefficients
            if (std::abs(c_i) < 1e-15) continue;

            const double H_ii = core_diag[i];
            const auto& occ_a_i = core_occ_a[i];
            const auto& occ_b_i = core_occ_b[i];

            // Enumerate all single/double excitations from det_i
            for_each_excitation_t(det_i, n_orb,
                [&](const DeterminantT<StorageType>& det_a)
            {
                // Bloom check: is det_a in the core?
                if (bloom.maybe_has(det_a)) {
                    ++local_bloom;
                    return;
                }

                // Signed H_ai via Slater-Condon rules
                double H_ai = compute_H_ij_t(det_i, det_a, h1, eri);
                if (std::abs(H_ai) < 1e-15) return;

                // Ablation: SHCI-style |H_ai * c_i| cutoff
                if (config.eps_hc_filter > 0 && std::abs(H_ai * c_i) < config.eps_hc_filter) return;

                // Incremental diagonal H_aa = H_ii + ΔH  (O(N_occ) vs O(N_occ²))
                double H_aa = compute_diagonal_incremental(
                    det_i, det_a, H_ii, occ_a_i, occ_b_i, h1, eri, n_orb);

                // Epstein-Nesbet denominator
                double d_a = energy_var - H_aa;
                if (std::abs(d_a) < config.intruder_threshold) {
                    ++local_intruder;
                    return;
                }

                // g contribution: H_ai * c_i / sqrt(|d_a|)
                double sigma_contrib = H_ai * c_i;
                double g_contrib = sigma_contrib / std::sqrt(std::abs(d_a));
                local_sketch.update(det_a, g_contrib);
                local_var.update(det_a, sigma_contrib);  // variance: no denominator
                local_hll.add(det_a);
                ++local_pairs;
            });

            // Update shared progress counter
            size_t done = progress_counter.fetch_add(1, std::memory_order_relaxed) + 1;

            // Time-based progress reporting (verbose >= 1, thread 0 only)
            if (config.verbose >= 1 && omp_get_thread_num() == 0 && done % 256 == 0) {
                auto now = std::chrono::high_resolution_clock::now();
                double since_last = std::chrono::duration<double>(now - last_report_time).count();
                if (since_last >= report_interval_s) {
                    double elapsed_loop = std::chrono::duration<double>(now - t_loop_start).count();
                    double pct = 100.0 * done / n_core;
                    double rate = done / elapsed_loop;
                    double eta = (n_core - done) / rate;
                    #pragma omp critical
                    {
                        std::cout << "  [Progress] " << done << "/" << n_core
                                  << " (" << std::fixed << std::setprecision(1) << pct << "%)"
                                  << "  " << std::setprecision(0) << rate << " dets/s"
                                  << "  elapsed=" << std::setprecision(0) << elapsed_loop << "s"
                                  << "  ETA=" << std::setprecision(0) << eta << "s"
                                  << std::endl;
                    }
                    last_report_time = now;
                }
            }
        }

        // Merge thread-local results into global
        #pragma omp critical
        {
            sketch.merge(local_sketch);
            var_sketch.merge(local_var);
            hll.merge(local_hll);
            total_pairs += local_pairs;
            total_bloom_filtered += local_bloom;
            total_intruder += local_intruder;
        }
    }

    // ---- Step 4: Estimate ΔE_PT2, variance, and cardinality ----
    double l2_norm_sq = sketch.estimate_l2_norm_sq();
    double var_ext = var_sketch.estimate_l2_norm_sq();
    double n_unique_est = hll.estimate();

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    result.l2_norm_sq = l2_norm_sq;
    result.variance_ext = var_ext;
    result.delta_pt2 = -l2_norm_sq;
    result.energy_pt2 = energy_var + result.delta_pt2;
    result.n_pairs_processed = total_pairs;
    result.n_bloom_filtered = total_bloom_filtered;
    result.n_intruder_skipped = total_intruder;
    result.n_unique_external_est = n_unique_est;
    result.avg_overlap_k = (n_unique_est > 0)
        ? static_cast<double>(total_pairs) / n_unique_est : 0.0;
    result.time_seconds = elapsed;

    if (config.verbose >= 1) {
        std::cout << std::fixed;
        std::cout << "  Pairs processed:  " << total_pairs << std::endl;
        std::cout << "  Unique externals: ~" << std::setprecision(0)
                  << n_unique_est << " (±"
                  << std::setprecision(1)
                  << hll.relative_error() * 100 << "%)" << std::endl;
        std::cout << "  Avg overlap k:    " << std::setprecision(1)
                  << result.avg_overlap_k << std::endl;
        std::cout << "  Bloom filtered:   " << total_bloom_filtered << std::endl;
        std::cout << "  Intruder skipped: " << total_intruder << std::endl;
        std::cout << "  ‖g‖² estimate:    " << std::setprecision(10)
                  << l2_norm_sq << std::endl;
        std::cout << "  σ²_ext estimate:  " << std::setprecision(10)
                  << var_ext << std::endl;
        std::cout << "  ΔE_PT2:           " << std::setprecision(10)
                  << result.delta_pt2 << std::endl;
        std::cout << "  E_var + ΔE_PT2:   " << std::setprecision(10)
                  << result.energy_pt2 << std::endl;
        std::cout << "  Time:             " << std::setprecision(2)
                  << elapsed << " s" << std::endl;
    }

    return result;
}
#endif  // DISABLED: old single-sketch compute_pt2

// ============================================================================
// Experimental: Exact (hash-table) PT2 for validation
// ============================================================================
template<typename StorageType>
PT2Result<StorageType> compute_pt2_exact_for_experiment(
    const std::vector<DeterminantT<StorageType>>& core_dets,
    const std::vector<double>& core_coeffs,
    double energy_var,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    int n_orb,
    const PT2Config<StorageType>& config)
{
    PT2Result<StorageType> result;
    const size_t n_core = core_dets.size();

    if (n_core == 0) {
        result.energy_pt2 = energy_var;
        return result;
    }

    auto t_start = std::chrono::high_resolution_clock::now();

    if (config.verbose >= 1) {
        std::cout << "[ExactPT2] Starting exact (hash-table) PT2 computation"
                  << std::endl;
        std::cout << "  Core dets: " << n_core << std::endl;
        std::cout << "  n_orb:     " << n_orb << std::endl;
    }

    // Exact core set for membership check (no false positives)
    using DET = DeterminantT<StorageType>;
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

    // External det accumulator: sigma_a and d_a
    fe_map<DET, double, DetHash<StorageType>> sigma_map;
    fe_map<DET, double, DetHash<StorageType>> denom_map;
    fe_set<DET, DetHash<StorageType>> intruder_set;

    size_t total_pairs = 0;
    size_t total_core_filtered = 0;
    size_t total_intruder = 0;

    for (size_t i = 0; i < n_core; ++i) {
        const auto& det_i = core_dets[i];
        const double c_i = core_coeffs[i];
        if (std::abs(c_i) < 1e-15) continue;

        const double H_ii = core_diag[i];
        const auto& occ_a_i = core_occ_a[i];
        const auto& occ_b_i = core_occ_b[i];

        for_each_excitation_t(det_i, n_orb,
            [&](const DeterminantT<StorageType>& det_a)
        {
            if (core_set.count(det_a)) {
                ++total_core_filtered;
                return;
            }
            if (intruder_set.count(det_a)) {
                ++total_intruder;
                return;
            }

            double H_ai = compute_H_ij_t(det_i, det_a, h1, eri);
            if (std::abs(H_ai) < 1e-15) return;

            // Ablation: SHCI-style |H_ai * c_i| cutoff
            if (config.eps_hc_filter > 0 && std::abs(H_ai * c_i) < config.eps_hc_filter) return;

            // Compute denominator on first encounter (incremental)
            auto [it_d, new_det] = denom_map.emplace(det_a, 0.0);
            if (new_det) {
                double H_aa = compute_diagonal_incremental(
                    det_i, det_a, H_ii, occ_a_i, occ_b_i, h1, eri, n_orb);
                double d_a = energy_var - H_aa;
                if (std::abs(d_a) < config.intruder_threshold) {
                    denom_map.erase(it_d);
                    intruder_set.insert(det_a);
                    ++total_intruder;
                    return;
                }
                it_d->second = d_a;
            }

            sigma_map[det_a] += H_ai * c_i;
            ++total_pairs;
        });

        if (config.verbose >= 2 && (i + 1) % 1000 == 0) {
            std::cout << "\r  Progress: " << (i + 1) << "/" << n_core
                      << " (" << sigma_map.size() << " unique externals)"
                      << std::flush;
        }
    }

    if (config.verbose >= 2) std::cout << std::endl;

    // Compute ΔE_PT2 = Σ_a σ_a² / d_a,  σ²_ext = Σ_a σ_a²,  ||Ψ^(1)||² = Σ_a (σ_a/d_a)²
    double delta_pt2 = 0.0;
    double variance_ext = 0.0;
    double sum_pt1_sq = 0.0;
    for (const auto& [det_a, sigma] : sigma_map) {
        double d_a = denom_map[det_a];
        double ratio = sigma / d_a;
        delta_pt2 += sigma * ratio;    // σ² / d_a
        variance_ext += sigma * sigma; // σ² (no denominator)
        sum_pt1_sq += ratio * ratio;   // (σ/d_a)²
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    size_t n_unique = sigma_map.size();

    result.delta_pt2 = delta_pt2;
    result.energy_pt2 = energy_var + delta_pt2;
    result.l2_norm_sq = -delta_pt2;  // ‖g‖² = -ΔE for comparison with sketch
    result.variance_ext = variance_ext;  // σ²_ext = Σ σ_a²
    result.sum_pt1_sq = sum_pt1_sq;  // ||Ψ^(1)||² for rPT2
    result.n_pairs_processed = total_pairs;
    result.n_bloom_filtered = total_core_filtered;
    result.n_intruder_skipped = total_intruder;
    result.n_unique_external_est = static_cast<double>(n_unique);
    result.avg_overlap_k = (n_unique > 0)
        ? static_cast<double>(total_pairs) / n_unique : 0.0;
    result.time_seconds = elapsed;

    if (config.verbose >= 1) {
        std::cout << std::fixed;
        std::cout << "  Pairs processed:  " << total_pairs << std::endl;
        std::cout << "  Unique externals: " << n_unique << " (exact)"
                  << std::endl;
        std::cout << "  Avg overlap k:    " << std::setprecision(1)
                  << result.avg_overlap_k << std::endl;
        std::cout << "  Core filtered:    " << total_core_filtered << std::endl;
        std::cout << "  Intruder skipped: " << total_intruder << std::endl;
        std::cout << "  ΔE_PT2:           " << std::setprecision(10)
                  << delta_pt2 << std::endl;
        std::cout << "  E_var + ΔE_PT2:   " << std::setprecision(10)
                  << result.energy_pt2 << std::endl;
        std::cout << "  Time:             " << std::setprecision(2)
                  << elapsed << " s" << std::endl;
    }

    return result;
}

// ============================================================================
// Experimental: SHCI-style PT2 with integral-level screening
// ============================================================================
template<typename StorageType>
PT2Result<StorageType> compute_pt2_shci_for_experiment(
    const std::vector<DeterminantT<StorageType>>& core_dets,
    const std::vector<double>& core_coeffs,
    double energy_var,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    int n_orb,
    const PT2Config<StorageType>& config)
{
    using BitOpsType = detail::HamiltonianBitOps<StorageType>;

    PT2Result<StorageType> result;
    const size_t n_core = core_dets.size();
    const double eps = config.eps_pt_shci;

    if (n_core == 0) {
        result.energy_pt2 = energy_var;
        return result;
    }

    auto t_start = std::chrono::high_resolution_clock::now();

    if (config.verbose >= 1) {
        std::cout << "[SHCI-PT2] Starting SHCI-style PT2 with integral screening"
                  << std::endl;
        std::cout << "  Core dets:    " << n_core << std::endl;
        std::cout << "  eps_pt_shci:  " << std::scientific << eps << std::fixed
                  << std::endl;
    }

    // Exact core set for membership check
    using DET = DeterminantT<StorageType>;
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

    // External det accumulators
    fe_map<DET, double, DetHash<StorageType>> sigma_map;
    fe_map<DET, double, DetHash<StorageType>> denom_map;
    fe_set<DET, DetHash<StorageType>> intruder_set;

    size_t total_pairs = 0;
    size_t total_core_filtered = 0;
    size_t total_intruder = 0;
    size_t total_screened = 0;

    for (size_t i = 0; i < n_core; ++i) {
        const auto& det_i = core_dets[i];
        const double c_i = core_coeffs[i];
        if (std::abs(c_i) < 1e-15) continue;

        const double H_ii = core_diag[i];
        const auto& occ_a_i = core_occ_a[i];
        const auto& occ_b_i = core_occ_b[i];

        // SHCI integral threshold: eps / |c_i|
        const double integral_threshold = (eps > 0.0) ? eps / std::abs(c_i) : 0.0;

        for_each_excitation_t(det_i, n_orb,
            [&](const DeterminantT<StorageType>& det_a)
        {
            if (core_set.count(det_a)) {
                ++total_core_filtered;
                return;
            }
            if (intruder_set.count(det_a)) {
                ++total_intruder;
                return;
            }

            // --- SHCI integral screening ---
            if (integral_threshold > 0.0) {
                const auto& ai = det_i.alpha;
                const auto& bi = det_i.beta;
                const auto& aj = det_a.alpha;
                const auto& bj = det_a.beta;

                int toggled_alpha = BitOpsType::count_differences(ai, aj);
                int toggled_beta  = BitOpsType::count_differences(bi, bj);

                if (toggled_alpha == 4 && toggled_beta == 0) {
                    // αα double: check individual ERI elements
                    int da_rem[2], da_add[2];
                    BitOpsType::storage_to_indices_inline(
                        BitOpsType::and_not(ai, aj), da_rem, 2);
                    BitOpsType::storage_to_indices_inline(
                        BitOpsType::and_not(aj, ai), da_add, 2);
                    int m = std::min(da_rem[0], da_rem[1]);
                    int n = std::max(da_rem[0], da_rem[1]);
                    int p = std::min(da_add[0], da_add[1]);
                    int q = std::max(da_add[0], da_add[1]);
                    double eri1 = std::abs(get_eri(eri, n_orb, m, p, n, q));
                    double eri2 = std::abs(get_eri(eri, n_orb, m, q, n, p));
                    if (std::max(eri1, eri2) < integral_threshold) {
                        ++total_screened;
                        return;
                    }
                } else if (toggled_alpha == 0 && toggled_beta == 4) {
                    // ββ double: check individual ERI elements
                    int db_rem[2], db_add[2];
                    BitOpsType::storage_to_indices_inline(
                        BitOpsType::and_not(bi, bj), db_rem, 2);
                    BitOpsType::storage_to_indices_inline(
                        BitOpsType::and_not(bj, bi), db_add, 2);
                    int m = std::min(db_rem[0], db_rem[1]);
                    int n = std::max(db_rem[0], db_rem[1]);
                    int p = std::min(db_add[0], db_add[1]);
                    int q = std::max(db_add[0], db_add[1]);
                    double eri1 = std::abs(get_eri(eri, n_orb, m, p, n, q));
                    double eri2 = std::abs(get_eri(eri, n_orb, m, q, n, p));
                    if (std::max(eri1, eri2) < integral_threshold) {
                        ++total_screened;
                        return;
                    }
                } else if (toggled_alpha == 2 && toggled_beta == 2) {
                    // αβ double: check single ERI element
                    int da_rem[2], da_add[2], db_rem[2], db_add[2];
                    BitOpsType::storage_to_indices_inline(
                        BitOpsType::and_not(ai, aj), da_rem, 2);
                    BitOpsType::storage_to_indices_inline(
                        BitOpsType::and_not(aj, ai), da_add, 2);
                    BitOpsType::storage_to_indices_inline(
                        BitOpsType::and_not(bi, bj), db_rem, 2);
                    BitOpsType::storage_to_indices_inline(
                        BitOpsType::and_not(bj, bi), db_add, 2);
                    int m = da_rem[0], p = da_add[0];
                    int n = db_rem[0], q = db_add[0];
                    double eri_val = std::abs(get_eri(eri, n_orb, m, p, n, q));
                    if (eri_val < integral_threshold) {
                        ++total_screened;
                        return;
                    }
                }
                // Singles: screen on |H_ai| after computing it (Fock element)
            }

            // Compute H_ai (Slater-Condon)
            double H_ai = compute_H_ij_t(det_i, det_a, h1, eri);
            if (std::abs(H_ai) < 1e-15) return;

            // Singles screening: |H_ai| < threshold (equivalent to SHCI's Fock screening)
            if (integral_threshold > 0.0) {
                int toggled_alpha = BitOpsType::count_differences(det_i.alpha, det_a.alpha);
                int toggled_beta  = BitOpsType::count_differences(det_i.beta, det_a.beta);
                if ((toggled_alpha == 2 && toggled_beta == 0) ||
                    (toggled_alpha == 0 && toggled_beta == 2)) {
                    if (std::abs(H_ai) < integral_threshold) {
                        ++total_screened;
                        return;
                    }
                }
            }

            // SHCI handler-level check: |H_ai * c_i| < eps
            if (eps > 0.0 && std::abs(H_ai * c_i) < eps) {
                ++total_screened;
                return;
            }

            // Compute denominator on first encounter (incremental)
            auto [it_d, new_det] = denom_map.emplace(det_a, 0.0);
            if (new_det) {
                double H_aa = compute_diagonal_incremental(
                    det_i, det_a, H_ii, occ_a_i, occ_b_i, h1, eri, n_orb);
                double d_a = energy_var - H_aa;
                if (std::abs(d_a) < config.intruder_threshold) {
                    denom_map.erase(it_d);
                    intruder_set.insert(det_a);
                    ++total_intruder;
                    return;
                }
                it_d->second = d_a;
            }

            sigma_map[det_a] += H_ai * c_i;
            ++total_pairs;
        });

        if (config.verbose >= 2 && (i + 1) % 1000 == 0) {
            std::cout << "\r  Progress: " << (i + 1) << "/" << n_core
                      << " (" << sigma_map.size() << " ext, "
                      << total_screened << " screened)" << std::flush;
        }
    }

    if (config.verbose >= 2) std::cout << std::endl;

    // Compute ΔE_PT2
    double delta_pt2 = 0.0;
    double variance_ext = 0.0;
    double sum_pt1_sq = 0.0;
    for (const auto& [det_a, sigma] : sigma_map) {
        double d_a = denom_map[det_a];
        double ratio = sigma / d_a;
        delta_pt2 += sigma * ratio;
        variance_ext += sigma * sigma;
        sum_pt1_sq += ratio * ratio;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    size_t n_unique = sigma_map.size();

    result.delta_pt2 = delta_pt2;
    result.energy_pt2 = energy_var + delta_pt2;
    result.l2_norm_sq = -delta_pt2;
    result.variance_ext = variance_ext;
    result.sum_pt1_sq = sum_pt1_sq;
    result.n_pairs_processed = total_pairs;
    result.n_bloom_filtered = total_core_filtered;
    result.n_intruder_skipped = total_intruder;
    result.n_integral_screened = total_screened;
    result.n_unique_external_est = static_cast<double>(n_unique);
    result.avg_overlap_k = (n_unique > 0)
        ? static_cast<double>(total_pairs) / n_unique : 0.0;
    result.time_seconds = elapsed;

    if (config.verbose >= 1) {
        std::cout << std::fixed;
        std::cout << "  Pairs processed:    " << total_pairs << std::endl;
        std::cout << "  Unique externals:   " << n_unique << " (exact)"
                  << std::endl;
        std::cout << "  Integral screened:  " << total_screened << std::endl;
        std::cout << "  Core filtered:      " << total_core_filtered << std::endl;
        std::cout << "  Intruder skipped:   " << total_intruder << std::endl;
        std::cout << "  σ²_ext:             " << std::setprecision(10)
                  << variance_ext << std::endl;
        std::cout << "  ΔE_PT2:             " << std::setprecision(10)
                  << delta_pt2 << std::endl;
        std::cout << "  E_var + ΔE_PT2:     " << std::setprecision(10)
                  << result.energy_pt2 << std::endl;
        std::cout << "  Time:               " << std::setprecision(2)
                  << elapsed << " s" << std::endl;
    }

    return result;
}

// ============================================================================
// Main: compute_pt2 — dual-sketch semistochastic PT2
// ============================================================================
template<typename StorageType>
PT2Result<StorageType> compute_pt2(
    const std::vector<DeterminantT<StorageType>>& core_dets,
    const std::vector<double>& core_coeffs,
    double energy_var,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    int n_orb,
    const PT2Config<StorageType>& config)
{
    PT2Result<StorageType> result;
    const size_t n_core = core_dets.size();

    if (n_core == 0) {
        result.energy_pt2 = energy_var;
        return result;
    }

    auto t_start = std::chrono::high_resolution_clock::now();

    // Sort indices by descending |c_i|
    std::vector<size_t> sorted_idx(n_core);
    for (size_t i = 0; i < n_core; ++i) sorted_idx[i] = i;
    std::sort(sorted_idx.begin(), sorted_idx.end(),
        [&](size_t a, size_t b) {
            return std::abs(core_coeffs[a]) > std::abs(core_coeffs[b]);
        });

    // ---- Determine split ----
    size_t n_det = config.n_deterministic;
    if (config.norm_target > 0.0 && config.norm_target < 1.0) {
        // Adaptive k: find smallest k s.t. cumulative ||c||² >= norm_target * total ||c||²
        double total_norm_sq = 0.0;
        for (size_t i = 0; i < n_core; ++i)
            total_norm_sq += core_coeffs[i] * core_coeffs[i];
        double target = config.norm_target * total_norm_sq;
        double cum = 0.0;
        n_det = n_core;
        for (size_t k = 0; k < n_core; ++k) {
            cum += core_coeffs[sorted_idx[k]] * core_coeffs[sorted_idx[k]];
            if (cum >= target) { n_det = k + 1; break; }
        }
        n_det = std::max(n_det, size_t(100));  // at least 100
    } else if (config.deterministic_fraction > 0.0) {
        n_det = std::max(size_t(100), static_cast<size_t>(n_core * config.deterministic_fraction));
    }
    if (n_det == 0 || n_det >= n_core) n_det = n_core;  // all deterministic
    // Compute N_s: stochastic_fraction (relative to N_core) takes priority over n_stochastic_samples
    size_t n_stoch_pool = (n_det < n_core) ? (n_core - n_det) : 0;
    size_t n_sample = config.n_stochastic_samples;
    if (config.stochastic_fraction > 0.0 && n_stoch_pool > 0) {
        n_sample = std::max(size_t(100),
            static_cast<size_t>(n_core * config.stochastic_fraction));
        n_sample = std::min(n_sample, n_stoch_pool);  // cap at pool size
    }

    bool fully_deterministic = (n_det >= n_core);

    if (config.verbose >= 1) {
        std::cout << "[SemiStochPT2] Starting Semistochastic Sketch PT2" << std::endl;
        std::cout << "  Core dets:       " << n_core << std::endl;
        std::cout << "  Deterministic k: " << n_det
                  << (fully_deterministic ? " (all)" : "");
        if (config.norm_target > 0.0)
            std::cout << " (norm_target=" << config.norm_target << ")";
        std::cout << std::endl;
        if (!fully_deterministic) {
            std::cout << "  Stochastic N_s:  " << n_sample << " (×2 draws)" << std::endl;
        }
        std::cout << "  Sketch:          w=" << config.sketch_width
                  << ", d=" << config.sketch_depth << std::endl;
    }

    // ---- Build Bloom filter from ALL core dets ----
    BloomFilter<StorageType> bloom;
    bloom.init(n_core, config.bloom_fp_rate);
    for (const auto& det : core_dets) bloom.add(det);
    result.bloom_memory_mb = bloom.memory_bytes() / (1024.0 * 1024.0);

    // ---- Initialize TWO PT2 sketches + TWO variance sketches ----
    CountSketch<StorageType> sketch1(config.sketch_width, config.sketch_depth,
                                     config.sketch_seed);
    CountSketch<StorageType> sketch2(config.sketch_width, config.sketch_depth,
                                     config.sketch_seed);
    CountSketch<StorageType> var_sketch1(config.sketch_width, config.sketch_depth,
                                         config.sketch_seed + 7919);
    CountSketch<StorageType> var_sketch2(config.sketch_width, config.sketch_depth,
                                         config.sketch_seed + 7919);
    HyperLogLog<StorageType> hll(config.hll_precision);
    result.sketch_memory_mb = 4.0 * sketch1.memory_bytes() / (1024.0 * 1024.0);

    if (config.verbose >= 1) {
        std::cout << "  Bloom:           " << std::fixed << std::setprecision(1)
                  << result.bloom_memory_mb << " MB" << std::endl;
        std::cout << "  Sketch mem:      " << std::setprecision(1)
                  << result.sketch_memory_mb << " MB (2 sketches)" << std::endl;
    }

    // ---- Precompute diagonals and occupied/virtual orbitals ----
    // OrbitalSums are computed on-the-fly per det in the parallel loop
    // to avoid O(N_core × N_orb) memory (112 GB at 100M dets).
    std::vector<double> core_diag(n_core);
    std::vector<std::vector<int>> core_occ_a(n_core), core_occ_b(n_core);
    std::vector<std::vector<int>> core_virt_a(n_core), core_virt_b(n_core);
    for (size_t i = 0; i < n_core; ++i) {
        core_diag[i] = compute_diagonal_element(core_dets[i], h1, eri, n_orb);
        core_occ_a[i] = core_dets[i].getOccupiedAlpha();
        core_occ_b[i] = core_dets[i].getOccupiedBeta();
        for (int o = 0; o < n_orb; ++o) {
            if (!BitOps<StorageType>::get_bit(core_dets[i].alpha, o))
                core_virt_a[i].push_back(o);
            if (!BitOps<StorageType>::get_bit(core_dets[i].beta, o))
                core_virt_b[i].push_back(o);
        }
    }

    // Lambda: process a set of (index, weight) pairs into a sketch
    size_t total_pairs = 0;
    size_t total_bloom_filtered = 0;
    size_t total_intruder = 0;
    size_t total_screened = 0;

    auto process_dets = [&](const std::vector<std::pair<size_t, double>>& idx_weights,
                            CountSketch<StorageType>& target_sketch,
                            CountSketch<StorageType>& target_var,
                            bool also_sketch2)
    {
        size_t n_process = idx_weights.size();
        std::atomic<size_t> progress_counter{0};
        auto t_phase = std::chrono::high_resolution_clock::now();
        auto last_report = t_phase;

        #pragma omp parallel
        {
            CountSketch<StorageType> local_s1(config.sketch_width, config.sketch_depth,
                                               config.sketch_seed);
            CountSketch<StorageType> local_s2(config.sketch_width, config.sketch_depth,
                                               config.sketch_seed);
            CountSketch<StorageType> local_v1(config.sketch_width, config.sketch_depth,
                                               config.sketch_seed + 7919);
            CountSketch<StorageType> local_v2(config.sketch_width, config.sketch_depth,
                                               config.sketch_seed + 7919);
            HyperLogLog<StorageType> local_hll(config.hll_precision);
            size_t local_pairs = 0, local_bloom = 0, local_intruder = 0;
            size_t local_screened = 0;

            #pragma omp for schedule(dynamic, 64)
            for (int idx = 0; idx < static_cast<int>(n_process); ++idx) {
                size_t i = idx_weights[idx].first;
                double weight = idx_weights[idx].second;
                const auto& det_i = core_dets[i];
                const double c_i = core_coeffs[i];
                if (std::abs(c_i) < 1e-15) continue;

                const double H_ii = core_diag[i];
                const auto& occ_a_i = core_occ_a[i];
                const auto& occ_b_i = core_occ_b[i];

                const auto& virt_a_i = core_virt_a[i];
                const auto& virt_b_i = core_virt_b[i];

                // SHCI-style integral screening: eps / |c_i|
                // Screens doubles by driving integral and singles by |H_ai|
                // BEFORE computing phase, diagonal, and Bloom check.
                double int_thresh = (config.eps_hc_filter > 0)
                    ? config.eps_hc_filter / std::abs(c_i) : 0.0;

                // Compute OrbitalSums on-the-fly (O(N_occ × N_orb) per det,
                // negligible vs excitation enumeration; avoids O(N_core) storage)
                auto sums_i = precompute_orbital_sums(occ_a_i, occ_b_i, eri, n_orb);

                for_each_excitation_H_diag_fast(det_i, H_ii,
                    occ_a_i, virt_a_i, occ_b_i, virt_b_i,
                    h1, eri, n_orb, sums_i,
                    [&](const DeterminantT<StorageType>& det_a,
                        double H_ai, double H_aa)
                {
                    if (bloom.maybe_has(det_a)) { ++local_bloom; return; }
                    if (std::abs(H_ai) < 1e-15) return;

                    double d_a = energy_var - H_aa;
                    if (std::abs(d_a) < config.intruder_threshold) {
                        ++local_intruder; return;
                    }

                    double sigma_contrib = H_ai * c_i * weight;
                    double g_contrib = sigma_contrib / std::sqrt(std::abs(d_a));
                    local_s1.update(det_a, g_contrib);
                    local_v1.update(det_a, sigma_contrib);
                    if (also_sketch2) {
                        local_s2.update(det_a, g_contrib);
                        local_v2.update(det_a, sigma_contrib);
                    }
                    local_hll.add(det_a);
                    ++local_pairs;
                }, int_thresh, &local_screened);

                // Progress reporting
                size_t done = progress_counter.fetch_add(1, std::memory_order_relaxed) + 1;
                if (config.verbose >= 1 && omp_get_thread_num() == 0 && done % 256 == 0) {
                    auto now = std::chrono::high_resolution_clock::now();
                    double since = std::chrono::duration<double>(now - last_report).count();
                    if (since >= 30.0) {
                        double elapsed = std::chrono::duration<double>(now - t_phase).count();
                        double rate = done / elapsed;
                        #pragma omp critical
                        {
                            std::cout << "  [Progress] " << done << "/" << n_process
                                      << " (" << std::fixed << std::setprecision(1)
                                      << 100.0 * done / n_process << "%)"
                                      << "  " << std::setprecision(0) << rate << " dets/s"
                                      << "  ETA=" << std::setprecision(0)
                                      << (n_process - done) / rate << "s" << std::endl;
                        }
                        last_report = now;
                    }
                }
            }

            #pragma omp critical
            {
                target_sketch.merge(local_s1);
                target_var.merge(local_v1);
                if (also_sketch2) {
                    sketch2.merge(local_s2);
                    var_sketch2.merge(local_v2);
                }
                hll.merge(local_hll);
                total_pairs += local_pairs;
                total_bloom_filtered += local_bloom;
                total_intruder += local_intruder;
                total_screened += local_screened;
            }
        }
    };

    // ---- Phase 1: Deterministic (top-k) → sketch1 only ----
    // Deterministic dets contribute identically to both sketches, so we only
    // update sketch1 during Phase 1 and copy to sketch2 afterward.  This halves
    // the sketch-update cost in Phase 1 (the dominant phase).
    if (config.verbose >= 1) {
        std::cout << "  Phase 1: Processing " << n_det << " deterministic dets..."
                  << std::endl;
    }
    {
        std::vector<std::pair<size_t, double>> det_idx_weights;
        det_idx_weights.reserve(n_det);
        for (size_t k = 0; k < n_det; ++k) {
            det_idx_weights.emplace_back(sorted_idx[k], 1.0);  // weight=1 for deterministic
        }
        process_dets(det_idx_weights, sketch1, var_sketch1, false);
    }
    // Copy deterministic contribution to sketch2 (identical data)
    if (!fully_deterministic) {
        sketch2 = sketch1;
        var_sketch2 = var_sketch1;
    }

    auto t_det_end = std::chrono::high_resolution_clock::now();
    double t_det = std::chrono::duration<double>(t_det_end - t_start).count();
    if (config.verbose >= 1) {
        std::cout << "  Phase 1 done: " << std::fixed << std::setprecision(1)
                  << t_det << "s, " << total_pairs << " pairs" << std::endl;
    }

    // ---- Phase 2: Stochastic samples (if not fully deterministic) ----
    if (!fully_deterministic) {
        // Build CDF for importance sampling: p_i ∝ |c_i|
        size_t n_stoch = n_core - n_det;
        std::vector<double> cdf(n_stoch);
        double sum_abs_c = 0.0;
        for (size_t k = n_det; k < n_core; ++k) {
            sum_abs_c += std::abs(core_coeffs[sorted_idx[k]]);
        }
        {
            double cumsum = 0.0;
            for (size_t k = 0; k < n_stoch; ++k) {
                cumsum += std::abs(core_coeffs[sorted_idx[n_det + k]]);
                cdf[k] = cumsum / sum_abs_c;
            }
        }

        // RNG for sampling
        auto sample_draw = [&](uint64_t seed_offset) -> std::vector<std::pair<size_t, double>> {
            std::vector<std::pair<size_t, double>> idx_weights;
            idx_weights.reserve(n_sample);

            // Simple SplitMix64 RNG
            uint64_t s = config.stochastic_seed + seed_offset;
            auto next_uniform = [&s]() -> double {
                s += 0x9e3779b97f4a7c15ULL;
                uint64_t z = s;
                z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
                z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
                z = z ^ (z >> 31);
                return static_cast<double>(z >> 11) * 0x1.0p-53;  // [0, 1)
            };

            for (size_t s_idx = 0; s_idx < n_sample; ++s_idx) {
                double u = next_uniform();
                // Binary search in CDF
                size_t lo = 0, hi = n_stoch;
                while (lo < hi) {
                    size_t mid = lo + (hi - lo) / 2;
                    if (cdf[mid] <= u) lo = mid + 1;
                    else hi = mid;
                }
                size_t stoch_k = (lo < n_stoch) ? lo : n_stoch - 1;
                size_t orig_idx = sorted_idx[n_det + stoch_k];
                double p_k = std::abs(core_coeffs[orig_idx]) / sum_abs_c;
                double weight = 1.0 / (n_sample * p_k);  // importance weight
                idx_weights.emplace_back(orig_idx, weight);
            }
            return idx_weights;
        };

        // Sample 1 → sketch1 only
        if (config.verbose >= 1) {
            std::cout << "  Phase 2a: Stochastic sample 1 (" << n_sample
                      << " draws from " << n_stoch << " dets)..." << std::endl;
        }
        auto sample1 = sample_draw(0);
        process_dets(sample1, sketch1, var_sketch1, false);

        auto t_s1_end = std::chrono::high_resolution_clock::now();
        if (config.verbose >= 1) {
            double t_s1 = std::chrono::duration<double>(t_s1_end - t_det_end).count();
            std::cout << "  Phase 2a done: " << std::fixed << std::setprecision(1)
                      << t_s1 << "s" << std::endl;
        }

        // Sample 2 → sketch2 only
        if (config.verbose >= 1) {
            std::cout << "  Phase 2b: Stochastic sample 2..." << std::endl;
        }
        auto sample2 = sample_draw(1000000007ULL);  // different seed offset
        process_dets(sample2, sketch2, var_sketch2, false);

        auto t_s2_end = std::chrono::high_resolution_clock::now();
        if (config.verbose >= 1) {
            double t_s2 = std::chrono::duration<double>(t_s2_end - t_s1_end).count();
            std::cout << "  Phase 2b done: " << std::fixed << std::setprecision(1)
                      << t_s2 << "s" << std::endl;
        }
    }

    // ---- Estimate ΔE_PT2 and variance ----
    double l2_est, var_est;
    if (fully_deterministic) {
        l2_est = sketch1.estimate_l2_norm_sq();
        var_est = var_sketch1.estimate_l2_norm_sq();
    } else {
        l2_est = sketch1.inner_product(sketch2);
        var_est = var_sketch1.inner_product(var_sketch2);
    }

    double n_unique_est = hll.estimate();
    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    result.l2_norm_sq = l2_est;
    result.variance_ext = var_est;
    result.delta_pt2 = -l2_est;
    result.energy_pt2 = energy_var + result.delta_pt2;
    result.n_pairs_processed = total_pairs;
    result.n_bloom_filtered = total_bloom_filtered;
    result.n_intruder_skipped = total_intruder;
    result.n_integral_screened = total_screened;
    result.n_unique_external_est = n_unique_est;
    result.avg_overlap_k = (n_unique_est > 0)
        ? static_cast<double>(total_pairs) / n_unique_est : 0.0;
    result.time_seconds = elapsed;

    if (config.verbose >= 1) {
        std::cout << std::fixed;
        std::cout << "  Pairs processed:  " << total_pairs << std::endl;
        std::cout << "  Integral screened:" << total_screened << std::endl;
        std::cout << "  Unique externals: ~" << std::setprecision(0)
                  << n_unique_est << std::endl;
        std::cout << "  σ²_ext:           " << std::setprecision(10)
                  << var_est << std::endl;
        std::cout << "  ΔE_PT2:           " << std::setprecision(10)
                  << result.delta_pt2 << std::endl;
        std::cout << "  E_var + ΔE_PT2:   " << std::setprecision(10)
                  << result.energy_pt2 << std::endl;
        std::cout << "  Time:             " << std::setprecision(2)
                  << elapsed << " s"
                  << (fully_deterministic ? " (fully deterministic)" : " (semistochastic)")
                  << std::endl;
    }

    // ---- Save sketches to disk if requested ----
    if (!config.sketch_save_path.empty()) {
        try {
            sketch1.save_to_file(config.sketch_save_path + "_1.bin");
            var_sketch1.save_to_file(config.sketch_save_path + "_var1.bin");
            if (!fully_deterministic) {
                sketch2.save_to_file(config.sketch_save_path + "_2.bin");
                var_sketch2.save_to_file(config.sketch_save_path + "_var2.bin");
            }
            if (config.verbose >= 1) {
                double mb = sketch1.memory_bytes() / (1024.0 * 1024.0);
                int n_files = fully_deterministic ? 2 : 4;
                std::cout << "  Sketch saved: " << n_files << " files, "
                          << std::fixed << std::setprecision(1)
                          << mb << " MB each -> " << config.sketch_save_path << "_*.bin"
                          << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "  [Warning] Failed to save sketch: " << e.what() << std::endl;
        }
    }

    return result;
}

// ============================================================================
// Explicit template instantiations
// ============================================================================
template PT2Result<uint64_t> compute_pt2<uint64_t>(
    const std::vector<DeterminantT<uint64_t>>&,
    const std::vector<double>&, double,
    const std::vector<std::vector<double>>&,
    const std::vector<double>&, int,
    const PT2Config<uint64_t>&);

template PT2Result<uint64_t> compute_pt2_exact_for_experiment<uint64_t>(
    const std::vector<DeterminantT<uint64_t>>&,
    const std::vector<double>&, double,
    const std::vector<std::vector<double>>&,
    const std::vector<double>&, int,
    const PT2Config<uint64_t>&);

template PT2Result<uint64_t> compute_pt2_shci_for_experiment<uint64_t>(
    const std::vector<DeterminantT<uint64_t>>&,
    const std::vector<double>&, double,
    const std::vector<std::vector<double>>&,
    const std::vector<double>&, int,
    const PT2Config<uint64_t>&);

// 128-bit instantiations
using Bit128 = std::array<uint64_t, 2>;
template PT2Result<Bit128> compute_pt2<Bit128>(
    const std::vector<DeterminantT<Bit128>>&,
    const std::vector<double>&, double,
    const std::vector<std::vector<double>>&,
    const std::vector<double>&, int,
    const PT2Config<Bit128>&);

template PT2Result<Bit128> compute_pt2_exact_for_experiment<Bit128>(
    const std::vector<DeterminantT<Bit128>>&,
    const std::vector<double>&, double,
    const std::vector<std::vector<double>>&,
    const std::vector<double>&, int,
    const PT2Config<Bit128>&);

template PT2Result<Bit128> compute_pt2_shci_for_experiment<Bit128>(
    const std::vector<DeterminantT<Bit128>>&,
    const std::vector<double>&, double,
    const std::vector<std::vector<double>>&,
    const std::vector<double>&, int,
    const PT2Config<Bit128>&);

}  // namespace fe
}  // namespace trimci_core
