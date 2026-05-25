#include "dressed_ci.hpp"
#include "count_sketch.hpp"
#include "fe_diagonal.hpp"
#include "determinant.hpp"
#include "hamiltonian.hpp"
#include "fe_types.hpp"
#include "detspace_davidson.hpp"
#include "detspace_matvec.hpp"
#include "ab_index.hpp"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <numeric>

namespace trimci_core {
namespace fe {

// ============================================================================
// Sketch-based Dressed CI: per-det sketches + dressed Davidson
//
// SIGN LIMITATION: This per-det sketch path encodes f_i(a) = H_ia/√|D_a|
// (unsigned) and applies a blanket σ[i] -= ⟨S_i, S_v⟩, which assumes all
// D_a = E_var - H_aa have the same (negative) sign.  This is correct for
// molecular systems where E_var < H_aa holds for nearly all external dets.
//
// For systems with random integrals (e.g. SYK model), D_a can be both
// positive and negative, and this path will produce INCORRECT K·v signs
// for terms where D_a > 0.  Use compute_dressed_ci_hybrid() instead,
// which separates build/query fingerprints to handle mixed D_a signs.
// ============================================================================

template<typename StorageType>
DressedCIResult<StorageType> compute_dressed_ci(
    const std::vector<DeterminantT<StorageType>>& core_dets,
    const std::vector<double>& core_coeffs,
    double energy_var,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    int n_orb,
    const DressedCIConfig& config)
{
    using DET = DeterminantT<StorageType>;
    using Clock = std::chrono::high_resolution_clock;

    DressedCIResult<StorageType> result;
    result.energy_var = energy_var;
    auto t_total_start = Clock::now();

    const size_t n_core = core_dets.size();

    // P3: Auto-tune sketch_width based on memory budget
    DressedCIConfig effective_config = config;
    if (config.memory_budget_mb > 0.0 && n_core > 0) {
        double budget_bytes = config.memory_budget_mb * 1024.0 * 1024.0;
        size_t w_auto = static_cast<size_t>(
            budget_bytes / (n_core * config.sketch_depth * sizeof(double)));
        w_auto = std::max(w_auto, config.auto_tune_w_min);
        w_auto = std::min(w_auto, config.auto_tune_w_max);
        effective_config.sketch_width = w_auto;
        if (config.verbose >= 1) {
            std::cout << "[DressedCI] Auto-tune: N_core=" << n_core
                      << ", budget=" << config.memory_budget_mb << " MB"
                      << " → w=" << w_auto << std::endl;
        }
    }
    const auto& cfg = effective_config;

    if (cfg.verbose >= 1) {
        std::cout << "[DressedCI] Starting sketch-based Dressed CI" << std::endl;
        std::cout << "  Core dets:     " << n_core << std::endl;
        std::cout << "  n_orb:         " << n_orb << std::endl;
        std::cout << "  Sketch:        w=" << cfg.sketch_width
                  << ", d=" << cfg.sketch_depth << std::endl;
        double mem_mb = double(n_core) * cfg.sketch_depth * cfg.sketch_width
                        * sizeof(double) / (1024.0 * 1024.0);
        std::cout << "  Sketch memory: " << std::fixed << std::setprecision(1)
                  << mem_mb << " MB" << std::endl;
        result.sketch_memory_mb = mem_mb;
    }

    // ================================================================
    // Step 1: Precompute core diagonals and occupied orbitals
    // ================================================================
    fe_set<DET, DetHash<StorageType>> core_set(core_dets.begin(), core_dets.end());

    std::vector<double> core_diag(n_core);
    std::vector<std::vector<int>> core_occ_a(n_core), core_occ_b(n_core);
    for (size_t i = 0; i < n_core; ++i) {
        core_diag[i] = compute_diagonal_element(core_dets[i], h1, eri, n_orb);
        core_occ_a[i] = core_dets[i].getOccupiedAlpha();
        core_occ_b[i] = core_dets[i].getOccupiedBeta();
    }

    // ================================================================
    // Step 2: Build per-det sketches (streaming pass)
    //   S_i encodes f_i(a) = H_{ia} / sqrt(|D_a|)
    //   Also accumulate exact PT2 for consistency check
    // ================================================================
    auto t_sketch_start = Clock::now();

    std::vector<CountSketch<StorageType>> per_det_sketches(n_core);
    for (size_t i = 0; i < n_core; ++i) {
        per_det_sketches[i] = CountSketch<StorageType>(
            cfg.sketch_width, cfg.sketch_depth, cfg.sketch_seed);
    }

    // Track sigma_a and H_aa for exact PT2 computation
    fe_map<DET, double, DetHash<StorageType>> sigma_map;
    fe_map<DET, double, DetHash<StorageType>> haa_map;
    fe_set<DET, DetHash<StorageType>> intruder_set;
    size_t total_intruder = 0;

    // Store per-ext-det connections (for exact K and/or PT2-on-dressed)
    struct ExtConnection {
        uint32_t core_idx;
        double H_ia;
    };
    struct ExtDetInfo {
        double D_a;   // D_a = E_var - H_aa
        double H_aa;
        std::vector<ExtConnection> connections;
    };
    const bool need_source_map = cfg.compute_exact_K || cfg.compute_pt2_dressed
                                || cfg.export_external_data;
    fe_map<DET, ExtDetInfo, DetHash<StorageType>> source_map;

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

            double H_ia = compute_H_ij_t(det_i, det_a, h1, eri);
            if (std::abs(H_ia) < 1e-15) return;

            auto [it_h, new_det] = haa_map.emplace(det_a, 0.0);
            if (new_det) {
                double H_aa = compute_diagonal_incremental(
                    det_i, det_a, H_ii, occ_a_i, occ_b_i, h1, eri, n_orb);
                double D_a = energy_var - H_aa;
                if (std::abs(D_a) < cfg.intruder_threshold) {
                    haa_map.erase(it_h);
                    intruder_set.insert(det_a);
                    ++total_intruder;
                    return;
                }
                it_h->second = H_aa;
            }

            double H_aa = it_h->second;
            double D_a = energy_var - H_aa;

            // Update per-det sketch: f_i(a) = H_{ia} / sqrt(|D_a|)
            double sqrt_abs_D = std::sqrt(std::abs(D_a));
            per_det_sketches[i].update(det_a, H_ia / sqrt_abs_D);

            // Accumulate sigma for exact PT2
            sigma_map[det_a] += H_ia * c_i;

            // Store connection for exact K and/or PT2-on-dressed
            if (need_source_map) {
                double H_aa_val = it_h->second;
                auto [it_s, _] = source_map.emplace(det_a, ExtDetInfo{D_a, H_aa_val, {}});
                it_s->second.connections.push_back(
                    {static_cast<uint32_t>(i), H_ia});
            }
        });

        if (cfg.verbose >= 2 && (i + 1) % 1000 == 0) {
            std::cout << "\r  Sketch pass: " << (i + 1) << "/" << n_core
                      << " (" << sigma_map.size() << " externals)"
                      << std::flush;
        }
    }

    if (cfg.verbose >= 2) std::cout << std::endl;

    // Exact PT2 and external variance from accumulated sigma
    double delta_pt2 = 0.0;
    double variance_ext = 0.0;
    for (const auto& [det_a, sigma] : sigma_map) {
        double sigma_sq = sigma * sigma;
        double H_aa = haa_map[det_a];
        double D_a = energy_var - H_aa;
        delta_pt2 += sigma_sq / D_a;
        variance_ext += sigma_sq;
    }

    result.delta_pt2 = delta_pt2;
    result.variance_ext = variance_ext;
    result.n_external = sigma_map.size();
    result.n_intruder_skipped = total_intruder;

    result.time_sketches = std::chrono::duration<double>(
        Clock::now() - t_sketch_start).count();

    if (cfg.verbose >= 1) {
        std::cout << "  Sketch pass:   " << result.n_external << " externals, "
                  << std::fixed << std::setprecision(2)
                  << result.time_sketches << " s" << std::endl;
        std::cout << std::setprecision(10);
        std::cout << "  ΔE_PT2 (exact) = " << delta_pt2 << std::endl;
        std::cout << "  Intruder:      " << total_intruder << std::endl;
    }

    // Free sigma/haa maps (no longer needed)
    sigma_map.clear();
    haa_map.clear();
    intruder_set.clear();

    // ================================================================
    // Step 3: Sketch consistency check — c^T K c vs ΔE_PT2
    //   K_{ij} = -⟨S_i, S_j⟩, so c^T K c = -Σ_{ij} c_i ⟨S_i, S_j⟩ c_j
    //   = -‖Σ_i c_i S_i‖² ≈ -|ΔE_PT2| = ΔE_PT2  (since ΔE_PT2 < 0)
    // ================================================================
    {
        // Build S_c = Σ_i c_i · S_i
        CountSketch<StorageType> S_c(cfg.sketch_width, cfg.sketch_depth,
                                      cfg.sketch_seed);
        for (size_t i = 0; i < n_core; ++i) {
            S_c.add_scaled(per_det_sketches[i], core_coeffs[i]);
        }
        double norm_sq = S_c.estimate_l2_norm_sq();
        result.cKc_sketch = -norm_sq;  // K = -Gram → c^T K c = -‖S_c‖²

        if (cfg.verbose >= 1) {
            std::cout << "  c^T K c (sketch) = " << result.cKc_sketch << std::endl;
            std::cout << "  ΔE_PT2 (exact)   = " << delta_pt2 << std::endl;
            double rel_err = std::abs(result.cKc_sketch - delta_pt2)
                           / std::abs(delta_pt2);
            std::cout << "  Relative error:  " << std::scientific
                      << rel_err << std::fixed << std::endl;
        }
    }

    // ================================================================
    // Step 4: Exact K computation (optional validation)
    // ================================================================
    std::vector<double> K_exact;
    if (cfg.compute_exact_K) {
        auto t_K_start = Clock::now();

        K_exact.resize(n_core * n_core, 0.0);
        for (const auto& [det_a, info] : source_map) {
            double inv_D = 1.0 / info.D_a;
            const auto& conns = info.connections;
            for (size_t p = 0; p < conns.size(); ++p) {
                uint32_t i = conns[p].core_idx;
                double H_ia = conns[p].H_ia;
                double val_i = H_ia * inv_D;

                K_exact[i * n_core + i] += val_i * H_ia;

                for (size_t q = p + 1; q < conns.size(); ++q) {
                    uint32_t j = conns[q].core_idx;
                    double H_ja = conns[q].H_ia;
                    double contrib = val_i * H_ja;
                    K_exact[i * n_core + j] += contrib;
                    K_exact[j * n_core + i] += contrib;
                }
            }
        }

        // c^T K_exact c
        double cKc_exact = 0.0;
        for (size_t i = 0; i < n_core; ++i) {
            double Kc_i = 0.0;
            for (size_t j = 0; j < n_core; ++j) {
                Kc_i += K_exact[i * n_core + j] * core_coeffs[j];
            }
            cKc_exact += core_coeffs[i] * Kc_i;
        }
        result.cKc_exact = cKc_exact;

        if (std::abs(cKc_exact) > 1e-15) {
            result.K_sketch_rel_error = std::abs(result.cKc_sketch - cKc_exact)
                                      / std::abs(cKc_exact);
        }

        result.time_exact_K = std::chrono::duration<double>(
            Clock::now() - t_K_start).count();

        // Store full K_exact matrix for eigenspectrum analysis
        result.K_exact_matrix = std::move(K_exact);

        if (cfg.verbose >= 1) {
            std::cout << "  c^T K c (exact)  = " << cKc_exact << std::endl;
            std::cout << "  Sketch rel error = " << std::scientific
                      << result.K_sketch_rel_error << std::fixed << std::endl;
            std::cout << "  K matrix:        "
                      << n_core * n_core * 8 / (1024*1024) << " MB, "
                      << std::setprecision(2) << result.time_exact_K << " s"
                      << std::endl;
        }

        // Build K_sketch matrix from per-det sketch inner products
        // K_sketch[i,j] = -⟨S_i, S_j⟩
        auto t_Ks_start = Clock::now();
        result.K_sketch_matrix.resize(n_core * n_core, 0.0);
        for (size_t i = 0; i < n_core; ++i) {
            result.K_sketch_matrix[i * n_core + i] =
                -per_det_sketches[i].estimate_l2_norm_sq();
            for (size_t j = i + 1; j < n_core; ++j) {
                double Kij = -per_det_sketches[i].inner_product(
                    per_det_sketches[j]);
                result.K_sketch_matrix[i * n_core + j] = Kij;
                result.K_sketch_matrix[j * n_core + i] = Kij;
            }
        }
        if (cfg.verbose >= 1) {
            double t_Ks = std::chrono::duration<double>(
                Clock::now() - t_Ks_start).count();
            std::cout << "  K_sketch matrix: " << std::setprecision(2)
                      << t_Ks << " s" << std::endl;
        }
    }
    if (!cfg.compute_pt2_dressed) {
        source_map.clear();
    }

    // ================================================================
    // Step 5: Build H_CC infrastructure for Davidson matvec
    // ================================================================
    ABIndex<StorageType> ab_index;
    ab_index.build(core_dets, n_orb);

    auto diag = compute_diagonals<StorageType>(core_dets, h1, eri, n_orb);

    // Dressed diagonal for preconditioner: H_ii + K_ii
    // K_ii = -‖S_i‖² (negative of sketch norm squared)
    std::vector<double> diag_dressed(n_core);
    for (size_t i = 0; i < n_core; ++i) {
        double K_ii = -per_det_sketches[i].estimate_l2_norm_sq();
        diag_dressed[i] = diag[i] + K_ii;
    }

    // Build connection cache for H_CC matvec
    ConnectionCache<StorageType> conn_cache;
    bool use_cache = (n_core <= 50000);
    if (use_cache) {
        conn_cache = build_connection_cache<StorageType>(
            ab_index, core_dets, h1, eri, n_orb);
    }

    MatvecContext<StorageType> ctx{core_dets, ab_index, h1, eri, n_orb, diag};

    // ================================================================
    // Step 6: Dressed Davidson
    //   (H_eff · v)_i = (H · v)_i - ⟨S_i, S_v⟩
    //   where S_v = Σ_j v_j · S_j
    // ================================================================
    auto t_dav_start = Clock::now();

    size_t w = cfg.sketch_width;
    size_t d = cfg.sketch_depth;
    uint64_t seed = cfg.sketch_seed;

    auto dressed_matvec = [&](const double* v, double* sigma, size_t n) {
        // H_CC · v (standard core Hamiltonian)
        std::fill(sigma, sigma + n, 0.0);
        if (use_cache) {
            matvec_cached<StorageType>(conn_cache, diag, v, sigma, n);
        } else {
            matvec_dressed<StorageType>(ctx, v, sigma, n);
        }

        // K · v via sketch: σ[i] -= ⟨S_i, S_v⟩
        // Step A: Build S_v = Σ_j v_j · S_j
        CountSketch<StorageType> S_v(w, d, seed);
        for (size_t j = 0; j < n; ++j) {
            if (std::abs(v[j]) > 1e-15) {
                S_v.add_scaled(per_det_sketches[j], v[j]);
            }
        }

        // Step B: For each i, compute ⟨S_i, S_v⟩ and subtract
        for (size_t i = 0; i < n; ++i) {
            sigma[i] -= per_det_sketches[i].inner_product(S_v);
        }
    };

    DavidsonParams dav_params = cfg.davidson_params;
    if (dav_params.max_iter == 0) dav_params.max_iter = 500;
    if (dav_params.max_subspace == 0) dav_params.max_subspace = 40;

    // Warm start with current coefficients
    std::vector<std::vector<double>> guess;
    guess.push_back(std::vector<double>(core_coeffs.begin(), core_coeffs.end()));

    if (cfg.verbose >= 1) {
        std::cout << "\n  Dressed Davidson (residual_tol=" << std::scientific
                  << dav_params.residual_tol << ", energy_tol="
                  << dav_params.energy_tol << std::fixed << ")..." << std::endl;
    }

    auto dav_result = davidson_solve(dressed_matvec, diag_dressed,
                                      n_core, dav_params, guess);

    result.time_davidson = std::chrono::duration<double>(
        Clock::now() - t_dav_start).count();

    // ================================================================
    // Step 7: Assemble result
    // ================================================================
    result.energy_dressed = dav_result.eigenvalues.empty()
                          ? energy_var : dav_result.eigenvalues[0];
    result.energy_improvement = result.energy_dressed - (energy_var + delta_pt2);
    result.davidson_iters = dav_result.n_iters;
    result.davidson_converged = dav_result.converged;

    if (!dav_result.eigenvectors.empty()) {
        result.coefficients_dressed = dav_result.eigenvectors[0];
    }

    // ================================================================
    // Step 8: PT2 on dressed coefficients
    //
    //   E_dressed = <c_d|H_CC + K|c_d>  (eigenvalue of dressed H)
    //   The physical reference is <c_d|H_CC|c_d> = E_dressed - <c_d|K|c_d>
    //
    //   <c_d|K|c_d> = Σ_a σ_a(c_d)² / (E_var - H_aa)  (K built with E_var)
    //   E_var_dressed = E_dressed - <c_d|K|c_d>
    //   ΔE_PT2' = Σ_a σ_a(c_d)² / (E_var_dressed - H_aa)
    //   E_final = E_var_dressed + ΔE_PT2'
    // ================================================================
    if (cfg.compute_pt2_dressed && !source_map.empty()
        && !result.coefficients_dressed.empty()) {
        auto t_pt2d_start = Clock::now();

        const auto& c_d = result.coefficients_dressed;

        // First pass: compute σ_a(c_d) and <c_d|K|c_d>
        double cKc_d = 0.0;
        std::vector<double> sigma_dressed;
        sigma_dressed.reserve(source_map.size());

        for (const auto& [det_a, info] : source_map) {
            double sigma_new = 0.0;
            for (const auto& conn : info.connections) {
                sigma_new += conn.H_ia * c_d[conn.core_idx];
            }
            sigma_dressed.push_back(sigma_new);
            // K was built with D_a = E_var - H_aa
            cKc_d += sigma_new * sigma_new / info.D_a;
        }

        result.cKc_dressed = cKc_d;
        // Physical reference: <c_d|H_CC|c_d>
        double E_var_d = result.energy_dressed - cKc_d;
        result.energy_var_dressed = E_var_d;

        // Second pass: PT2 with physical reference energy
        double pt2_dressed = 0.0;
        size_t idx = 0;
        for (const auto& [det_a, info] : source_map) {
            double D_a_new = E_var_d - info.H_aa;
            if (std::abs(D_a_new) < cfg.intruder_threshold) { ++idx; continue; }
            pt2_dressed += sigma_dressed[idx] * sigma_dressed[idx] / D_a_new;
            ++idx;
        }

        result.delta_pt2_dressed = pt2_dressed;
        result.time_pt2_dressed = std::chrono::duration<double>(
            Clock::now() - t_pt2d_start).count();

        if (cfg.verbose >= 1) {
            std::cout << "\n  PT2 on dressed:" << std::endl;
            std::cout << std::setprecision(10);
            std::cout << "  <c_d|K|c_d>          = " << cKc_d << std::endl;
            std::cout << "  <c_d|H_CC|c_d>       = " << E_var_d << std::endl;
            std::cout << "  ΔE_PT2(c_d)          = " << pt2_dressed << std::endl;
            std::cout << "  E_final = E_ref+PT2  = "
                      << (E_var_d + pt2_dressed) << std::endl;
            std::cout << std::setprecision(2);
            std::cout << "  Time:                " << result.time_pt2_dressed
                      << " s" << std::endl;
        }
    }
    // Export per-external-det coupling data (for Σ(ω) computation etc.)
    if (cfg.export_external_data && !source_map.empty()) {
        result.external_H_aa.reserve(source_map.size());
        size_t total_conns = 0;
        for (const auto& [det_a, info] : source_map)
            total_conns += info.connections.size();
        result.coupling_ext_idx.reserve(total_conns);
        result.coupling_core_idx.reserve(total_conns);
        result.coupling_H_ia.reserve(total_conns);

        int32_t a_idx = 0;
        for (const auto& [det_a, info] : source_map) {
            result.external_H_aa.push_back(info.H_aa);
            for (const auto& conn : info.connections) {
                result.coupling_ext_idx.push_back(a_idx);
                result.coupling_core_idx.push_back(
                    static_cast<int32_t>(conn.core_idx));
                result.coupling_H_ia.push_back(conn.H_ia);
            }
            ++a_idx;
        }
        if (cfg.verbose >= 1) {
            std::cout << "  Exported " << result.external_H_aa.size()
                      << " external dets, " << total_conns
                      << " couplings" << std::endl;
        }
    }
    source_map.clear();

    result.time_total = std::chrono::duration<double>(
        Clock::now() - t_total_start).count();

    if (cfg.verbose >= 1) {
        std::cout << "  Davidson: " << dav_result.n_iters << " iters, "
                  << (dav_result.converged ? "converged" : "NOT converged")
                  << ", |r| = " << std::scientific << dav_result.residual_norm
                  << std::fixed << std::endl;
        std::cout << "\n  --- Results ---" << std::endl;
        std::cout << std::setprecision(10);
        std::cout << "  E_var:               " << energy_var << std::endl;
        std::cout << "  E_var + PT2:         " << (energy_var + delta_pt2) << std::endl;
        std::cout << "  E_dressed:           " << result.energy_dressed << std::endl;
        std::cout << "  Improvement:         " << std::scientific
                  << result.energy_improvement << std::fixed;
        if (std::abs(delta_pt2) > 1e-15) {
            double pct = result.energy_improvement / delta_pt2 * 100.0;
            std::cout << " (" << std::setprecision(1) << pct
                      << "% of PT2 correction)";
        }
        std::cout << std::endl;
        std::cout << std::setprecision(2);
        std::cout << "  Time: sketches=" << result.time_sketches << "s";
        if (cfg.compute_exact_K) {
            std::cout << ", exactK=" << result.time_exact_K << "s";
        }
        std::cout << ", davidson=" << result.time_davidson
                  << "s, total=" << result.time_total << "s" << std::endl;
    }

    return result;
}

// ============================================================================
// Explicit template instantiation
// ============================================================================
template DressedCIResult<uint64_t> compute_dressed_ci<uint64_t>(
    const std::vector<DeterminantT<uint64_t>>&,
    const std::vector<double>&, double,
    const std::vector<std::vector<double>>&,
    const std::vector<double>&, int,
    const DressedCIConfig&);

// 128-bit instantiations
using Bit128 = std::array<uint64_t, 2>;
template DressedCIResult<Bit128> compute_dressed_ci<Bit128>(
    const std::vector<DeterminantT<Bit128>>&,
    const std::vector<double>&, double,
    const std::vector<std::vector<double>>&,
    const std::vector<double>&, int,
    const DressedCIConfig&);

}  // namespace fe
}  // namespace trimci_core
