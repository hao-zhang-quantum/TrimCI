#include "dressed_ci.hpp"
#include "streaming_pt2.hpp"
#include "count_sketch.hpp"
#include "bloom_filter.hpp"
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
#include <atomic>

namespace trimci_core {
namespace fe {

// ============================================================================
// Hybrid Dressed CI with adaptive PSF (Precomputed Sketch Fingerprints).
//
// Dressed matvec: σ = (H_CC + K) · v
//   Step A: σ = H_CC · v                (standard matvec via ABIndex)
//   Step B: Build sketch S from v       (PSF scatter-add or on-the-fly enum)
//   Step C: Query K·v from S            (PSF sparse dot or on-the-fly enum)
//
// PSF mode (small-medium systems, fingerprint fits in RAM):
//   Precompute F_i[r][k] = Σ_{a:h_r(a)=k} ξ_r(a)·f_i(a) once during K_ii.
//   All subsequent matvecs are pure arithmetic — no excitation enumeration.
//   Cost: O(N × enum) one-time, O(N × d × avg_entries) per-mv.
//
// On-the-fly mode (large systems, fingerprint exceeds memory limit):
//   Each matvec enumerates all excitations for build + query.
//   Same as original approach but with K_ii preconditioner.
//   Cost: O(N × enum) per-mv (build + query, 2 passes).
//
// Mode is selected automatically based on estimated PSF memory.
// ============================================================================

template<typename StorageType>
DressedCIResult<StorageType> compute_dressed_ci_hybrid(
    const std::vector<DeterminantT<StorageType>>& core_dets,
    const std::vector<double>& core_coeffs,
    double energy_var,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    int n_orb,
    const PT2Config<StorageType>& pt2_config,
    const DavidsonParams& dav_params_in)
{
    using DET = DeterminantT<StorageType>;
    using Clock = std::chrono::high_resolution_clock;

    DressedCIResult<StorageType> result;
    result.energy_var = energy_var;
    auto t_total_start = Clock::now();

    const size_t n_core = core_dets.size();
    if (n_core == 0) return result;

    // --- Sketch/screening params from PT2 config ---
    // Dressed CI is much more sensitive to sketch width than PT2:
    // sketch estimates K*v (vector), errors compound via eigenvalue problem.
    // Default: use same width as PT2 (not /4, which causes divergence at large N).
    const size_t w = (pt2_config.dressed_sketch_width > 0)
        ? static_cast<size_t>(pt2_config.dressed_sketch_width)
        : pt2_config.sketch_width;
    const size_t d = pt2_config.sketch_depth;
    const uint64_t seed = pt2_config.sketch_seed;
    const double eps_hc = pt2_config.eps_hc_filter;
    const double intruder_thresh = pt2_config.intruder_threshold;
    const int verbose = pt2_config.verbose;
    const double psf_max_gb = pt2_config.psf_max_memory_gb;

    if (verbose >= 1) {
        double mem_mb = 2.0 * d * w * sizeof(double) / (1024.0 * 1024.0);
        std::cout << "[DressedCI-Hybrid] On-the-fly dressed CI (PT2 config)"
                  << std::endl;
        std::cout << "  Core dets:     " << n_core << std::endl;
        std::cout << "  Sketch:        w=" << w << ", d=" << d << std::endl;
        std::cout << "  eps_hc_filter: " << std::scientific << eps_hc
                  << std::fixed << std::endl;
        if (pt2_config.dressed_query_norm_target < 1.0 - 1e-12) {
            std::cout << "  Build+Query:   synchronized, norm_target="
                      << std::setprecision(4)
                      << pt2_config.dressed_query_norm_target << std::endl;
        } else {
            std::cout << "  Build+Query:   full (no truncation)" << std::endl;
        }
        std::cout << "  Sketch memory: " << std::setprecision(1)
                  << mem_mb << " MB (per-matvec temp)" << std::endl;
        result.sketch_memory_mb = mem_mb;
    }

    // ================================================================
    // Precompute: Bloom filter, diagonals, occupied/virtual orbitals
    // ================================================================
    BloomFilter<StorageType> bloom;
    bloom.init(n_core, pt2_config.bloom_fp_rate);
    for (const auto& det : core_dets) bloom.add(det);

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

    // ================================================================
    // H_CC infrastructure (ABIndex + optional connection cache)
    // ================================================================
    ABIndex<StorageType> ab_index;
    ab_index.build(core_dets, n_orb);

    auto diag = compute_diagonals<StorageType>(core_dets, h1, eri, n_orb);

    ConnectionCache<StorageType> conn_cache;
    bool use_cache = (n_core <= 50000);
    if (use_cache) {
        conn_cache = build_connection_cache<StorageType>(
            ab_index, core_dets, h1, eri, n_orb);
    }

    MatvecContext<StorageType> ctx{core_dets, ab_index, h1, eri, n_orb, diag};

    auto t_prep_end = Clock::now();
    double t_prep = std::chrono::duration<double>(t_prep_end - t_total_start).count();
    if (verbose >= 1) {
        std::cout << "  Preparation:   " << std::setprecision(2) << t_prep
                  << " s" << std::endl;
    }

    // ================================================================
    // Query set: top dets covering query_norm_target of ||c||²
    // ================================================================
    const double query_norm_target = pt2_config.dressed_query_norm_target;
    std::vector<size_t> query_indices;

    if (query_norm_target < 1.0 - 1e-12) {
        std::vector<size_t> order(n_core);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return core_coeffs[a] * core_coeffs[a] > core_coeffs[b] * core_coeffs[b];
        });
        double total_norm2 = 0.0;
        for (const auto& c : core_coeffs) total_norm2 += c * c;
        double accum = 0.0;
        double target = query_norm_target * total_norm2;
        for (auto idx : order) {
            query_indices.push_back(idx);
            accum += core_coeffs[idx] * core_coeffs[idx];
            if (accum >= target) break;
        }
        std::sort(query_indices.begin(), query_indices.end());
    } else {
        query_indices.resize(n_core);
        std::iota(query_indices.begin(), query_indices.end(), 0);
    }
    const size_t n_query = query_indices.size();

    if (verbose >= 1) {
        std::cout << "  Query set:     " << n_query << "/" << n_core
                  << " (" << std::setprecision(1) << 100.0 * n_query / n_core
                  << "%, norm_target=" << std::setprecision(4) << query_norm_target << ")"
                  << std::endl;
    }

    // ================================================================
    // PSF memory estimation via sampling
    // ================================================================
    CountSketch<StorageType> hash_helper(w, d, seed);
    const size_t actual_w = hash_helper.width();

    size_t n_sample = std::min<size_t>(std::max<size_t>(n_query / 100, 10), n_query);
    std::vector<size_t> sample_entries(n_sample, 0);

    #pragma omp parallel for schedule(dynamic, 4)
    for (int si = 0; si < static_cast<int>(n_sample); ++si) {
        // Sample evenly across query set
        size_t qi = si * n_query / n_sample;
        size_t i = query_indices[qi];

        double int_thresh = (eps_hc > 0 && std::abs(core_coeffs[i]) > 1e-15)
            ? eps_hc / std::abs(core_coeffs[i]) : 0.0;

        auto sums_i = precompute_orbital_sums(
            core_occ_a[i], core_occ_b[i], eri, n_orb);

        size_t count = 0;
        for_each_excitation_H_diag_fast(core_dets[i], core_diag[i],
            core_occ_a[i], core_virt_a[i],
            core_occ_b[i], core_virt_b[i],
            h1, eri, n_orb, sums_i,
            [&](const DET& det_a, double H_ia, double H_aa) {
                if (bloom.maybe_has(det_a)) return;
                if (std::abs(H_ia) < 1e-15) return;
                double D_a = energy_var - H_aa;
                if (std::abs(D_a) < intruder_thresh) return;
                ++count;
            }, int_thresh, nullptr);
        sample_entries[si] = count;
    }

    double avg_entries_per_det = 0.0;
    for (auto c : sample_entries) avg_entries_per_det += c;
    avg_entries_per_det /= n_sample;

    // Each entry: d rows × (4B bucket + 4B build_val + 4B query_val) = 12*d bytes
    // After merge, #entries per row ≈ avg_entries_per_det (most hash to unique buckets)
    double est_total_entries = avg_entries_per_det * n_query * d;
    double est_psf_gb = est_total_entries * 12.0 / (1024.0 * 1024.0 * 1024.0);
    bool use_psf = (est_psf_gb <= psf_max_gb);

    if (verbose >= 1) {
        std::cout << "  PSF estimate:  " << std::setprecision(1)
                  << est_psf_gb << " GB (sampled " << n_sample << " dets, ~"
                  << std::setprecision(0) << avg_entries_per_det
                  << " exc/det)";
        if (use_psf) {
            std::cout << " -> PSF mode" << std::endl;
        } else {
            std::cout << " -> on-the-fly mode (limit "
                      << std::setprecision(0) << psf_max_gb << " GB)" << std::endl;
        }
    }

    // ================================================================
    // Phase 1: K_ii precomputation (always needed for preconditioner).
    // If use_psf: also build sketch fingerprints (combined single pass).
    // ================================================================
    auto t_kii_start = Clock::now();
    std::vector<double> precond_diag(diag.begin(), diag.end());

    // PSF data (populated only when use_psf == true)
    // Build and query fingerprints are separate to handle the sign of D_a:
    //   build: sgn * H_ia / sqrt(|D_a|)             (symmetric, unsigned)
    //   query: sgn * sign(D_a) * H_ia / sqrt(|D_a|) (carries D_a sign)
    // This ensures K*v is correctly signed for both D_a<0 and D_a>0 terms.
    std::vector<size_t> fp_offsets;
    std::vector<uint32_t> fp_buckets;
    std::vector<float> fp_values_build;
    std::vector<float> fp_values_query;
    size_t total_fp_entries = 0;

    if (use_psf) {
        // Combined K_ii + PSF fingerprint construction (single pass)
        // Build values: sgn * H_ia / sqrt(|D_a|)   (symmetric, used to construct sketch)
        // Query values: sgn * sign(D_a) * H_ia / sqrt(|D_a|)  (asymmetric, carries D_a sign)
        struct FPEntry { uint32_t bucket; float build_val; float query_val; };
        struct DetFP { std::vector<FPEntry> rows[8]; };  // d <= 8
        std::vector<DetFP> det_fps(n_query);

        #pragma omp parallel for schedule(dynamic, 64)
        for (int qi = 0; qi < static_cast<int>(n_query); ++qi) {
            size_t i = query_indices[qi];
            double kii = 0.0;

            double int_thresh = (eps_hc > 0 && std::abs(core_coeffs[i]) > 1e-15)
                ? eps_hc / std::abs(core_coeffs[i]) : 0.0;

            auto sums_i = precompute_orbital_sums(
                core_occ_a[i], core_occ_b[i], eri, n_orb);

            auto& fp = det_fps[qi];

            for_each_excitation_H_diag_fast(core_dets[i], core_diag[i],
                core_occ_a[i], core_virt_a[i],
                core_occ_b[i], core_virt_b[i],
                h1, eri, n_orb, sums_i,
                [&](const DET& det_a, double H_ia, double H_aa) {
                    if (bloom.maybe_has(det_a)) return;
                    if (std::abs(H_ia) < 1e-15) return;

                    double D_a = energy_var - H_aa;
                    if (std::abs(D_a) < intruder_thresh) return;

                    kii += H_ia * H_ia / D_a;

                    float f_unsigned = static_cast<float>(
                        H_ia / std::sqrt(std::abs(D_a)));
                    float sign_D = (D_a < 0) ? -1.0f : 1.0f;
                    uint64_t dk = CountSketch<StorageType>::det_key(det_a);
                    for (size_t r = 0; r < d; ++r) {
                        uint32_t bkt = static_cast<uint32_t>(
                            hash_helper.bucket_for(r, dk));
                        float sgn = static_cast<float>(
                            hash_helper.sign_for(r, dk));
                        fp.rows[r].push_back({bkt,
                            sgn * f_unsigned,            // build
                            sgn * sign_D * f_unsigned});  // query (with D_a sign)
                    }
                }, int_thresh, nullptr);

            precond_diag[i] = diag[i] + kii;

            // Merge duplicate buckets per row
            for (size_t r = 0; r < d; ++r) {
                auto& row = fp.rows[r];
                if (row.size() <= 1) continue;
                std::sort(row.begin(), row.end(),
                    [](const FPEntry& a, const FPEntry& b) {
                        return a.bucket < b.bucket;
                    });
                size_t wr = 0;
                for (size_t rd = 1; rd < row.size(); ++rd) {
                    if (row[rd].bucket == row[wr].bucket) {
                        row[wr].build_val += row[rd].build_val;
                        row[wr].query_val += row[rd].query_val;
                    } else {
                        row[++wr] = row[rd];
                    }
                }
                row.resize(wr + 1);
            }
        }

        // Flatten to CSR
        fp_offsets.resize(n_query * d + 1);
        for (size_t qi = 0; qi < n_query; ++qi) {
            for (size_t r = 0; r < d; ++r) {
                fp_offsets[qi * d + r] = total_fp_entries;
                total_fp_entries += det_fps[qi].rows[r].size();
            }
        }
        fp_offsets[n_query * d] = total_fp_entries;

        fp_buckets.resize(total_fp_entries);
        fp_values_build.resize(total_fp_entries);
        fp_values_query.resize(total_fp_entries);
        for (size_t qi = 0; qi < n_query; ++qi) {
            for (size_t r = 0; r < d; ++r) {
                size_t start = fp_offsets[qi * d + r];
                const auto& row = det_fps[qi].rows[r];
                for (size_t e = 0; e < row.size(); ++e) {
                    fp_buckets[start + e] = row[e].bucket;
                    fp_values_build[start + e] = row[e].build_val;
                    fp_values_query[start + e] = row[e].query_val;
                }
            }
        }
    } else {
        // K_ii only (no fingerprint storage)
        #pragma omp parallel for schedule(dynamic, 64)
        for (int qi = 0; qi < static_cast<int>(n_query); ++qi) {
            size_t i = query_indices[qi];
            double kii = 0.0;

            double int_thresh = (eps_hc > 0 && std::abs(core_coeffs[i]) > 1e-15)
                ? eps_hc / std::abs(core_coeffs[i]) : 0.0;

            auto sums_i = precompute_orbital_sums(
                core_occ_a[i], core_occ_b[i], eri, n_orb);

            for_each_excitation_H_diag_fast(core_dets[i], core_diag[i],
                core_occ_a[i], core_virt_a[i],
                core_occ_b[i], core_virt_b[i],
                h1, eri, n_orb, sums_i,
                [&](const DET& det_a, double H_ia, double H_aa) {
                    if (bloom.maybe_has(det_a)) return;
                    if (std::abs(H_ia) < 1e-15) return;
                    double D_a = energy_var - H_aa;
                    if (std::abs(D_a) < intruder_thresh) return;
                    kii += H_ia * H_ia / D_a;
                }, int_thresh, nullptr);

            precond_diag[i] = diag[i] + kii;
        }
    }

    double t_kii = std::chrono::duration<double>(Clock::now() - t_kii_start).count();
    if (verbose >= 1) {
        if (use_psf) {
            double fp_mem_gb = (fp_offsets.size() * 8 + fp_buckets.size() * 4
                              + fp_values_build.size() * 4
                              + fp_values_query.size() * 4) / (1024.0 * 1024.0 * 1024.0);
            double avg_e = n_query > 0
                ? static_cast<double>(total_fp_entries) / n_query / d : 0;
            std::cout << "  K_ii + PSF:    " << std::setprecision(2) << t_kii
                      << " s (" << total_fp_entries << " entries, "
                      << std::setprecision(2) << fp_mem_gb << " GB, ~"
                      << std::setprecision(0) << avg_e << " buckets/det/row)"
                      << std::endl;
        } else {
            std::cout << "  K_ii:          " << std::setprecision(2) << t_kii
                      << " s" << std::endl;
        }
    }

    // ================================================================
    // Dressed matvec lambda: σ = (H_CC + K) · v
    // ================================================================
    size_t matvec_calls = 0;
    double total_build_time = 0.0;
    double total_query_time = 0.0;
    size_t total_build_dets_processed = 0;
    size_t total_query_dets_processed = 0;

    auto dressed_matvec = [&](const double* v, double* sigma, size_t n) {
        ++matvec_calls;

        // --- Step A: H_CC · v ---
        std::fill(sigma, sigma + n, 0.0);
        if (use_cache) {
            matvec_cached<StorageType>(conn_cache, diag, v, sigma, n);
        } else {
            matvec_dressed<StorageType>(ctx, v, sigma, n);
        }

        auto t_build = Clock::now();

        if (use_psf) {
            // === PSF mode: pure arithmetic, no excitation enumeration ===

            // --- Step B: Build sketch S via scatter-add (unsigned fingerprints) ---
            total_build_dets_processed += n_query;
            CountSketch<StorageType> sketch_v(w, d, seed);

            #pragma omp parallel
            {
                std::vector<double> local_S(d * actual_w, 0.0);
                #pragma omp for schedule(static) nowait
                for (int bj = 0; bj < static_cast<int>(n_query); ++bj) {
                    size_t j = query_indices[bj];
                    if (std::abs(v[j]) < 1e-15) continue;
                    double vj = v[j];
                    for (size_t r = 0; r < d; ++r) {
                        size_t start = fp_offsets[bj * d + r];
                        size_t end = fp_offsets[bj * d + r + 1];
                        double* row = local_S.data() + r * actual_w;
                        for (size_t e = start; e < end; ++e) {
                            row[fp_buckets[e]] += vj * static_cast<double>(fp_values_build[e]);
                        }
                    }
                }
                double* S = sketch_v.data();
                #pragma omp critical
                {
                    for (size_t i = 0; i < d * actual_w; ++i) {
                        S[i] += local_S[i];
                    }
                }
            }

            auto t_query = Clock::now();
            total_build_time += std::chrono::duration<double>(t_query - t_build).count();

            // --- Step C: Query K·v via sparse dot product (signed fingerprints) ---
            // Uses fp_values_query which includes sign(D_a), giving correct K*v sign.
            total_query_dets_processed += n_query;
            const double* S = sketch_v.data();

            #pragma omp parallel for schedule(static)
            for (int qi = 0; qi < static_cast<int>(n_query); ++qi) {
                size_t i = query_indices[qi];
                double row_accum[8] = {};
                for (size_t r = 0; r < d; ++r) {
                    size_t start = fp_offsets[qi * d + r];
                    size_t end = fp_offsets[qi * d + r + 1];
                    const double* row = S + r * actual_w;
                    double sum = 0.0;
                    for (size_t e = start; e < end; ++e) {
                        sum += static_cast<double>(fp_values_query[e]) * row[fp_buckets[e]];
                    }
                    row_accum[r] = sum;
                }
                std::vector<double> ra(row_accum, row_accum + d);
                double Kv_i = CountSketch<StorageType>::median_estimate(ra);
                sigma[i] += Kv_i;
            }

            auto t_mv_end = Clock::now();
            total_query_time += std::chrono::duration<double>(t_mv_end - t_query).count();

        } else {
            // === On-the-fly mode: enumerate excitations each matvec ===

            // --- Step B: Build sketch via excitation enumeration ---
            total_build_dets_processed += n_query;
            CountSketch<StorageType> sketch_v(w, d, seed);

            #pragma omp parallel
            {
                std::vector<double> local_S(d * actual_w, 0.0);
                #pragma omp for schedule(dynamic, 64) nowait
                for (int bj = 0; bj < static_cast<int>(n_query); ++bj) {
                    size_t j = query_indices[bj];
                    double vj = v[j];
                    if (std::abs(vj) < 1e-15) continue;

                    double int_thresh = (eps_hc > 0 && std::abs(core_coeffs[j]) > 1e-15)
                        ? eps_hc / std::abs(core_coeffs[j]) : 0.0;
                    auto sums_j = precompute_orbital_sums(
                        core_occ_a[j], core_occ_b[j], eri, n_orb);

                    for_each_excitation_H_diag_fast(core_dets[j], core_diag[j],
                        core_occ_a[j], core_virt_a[j],
                        core_occ_b[j], core_virt_b[j],
                        h1, eri, n_orb, sums_j,
                        [&](const DET& det_a, double H_ja, double H_aa) {
                            if (bloom.maybe_has(det_a)) return;
                            if (std::abs(H_ja) < 1e-15) return;
                            double D_a = energy_var - H_aa;
                            if (std::abs(D_a) < intruder_thresh) return;

                            double f_val = H_ja / std::sqrt(std::abs(D_a));
                            uint64_t dk = CountSketch<StorageType>::det_key(det_a);
                            for (size_t r = 0; r < d; ++r) {
                                size_t bkt = hash_helper.bucket_for(r, dk);
                                double sgn = hash_helper.sign_for(r, dk);
                                local_S[r * actual_w + bkt] += vj * sgn * f_val;
                            }
                        }, int_thresh, nullptr);
                }
                double* S = sketch_v.data();
                #pragma omp critical
                {
                    for (size_t i = 0; i < d * actual_w; ++i) {
                        S[i] += local_S[i];
                    }
                }
            }

            auto t_query = Clock::now();
            total_build_time += std::chrono::duration<double>(t_query - t_build).count();

            // --- Step C: Query sketch via excitation enumeration (with D_a sign) ---
            total_query_dets_processed += n_query;
            const double* S = sketch_v.data();

            #pragma omp parallel for schedule(dynamic, 64)
            for (int qi = 0; qi < static_cast<int>(n_query); ++qi) {
                size_t i = query_indices[qi];

                double int_thresh = (eps_hc > 0 && std::abs(core_coeffs[i]) > 1e-15)
                    ? eps_hc / std::abs(core_coeffs[i]) : 0.0;
                auto sums_i = precompute_orbital_sums(
                    core_occ_a[i], core_occ_b[i], eri, n_orb);

                double row_accum[8] = {};

                for_each_excitation_H_diag_fast(core_dets[i], core_diag[i],
                    core_occ_a[i], core_virt_a[i],
                    core_occ_b[i], core_virt_b[i],
                    h1, eri, n_orb, sums_i,
                    [&](const DET& det_a, double H_ia, double H_aa) {
                        if (bloom.maybe_has(det_a)) return;
                        if (std::abs(H_ia) < 1e-15) return;
                        double D_a = energy_var - H_aa;
                        if (std::abs(D_a) < intruder_thresh) return;

                        // Include sign(D_a) to correctly handle both D_a<0 and D_a>0
                        double sign_D = (D_a < 0) ? -1.0 : 1.0;
                        double f_val = H_ia / std::sqrt(std::abs(D_a));
                        uint64_t dk = CountSketch<StorageType>::det_key(det_a);
                        for (size_t r = 0; r < d; ++r) {
                            size_t bkt = hash_helper.bucket_for(r, dk);
                            double sgn = hash_helper.sign_for(r, dk);
                            row_accum[r] += sign_D * f_val * sgn * S[r * actual_w + bkt];
                        }
                    }, int_thresh, nullptr);

                std::vector<double> ra(row_accum, row_accum + d);
                double Kv_i = CountSketch<StorageType>::median_estimate(ra);
                sigma[i] += Kv_i;
            }

            auto t_mv_end = Clock::now();
            total_query_time += std::chrono::duration<double>(t_mv_end - t_query).count();
        }
    };

    // ================================================================
    // Davidson solve with dressed matvec
    // ================================================================
    auto t_dav_start = Clock::now();

    DavidsonParams dav_params = dav_params_in;
    if (dav_params.max_iter == 0) dav_params.max_iter = 500;
    if (dav_params.max_subspace == 0) dav_params.max_subspace = 40;
    const double det = pt2_config.dressed_davidson_energy_tol;
    if (det > 0.0) {
        dav_params.energy_tol = std::max(dav_params.energy_tol, det);
    }

    std::vector<std::vector<double>> guess;
    guess.push_back(std::vector<double>(core_coeffs.begin(), core_coeffs.end()));

    if (verbose >= 1) {
        std::cout << "\n  Dressed Davidson (residual_tol=" << std::scientific
                  << dav_params.residual_tol << ", energy_tol="
                  << dav_params.energy_tol << std::fixed << ")..." << std::endl;
    }

    auto dav_result = davidson_solve(dressed_matvec, precond_diag, n_core,
                                      dav_params, guess);

    result.time_davidson = std::chrono::duration<double>(
        Clock::now() - t_dav_start).count();

    // ================================================================
    // Assemble result
    // ================================================================
    result.energy_dressed = dav_result.eigenvalues.empty()
                          ? energy_var : dav_result.eigenvalues[0];
    result.davidson_iters = dav_result.n_iters;
    result.davidson_converged = dav_result.converged;

    if (!dav_result.eigenvectors.empty()) {
        result.coefficients_dressed = dav_result.eigenvectors[0];
    }

    result.time_total = std::chrono::duration<double>(
        Clock::now() - t_total_start).count();

    if (verbose >= 1) {
        std::cout << "  Davidson: " << dav_result.n_iters << " iters, "
                  << (dav_result.converged ? "converged" : "NOT converged")
                  << ", |r| = " << std::scientific << dav_result.residual_norm
                  << std::fixed << std::endl;
        std::cout << "  Matvec calls:    " << matvec_calls << std::endl;
        if (matvec_calls > 0) {
            size_t avg_build = total_build_dets_processed / matvec_calls;
            size_t avg_query = total_query_dets_processed / matvec_calls;
            std::cout << "  Build dets/mv:   " << avg_build << "/" << n_core
                      << " (" << std::setprecision(1)
                      << 100.0 * avg_build / n_core << "%)" << std::endl;
            std::cout << "  Query dets/mv:   " << avg_query << "/" << n_core
                      << " (" << std::setprecision(1)
                      << 100.0 * avg_query / n_core << "%)" << std::endl;
        }
        std::cout << "  Build time:      " << std::setprecision(2)
                  << total_build_time << " s" << std::endl;
        std::cout << "  Query time:      " << total_query_time << " s"
                  << std::endl;
        std::cout << "\n  --- Results ---" << std::endl;
        std::cout << std::setprecision(10);
        std::cout << "  E_var:           " << energy_var << std::endl;
        std::cout << "  E_dressed:       " << result.energy_dressed << std::endl;
        std::cout << std::setprecision(2);
        std::cout << "  Time: prep=" << t_prep << "s, davidson="
                  << result.time_davidson << "s, total="
                  << result.time_total << "s" << std::endl;
    }

    return result;
}

// ============================================================================
// Explicit template instantiation
// ============================================================================
template DressedCIResult<uint64_t> compute_dressed_ci_hybrid<uint64_t>(
    const std::vector<DeterminantT<uint64_t>>&,
    const std::vector<double>&, double,
    const std::vector<std::vector<double>>&,
    const std::vector<double>&, int,
    const PT2Config<uint64_t>&,
    const DavidsonParams&);

// 128-bit instantiations
using Bit128 = std::array<uint64_t, 2>;
template DressedCIResult<Bit128> compute_dressed_ci_hybrid<Bit128>(
    const std::vector<DeterminantT<Bit128>>&,
    const std::vector<double>&, double,
    const std::vector<std::vector<double>>&,
    const std::vector<double>&, int,
    const PT2Config<Bit128>&,
    const DavidsonParams&);

}  // namespace fe
}  // namespace trimci_core
