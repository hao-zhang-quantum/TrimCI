#include "screening_scalable.hpp"
#include "hamiltonian_scalable.hpp"
#include "screening.hpp"
#include <algorithm>
#include <random>
#include <cmath>
#include <unordered_set>
#include <atomic>
#include <iostream>
#include <fstream>
#ifdef _OPENMP
#  include <omp.h>
#endif
#include "omp_compat.hpp"

namespace trimci_core {

// Precompute double excitation table (same implementation for all types)


// Template implementation for processing parent determinants
template<typename StorageType>
std::vector<std::pair<DeterminantT<StorageType>, double>>
process_parent_worker_t(
    const DeterminantT<StorageType>& det,
    int n_orb,
    double thr,
    const DoubleExcTable& table,
    const std::vector<std::vector<double>>& h1,
    const std::vector<std::vector<std::vector<std::vector<double>>>>& eri
) {
    std::vector<std::pair<DeterminantT<StorageType>, double>> new_pairs;
    auto occ_a = det.getOccupiedAlpha();
    auto occ_b = det.getOccupiedBeta();

    // αα same-spin double excitations
    for (size_t ia = 0; ia < occ_a.size(); ++ia) {
        for (size_t ib = ia+1; ib < occ_a.size(); ++ib) {
            int i = occ_a[ia], j = occ_a[ib];
            auto it = table.find({i,j});
            if (it == table.end()) continue;
            
            for (const auto& t : it->second) {
                int p, q;
                double h_val;
                std::tie(p, q, h_val) = t;
                
                // Check if orbitals p and q are unoccupied in alpha
                if (BitOps<StorageType>::get_bit(det.alpha, p) || 
                    BitOps<StorageType>::get_bit(det.alpha, q)) continue;
                
                auto dj = det.doubleExcite(i, j, p, q, true);
                int ph = detail::double_phase_t(det.alpha, i, j, p, q);
                new_pairs.emplace_back(dj, ph * h_val);
            }
        }
    }
    
    // ββ same-spin double excitations
    for (size_t ia = 0; ia < occ_b.size(); ++ia) {
        for (size_t ib = ia+1; ib < occ_b.size(); ++ib) {
            int i = occ_b[ia], j = occ_b[ib];
            auto it = table.find({i,j});
            if (it == table.end()) continue;
            
            for (const auto& t : it->second) {
                int p, q;
                double h_val;
                std::tie(p, q, h_val) = t;
                
                // Check if orbitals p and q are unoccupied in beta
                if (BitOps<StorageType>::get_bit(det.beta, p) || 
                    BitOps<StorageType>::get_bit(det.beta, q)) continue;
                
                auto dj = det.doubleExcite(i, j, p, q, false);
                int ph = detail::double_phase_t(det.beta, i, j, p, q);
                new_pairs.emplace_back(dj, ph * h_val);
            }
        }
    }
    
    // Mixed αβ double excitations
    for (int i : occ_a) {
        for (int j : occ_b) {
            for (int p = 0; p < n_orb; ++p) {
                if (BitOps<StorageType>::get_bit(det.alpha, p)) continue;
                for (int q = 0; q < n_orb; ++q) {
                    if (BitOps<StorageType>::get_bit(det.beta, q)) continue;
                    double h_val = eri[i][p][j][q];
                    if (std::abs(h_val) <= thr) continue;
                    
                    // Create new determinant
                    StorageType new_alpha = det.alpha;
                    StorageType new_beta = det.beta;
                    BitOps<StorageType>::clear_bit(new_alpha, i);
                    BitOps<StorageType>::set_bit(new_alpha, p);
                    BitOps<StorageType>::clear_bit(new_beta, j);
                    BitOps<StorageType>::set_bit(new_beta, q);
                    
                    DeterminantT<StorageType> dj(new_alpha, new_beta);
                    int pa = detail::single_phase_t(det.alpha, i, p);
                    int pb = detail::single_phase_t(det.beta, j, q);
                    new_pairs.emplace_back(dj, pa * pb * h_val);
                }
            }
        }
    }
    
    // α single excitations
    for (int i : occ_a) {
        for (int p = 0; p < n_orb; ++p) {
            if (BitOps<StorageType>::get_bit(det.alpha, p)) continue;
            auto dj = det.singleExcite(i, p, true);
            double hij = compute_H_ij_t(det, dj, h1, eri);
            if (std::abs(hij) > thr) new_pairs.emplace_back(dj, hij);
        }
    }
    
    // β single excitations
    for (int j : occ_b) {
        for (int q = 0; q < n_orb; ++q) {
            if (BitOps<StorageType>::get_bit(det.beta, q)) continue;
            auto dj = det.singleExcite(j, q, false);
            double hij = compute_H_ij_t(det, dj, h1, eri);
            if (std::abs(hij) > thr) new_pairs.emplace_back(dj, hij);
        }
    }

    return new_pairs;
}

// Explicit instantiations
template std::vector<std::pair<DeterminantT<uint64_t>, double>>
process_parent_worker_t<uint64_t>(const DeterminantT<uint64_t>&, int, double, const DoubleExcTable&,
                                  const std::vector<std::vector<double>>&,
                                  const std::vector<std::vector<std::vector<std::vector<double>>>>&);

template std::vector<std::pair<DeterminantT<std::array<uint64_t, 2>>, double>>
process_parent_worker_t<std::array<uint64_t, 2>>(const DeterminantT<std::array<uint64_t, 2>>&, int, double, const DoubleExcTable&,
                                                 const std::vector<std::vector<double>>&,
                                                 const std::vector<std::vector<std::vector<std::vector<double>>>>&);

template std::vector<std::pair<DeterminantT<std::array<uint64_t, 3>>, double>>
process_parent_worker_t<std::array<uint64_t, 3>>(const DeterminantT<std::array<uint64_t, 3>>&, int, double, const DoubleExcTable&,
                                                 const std::vector<std::vector<double>>&,
                                                 const std::vector<std::vector<std::vector<std::vector<double>>>>&);

// Template implementation for pool building
template<typename StorageType>
std::pair<std::vector<DeterminantT<StorageType>>, double>
pool_build_t(
    const std::vector<DeterminantT<StorageType>>& initial_pool,
    const std::vector<double>& initial_coeffs,
    int n_orb,
    const std::vector<std::vector<double>>& h1,
    const std::vector<std::vector<std::vector<std::vector<double>>>>& eri,
    double threshold,
    size_t target_size,
    HijCacheT<StorageType>& cache,
    const std::string& cache_file,
    int max_rounds
) {
    std::cout << "[PoolBuild] Starting pool build: "
              << "target_size=" << target_size
              << ", threshold=" << threshold
              << std::endl;

    auto table = precompute_double_exc_table(n_orb, eri, threshold);
    std::unordered_set<DeterminantT<StorageType>> pool_set(
        initial_pool.begin(), initial_pool.end()
    );
    pool_set.reserve(target_size);
    std::vector<DeterminantT<StorageType>> frontier = initial_pool;
    
    // Control whether to use coefficient map
    bool use_coeffs = !initial_coeffs.empty();
    std::cout << "[PoolBuild] use_coeffs: " << use_coeffs << std::endl;

    std::unordered_map<DeterminantT<StorageType>, double> coeff_map;
    if (use_coeffs) {
        coeff_map.reserve(initial_pool.size());
        for (size_t i = 0; i < initial_pool.size(); ++i) {
            coeff_map[initial_pool[i]] = initial_coeffs[i];
        }
    }

    int round = 1;
    std::atomic<bool> reached{false};
    double min_threshold = threshold * 1e-8;
    size_t prev_pool_size = pool_set.size();
    int stagnant_rounds = 0;
    const int max_stagnant_rounds = 1000;

    while (pool_set.size() < target_size) {
        if (frontier.empty() || (max_rounds > 0 && round > max_rounds)) {
            if (pool_set.size() < target_size) {
                // Check if threshold is too small or no progress is being made
                if (threshold < min_threshold) {
                    std::cout << "[PoolBuild] threshold too small (" << threshold 
                              << " < " << min_threshold << "), stopping to prevent infinite loop." << std::endl;
                    break;
                }
                if (pool_set.size() == prev_pool_size) {
                    stagnant_rounds++;
                    if (stagnant_rounds >= max_stagnant_rounds) {
                        std::cout << "[PoolBuild] no progress for " << max_stagnant_rounds 
                                  << " threshold reductions, stopping." << std::endl;
                        break;
                    }
                } else {
                    stagnant_rounds = 0;
                    prev_pool_size = pool_set.size();
                }
                
                // 🔹 降低 threshold 并重启
                threshold *= 0.9;
                round = 1;
                frontier = std::vector<DeterminantT<StorageType>>(initial_pool.begin(), initial_pool.end());
                std::cout << "[PoolBuild] threshold relaxed to " << threshold
                          << ", restarting from initial pool=" << initial_pool.size()
                          << std::endl;
            } else {
                break;
            }
        }

        std::cout << "[PoolBuild] Round " << round
                  << ": pool_size=" << pool_set.size()
                  << ", frontier_size=" << frontier.size()
                  << std::endl;

        std::vector<DeterminantT<StorageType>> new_frontier;

        // 每个线程局部 coeff_map
        // 只有在 use_coeffs 时才创建 local_maps
        std::vector<std::unordered_map<DeterminantT<StorageType>,double>> local_coeff_maps;
        if (use_coeffs) {
            #ifdef _OPENMP
            local_coeff_maps.resize(omp_get_max_threads());
            #else
            local_coeff_maps.resize(1);
            #endif
        }

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            auto* local_map = use_coeffs ? &local_coeff_maps[tid] : nullptr;

            #pragma omp for schedule(dynamic)
            for (int idx = 0; idx < (int)frontier.size(); ++idx) {
                //if (reached.load()) continue;

                auto det = frontier[idx];
                double ci = 1.0;
                if (use_coeffs) {
                    auto it = coeff_map.find(det);
                    ci = (it != coeff_map.end()) ? it->second : 1.0;
                }
                double local_threshold = threshold / std::max(std::abs(ci), 1e-12);

                auto locals = process_parent_worker_t(det, n_orb, local_threshold, table, h1, eri);

                for (auto& pr : locals) {
                    const auto& dj  = pr.first;
                    double hij = pr.second;

                    if (std::abs(hij) > local_threshold) {
                        bool inserted = false;
                        #pragma omp critical(pool_update)
                        {
                            if (pool_set.insert(dj).second) {
                                cache[pair_key_t(det, dj)] = hij;
                                new_frontier.push_back(dj);
                                inserted = true;
                            }
                        }

                        if (inserted && use_coeffs) {
                            //double est_cj = std::abs(hij) * std::abs(ci);
                            double est_cj = std::abs(ci);
                            auto it = local_map->find(dj);
                            if (it == local_map->end()) {
                                (*local_map)[dj] = est_cj;
                            } else {
                                //it->second += est_cj;
                                it->second = std::max(it->second, est_cj);
                            }
                        }

                        if (pool_set.size() >= target_size) {
                            reached.store(true);
                        }
                    }
                }
            }
        } // omp parallel 结束

        // 🔹 最后归并 local_map 到 coeff_map（仅当 use_coeffs）
        if (use_coeffs) {
            for (auto& lm : local_coeff_maps) {
                for (auto& kv : lm) {
                    auto it = coeff_map.find(kv.first);
                    if (it == coeff_map.end()) {
                        coeff_map[kv.first] = kv.second;
                    } else {
                        it->second = std::max(it->second, kv.second);
                    }
                }
            }
        }

        if (reached.load()) {
            std::cout << "[PoolBuild] target_size reached, stopping.\n";
            frontier.swap(new_frontier);
            break;
        }

        frontier.swap(new_frontier);
        ++round;
    }

    std::cout << "[PoolBuild] Final pool size: " << pool_set.size() 
              << ", final threshold: " << threshold << std::endl;

    return {std::vector<DeterminantT<StorageType>>(pool_set.begin(), pool_set.end()), threshold};
}

// Explicit instantiations
template std::pair<std::vector<DeterminantT<uint64_t>>, double>
pool_build_t<uint64_t>(const std::vector<DeterminantT<uint64_t>>&, const std::vector<double>&, int,
                       const std::vector<std::vector<double>>&,
                       const std::vector<std::vector<std::vector<std::vector<double>>>>&,
                       double, size_t, HijCacheT<uint64_t>&, const std::string&, int);

template std::pair<std::vector<DeterminantT<std::array<uint64_t, 2>>>, double>
pool_build_t<std::array<uint64_t, 2>>(const std::vector<DeterminantT<std::array<uint64_t, 2>>>&, const std::vector<double>&, int,
                                      const std::vector<std::vector<double>>&,
                                      const std::vector<std::vector<std::vector<std::vector<double>>>>&,
                                      double, size_t, HijCacheT<std::array<uint64_t, 2>>&, const std::string&, int);

template std::pair<std::vector<DeterminantT<std::array<uint64_t, 3>>>, double>
pool_build_t<std::array<uint64_t, 3>>(const std::vector<DeterminantT<std::array<uint64_t, 3>>>&, const std::vector<double>&, int,
                                      const std::vector<std::vector<double>>&,
                                      const std::vector<std::vector<std::vector<std::vector<double>>>>&,
                                      double, size_t, HijCacheT<std::array<uint64_t, 3>>&, const std::string&, int);

// Backward compatible version
std::pair<std::vector<Determinant>, double>
pool_build_compat(
    const std::vector<Determinant>& initial_pool,
    const std::vector<double>& initial_coeff,
    int n_orb,
    const std::vector<std::vector<double>>& h1,
    const std::vector<std::vector<std::vector<std::vector<double>>>>& eri,
    double threshold,
    size_t target_size,
    HijCache& cache,
    const std::string& cache_file,
    int max_rounds
) {
    return ::trimci_core::pool_build(initial_pool, initial_coeff, n_orb, h1, eri, 
                                    threshold, target_size, cache, cache_file, max_rounds);
}

} // namespace trimci_core