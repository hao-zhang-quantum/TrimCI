// screening.cpp
#include "screening.hpp"
#include <algorithm>
#include <random>
#include <cmath>
#include <unordered_set>
#include <atomic>
#include <iostream>    // 新增：用于输出进度信息
#include <fstream>
#ifdef _OPENMP
#  include <omp.h>
#endif
#include "omp_compat.hpp"
#include "bit_compat.hpp"

namespace trimci_core {

// —— 单激发相位 —— 
static int single_phase(uint64_t mask, int i, int p) {
    int n_i = __builtin_popcountll(mask & ((1ULL<<i)-1));
    uint64_t m1 = mask & ~(1ULL<<i);
    int n_p = __builtin_popcountll(m1 & ((1ULL<<p)-1));
    return ((n_i + n_p) % 2) ? -1 : 1;
}

// —— 双激发相位 —— 
static int double_phase(uint64_t mask, int i, int j, int p, int q) {
    int n_i = __builtin_popcountll(mask & ((1ULL<<i)-1));
    uint64_t m1 = mask & ~(1ULL<<i);
    int n_j = __builtin_popcountll(m1 & ((1ULL<<j)-1));
    uint64_t m2 = m1 & ~(1ULL<<j);
    int n_p = __builtin_popcountll(m2 & ((1ULL<<p)-1));
    uint64_t m3 = m2 | (1ULL<<p);
    int n_q = __builtin_popcountll(m3 & ((1ULL<<q)-1));
    return ((n_i + n_j + n_p + n_q) % 2) ? -1 : 1;
}

// —— 1) 预计算双激发表 —— 
DoubleExcTable precompute_double_exc_table(
    int n_orb,
    const std::vector<std::vector<std::vector<std::vector<double>>>>& eri,
    double thr
) {
    DoubleExcTable table;
    for (int i = 0; i < n_orb; ++i) {
        for (int j = i+1; j < n_orb; ++j) {
            std::vector<std::tuple<int,int,double>> entries;
            for (int p = 0; p < n_orb; ++p) {
                for (int q = p+1; q < n_orb; ++q) {
                    double h_val = eri[i][j][p][q] - eri[i][j][q][p];
                    if (std::abs(h_val) > thr) {
                        entries.emplace_back(p, q, h_val);
                    }
                }
            }
            std::sort(entries.begin(), entries.end(),
                      [](auto& a, auto& b){
                          return std::abs(std::get<2>(a))
                               > std::abs(std::get<2>(b));
                      });
            table[{i,j}] = std::move(entries);
        }
    }
    return table;
}

// —— 2) processing parent determinants
std::vector<std::pair<Determinant,double>>
process_parent_worker(
    const Determinant& det,
    int n_orb,
    double thr,
    const DoubleExcTable& table,
    const std::vector<std::vector<double>>& h1,
    const std::vector<std::vector<std::vector<std::vector<double>>>>& eri
) {
    std::vector<std::pair<Determinant,double>> new_pairs;
    auto occ_a = det.getOccupiedAlpha();
    auto occ_b = det.getOccupiedBeta();

    // αα same-spin double excitations
    for (size_t ia = 0; ia < occ_a.size(); ++ia) {
        for (size_t ib = ia+1; ib < occ_a.size(); ++ib) {
            int i = occ_a[ia], j = occ_a[ib];
            for (auto& t : table.at({i,j})) {
                int p,q; double h_val;
                std::tie(p,q,h_val) = t;
                if (((det.alpha >> p) & 1) || ((det.alpha >> q) & 1)) continue;
                auto dj = det.doubleExcite(i, j, p, q, true);
                int ph = double_phase(det.alpha, i, j, p, q);
                new_pairs.emplace_back(dj, ph * h_val);
            }
        }
    }

    // ββ same-spin double excitations
    for (size_t ia = 0; ia < occ_b.size(); ++ia) {
        for (size_t ib = ia+1; ib < occ_b.size(); ++ib) {
            int i = occ_b[ia], j = occ_b[ib];
            for (auto& t : table.at({i,j})) {
                int p,q; double h_val;
                std::tie(p,q,h_val) = t;
                if (((det.beta >> p) & 1) || ((det.beta >> q) & 1)) continue;
                auto dj = det.doubleExcite(i, j, p, q, false);
                int ph = double_phase(det.beta, i, j, p, q);
                new_pairs.emplace_back(dj, ph * h_val);
            }
        }
    }
    // Mixed αβ double excitations
    for (int i : occ_a) {
        for (int j : occ_b) {
            for (int p = 0; p < n_orb; ++p) {
                if ((det.alpha >> p) & 1) continue;
                for (int q = 0; q < n_orb; ++q) {
                    if ((det.beta  >> q) & 1) continue;
                    double h_val = eri[i][p][j][q];
                    if (std::abs(h_val) <= thr) continue;
                    Determinant dj(
                        (det.alpha & ~(1ULL<<i)) | (1ULL<<p),
                        (det.beta  & ~(1ULL<<j)) | (1ULL<<q)
                    );
                    int pa = single_phase(det.alpha, i, p);
                    int pb = single_phase(det.beta,  j, q);
                    new_pairs.emplace_back(dj, pa * pb * h_val);
                }
            }
        }
    }

    // α single excitations
    for (int i : occ_a) {
        for (int p = 0; p < n_orb; ++p) {
            if ((det.alpha >> p) & 1) continue;
            auto dj = det.singleExcite(i, p, true);
            double hij = compute_H_ij(det, dj, h1, eri);
            if (std::abs(hij) > thr) new_pairs.emplace_back(dj, hij);
        }
    }
    // β single excitations

    for (int j : occ_b) {
        for (int q = 0; q < n_orb; ++q) {
            if ((det.beta >> q) & 1) continue;
            auto dj = det.singleExcite(j, q, false);
            double hij = compute_H_ij(det, dj, h1, eri);
            if (std::abs(hij) > thr) new_pairs.emplace_back(dj, hij);
        }
    }

    return new_pairs;
}

// —— 3) 并行筛选主流程：直接返回完整的池和调整后的threshold —— 
std::pair<std::vector<Determinant>, double>
pool_build(
    const std::vector<Determinant>& initial_pool,
    const std::vector<double>& initial_coeffs,
    int n_orb,
    const std::vector<std::vector<double>>& h1,
    const std::vector<std::vector<std::vector<std::vector<double>>>>& eri,
    double threshold,
    size_t target_size,
    HijCache& cache,
    const std::string& cache_file,
    int max_rounds
) {
    std::cout << "[PoolBuild] Starting pool build: "
              << "target_size=" << target_size
              << ", threshold=" << threshold
              << std::endl;

    auto table = precompute_double_exc_table(n_orb, eri, threshold);
    std::unordered_set<Determinant> pool_set(
        initial_pool.begin(), initial_pool.end()
    );
    pool_set.reserve(target_size);
    std::vector<Determinant> frontier = initial_pool;
    
    // --- 控制是否使用 coeff_map ---
    bool use_coeffs = !initial_coeffs.empty();
    std::cout << "[PoolBuild] use_coeffs: " << use_coeffs << std::endl;

    std::unordered_map<Determinant, double> coeff_map;
    if (use_coeffs) {
        coeff_map.reserve(initial_pool.size());
        for (size_t i = 0; i < initial_pool.size(); ++i) {
            coeff_map[initial_pool[i]] = initial_coeffs[i];
        }
    }

    int round = 1;
    std::atomic<bool> reached{false};
    double min_threshold = threshold * 1e-8;  // Minimum threshold to prevent infinite loop
    size_t prev_pool_size = pool_set.size();
    int stagnant_rounds = 0;
    const int max_stagnant_rounds = 1000;  // Maximum rounds without progress


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
                frontier = std::vector<Determinant>(initial_pool.begin(), initial_pool.end());
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

        std::vector<Determinant> new_frontier;

        // 每个线程局部 coeff_map
        // 只有在 use_coeffs 时才创建 local_maps
        std::vector<std::unordered_map<Determinant,double>> local_coeff_maps;
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

                auto locals = process_parent_worker(det, n_orb, local_threshold, table, h1, eri);

                for (auto& pr : locals) {
                    const auto& dj  = pr.first;
                    double hij = pr.second;

                    if (std::abs(hij) > local_threshold) {
                        bool inserted = false;
                        #pragma omp critical(pool_update)
                        {
                            if (pool_set.insert(dj).second) {
                                cache[pair_key(det, dj)] = hij;
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

    std::cout << "[PoolBuild] Finished pool size="
              << pool_set.size()
              << ", final threshold=" << threshold
              << std::endl;

    return std::make_pair(std::vector<Determinant>(pool_set.begin(), pool_set.end()), threshold);

    // // --- 保证最终大小 = target_size ---
    // std::vector<Determinant> result;
    // result.reserve(target_size);

    // // 先放入 initial_pool
    // result.insert(result.end(), initial_pool.begin(), initial_pool.end());
    // std::unordered_set<Determinant> init_set(initial_pool.begin(), initial_pool.end());

    // // 再放入 pool_set - frontier
    // std::unordered_set<Determinant> frontier_set(frontier.begin(), frontier.end());
    // for (const auto& det : pool_set) {
    //     if (!init_set.count(det) && !frontier_set.count(det)) {
    //         result.push_back(det);
    //     }
    // }

    // size_t remaining = (target_size > result.size())
    //                 ? (target_size - result.size())
    //                 : 0;

    // if (remaining > 0) {
    //     // 🔹 从 frontier 中随机挑选
    //     std::vector<Determinant> candidates;
    //     candidates.reserve(frontier.size());
    //     for (auto& det : frontier) {
    //         if (!init_set.count(det)) {
    //             candidates.push_back(det);
    //         }
    //     }

    //     std::random_device rd;
    //     std::mt19937 gen(rd());
    //     std::shuffle(candidates.begin(), candidates.end(), gen);

    //     if (candidates.size() < remaining) {
    //         remaining = candidates.size();  // 不够就全放
    //     }
    //     result.insert(result.end(), candidates.begin(), candidates.begin() + remaining);
    // }

    // // ✅ 最终返回精简后的 result（保留了所有 earlier-round dets + 随机选取 last-round dets）
    // return result;
}



} // namespace trimci_core
