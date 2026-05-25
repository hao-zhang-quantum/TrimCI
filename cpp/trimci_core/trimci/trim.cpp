#include "trim.hpp"
#include "hamiltonian.hpp"
#include <random>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <fstream>
#include <iostream>
#include <chrono>
#include <numeric>
#include <iomanip>

#ifdef _OPENMP
#  include <omp.h>
#endif
#include "omp_compat.hpp"

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/MatrixFunctions>  // For matrix exponential
#include "bit_compat.hpp"
#include <cstdint>

// Timestamp-based seed helpers
static inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
static inline uint64_t timestamp_seed() {
    uint64_t ts = static_cast<uint64_t>(
        std::chrono::system_clock::now().time_since_epoch().count()
    );
    return splitmix64(ts);
}

namespace trimci_core {

using SpMat   = Eigen::SparseMatrix<double, Eigen::RowMajor>;
using Triplet = Eigen::Triplet<double>;
using Vec     = Eigen::VectorXd;
using Mat     = Eigen::MatrixXd;

// =================================================================================
// 1) partition_pool_t
// =================================================================================
template<typename StorageType>
std::vector<std::vector<DeterminantT<StorageType>>>
partition_pool_t(const std::vector<DeterminantT<StorageType>>& pool, int m)
{
    size_t n = pool.size();
    if (m <= 0) throw std::invalid_argument("m must be positive");

    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);

    // Use thread_local RNG for thread safety.
    static thread_local std::mt19937_64 rng{ timestamp_seed() };
    std::shuffle(idx.begin(), idx.end(), rng);

    std::vector<std::vector<DeterminantT<StorageType>>> subsets(m);
    size_t base = (n + m - 1) / m;
    for (auto& sub : subsets) sub.reserve(base);

    int num_threads = omp_get_max_threads();
    std::vector<std::vector<std::vector<DeterminantT<StorageType>>>> thread_subsets(
        num_threads, std::vector<std::vector<DeterminantT<StorageType>>>(m)
    );
    for (int t = 0; t < num_threads; ++t) {
        for (auto& sub : thread_subsets[t]) sub.reserve(base / num_threads + 1);
    }

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        #pragma omp for
        for (int i = 0; i < static_cast<int>(n); ++i) {
             int group = i % m;
            thread_subsets[tid][group].push_back(pool[idx[static_cast<size_t>(i)]]);
        }
    }

    for (int t = 0; t < num_threads; ++t)
        for (int j = 0; j < m; ++j)
            subsets[j].insert(subsets[j].end(),
                              thread_subsets[t][j].begin(),
                              thread_subsets[t][j].end());
    
    return subsets;
}

// Wrapper
std::vector<std::vector<Determinant>>
partition_pool(const std::vector<Determinant>& pool, int m) {
    return partition_pool_t<uint64_t>(pool, m);
}

// =================================================================================
// 1b) build_alpha_topology — Pre-compute alpha single-excitation connectivity
// =================================================================================
template<typename StorageType>
static AlphaTopologyT<StorageType> build_alpha_topology(
    const std::vector<DeterminantT<StorageType>>& dets,
    int n_orb
) {
    // Collect unique alpha keys
    std::unordered_map<StorageType, bool, StorageHashT<StorageType>> alpha_set;
    alpha_set.reserve(dets.size());
    for (const auto& d : dets) alpha_set[d.alpha] = true;

    // Collect keys into vector for parallel iteration
    std::vector<StorageType> all_keys;
    all_keys.reserve(alpha_set.size());
    for (auto& [key, _] : alpha_set) all_keys.push_back(key);
    int n_keys = static_cast<int>(all_keys.size());

    // Parallel: enumerate single excitations per key, check against alpha_set (read-only)
    std::vector<std::vector<StorageType>> per_key_conns(n_keys);

    #pragma omp parallel for schedule(dynamic, 64)
    for (int ki = 0; ki < n_keys; ++ki) {
        auto& conns = per_key_conns[ki];
        const auto& alpha_key = all_keys[ki];
        if constexpr (std::is_same_v<StorageType, uint64_t>) {
            uint64_t c = alpha_key;
            uint64_t holes = (~c) & (n_orb == 64 ? ~0ULL : ((1ULL << n_orb) - 1));
            while (c) {
                int p = ctz64(c);
                c &= ~(1ULL << p);
                uint64_t h = holes;
                while (h) {
                    int q = ctz64(h);
                    h &= ~(1ULL << q);
                    uint64_t new_conf = (alpha_key ^ (1ULL << p)) | (1ULL << q);
                    if (alpha_set.count(new_conf)) {
                        conns.push_back(new_conf);
                    }
                }
            }
        } else {
            for (int p = 0; p < n_orb; ++p) {
                if (BitOps<StorageType>::get_bit(alpha_key, p)) {
                    for (int q = 0; q < n_orb; ++q) {
                        if (!BitOps<StorageType>::get_bit(alpha_key, q)) {
                            StorageType new_conf = alpha_key;
                            BitOps<StorageType>::flip_bit(new_conf, p);
                            BitOps<StorageType>::flip_bit(new_conf, q);
                            if (alpha_set.count(new_conf)) {
                                conns.push_back(new_conf);
                            }
                        }
                    }
                }
            }
        }
    }

    // Transfer to topology map (sequential)
    AlphaTopologyT<StorageType> topology;
    topology.reserve(alpha_set.size());
    for (int ki = 0; ki < n_keys; ++ki) {
        if (!per_key_conns[ki].empty()) {
            topology[all_keys[ki]] = std::move(per_key_conns[ki]);
        }
    }
    return topology;
}

// =================================================================================
// 2a) build_hamiltonian_sparse_t — Build sparse H matrix from determinants
// =================================================================================
template<typename StorageType>
SpMat build_hamiltonian_sparse_t(
    const std::vector<DeterminantT<StorageType>>& dets,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    bool quantization,
    int n_orb,
    const IntegralSparsityInfo* sparsity,
    int verbosity,
    int core_start,
    const std::vector<Triplet>* precomputed_core_triplets,
    const AlphaTopologyT<StorageType>* alpha_topology
) {
    auto t0 = std::chrono::high_resolution_clock::now();
    int dim = static_cast<int>(dets.size());

    using SHash = StorageHashT<StorageType>;

    // Build lookup maps (group dets by alpha / beta string)
    std::unordered_map<StorageType, std::vector<int>, SHash> alpha_map;
    std::unordered_map<StorageType, std::vector<int>, SHash> beta_map;

    alpha_map.reserve(dim);
    beta_map.reserve(dim);

    for(int i = 0; i < dim; ++i) {
        alpha_map[dets[i].alpha].push_back(i);
        beta_map[dets[i].beta].push_back(i);
    }

    // Sort map values for early termination via lower_bound in Cases
    for (auto& [key, indices] : alpha_map) std::sort(indices.begin(), indices.end());
    for (auto& [key, indices] : beta_map) std::sort(indices.begin(), indices.end());

    auto t1_maps = std::chrono::high_resolution_clock::now();

    // Helper to generate single excitations (used for Case 3)
    auto for_each_single = [&](const StorageType& config, auto&& callback) {
        if constexpr (std::is_same_v<StorageType, uint64_t>) {
             uint64_t c = config;
             uint64_t holes = (~c) & (n_orb == 64 ? ~0ULL : ((1ULL << n_orb) - 1));
             while (c) {
                 int p = ctz64(c);
                 c &= ~(1ULL << p);
                 uint64_t h = holes;
                 while(h) {
                     int q = ctz64(h);
                     h &= ~(1ULL << q);
                     uint64_t new_conf = (config ^ (1ULL << p)) | (1ULL << q);
                     callback(new_conf);
                 }
             }
        } else {
             for(int p=0; p<n_orb; ++p) {
                 if(BitOps<StorageType>::get_bit(config, p)) {
                     for(int q=0; q<n_orb; ++q) {
                         if(!BitOps<StorageType>::get_bit(config, q)) {
                             StorageType new_conf = config;
                             BitOps<StorageType>::flip_bit(new_conf, p);
                             BitOps<StorageType>::flip_bit(new_conf, q);
                             callback(new_conf);
                         }
                     }
                 }
             }
        }
    };

    const bool eri_diag = sparsity && sparsity->eri_is_diagonal;

    // Element-wise XOR helper (works for both uint64_t and std::array<uint64_t, N>)
    auto xor_storage = [](const StorageType& a, const StorageType& b) -> StorageType {
        if constexpr (std::is_same_v<StorageType, uint64_t>) {
            return a ^ b;
        } else {
            StorageType result;
            for (size_t k = 0; k < a.size(); ++k)
                result[k] = a[k] ^ b[k];
            return result;
        }
    };

    // Pre-build Case 3 connections: for each unique alpha key, find connected alpha keys.
    // When alpha_topology is provided, resolve pre-computed connectivity instead of
    // re-enumerating single excitations (saves K × n_singles hash lookups per call).
    // The resolution does K × avg_connections hash lookups in alpha_map. Parallelizing
    // over alpha keys distributes these lookups across threads.
    std::unordered_map<StorageType, std::vector<const std::vector<int>*>, SHash> alpha_connections;
    if (!eri_diag) {
        // Collect alpha keys for parallel iteration
        std::vector<StorageType> alpha_keys_vec;
        alpha_keys_vec.reserve(alpha_map.size());
        for (auto& [key, _] : alpha_map) alpha_keys_vec.push_back(key);

        // Build connections in parallel per alpha key
        int n_akeys = static_cast<int>(alpha_keys_vec.size());
        std::vector<std::vector<const std::vector<int>*>> conns_per_key(n_akeys);

        if (alpha_topology) {
            #pragma omp parallel for schedule(dynamic, 64)
            for (int ki = 0; ki < n_akeys; ++ki) {
                auto topo_it = alpha_topology->find(alpha_keys_vec[ki]);
                if (topo_it != alpha_topology->end()) {
                    auto& conns = conns_per_key[ki];
                    for (const auto& connected_key : topo_it->second) {
                        auto it = alpha_map.find(connected_key);
                        if (it != alpha_map.end()) {
                            conns.push_back(&(it->second));
                        }
                    }
                }
            }
        } else {
            #pragma omp parallel for schedule(dynamic, 64)
            for (int ki = 0; ki < n_akeys; ++ki) {
                auto& conns = conns_per_key[ki];
                for_each_single(alpha_keys_vec[ki], [&](const StorageType& a_new) {
                    auto it = alpha_map.find(a_new);
                    if (it != alpha_map.end()) {
                        conns.push_back(&(it->second));
                    }
                });
            }
        }

        // Transfer to alpha_connections map
        alpha_connections.reserve(alpha_map.size());
        for (int ki = 0; ki < n_akeys; ++ki) {
            if (!conns_per_key[ki].empty()) {
                alpha_connections[alpha_keys_vec[ki]] = std::move(conns_per_key[ki]);
            }
        }
    }

    auto t2_conns = std::chrono::high_resolution_clock::now();

    // Parallel build: group-based scan
    // Use per-thread vectors to avoid omp critical serialization
    int n_threads_max = omp_get_max_threads();
    std::vector<std::vector<Triplet>> thread_triplets(n_threads_max);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto& local_triplets = thread_triplets[tid];
        local_triplets.reserve(dim * 30 / omp_get_num_threads() + 1000);

        #pragma omp for schedule(dynamic)
        for (int i = 0; i < dim; ++i) {

            // Case 1: Same Beta — alpha differs (j <= i, lower triangle only)
            {
                auto it = beta_map.find(dets[i].beta);
                if (it != beta_map.end()) {
                    const auto& indices = it->second;
                    auto end_it = std::upper_bound(indices.begin(), indices.end(), i);
                    // Skip core-core pairs: when i is core, only iterate pool dets (j < core_start)
                    if (core_start >= 0 && i >= core_start) {
                        auto core_end = std::lower_bound(indices.begin(), end_it, core_start);
                        end_it = core_end;
                    }
                    for (auto jt = indices.begin(); jt != end_it; ++jt) {
                        int j = *jt;
                        int diff = detail::HamiltonianBitOps<StorageType>::count_differences(dets[i].alpha, dets[j].alpha);
                        if (diff > 4 || (diff == 4 && eri_diag)) continue;
                        double Hij = compute_H_ij_t(dets[i], dets[j], h1, eri, sparsity);
                        if (std::abs(Hij) > 1e-12) {
                            local_triplets.emplace_back(i, j, Hij);
                        }
                    }
                }
            }

            // Case 2: Same Alpha — beta differs (j < i, lower triangle only)
            {
                auto it = alpha_map.find(dets[i].alpha);
                if (it != alpha_map.end()) {
                    const auto& indices = it->second;
                    auto end_it = std::lower_bound(indices.begin(), indices.end(), i);
                    // Skip core-core pairs
                    if (core_start >= 0 && i >= core_start) {
                        auto core_end = std::lower_bound(indices.begin(), end_it, core_start);
                        end_it = core_end;
                    }
                    for (auto jt = indices.begin(); jt != end_it; ++jt) {
                        int j = *jt;
                        int diff = detail::HamiltonianBitOps<StorageType>::count_differences(dets[i].beta, dets[j].beta);
                        if (diff > 4 || (diff == 4 && eri_diag)) continue;
                        double Hij = compute_H_ij_t(dets[i], dets[j], h1, eri, sparsity);
                        if (std::abs(Hij) > 1e-12) {
                            local_triplets.emplace_back(i, j, Hij);
                        }
                    }
                }
            }

            // Case 3: Mixed doubles (j < i, emit both directions)
            // Inline H_ij: for mixed αβ double, H_ij = phase × eri(m,p,n,q)
            if (!eri_diag) {
                auto conn_it = alpha_connections.find(dets[i].alpha);
                if (conn_it != alpha_connections.end()) {
                    for (auto* partner_indices : conn_it->second) {
                        auto end_it = std::lower_bound(partner_indices->begin(), partner_indices->end(), i);
                        // Skip core-core pairs
                        if (core_start >= 0 && i >= core_start) {
                            auto core_end = std::lower_bound(partner_indices->begin(), end_it, core_start);
                            end_it = core_end;
                        }
                        for (auto jt = partner_indices->begin(); jt != end_it; ++jt) {
                            int j = *jt;
                            int b_diff = detail::HamiltonianBitOps<StorageType>::count_differences(dets[i].beta, dets[j].beta);
                            if (b_diff == 2) {
                                auto diff_a = xor_storage(dets[i].alpha, dets[j].alpha);
                                int da_buf[2];
                                detail::HamiltonianBitOps<StorageType>::storage_to_indices_inline(diff_a, da_buf, 2);
                                int m, p;
                                if constexpr (std::is_same_v<StorageType, uint64_t>) {
                                    bool bit0_in_i = (dets[i].alpha >> da_buf[0]) & 1;
                                    m = bit0_in_i ? da_buf[0] : da_buf[1];
                                    p = bit0_in_i ? da_buf[1] : da_buf[0];
                                } else {
                                    m = BitOps<StorageType>::get_bit(dets[i].alpha, da_buf[0]) ? da_buf[0] : da_buf[1];
                                    p = (m == da_buf[0]) ? da_buf[1] : da_buf[0];
                                }
                                auto diff_b = xor_storage(dets[i].beta, dets[j].beta);
                                int db_buf[2];
                                detail::HamiltonianBitOps<StorageType>::storage_to_indices_inline(diff_b, db_buf, 2);
                                int n, q;
                                if constexpr (std::is_same_v<StorageType, uint64_t>) {
                                    bool bit0_in_i = (dets[i].beta >> db_buf[0]) & 1;
                                    n = bit0_in_i ? db_buf[0] : db_buf[1];
                                    q = bit0_in_i ? db_buf[1] : db_buf[0];
                                } else {
                                    n = BitOps<StorageType>::get_bit(dets[i].beta, db_buf[0]) ? db_buf[0] : db_buf[1];
                                    q = (n == db_buf[0]) ? db_buf[1] : db_buf[0];
                                }
                                int phase = detail::cre_des_sign_t(m, p, dets[j].alpha)
                                          * detail::cre_des_sign_t(n, q, dets[j].beta);
                                double Hij = phase * get_eri(eri, n_orb, m, p, n, q);
                                if (std::abs(Hij) > 1e-12) {
                                    local_triplets.emplace_back(i, j, Hij);
                                }
                            }
                        }
                    }
                }
            }

        } // end for i
    }

    auto t3_parallel = std::chrono::high_resolution_clock::now();

    // Merge per-thread triplets
    size_t total_triplets = 0;
    for (auto& t : thread_triplets) total_triplets += t.size();

    std::vector<Triplet> triplets;
    triplets.reserve(total_triplets + (precomputed_core_triplets ? precomputed_core_triplets->size() : 0));
    for (auto& t : thread_triplets) {
        triplets.insert(triplets.end(), t.begin(), t.end());
        t.clear();
        t.shrink_to_fit();
    }

    // Insert precomputed core-core triplets (offset by core_start)
    if (precomputed_core_triplets && core_start >= 0) {
        for (const auto& t : *precomputed_core_triplets) {
            triplets.emplace_back(t.row() + core_start, t.col() + core_start, t.value());
        }
    }

    auto t4_merge = std::chrono::high_resolution_clock::now();

    // Sparse Matrix Setup
    SpMat H(dim, dim);
    H.setFromTriplets(triplets.begin(), triplets.end());
    if (quantization) {
        Eigen::SparseMatrix<float, Eigen::RowMajor> Hf = H.cast<float>();
        H = Hf.cast<double>();
    }

    auto t5_assemble = std::chrono::high_resolution_clock::now();

    if (verbosity >= 2) {
        auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        std::cout << "  [BuildH] dim=" << dim
                  << " nnz=" << H.nonZeros()
                  << " triplets=" << total_triplets
                  << " alpha_keys=" << alpha_map.size()
                  << " beta_keys=" << beta_map.size()
                  << "\n  [BuildH] maps=" << ms(t0, t1_maps) << "ms"
                  << " conns=" << ms(t1_maps, t2_conns) << "ms"
                  << " parallel=" << ms(t2_conns, t3_parallel) << "ms"
                  << " merge=" << ms(t3_parallel, t4_merge) << "ms"
                  << " assemble=" << ms(t4_merge, t5_assemble) << "ms"
                  << " total=" << ms(t0, t5_assemble) << "ms\n";
    }

    return H;
}

// =================================================================================
// 2b) davidson_solve — Run Davidson on a pre-built sparse H
// =================================================================================
static std::tuple<double, std::vector<double>>
davidson_solve(
    const SpMat& H,
    int dim,
    int max_iter,
    double tol,
    int verbosity,
    const std::vector<double>& initial_guess,
    const std::string& init_strategy
) {
    // Davidson loop — incremental SpMV caching
    // H stores lower triangle only; use selfadjointView for symmetric SpMV
    auto H_sym = H.selfadjointView<Eigen::Lower>();
    Vec H_diag = H.diagonal();
    const int restart_size = std::min(max_iter / 2, 10);
    const int max_subspace = restart_size + 1;
    Mat V = Mat::Zero(dim, max_subspace);
    Mat HV = Mat::Zero(dim, max_subspace);
    Mat H_sub_cached = Mat::Zero(max_subspace, max_subspace);

    Vec v0 = Vec::Zero(dim);

    // Select initialization strategy
    bool used_warm = false;
    if (init_strategy == "lowest_diag" || init_strategy == "lowest_diag_noise") {
        // Start from the det with lowest diagonal element (+ optional noise)
        int min_idx = 0;
        double min_val = H_diag(0);
        for (int i = 1; i < dim; ++i) {
            if (H_diag(i) < min_val) { min_val = H_diag(i); min_idx = i; }
        }
        v0(min_idx) = 1.0;
        if (init_strategy == "lowest_diag_noise") {
            static thread_local std::mt19937_64 rng_ld{timestamp_seed()};
            std::normal_distribution<double> dist(0.0, 0.01);
            for (int i = 0; i < dim; ++i) v0(i) += dist(rng_ld);
        }
        v0.normalize();
        if (verbosity >= 2)
            std::cout << "  [Davidson] Init: " << init_strategy
                      << " (idx=" << min_idx << ", H_ii=" << min_val << ")\n";
    } else if (init_strategy == "random") {
        static thread_local std::mt19937_64 rng_r{timestamp_seed()};
        std::normal_distribution<double> dist(0.0, 1.0);
        for (int i = 0; i < dim; ++i) v0(i) = dist(rng_r);
        v0.normalize();
        if (verbosity >= 2) std::cout << "  [Davidson] Init: random\n";
    } else if (init_strategy == "multi_random") {
        // Run Davidson with multiple random starts, keep best
        static thread_local std::mt19937_64 rng_mr{timestamp_seed()};
        std::normal_distribution<double> dist(0.0, 1.0);
        double best_e = std::numeric_limits<double>::infinity();
        Vec best_v;
        const int n_trials = 5;
        for (int trial = 0; trial < n_trials; ++trial) {
            Vec vt = Vec::Zero(dim);
            for (int i = 0; i < dim; ++i) vt(i) = dist(rng_mr);
            vt.normalize();
            // Quick Davidson with fewer iterations
            auto [e_trial, c_trial] = davidson_solve(H, dim,
                std::min(max_iter, 50), tol, 0, {}, "random");
            if (e_trial < best_e) {
                best_e = e_trial;
                best_v = Vec::Zero(dim);
                for (int i = 0; i < dim; ++i) best_v(i) = c_trial[i];
            }
        }
        v0 = best_v;
        v0.normalize();
        if (verbosity >= 2)
            std::cout << "  [Davidson] Init: multi_random (" << n_trials
                      << " trials, best E=" << best_e << ")\n";
    } else {
        // "warm" (default): use provided initial guess, fallback to random
        if (!initial_guess.empty() && static_cast<int>(initial_guess.size()) == dim) {
            for (int i = 0; i < dim; ++i) v0(i) = initial_guess[i];
            v0.normalize();
            used_warm = true;
            if (verbosity >= 2) std::cout << "  [Davidson] Init: warm start\n";
        } else {
            static thread_local std::mt19937_64 rng_f{timestamp_seed()};
            std::normal_distribution<double> dist(0.0, 1.0);
            for (int i = 0; i < dim; ++i) v0(i) = dist(rng_f);
            v0.normalize();
            if (verbosity >= 2) std::cout << "  [Davidson] Init: random (no warm data)\n";
        }
    }
    V.col(0) = v0;
    HV.col(0) = H_sym * v0;
    H_sub_cached(0, 0) = v0.dot(HV.col(0));

    int current_subspace_size = 1;
    double current_energy = 0.0;
    Vec current_ritz_vec;

    double prev_energy = 0.0;

    for (int iter = 0; iter < max_iter; ++iter) {
        Mat H_sub = H_sub_cached.topLeftCorner(current_subspace_size, current_subspace_size);

        Eigen::SelfAdjointEigenSolver<Mat> sub_solver(H_sub);
        current_energy = sub_solver.eigenvalues()(0);
        Vec sub_ritz_vec = sub_solver.eigenvectors().col(0);

        Mat V_current = V.leftCols(current_subspace_size);
        current_ritz_vec = V_current * sub_ritz_vec;

        Vec Hv = HV.leftCols(current_subspace_size) * sub_ritz_vec;
        Vec residual = Hv - current_energy * current_ritz_vec;
        double res_norm = residual.norm();
        double energy_change = std::abs(current_energy - prev_energy);
        prev_energy = current_energy;

        if (verbosity >= 2) {
            std::cout << "  [Davidson] " << iter << ": E="
                      << std::fixed << std::setprecision(10) << current_energy
                      << " |r|=" << std::scientific << res_norm << std::defaultfloat << "\n";
        }

        if (iter > 0 && res_norm < tol && energy_change < tol * 1e-2) break;

        if (current_subspace_size >= restart_size && current_subspace_size < max_iter && current_subspace_size < dim) {
            double ritz_norm = current_ritz_vec.norm();
            if (ritz_norm > 1e-15) current_ritz_vec /= ritz_norm;
            V.col(0) = current_ritz_vec;
            HV.col(0) = H_sym * current_ritz_vec;
            H_sub_cached(0, 0) = current_ritz_vec.dot(HV.col(0));
            current_subspace_size = 1;
        } else if (current_subspace_size >= max_iter || current_subspace_size >= dim) {
            break;
        }

        const double shift = 0.1;
        Vec correction = Vec::Zero(dim);
        for (int i = 0; i < dim; ++i) {
            double denom = H_diag(i) - current_energy + shift;
            if (std::abs(denom) > 1e-12) correction(i) = -residual(i) / denom;
        }

        if (current_subspace_size > 0) {
            Mat V_proj = V.leftCols(current_subspace_size);
            correction -= V_proj * (V_proj.transpose() * correction);
        }

        double c_norm = correction.norm();
        if (c_norm > 1e-12) {
            correction /= c_norm;
            int k = current_subspace_size;
            V.col(k) = correction;
            HV.col(k) = H_sym * correction;
            for (int j = 0; j <= k; ++j) {
                double val = V.col(j).dot(HV.col(k));
                H_sub_cached(j, k) = val;
                H_sub_cached(k, j) = val;
            }
            current_subspace_size++;
        } else {
             if (iter == 0 && res_norm > 1e-12) {
                 Vec res_normalized = residual / res_norm;
                 int k = current_subspace_size;
                 V.col(k) = res_normalized;
                 HV.col(k) = H_sym * res_normalized;
                 for (int j = 0; j <= k; ++j) {
                     double val = V.col(j).dot(HV.col(k));
                     H_sub_cached(j, k) = val;
                     H_sub_cached(k, j) = val;
                 }
                 current_subspace_size++;
                 continue;
             }
        }
    }

    // Normalize the Ritz vector before returning
    double final_norm = current_ritz_vec.norm();
    if (final_norm > 1e-15) current_ritz_vec /= final_norm;

    std::vector<double> coeffs(dim);
    for (int i = 0; i < dim; ++i) coeffs[i] = current_ritz_vec(i);
    return {current_energy, coeffs};
}

// =================================================================================
// 2c) diagonalize_subspace_davidson_t — wrapper calling build + solve
// =================================================================================
// Template implementation: diagonalize_subspace_davidson_t
template<typename StorageType>
std::tuple<double, std::vector<double>>
diagonalize_subspace_davidson_t(
    const std::vector<DeterminantT<StorageType>>& dets,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    HijCacheT<StorageType>& cache,
    bool quantization,
    int max_iter,
    double tol,
    int verbosity,
    int n_orb,
    const std::vector<double>& initial_guess,
    const IntegralSparsityInfo* sparsity,
    const std::string& init_strategy
) {
    int dim = static_cast<int>(dets.size());
    if (dim == 0) return {0.0, {}};

    // Tiny subspace handling
    if (dim == 1) {
        double Hij = compute_H_ij_t(dets[0], dets[0], h1, eri);
        return {Hij, {1.0}};
    }
    if (dim == 2) {
        double h00 = compute_H_ij_t(dets[0], dets[0], h1, eri);
        double h11 = compute_H_ij_t(dets[1], dets[1], h1, eri);
        double h01 = 0.0;
        int diff = detail::HamiltonianBitOps<StorageType>::count_differences(dets[0].alpha, dets[1].alpha)
                 + detail::HamiltonianBitOps<StorageType>::count_differences(dets[0].beta, dets[1].beta);
        if (diff <= 4) {
            h01 = compute_H_ij_t(dets[0], dets[1], h1, eri);
        }

        const double eps_off = 1e-12;
        if (std::abs(h01) < eps_off) {
            if (h00 <= h11) return {h00, {1.0, 0.0}};
            else            return {h11, {0.0, 1.0}};
        }

        double delta = h11 - h00;
        double sqrt_term = std::sqrt(delta * delta + 4.0 * h01 * h01);
        double E0 = 0.5 * (h00 + h11 - sqrt_term);
        double c0 = 2.0 * h01;
        double c1 = delta + sqrt_term;
        double norm = std::sqrt(c0 * c0 + c1 * c1);
        c0 /= norm; c1 /= norm;
        return {E0, {c0, c1}};
    }

    // Build H then solve
    SpMat H = build_hamiltonian_sparse_t<StorageType>(dets, h1, eri, quantization, n_orb, sparsity, verbosity);
    return davidson_solve(H, dim, max_iter, tol, verbosity, initial_guess, init_strategy);
}

// Wrapper
std::tuple<double, std::vector<double>>
diagonalize_subspace_davidson(
    const std::vector<Determinant>& dets,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    HijCache& cache,
    bool quantization,
    int max_iter,
    double tol,
    int verbosity,
    int n_orb,
    const std::vector<double>& initial_guess
) {
    return diagonalize_subspace_davidson_t<uint64_t>(dets, h1, eri, cache, quantization, max_iter, tol, verbosity, n_orb, initial_guess);
}

// =================================================================================
// 3) select_top_k_dets_t
// =================================================================================
template<typename StorageType>
std::vector<DeterminantT<StorageType>>
select_top_k_dets_t(
    const std::vector<DeterminantT<StorageType>>& dets,
    const std::vector<double>& coeffs,
    size_t k,
    const std::vector<DeterminantT<StorageType>>& core_vec,
    bool keep_core
)
{
    std::unordered_set<DeterminantT<StorageType>> core_set(core_vec.begin(), core_vec.end());
    size_t n = dets.size();
    if (n == 0 || k == 0) return {};

    std::vector<std::pair<double, DeterminantT<StorageType>>> scored;
    scored.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        if (!core_set.empty() && core_set.count(dets[i])) continue;
        scored.emplace_back(std::abs(coeffs[i]), dets[i]);
    }

    if (scored.empty()) return {}; 
    
    if (k > scored.size()) k = scored.size();

    auto mid = scored.begin() + k;
    std::nth_element(scored.begin(), mid, scored.end(),
                     [](auto& a, auto& b) { return a.first > b.first; });

    std::vector<DeterminantT<StorageType>> top;
    if (keep_core) {
        for (const auto& d : core_set) top.push_back(d);
    }
    
    for (size_t i = 0; i < k; ++i) top.push_back(scored[i].second);
    
    return top;
}

// Wrapper
std::vector<Determinant>
select_top_k_dets(
    const std::vector<Determinant>& dets,
    const std::vector<double>& coeffs,
    size_t k,
    const std::vector<Determinant>& core_vec,
    bool keep_core
) {
    return select_top_k_dets_t<uint64_t>(dets, coeffs, k, core_vec, keep_core);
}


// =================================================================================
// 4) run_trim_t
// =================================================================================// Template implementation: run_trim_t
template<typename StorageType>
std::tuple<double, std::vector<DeterminantT<StorageType>>, std::vector<double>>
run_trim_t(
    const std::vector<DeterminantT<StorageType>>& params_initial_pool,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    const std::string& mol_name,
    int n_elec,
    int n_orb,
    const std::vector<int>& group_sizes,
    const std::vector<int>& keep_sizes,
    bool quantization,
    bool save_cache,
    const std::vector<DeterminantT<StorageType>>& external_core_dets,
    double tol,
    int verbosity,
    const IntegralSparsityInfo* precomputed_sparsity,
    const std::vector<double>& core_coeffs,
    const std::string& davidson_init
) {
    if (verbosity >= 1) {
        std::cout << "[Trim] Start. Pool=" << params_initial_pool.size() << "\n";
    }

    HijCacheT<StorageType> cache;
    std::string cache_file;
    std::tie(cache, cache_file) = load_or_create_Hij_cache_t<StorageType>(mol_name, n_elec, n_orb);

    // Use precomputed sparsity info if available, otherwise build
    IntegralSparsityInfo local_sparsity_info;
    const IntegralSparsityInfo* sparsity_ptr = precomputed_sparsity;
    if (!sparsity_ptr) {
        local_sparsity_info = build_sparsity_info(n_orb, h1, eri);
        sparsity_ptr = local_sparsity_info.is_sparse ? &local_sparsity_info : nullptr;
    }
    if (verbosity >= 1 && sparsity_ptr && sparsity_ptr->is_sparse) {
        std::cout << "[Trim] Sparsity detected: h1=" << sparsity_ptr->h1_sparsity
                  << ", eri=" << sparsity_ptr->eri_sparsity
                  << ", diagonal=" << sparsity_ptr->eri_is_diagonal << std::endl;
    }

    auto current_pool = params_initial_pool;
    auto core_dets = external_core_dets;

    // Build core coefficient map for Davidson warm start
    std::unordered_map<DeterminantT<StorageType>, double> core_coeff_map;
    if (!core_coeffs.empty() && core_coeffs.size() == core_dets.size()) {
        for (size_t i = 0; i < core_dets.size(); ++i) {
            core_coeff_map[core_dets[i]] = core_coeffs[i];
        }
    }

    // Helper: construct initial guess for a det set from core coefficients
    auto make_initial_guess = [&](const std::vector<DeterminantT<StorageType>>& dets) -> std::vector<double> {
        if (core_coeff_map.empty()) return {};
        std::vector<double> guess(dets.size(), 0.0);
        bool has_nonzero = false;
        for (size_t i = 0; i < dets.size(); ++i) {
            auto it = core_coeff_map.find(dets[i]);
            if (it != core_coeff_map.end()) {
                guess[i] = it->second;
                has_nonzero = true;
            }
        }
        return has_nonzero ? guess : std::vector<double>{};
    };

    // Pre-compute core-core H triplets once (reused across all rounds and groups)
    int n_core = static_cast<int>(core_dets.size());
    std::vector<Triplet> core_core_triplets;
    if (n_core > 0 && !group_sizes.empty()) {
        auto t_cc_start = std::chrono::high_resolution_clock::now();
        SpMat core_H = build_hamiltonian_sparse_t<StorageType>(
            core_dets, h1, eri, quantization, n_orb, sparsity_ptr, 0);
        core_core_triplets.reserve(core_H.nonZeros());
        for (int k = 0; k < core_H.outerSize(); ++k) {
            for (SpMat::InnerIterator it(core_H, k); it; ++it) {
                core_core_triplets.emplace_back(it.row(), it.col(), it.value());
            }
        }
        if (verbosity >= 1) {
            double dt = std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - t_cc_start).count();
            std::cout << "[Trim] Core-core H precomputed: " << n_core << " dets, "
                      << core_core_triplets.size() << " nnz, " << dt << "s\n";
        }
    }

    // Build core lookup once (core_dets is constant across rounds)
    std::unordered_set<DeterminantT<StorageType>> core_set_lookup(core_dets.begin(), core_dets.end());

    // Rounds
    for (size_t r = 0; r < group_sizes.size(); ++r) {
        int m = group_sizes[r];
        int k = keep_sizes[r];

        if (verbosity >= 1) {
            std::cout << "[Trim] Round " << r+1 << " (m=" << m << ", k=" << k << ")\n";
        }

        // Remove core dets from pool before partitioning to avoid duplicates
        // (pool can contain core dets: initially from pool_build, later from round-end merge)
        std::vector<DeterminantT<StorageType>> pool_no_core;
        pool_no_core.reserve(current_pool.size());
        for (auto& d : current_pool) {
            if (core_set_lookup.find(d) == core_set_lookup.end()) {
                pool_no_core.push_back(d);
            }
        }
        if (verbosity >= 2 && pool_no_core.size() < current_pool.size()) {
            std::cout << "[Trim] Pool: removed " << (current_pool.size() - pool_no_core.size())
                      << " core dets before partitioning\n";
        }

        auto subsets = partition_pool_t(pool_no_core, m);
        // Add core to all subsets (now guaranteed no duplicates)
        for(auto& s : subsets) s.insert(s.end(), core_dets.begin(), core_dets.end());

        // Pre-compute alpha topology for current_pool + core_dets (reused across m groups)
        auto t_topo_start = std::chrono::high_resolution_clock::now();
        std::vector<DeterminantT<StorageType>> all_dets;
        all_dets.reserve(current_pool.size() + core_dets.size());
        all_dets.insert(all_dets.end(), current_pool.begin(), current_pool.end());
        all_dets.insert(all_dets.end(), core_dets.begin(), core_dets.end());
        bool eri_diag = sparsity_ptr && sparsity_ptr->eri_is_diagonal;
        AlphaTopologyT<StorageType> alpha_topo;
        if (!eri_diag) {
            alpha_topo = build_alpha_topology<StorageType>(all_dets, n_orb);
        }
        if (verbosity >= 2) {
            double dt = std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - t_topo_start).count();
            std::cout << "[Trim] Alpha topology: " << alpha_topo.size()
                      << " keys, " << dt << "s\n";
        }

        std::vector<DeterminantT<StorageType>> selected;

        auto t_phase1_start = std::chrono::high_resolution_clock::now();

        // Two-phase approach to avoid nested OMP parallelism:
        // Phase 1: Build H matrices sequentially, each with full OMP parallel
        // Core-core block is precomputed; alpha topology resolves cached connectivity
        std::vector<SpMat> group_H(subsets.size());
        for (size_t i = 0; i < subsets.size(); ++i) {
            if (subsets[i].empty()) continue;
            int core_start_idx = static_cast<int>(subsets[i].size()) - n_core;
            group_H[i] = build_hamiltonian_sparse_t<StorageType>(
                subsets[i], h1, eri, quantization, n_orb, sparsity_ptr, 0,
                (n_core > 0 ? core_start_idx : -1),
                (n_core > 0 ? &core_core_triplets : nullptr),
                (!eri_diag ? &alpha_topo : nullptr));
        }

        auto t_phase2_start = std::chrono::high_resolution_clock::now();

        // Phase 2: Run Davidson + selection in parallel over groups
        #pragma omp parallel
        {
            std::vector<DeterminantT<StorageType>> local_sel;
            #pragma omp for schedule(dynamic)
            for (int i = 0; i < static_cast<int>(subsets.size()); ++i) {
                if (subsets[i].empty()) continue;
                auto guess = make_initial_guess(subsets[i]);
                auto [e, c] = davidson_solve(
                    group_H[i], static_cast<int>(subsets[i].size()),
                    100, tol, 0, guess, davidson_init);
                auto top = select_top_k_dets_t(subsets[i], c, k, core_dets, false);
                local_sel.insert(local_sel.end(), top.begin(), top.end());
            }
            #pragma omp critical
            selected.insert(selected.end(), local_sel.begin(), local_sel.end());
        }
        group_H.clear();  // Free H matrices after use

        auto t_phase2_end = std::chrono::high_resolution_clock::now();
        if (verbosity >= 2) {
            double dt1 = std::chrono::duration<double>(t_phase2_start - t_phase1_start).count();
            double dt2 = std::chrono::duration<double>(t_phase2_end - t_phase2_start).count();
            std::cout << "[Trim] Phase1(H build)=" << dt1 << "s  Phase2(Davidson+sel)=" << dt2 << "s\n";
        }

        selected.insert(selected.end(), core_dets.begin(), core_dets.end());
        std::unordered_set<DeterminantT<StorageType>> uniq(selected.begin(), selected.end());
        current_pool.assign(uniq.begin(), uniq.end());

        if (verbosity >= 1) {
            std::cout << "[Trim] Round " << r+1 << " end. Pool=" << current_pool.size() << "\n";
        }
    }

    // Final diagonalization with warm start
    if (verbosity >= 1) {
        std::cout << "[Trim] Final diagonalization...\n";
    }
    auto final_guess = make_initial_guess(current_pool);
    auto [fe, fc] = diagonalize_subspace_davidson_t(current_pool, h1, eri, cache, quantization, 200, tol, verbosity, n_orb, final_guess, sparsity_ptr, davidson_init);
    if (verbosity >= 1) {
        std::cout << "[Trim] Final E=" << std::fixed << std::setprecision(10) << fe << std::endl;
    }

    if (save_cache) {
        // Cache saving logic (omitted)
    }

    return {fe, current_pool, fc};
}


// Wrapper
std::tuple<double, std::vector<Determinant>, std::vector<double>>
run_trim(
    const std::vector<Determinant>& pool,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    const std::string& mol_name,
    int n_elec,
    int n_orb,
    const std::vector<int>& group_sizes,
    const std::vector<int>& keep_sizes,
    bool quantization,
    bool save_cache,
    const std::vector<Determinant>& external_core_dets,
    double tol,
    int verbosity
) {
    return run_trim_t<uint64_t>(pool, h1, eri, mol_name, n_elec, n_orb, group_sizes, keep_sizes, 
                                quantization, save_cache, external_core_dets, tol, verbosity);
}

// Explicit instantiations
template std::vector<std::vector<DeterminantT<uint64_t>>> partition_pool_t<uint64_t>(const std::vector<DeterminantT<uint64_t>>&, int);
template std::vector<std::vector<DeterminantT<std::array<uint64_t, 2>>>> partition_pool_t<std::array<uint64_t, 2>>(const std::vector<DeterminantT<std::array<uint64_t, 2>>>&, int);
template std::vector<std::vector<DeterminantT<std::array<uint64_t, 3>>>> partition_pool_t<std::array<uint64_t, 3>>(const std::vector<DeterminantT<std::array<uint64_t, 3>>>&, int);

template SpMat build_hamiltonian_sparse_t<uint64_t>(const std::vector<DeterminantT<uint64_t>>&, const std::vector<std::vector<double>>&, const std::vector<double>&, bool, int, const IntegralSparsityInfo*, int, int, const std::vector<Triplet>*, const AlphaTopologyT<uint64_t>*);
template SpMat build_hamiltonian_sparse_t<std::array<uint64_t, 2>>(const std::vector<DeterminantT<std::array<uint64_t, 2>>>&, const std::vector<std::vector<double>>&, const std::vector<double>&, bool, int, const IntegralSparsityInfo*, int, int, const std::vector<Triplet>*, const AlphaTopologyT<std::array<uint64_t, 2>>*);
template SpMat build_hamiltonian_sparse_t<std::array<uint64_t, 3>>(const std::vector<DeterminantT<std::array<uint64_t, 3>>>&, const std::vector<std::vector<double>>&, const std::vector<double>&, bool, int, const IntegralSparsityInfo*, int, int, const std::vector<Triplet>*, const AlphaTopologyT<std::array<uint64_t, 3>>*);

template std::tuple<double, std::vector<double>>
diagonalize_subspace_davidson_t<uint64_t>(const std::vector<DeterminantT<uint64_t>>&, const std::vector<std::vector<double>>&,
                                          const std::vector<double>&, HijCacheT<uint64_t>&, bool, int, double, int, int, const std::vector<double>&, const IntegralSparsityInfo*, const std::string&);
template std::tuple<double, std::vector<double>>
diagonalize_subspace_davidson_t<std::array<uint64_t, 2>>(const std::vector<DeterminantT<std::array<uint64_t, 2>>>&, const std::vector<std::vector<double>>&,
                                                         const std::vector<double>&, HijCacheT<std::array<uint64_t, 2>>&, bool, int, double, int, int, const std::vector<double>&, const IntegralSparsityInfo*, const std::string&);
template std::tuple<double, std::vector<double>>
diagonalize_subspace_davidson_t<std::array<uint64_t, 3>>(const std::vector<DeterminantT<std::array<uint64_t, 3>>>&, const std::vector<std::vector<double>>&,
                                                         const std::vector<double>&, HijCacheT<std::array<uint64_t, 3>>&, bool, int, double, int, int, const std::vector<double>&, const IntegralSparsityInfo*, const std::string&);

template std::vector<DeterminantT<uint64_t>> select_top_k_dets_t<uint64_t>(const std::vector<DeterminantT<uint64_t>>&, const std::vector<double>&, size_t, const std::vector<DeterminantT<uint64_t>>&, bool);
template std::vector<DeterminantT<std::array<uint64_t, 2>>> select_top_k_dets_t<std::array<uint64_t, 2>>(const std::vector<DeterminantT<std::array<uint64_t, 2>>>&, const std::vector<double>&, size_t, const std::vector<DeterminantT<std::array<uint64_t, 2>>>&, bool);
template std::vector<DeterminantT<std::array<uint64_t, 3>>> select_top_k_dets_t<std::array<uint64_t, 3>>(const std::vector<DeterminantT<std::array<uint64_t, 3>>>&, const std::vector<double>&, size_t, const std::vector<DeterminantT<std::array<uint64_t, 3>>>&, bool);

template std::tuple<double, std::vector<DeterminantT<uint64_t>>, std::vector<double>>
run_trim_t<uint64_t>(const std::vector<DeterminantT<uint64_t>>&, const std::vector<std::vector<double>>&,
                     const std::vector<double>&, const std::string&, int, int,
                     const std::vector<int>&, const std::vector<int>&, bool, bool,
                     const std::vector<DeterminantT<uint64_t>>&, double, int, const IntegralSparsityInfo*, const std::vector<double>&, const std::string&);
template std::tuple<double, std::vector<DeterminantT<std::array<uint64_t, 2>>>, std::vector<double>>
run_trim_t<std::array<uint64_t, 2>>(const std::vector<DeterminantT<std::array<uint64_t, 2>>>&, const std::vector<std::vector<double>>&,
                                    const std::vector<double>&, const std::string&, int, int,
                                    const std::vector<int>&, const std::vector<int>&, bool, bool,
                                    const std::vector<DeterminantT<std::array<uint64_t, 2>>>&, double, int, const IntegralSparsityInfo*, const std::vector<double>&, const std::string&);
template std::tuple<double, std::vector<DeterminantT<std::array<uint64_t, 3>>>, std::vector<double>>
run_trim_t<std::array<uint64_t, 3>>(const std::vector<DeterminantT<std::array<uint64_t, 3>>>&, const std::vector<std::vector<double>>&,
                                    const std::vector<double>&, const std::string&, int, int,
                                    const std::vector<int>&, const std::vector<int>&, bool, bool,
                                    const std::vector<DeterminantT<std::array<uint64_t, 3>>>&, double, int, const IntegralSparsityInfo*, const std::vector<double>&, const std::string&);

template std::vector<std::vector<DeterminantT<std::array<uint64_t, 4>>>> partition_pool_t<std::array<uint64_t, 4>>(const std::vector<DeterminantT<std::array<uint64_t, 4>>>&, int);
template std::vector<std::vector<DeterminantT<std::array<uint64_t, 5>>>> partition_pool_t<std::array<uint64_t, 5>>(const std::vector<DeterminantT<std::array<uint64_t, 5>>>&, int);
template std::vector<std::vector<DeterminantT<std::array<uint64_t, 6>>>> partition_pool_t<std::array<uint64_t, 6>>(const std::vector<DeterminantT<std::array<uint64_t, 6>>>&, int);
template std::vector<std::vector<DeterminantT<std::array<uint64_t, 7>>>> partition_pool_t<std::array<uint64_t, 7>>(const std::vector<DeterminantT<std::array<uint64_t, 7>>>&, int);
template std::vector<std::vector<DeterminantT<std::array<uint64_t, 8>>>> partition_pool_t<std::array<uint64_t, 8>>(const std::vector<DeterminantT<std::array<uint64_t, 8>>>&, int);

template SpMat build_hamiltonian_sparse_t<std::array<uint64_t, 4>>(const std::vector<DeterminantT<std::array<uint64_t, 4>>>&, const std::vector<std::vector<double>>&, const std::vector<double>&, bool, int, const IntegralSparsityInfo*, int, int, const std::vector<Triplet>*, const AlphaTopologyT<std::array<uint64_t, 4>>*);
template SpMat build_hamiltonian_sparse_t<std::array<uint64_t, 5>>(const std::vector<DeterminantT<std::array<uint64_t, 5>>>&, const std::vector<std::vector<double>>&, const std::vector<double>&, bool, int, const IntegralSparsityInfo*, int, int, const std::vector<Triplet>*, const AlphaTopologyT<std::array<uint64_t, 5>>*);
template SpMat build_hamiltonian_sparse_t<std::array<uint64_t, 6>>(const std::vector<DeterminantT<std::array<uint64_t, 6>>>&, const std::vector<std::vector<double>>&, const std::vector<double>&, bool, int, const IntegralSparsityInfo*, int, int, const std::vector<Triplet>*, const AlphaTopologyT<std::array<uint64_t, 6>>*);
template SpMat build_hamiltonian_sparse_t<std::array<uint64_t, 7>>(const std::vector<DeterminantT<std::array<uint64_t, 7>>>&, const std::vector<std::vector<double>>&, const std::vector<double>&, bool, int, const IntegralSparsityInfo*, int, int, const std::vector<Triplet>*, const AlphaTopologyT<std::array<uint64_t, 7>>*);
template SpMat build_hamiltonian_sparse_t<std::array<uint64_t, 8>>(const std::vector<DeterminantT<std::array<uint64_t, 8>>>&, const std::vector<std::vector<double>>&, const std::vector<double>&, bool, int, const IntegralSparsityInfo*, int, int, const std::vector<Triplet>*, const AlphaTopologyT<std::array<uint64_t, 8>>*);

template std::tuple<double, std::vector<double>>
diagonalize_subspace_davidson_t<std::array<uint64_t, 4>>(const std::vector<DeterminantT<std::array<uint64_t, 4>>>&, const std::vector<std::vector<double>>&,
                                                         const std::vector<double>&, HijCacheT<std::array<uint64_t, 4>>&, bool, int, double, int, int, const std::vector<double>&, const IntegralSparsityInfo*, const std::string&);
template std::tuple<double, std::vector<double>>
diagonalize_subspace_davidson_t<std::array<uint64_t, 5>>(const std::vector<DeterminantT<std::array<uint64_t, 5>>>&, const std::vector<std::vector<double>>&,
                                                         const std::vector<double>&, HijCacheT<std::array<uint64_t, 5>>&, bool, int, double, int, int, const std::vector<double>&, const IntegralSparsityInfo*, const std::string&);
template std::tuple<double, std::vector<double>>
diagonalize_subspace_davidson_t<std::array<uint64_t, 6>>(const std::vector<DeterminantT<std::array<uint64_t, 6>>>&, const std::vector<std::vector<double>>&,
                                                         const std::vector<double>&, HijCacheT<std::array<uint64_t, 6>>&, bool, int, double, int, int, const std::vector<double>&, const IntegralSparsityInfo*, const std::string&);
template std::tuple<double, std::vector<double>>
diagonalize_subspace_davidson_t<std::array<uint64_t, 7>>(const std::vector<DeterminantT<std::array<uint64_t, 7>>>&, const std::vector<std::vector<double>>&,
                                                         const std::vector<double>&, HijCacheT<std::array<uint64_t, 7>>&, bool, int, double, int, int, const std::vector<double>&, const IntegralSparsityInfo*, const std::string&);
template std::tuple<double, std::vector<double>>
diagonalize_subspace_davidson_t<std::array<uint64_t, 8>>(const std::vector<DeterminantT<std::array<uint64_t, 8>>>&, const std::vector<std::vector<double>>&,
                                                         const std::vector<double>&, HijCacheT<std::array<uint64_t, 8>>&, bool, int, double, int, int, const std::vector<double>&, const IntegralSparsityInfo*, const std::string&);

template std::vector<DeterminantT<std::array<uint64_t, 4>>> select_top_k_dets_t<std::array<uint64_t, 4>>(const std::vector<DeterminantT<std::array<uint64_t, 4>>>&, const std::vector<double>&, size_t, const std::vector<DeterminantT<std::array<uint64_t, 4>>>&, bool);
template std::vector<DeterminantT<std::array<uint64_t, 5>>> select_top_k_dets_t<std::array<uint64_t, 5>>(const std::vector<DeterminantT<std::array<uint64_t, 5>>>&, const std::vector<double>&, size_t, const std::vector<DeterminantT<std::array<uint64_t, 5>>>&, bool);
template std::vector<DeterminantT<std::array<uint64_t, 6>>> select_top_k_dets_t<std::array<uint64_t, 6>>(const std::vector<DeterminantT<std::array<uint64_t, 6>>>&, const std::vector<double>&, size_t, const std::vector<DeterminantT<std::array<uint64_t, 6>>>&, bool);
template std::vector<DeterminantT<std::array<uint64_t, 7>>> select_top_k_dets_t<std::array<uint64_t, 7>>(const std::vector<DeterminantT<std::array<uint64_t, 7>>>&, const std::vector<double>&, size_t, const std::vector<DeterminantT<std::array<uint64_t, 7>>>&, bool);
template std::vector<DeterminantT<std::array<uint64_t, 8>>> select_top_k_dets_t<std::array<uint64_t, 8>>(const std::vector<DeterminantT<std::array<uint64_t, 8>>>&, const std::vector<double>&, size_t, const std::vector<DeterminantT<std::array<uint64_t, 8>>>&, bool);

template std::tuple<double, std::vector<DeterminantT<std::array<uint64_t, 4>>>, std::vector<double>>
run_trim_t<std::array<uint64_t, 4>>(const std::vector<DeterminantT<std::array<uint64_t, 4>>>&, const std::vector<std::vector<double>>&,
                                    const std::vector<double>&, const std::string&, int, int,
                                    const std::vector<int>&, const std::vector<int>&, bool, bool,
                                    const std::vector<DeterminantT<std::array<uint64_t, 4>>>&, double, int, const IntegralSparsityInfo*, const std::vector<double>&, const std::string&);
template std::tuple<double, std::vector<DeterminantT<std::array<uint64_t, 5>>>, std::vector<double>>
run_trim_t<std::array<uint64_t, 5>>(const std::vector<DeterminantT<std::array<uint64_t, 5>>>&, const std::vector<std::vector<double>>&,
                                    const std::vector<double>&, const std::string&, int, int,
                                    const std::vector<int>&, const std::vector<int>&, bool, bool,
                                    const std::vector<DeterminantT<std::array<uint64_t, 5>>>&, double, int, const IntegralSparsityInfo*, const std::vector<double>&, const std::string&);
template std::tuple<double, std::vector<DeterminantT<std::array<uint64_t, 6>>>, std::vector<double>>
run_trim_t<std::array<uint64_t, 6>>(const std::vector<DeterminantT<std::array<uint64_t, 6>>>&, const std::vector<std::vector<double>>&,
                                    const std::vector<double>&, const std::string&, int, int,
                                    const std::vector<int>&, const std::vector<int>&, bool, bool,
                                    const std::vector<DeterminantT<std::array<uint64_t, 6>>>&, double, int, const IntegralSparsityInfo*, const std::vector<double>&, const std::string&);
template std::tuple<double, std::vector<DeterminantT<std::array<uint64_t, 7>>>, std::vector<double>>
run_trim_t<std::array<uint64_t, 7>>(const std::vector<DeterminantT<std::array<uint64_t, 7>>>&, const std::vector<std::vector<double>>&,
                                    const std::vector<double>&, const std::string&, int, int,
                                    const std::vector<int>&, const std::vector<int>&, bool, bool,
                                    const std::vector<DeterminantT<std::array<uint64_t, 7>>>&, double, int, const IntegralSparsityInfo*, const std::vector<double>&, const std::string&);
template std::tuple<double, std::vector<DeterminantT<std::array<uint64_t, 8>>>, std::vector<double>>
run_trim_t<std::array<uint64_t, 8>>(const std::vector<DeterminantT<std::array<uint64_t, 8>>>&, const std::vector<std::vector<double>>&,
                                    const std::vector<double>&, const std::string&, int, int,
                                    const std::vector<int>&, const std::vector<int>&, bool, bool,
                                    const std::vector<DeterminantT<std::array<uint64_t, 8>>>&, double, int, const IntegralSparsityInfo*, const std::vector<double>&, const std::string&);

// =================================================================================
// 5) transform_integrals
// =================================================================================
// ATTENTIVE MODE OPTIMIZATION:
// When attentive_orbitals is provided, U is assumed to be block-diagonal:
//   U = [ U_att   0  ]
//       [   0    I  ]
// In this case, only the attentive sub-block of integrals needs transformation,
// reducing complexity from O(N^5) to O(k^5) for the ERI transform.
// =================================================================================
std::pair<std::vector<std::vector<double>>, std::vector<double>>
transform_integrals(
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    const std::vector<std::vector<double>>& U,
    const std::vector<int>& attentive_orbitals
) {
    int n = static_cast<int>(U.size());
    if (n == 0) return {{}, {}};
    
    // Convert U to Eigen matrix
    Mat Um(n, n);
    for(int i=0; i<n; ++i)
        for(int j=0; j<n; ++j)
            Um(i, j) = U[i][j];
    
    // Convert h1 to Eigen matrix
    Mat H(n, n);
    for(int i=0; i<n; ++i)
        for(int j=0; j<n; ++j)
            H(i, j) = h1[i][j];
    
    // Check if attentive mode is enabled
    bool attentive_mode = !attentive_orbitals.empty() && 
                          (int)attentive_orbitals.size() < n;
    
    // 1. Transform 1-body: h' = U^T h U
    // This is O(N^3) regardless of mode, but fast
    Mat H_new = Um.transpose() * H * Um;
    
    // Convert h1_new back to vector
    std::vector<std::vector<double>> h1_new(n, std::vector<double>(n));
    for(int i=0; i<n; ++i)
        for(int j=0; j<n; ++j)
            h1_new[i][j] = H_new(i, j);
    
    // 2. Transform 2-body ERI
    std::vector<double> eri_new;
    
    if (attentive_mode) {
        // ========== ATTENTIVE MODE: O(k * N^3) instead of O(N^5) ==========
        // Algorithm based on block-diagonal U structure:
        //   U = [ U_att   0  ]
        //       [   0     I  ]
        //
        // All 4 passes use U_att^T (same as full mode uses U.T for all passes)
        // The key insight: each pass does sum_x U[x,y] * tensor[x,...]
        // which in matrix form is U.T @ tensor.reshape(n, n^3)
        //
        // Complexity: O(k * N^3) per pass, vs O(N^5) for full transform
        
        int k = static_cast<int>(attentive_orbitals.size());
        size_t N = static_cast<size_t>(n);
        size_t N2 = N*N;
        size_t N3 = N*N*N;
        
        // Extract U_att (k x k submatrix)
        Mat U_att(k, k);
        for(int i=0; i<k; ++i)
            for(int j=0; j<k; ++j)
                U_att(i, j) = Um(attentive_orbitals[i], attentive_orbitals[j]);
        
        // Work buffer for the ERI tensor
        std::vector<double> g = eri;
        
        // Temporary buffer for attentive slice
        std::vector<double> att_slice(k * N3);
        std::vector<double> att_transformed(k * N3);
        
        // === Pass 1: Transform first index ===
        // g[att[m], j, k, l] = sum_{p in att} U_att.T[m, p] * g[att[p], j, k, l]
        using MatRow = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        
        // Extract attentive rows
        for(int m = 0; m < k; ++m) {
            int orb = attentive_orbitals[m];
            std::copy(g.begin() + orb * N3, g.begin() + (orb + 1) * N3, 
                      att_slice.begin() + m * N3);
        }
        
        // Transform: U_att^T @ att_slice(k, N^3)
        Eigen::Map<MatRow> slice_mat(att_slice.data(), k, N3);
        Eigen::Map<MatRow> trans_mat(att_transformed.data(), k, N3);
        trans_mat = U_att.transpose() * slice_mat;
        
        // Write back
        for(int m = 0; m < k; ++m) {
            int orb = attentive_orbitals[m];
            std::copy(att_transformed.begin() + m * N3, 
                      att_transformed.begin() + (m + 1) * N3,
                      g.begin() + orb * N3);
        }
        
        // === Pass 2: Transform second index ===
        // g[i, att[m], k, l] = sum_{j in att} U_att^T[m, j] * g[i, att[j], k, l]
        // 
        // Strategy: For each i, extract g[i, att, :, :] as a (k, N²) matrix,
        // apply U_att @ (k, N²), and write back.
        
        #pragma omp parallel for
        for(int i = 0; i < static_cast<int>(N); ++i) {
            // Extract g[i, att, :, :] -> (k, N²)
            std::vector<double> slice_in(k * N2);
            for(int m = 0; m < k; ++m) {
                int j = attentive_orbitals[m];
                for(size_t kk = 0; kk < N; ++kk) {
                    for(size_t l = 0; l < N; ++l) {
                        slice_in[m * N2 + kk * N + l] = g[i*N3 + j*N2 + kk*N + l];
                    }
                }
            }
            
            // Transform: U_att^T @ slice_in
            Eigen::Map<MatRow> in_mat(slice_in.data(), k, N2);
            MatRow out_mat = U_att.transpose() * in_mat;
            
            // Write back
            for(int m = 0; m < k; ++m) {
                int j = attentive_orbitals[m];
                for(size_t kk = 0; kk < N; ++kk) {
                    for(size_t l = 0; l < N; ++l) {
                        g[i*N3 + j*N2 + kk*N + l] = out_mat(m, kk * N + l);
                    }
                }
            }
        }
        
        // === Pass 3: Transform third index ===
        // g[i, j, att[m], l] = sum_{k in att} U_att^T[m, k] * g[i, j, att[k], l]
        
        #pragma omp parallel for
        for(int i = 0; i < static_cast<int>(N); ++i) {
            for(size_t j = 0; j < N; ++j) {
                // Extract g[i, j, att, :] -> (k, N)
                std::vector<double> slice_in(k * N);
                for(int m = 0; m < k; ++m) {
                    int kk = attentive_orbitals[m];
                    for(size_t l = 0; l < N; ++l) {
                        slice_in[m * N + l] = g[i*N3 + j*N2 + kk*N + l];
                    }
                }
                
                Eigen::Map<MatRow> in_mat(slice_in.data(), k, N);
                MatRow out_mat = U_att.transpose() * in_mat;
                
                for(int m = 0; m < k; ++m) {
                    int kk = attentive_orbitals[m];
                    for(size_t l = 0; l < N; ++l) {
                        g[i*N3 + j*N2 + kk*N + l] = out_mat(m, l);
                    }
                }
            }
        }
        
        // === Pass 4: Transform fourth index ===
        // g[i, j, k, att[m]] = sum_{l in att} U_att^T[m, l] * g[i, j, k, att[l]]
        
        #pragma omp parallel for
        for(int i = 0; i < static_cast<int>(N); ++i) {
            for(size_t j = 0; j < N; ++j) {
                for(size_t kk = 0; kk < N; ++kk) {
                    // Extract g[i, j, k, att] -> (k, 1) = vector
                    std::vector<double> vec_in(k), vec_out(k);
                    for(int m = 0; m < k; ++m) {
                        int l = attentive_orbitals[m];
                        vec_in[m] = g[i*N3 + j*N2 + kk*N + l];
                    }
                    
                    Eigen::Map<Eigen::VectorXd> v_in(vec_in.data(), k);
                    Eigen::Map<Eigen::VectorXd> v_out(vec_out.data(), k);
                    v_out = U_att.transpose() * v_in;
                    
                    for(int m = 0; m < k; ++m) {
                        int l = attentive_orbitals[m];
                        g[i*N3 + j*N2 + kk*N + l] = vec_out[m];
                    }
                }
            }
        }
        
        eri_new = g;
    } else {
        // ========== FULL MODE: O(N^5) ==========
        // Original 4-pass algorithm
        
        size_t N = static_cast<size_t>(n);
        size_t N2 = N*N;
        size_t N3 = N*N*N;
        
        std::vector<double> buf1 = eri;
        std::vector<double> buf2(eri.size());
        std::vector<double>* p_in = &buf1;
        std::vector<double>* p_out = &buf2;
        
        for(int pass=0; pass<4; ++pass) {
            using MatRow = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
            Eigen::Map<const MatRow> MatIn(p_in->data(), n, static_cast<Eigen::Index>(N3));
            
            // All passes use U.T because the permute step rotates indices such that
            // each pass transforms a different original index (p,q,r,s) to (i,j,k,l)
            // using the same operation: sum_x U[x,y] * tensor[x,...]
            MatRow MatOut = Um.transpose() * MatIn;
            
            const double* src = MatOut.data();
            double* dst = p_out->data();
            
            #pragma omp parallel for
            for (int p = 0; p < static_cast<int>(N); ++p) {
                for (size_t j = 0; j < N; ++j) {
                    for (size_t k = 0; k < N; ++k) {
                        for (size_t l = 0; l < N; ++l) {
                            size_t src_idx = p*N3 + j*N2 + k*N + l;
                            size_t dst_idx = j*N3 + k*N2 + l*N + p;
                            dst[dst_idx] = src[src_idx];
                        }
                    }
                }
            }
            std::swap(p_in, p_out);
        }
        
        eri_new = *p_in;
    }
    
    return {h1_new, eri_new};
}

// =================================================================================
// Finite Difference Gradient for Orbital Optimization
// =================================================================================
// Computes forward FD gradient: grad[k] = (E(x + eps*e_k) - E(x)) / eps
//
// Algorithm:
//   1. Compute center energy E(x) by transforming integrals with U = expm(K(x))
//   2. For each active index k (in parallel via OpenMP):
//      - Perturb: x_plus = x; x_plus[k] += eps
//      - Compute U_plus = expm(K(x_plus)) using Taylor series
//      - Transform integrals and compute E(x_plus)
//   3. Return gradient vector
//
// Performance Notes (2026-01-20 benchmark, N=36, k=12):
//   - C++ Pure BD:      ~18 ms per transform (used here, data already in C++)
//   - Python BD:        ~10 ms per transform (but would need Python↔C++ roundtrip)
//   - C++ via Python:   ~89 ms per transform (includes data conversion overhead)
//
// Using C++ transform here avoids n_active Python↔C++ roundtrips per gradient.
// For n_active=100 active parameters, this saves ~7.1 seconds per gradient call.
// =================================================================================

template<typename StorageType>
std::vector<double> compute_fd_gradient_parallel_t(
    const std::vector<DeterminantT<StorageType>>& dets,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    HijCacheT<StorageType>& cache,
    const std::vector<double>& x,
    const std::vector<int>& active_indices,
    int n_orb,
    int n_elec,
    double eps,
    double davidson_tol,
    int davidson_max_iter
) {
    size_t n_params = x.size();
    size_t n_active = active_indices.size();
    std::vector<double> grad(n_params, 0.0);
    
    if (n_active == 0) return grad;
    
    // Unpack kappa parameters to orbital rotation matrix U = expm(K)
    // K is antisymmetric with K(i,j) = kappa[idx] for i < j
    // Uses Taylor series expansion (20 terms) for matrix exponential
    auto unpack_kappa = [n_orb](const std::vector<double>& kappa) {
        Mat K = Mat::Zero(n_orb, n_orb);
        int idx = 0;
        for (int i = 0; i < n_orb; ++i) {
            for (int j = i + 1; j < n_orb; ++j) {
                K(i, j) = kappa[idx];
                K(j, i) = -kappa[idx];
                idx++;
            }
        }
        // Taylor series: exp(K) = I + K + K^2/2! + K^3/3! + ...
        Mat U = Mat::Identity(n_orb, n_orb);
        Mat K_power = Mat::Identity(n_orb, n_orb);
        double factorial = 1.0;
        
        for (int term = 1; term <= 20; ++term) {
            K_power = K_power * K;
            factorial *= term;
            U += K_power / factorial;
        }
        return U;
    };
    
    // Convert Eigen matrix to std::vector for interface compatibility
    auto mat_to_vec = [n_orb](const Mat& M) {
        std::vector<std::vector<double>> V(n_orb, std::vector<double>(n_orb));
        for (int i = 0; i < n_orb; ++i)
            for (int j = 0; j < n_orb; ++j)
                V[i][j] = M(i, j);
        return V;
    };
    
    // Step 1: Compute center energy E(x)
    Mat U_center = unpack_kappa(x);
    auto U_center_vec = mat_to_vec(U_center);
    auto [h1_c, eri_c] = transform_integrals(h1, eri, U_center_vec);
    
    auto result_center = diagonalize_subspace_davidson_t<StorageType>(
        dets, h1_c, eri_c, cache, false, davidson_max_iter, davidson_tol, false, n_orb, {}
    );
    double e_center = std::get<0>(result_center);
    
    // Step 2: Compute E(x + eps*e_k) for each active k (OpenMP parallel)
    std::vector<double> energies_plus(n_active);
    
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < static_cast<int>(n_active); ++i) {
        int k = active_indices[i];
        
        // Perturb parameter k
        std::vector<double> x_plus = x;
        x_plus[k] += eps;
        
        // Compute rotation matrix and transform integrals
        Mat U_plus = unpack_kappa(x_plus);
        auto U_plus_vec = mat_to_vec(U_plus);
        auto transformed = transform_integrals(h1, eri, U_plus_vec);
        
        // Thread-local cache for thread safety
        HijCacheT<StorageType> cache_local;
        
        // Diagonalize and store energy
        auto result = diagonalize_subspace_davidson_t<StorageType>(
            dets, transformed.first, transformed.second, cache_local, false, davidson_max_iter, davidson_tol, false, n_orb, {}
        );
        energies_plus[i] = std::get<0>(result);
    }
    
    // Step 3: Compute gradient via forward difference
    for (size_t i = 0; i < n_active; ++i) {
        int k = active_indices[i];
        grad[k] = (energies_plus[i] - e_center) / eps;
    }
    
    return grad;
}

// Wrapper for default storage type
std::vector<double> compute_fd_gradient_parallel(
    const std::vector<Determinant>& dets,
    const std::vector<std::vector<double>>& h1,  // ORIGINAL integrals
    const std::vector<double>& eri,              // ORIGINAL integrals
    HijCache& cache,
    const std::vector<double>& x,
    const std::vector<int>& active_indices,
    int n_orb,
    int n_elec,
    double eps,
    double davidson_tol,
    int davidson_max_iter
) {
    return compute_fd_gradient_parallel_t<uint64_t>(
        dets, h1, eri, cache, x, active_indices, n_orb, n_elec, eps, davidson_tol, davidson_max_iter
    );
}

} // namespace trimci_core

