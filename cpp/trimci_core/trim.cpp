// =================================================================================
// FILE: trim.cpp
// =================================================================================
#include "trim.hpp"
#include "hamiltonian.hpp" // Include the header that provides the function definitions.
#include "trim_excitation_enumerator.hpp"
#include <random>
#include <algorithm>
#include <unordered_set>
#include <fstream>
#include <iostream>
#include <chrono>
#include <numeric> // For std::iota
#include <iomanip> // For std::setprecision
#include "bit_compat.hpp"

#ifdef _OPENMP
#  include <omp.h>
#endif
#include "omp_compat.hpp"

// Eigen
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <cstdint>

// Timestamp-based seed helpers (non-reproducible by design)
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

using SpMat   = Eigen::SparseMatrix<double>;
using Triplet = Eigen::Triplet<double>;
using Vec     = Eigen::VectorXd;
using Mat     = Eigen::MatrixXd;

// 1) Randomly partition the pool of determinants (Unchanged)
std::vector<std::vector<Determinant>>
partition_pool(const std::vector<Determinant>& pool, int m)
{
    size_t n = pool.size();
    if (m <= 0) {
        throw std::invalid_argument("m must be positive");
    }

    // 1) 生成并打乱索引，而不是复制整个对象池
    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);

    static thread_local std::mt19937_64 rng{ timestamp_seed() };
    std::shuffle(idx.begin(), idx.end(), rng);

    // 2) 准备全局子集并预分配
    std::vector<std::vector<Determinant>> subsets(m);
    size_t base = (n + m - 1) / m;  // 向上取整
    for (auto& sub : subsets) {
        sub.reserve(base);
    }

    // 3) 为每个线程创建本地子集缓冲区
    int num_threads = omp_get_max_threads();
    std::vector<std::vector<std::vector<Determinant>>> thread_subsets(
        num_threads,
        std::vector<std::vector<Determinant>>(m)
    );
    // 给本地缓冲区也预留一点空间
    for (int t = 0; t < num_threads; ++t) {
        for (auto& sub : thread_subsets[t]) {
            sub.reserve(base / num_threads + 1);
        }
    }

    // 4) 并行分配
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        #pragma omp for
        for (int i = 0; i < static_cast<int>(n); ++i) {
            int group = i % m;
            thread_subsets[tid][group].push_back(pool[idx[static_cast<size_t>(i)]]);
        }
    }

    // 5) 合并所有线程的本地子集到全局子集中
    for (int t = 0; t < num_threads; ++t) {
        for (int j = 0; j < m; ++j) {
            auto& local = thread_subsets[t][j];
            subsets[j].insert(subsets[j].end(), local.begin(), local.end());
        }
    }

    std::cout << "[Trim] Partition into " << m
              << " subsets, each with approx. size " << base << "\n";
    return subsets;
}

// 2) Diagonally Preconditioned Davidson Method (Unchanged)
std::tuple<double, std::vector<double>>
diagonalize_subspace_davidson(
    const std::vector<Determinant>& dets,
    const std::vector<std::vector<double>>& h1,
    const std::vector<std::vector<std::vector<std::vector<double>>>>& eri,
    HijCache& cache,
    bool quantization,
    int max_iter,
    double tol,
    bool verbose,
    int n_orb
)
{
    auto total_start = std::chrono::high_resolution_clock::now();
    
    int dim = (int)dets.size();
    if (dim == 0) return {0.0, {}};
    if (dim == 1) {
        double Hij = compute_H_ij(dets[0], dets[0], h1, eri);
        return {Hij, {1.0}};
    }
    if (dim == 2) {
        // 1) 计算对角哈密顿矩阵元
        double h00 = compute_H_ij(dets[0], dets[0], h1, eri);
        double h11 = compute_H_ij(dets[1], dets[1], h1, eri);

        // 2) 计算非对角元前，先检查激发差异
        double h01 = 0.0;
        uint64_t a0 = dets[0].alpha, b0 = dets[0].beta;
        uint64_t a1 = dets[1].alpha, b1 = dets[1].beta;
        int diff = __builtin_popcountll(a0 ^ a1) + __builtin_popcountll(b0 ^ b1);

        // 只有在最多为双激发（差异<=4）时，矩阵元才可能非零
        if (diff <= 4) {
            h01 = compute_H_ij(dets[0], dets[1], h1, eri);
        }

        // 3) 处理非对角元可忽略不计的特殊情况
        const double eps_off = 1e-12;
        if (std::abs(h01) < eps_off) {
            if (h00 <= h11) {
                return {h00, std::vector<double>{1.0, 0.0}};
            } else {
                return {h11, std::vector<double>{0.0, 1.0}};
            }
        }

        // 4) 解析求解本征值
        double delta      = h11 - h00;
        double sqrt_term  = std::sqrt(delta * delta + 4.0 * h01 * h01);
        double E0         = 0.5 * (h00 + h11 - sqrt_term);

        // 5) 计算基态的未归一化本征向量
        double c0 = 2.0 * h01;
        double c1 = delta + sqrt_term;

        // 6) 归一化向量
        double norm = std::sqrt(c0 * c0 + c1 * c1);
        assert(norm > 1e-16);
        c0 /= norm;
        c1 /= norm;

        // 7) 返回能量和归一化的系数
        return { E0, std::vector<double>{ c0, c1 } };
    }


    auto matrix_build_start = std::chrono::high_resolution_clock::now();
    
    std::vector<Triplet> triplets;

    triplets.reserve(dim * 50);
    //if (dim<=8){
    if (true){
        //—— 双重循环 + popcount 过滤 ——  
        for (int i = 0; i < dim; ++i) {
            for (int j = 0; j <= i; ++j) {
                uint64_t ai = dets[i].alpha, bi = dets[i].beta;
                uint64_t aj = dets[j].alpha, bj = dets[j].beta;
                int diff = __builtin_popcountll(ai ^ aj)
                        + __builtin_popcountll(bi ^ bj);
                if (diff > 4) continue; 
                double Hij = compute_H_ij(dets[i], dets[j], h1, eri);
                if (std::abs(Hij) > 1e-12) {
                    triplets.emplace_back(i, j, Hij);
                    if (i != j) {
                        triplets.emplace_back(j, i, Hij);
                    }
                }
            }
        }
    }
    else {
        // —— 激发枚举 ——  
        generate_triplets_by_excitations(dets, h1, eri, triplets, n_orb, 1e-12);
    }
    
    auto matrix_build_end = std::chrono::high_resolution_clock::now();
    double matrix_build_time = std::chrono::duration<double>(matrix_build_end - matrix_build_start).count();
    
    if (verbose) {
        std::cout << "  [Davidson] Matrix build completed: " << std::fixed << std::setprecision(6) << matrix_build_time << " s" << std::endl;
    }


    auto sparse_matrix_start = std::chrono::high_resolution_clock::now();
    
    SpMat H(dim, dim);
    H.setFromTriplets(triplets.begin(), triplets.end());

    if (quantization) {
        Eigen::SparseMatrix<float> Hf = H.cast<float>();
        H = Hf.cast<double>();
    }
    
    auto sparse_matrix_end = std::chrono::high_resolution_clock::now();
    double sparse_matrix_time = std::chrono::duration<double>(sparse_matrix_end - sparse_matrix_start).count();
    
    if (verbose) {
        std::cout << "  [Davidson] Sparse matrix setup completed: " << std::fixed << std::setprecision(6) << sparse_matrix_time << " s" << std::endl;
    }

    auto davidson_init_start = std::chrono::high_resolution_clock::now();
    
    Vec H_diag = H.diagonal();
    Mat V = Mat::Zero(dim, max_iter);
    Vec v0 = Vec::Zero(dim);
    int min_diag_idx = 0;
    H_diag.minCoeff(&min_diag_idx);
    v0(min_diag_idx) = 1.0;
    V.col(0) = v0;

    int current_subspace_size = 1;
    double current_energy = 0.0;
    Vec current_ritz_vec;
    
    auto davidson_init_end = std::chrono::high_resolution_clock::now();
    double davidson_init_time = std::chrono::duration<double>(davidson_init_end - davidson_init_start).count();
    
    if (verbose) {
        std::cout << "  [Davidson] Initialization completed: " << std::fixed << std::setprecision(6) << davidson_init_time << " s" << std::endl;
    }
    
    auto davidson_iter_start = std::chrono::high_resolution_clock::now();

    // 并行化设置：建议线程数
    #ifdef _OPENMP
    int num_threads = omp_get_max_threads();
    if (verbose) {
        std::cout << "  [Davidson] Using OpenMP with " << num_threads << " threads" << std::endl;
    }
    // 对于大矩阵，限制线程数避免内存带宽瓶颈
    //if (dim > 10000 && num_threads > 8) {
    //    omp_set_num_threads(8);
    //    if (verbose) std::cout << "  [Davidson] Limited to 8 threads for large matrix" << std::endl;
    //}
    #endif

    // 子空间重启参数
    const int restart_size = std::min(max_iter / 2, 10);
    double prev_energy = 0.0;
    
    for (int iter = 0; iter < max_iter; ++iter) {
        Mat V_current = V.leftCols(current_subspace_size);
        
        // 优化：预计算 H*V_current 避免重复计算
        Mat HV_current = H * V_current;
        Mat H_sub = V_current.transpose() * HV_current;
        
        Eigen::SelfAdjointEigenSolver<Mat> sub_solver(H_sub);
        current_energy = sub_solver.eigenvalues()(0);
        Vec sub_ritz_vec = sub_solver.eigenvectors().col(0);
        current_ritz_vec = V_current * sub_ritz_vec;
        
        // 优化：使用预计算的 HV_current
        Vec Hv = HV_current * sub_ritz_vec;
        Vec residual = Hv - current_energy * current_ritz_vec;
        double res_norm = residual.norm();
        
        // 添加能量收敛判据
        double energy_change = std::abs(current_energy - prev_energy);
        prev_energy = current_energy;

        if (verbose) {
            std::cout << "  [Davidson] iter=" << std::setw(3) << iter
                      << ", subspace_size=" << std::setw(4) << current_subspace_size
                      << ", E=" << std::fixed << std::setprecision(12) << current_energy
                      << ", |r|=" << std::scientific << res_norm 
                      << ", ΔE=" << energy_change << std::defaultfloat << std::endl;
        }

        if (iter > 0 && res_norm < tol && energy_change < tol * 1e-2) {
            if (verbose) std::cout << "  [Davidson] Converged in " << iter + 1 << " iterations." << std::endl;
            break;
        }
        
        // 子空间重启机制
        if (current_subspace_size >= restart_size && current_subspace_size < max_iter && current_subspace_size < dim) {
            if (verbose) std::cout << "  [Davidson] Restarting subspace at iter " << iter << std::endl;
            // 保留最重要的向量
            V.col(0) = current_ritz_vec;
            current_subspace_size = 1;
        } else if (current_subspace_size >= max_iter || current_subspace_size >= dim) {
            if (verbose) std::cerr << "  [Davidson] Warning: Stopped because subspace size limit reached." << std::endl;
            break;
        }

        // 改进的预条件子：使用更小的阈值和shift
        const double shift = 0.1;
        Vec correction = Vec::Zero(dim);
        for(int i = 0; i < dim; ++i) {
            double denom = H_diag(i) - current_energy + shift;
            if (std::abs(denom) > 1e-12) {
                correction(i) = -residual(i) / denom;
            }
        }
        
        // 优化：矩阵化的Gram-Schmidt正交化
        if (current_subspace_size > 0) {
            Mat V_current_for_proj = V.leftCols(current_subspace_size);
            correction -= V_current_for_proj * (V_current_for_proj.transpose() * correction);
        }

        double correction_norm = correction.norm();
        // ✅ 检查首轮塌缩 → 重启
        if (iter == 0 && correction_norm < 1e-12 && res_norm < 1e-12) {
            if (verbose) std::cout << "  [Davidson] Warning: Initial vector collapsed, reinitializing..." << std::endl;
            V.col(0).setRandom();
            V.col(0).normalize();
            current_subspace_size = 1;
            iter = -1; // 重置迭代，从头再来
            continue;
        }

        if (correction_norm > 1e-12) {
            correction /= correction_norm;
            V.col(current_subspace_size) = correction;
            current_subspace_size++;
        } else {
            // 如果在首轮迭代 preconditioned 向量过小，尝试使用归一化残差向量继续迭代
            if (iter == 0 && res_norm > 1e-12) {
                if (verbose) std::cout << "  [Davidson] Warning: Preconditioner failed at iter 0, using residual vector." << std::endl;
                Vec alt = residual / res_norm;
                V.col(current_subspace_size) = alt;
                current_subspace_size++;
                continue; // 继续下一轮迭代
            }

            //if (res_norm < tol) {
            //    if (verbose) std::cout << "  [Davidson] Converged with small correction at iter " << iter << std::endl;
            //} else {
            //    if (verbose) std::cout << "  [Davidson] Warning: Correction vector too small, stopping." << std::endl;
            //}
            //break;
        }
    }
    
    auto davidson_iter_end = std::chrono::high_resolution_clock::now();
    double davidson_iter_time = std::chrono::duration<double>(davidson_iter_end - davidson_iter_start).count();
    
    if (verbose) {
        std::cout << "  [Davidson] Iterations completed: " << std::fixed << std::setprecision(6) << davidson_iter_time << " s" << std::endl;
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(total_end - total_start).count();
    
    if (verbose) {
        std::cout << "  [Davidson] Total diagonalization time: " << std::fixed << std::setprecision(6) << total_time << " s" << std::endl;
        std::cout << "  [Davidson] Timing summary - Matrix build: " << std::fixed << std::setprecision(6) << matrix_build_time 
                  << "s, Sparse setup: " << sparse_matrix_time 
                  << "s, Init: " << davidson_init_time 
                  << "s, Iterations: " << davidson_iter_time << "s" << std::endl;
    }

    std::vector<double> coeffs(dim);
    for (int i = 0; i < dim; ++i) coeffs[i] = current_ritz_vec(i);
    return {current_energy, coeffs};
}

// 3) Select top-k determinants based on coefficient magnitude (excluding core set)
std::vector<Determinant>
select_top_k_dets(
    const std::vector<Determinant>& dets,
    const std::vector<double>& coeffs,
    size_t k,
    const std::vector<Determinant>& core_vec,
    bool keep_core
)
{
    std::unordered_set<Determinant> core_set(core_vec.begin(), core_vec.end());
    size_t n = dets.size();
    if (n == 0 || k == 0) return {};

    std::vector<std::pair<double, Determinant>> scored;
    scored.reserve(n);

    if (!core_set.empty()) {
        // 有 core_set 的情况：要过滤
        for (size_t i = 0; i < n; ++i) {
            if (core_set.count(dets[i])) continue; 
            scored.emplace_back(std::abs(coeffs[i]), dets[i]);
        }
    } else {
        // 没有 core_set：直接全部加入
        for (size_t i = 0; i < n; ++i) {
            scored.emplace_back(std::abs(coeffs[i]), dets[i]);
        }
    }

    if (scored.empty()) return {};

    if (k > scored.size()) k = scored.size();

    auto mid = scored.begin() + k;
    std::nth_element(
        scored.begin(), mid, scored.end(),
        [](auto& a, auto& b) { return a.first > b.first; }
    );

    std::vector<Determinant> top;
    top.reserve(k + (keep_core ? core_set.size() : 0));

    if (keep_core && !core_set.empty()) {
        for (auto const& d : core_set) {
            top.push_back(d);
        }
    }

    for (size_t i = 0; i < k; ++i) {
        top.push_back(scored[i].second);
    }

    return top;
}



// 4) Main Trim workflow with multi-round
std::tuple<double, std::vector<Determinant>, std::vector<double>>
run_trim(
    const std::vector<Determinant>& pool,
    const std::vector<std::vector<double>>& h1,
    const std::vector<std::vector<std::vector<std::vector<double>>>>& eri,
    const std::string& mol_name,
    int n_elec,
    int n_orb,
    const std::vector<int>& group_sizes,
    const std::vector<int>& keep_sizes,
    bool quantization,
    bool save_cache,
    bool verbose,
    const std::vector<Determinant>& external_core_dets
)
{
    if (group_sizes.size() != keep_sizes.size()) {
        throw std::runtime_error("[Trim] Error: group_sizes and keep_sizes must have the same length.");
    }

    // Initial summary
    std::cout << "[Trim] Start: initial pool=" << pool.size() << " determinants" << std::endl;
    for (size_t i = 0; i < group_sizes.size(); ++i) {
        std::cout << "[Trim] Round " << i + 1
                  << ": m=" << group_sizes[i]
                  << ", k=" << keep_sizes[i] << std::endl;
    }

    // -- Setup core determinants --
    std::vector<Determinant> core_dets;
    
    // 检查是否提供了外部的core_dets
    if (!external_core_dets.empty()) {
        // 使用外部提供的core_dets
        core_dets = external_core_dets;
        std::cout << "[Trim] Using external core determinants: " << core_dets.size() << std::endl;
    } else {
        // 没有提供外部core_dets，生成默认的core_dets
        // 首先生成 HF（core）Determinant
        // 这里不具有一般性，后续需要修改
        unsigned n_up = n_elec/2 + (n_elec % 2);  // 多给 α 自旋
        unsigned n_dn = n_elec/2;
        if (n_up > n_orb || n_dn > n_orb) {
            throw std::runtime_error("[Trim] Error: n_elec exceeds available orbitals!");
        }
        // 掩码：只保留低 n_orb 位
        uint64_t mask = (n_orb < 64 ? (1ULL << n_orb) - 1 : ~0ULL);
        // 低 n_up 个 1 → α 位图；再 & mask 保证不超范围
        uint64_t alpha_bits = ((n_up < 64 ? (1ULL << n_up) - 1 : ~0ULL) & mask);
        // 低 n_dn 个 1 → β 位图；再 & mask
        uint64_t beta_bits  = ((n_dn < 64 ? (1ULL << n_dn) - 1 : ~0ULL) & mask);
        
        // 添加HF行列式
        Determinant hf_det(alpha_bits, beta_bits);
        core_dets.push_back(hf_det);
                
        std::cout << "[Trim] Generated default core determinants: " << core_dets.size() << std::endl;
    }
    
    // 输出所有core_dets的信息
    if (verbose) {
        for (size_t i = 0; i < core_dets.size(); ++i) {
            std::cout << "[Trim] core_det " << i << ": " << core_dets[i] << std::endl;
        }
    }


    // Load or create cache
    HijCache cache;
    std::string cache_file;
    std::tie(cache, cache_file) = load_or_create_Hij_cache(mol_name, n_elec, n_orb);
    std::cout << "[Trim] Cache file: " << cache_file
              << " (entries=" << cache.size() << ")" << std::endl;

    std::vector<Determinant> current_pool = pool;

    // Loop over trimming rounds
    for (size_t round = 0; round < group_sizes.size(); ++round) {
        auto round_start = std::chrono::high_resolution_clock::now();
        int m = group_sizes[round];
        int k = keep_sizes[round];
        size_t initial_size = current_pool.size();

        std::cout << std::endl
                  << "--- [Trim] Round " << round + 1 << "/" << group_sizes.size()
                  << " (m=" << m << ", k=" << k << ") with "
                  << initial_size << " determinants ---" << std::endl;

        if (initial_size == 0) {
            std::cout << "[Trim] Warning: Pool is empty. Stopping." << std::endl;
            break;
        }

        // Partition and select
        auto subsets = partition_pool(current_pool, m);
        // 将所有core_dets添加到每个子集中
        // 注意，有些子集中可能已经包含了部分core_dets，但这是中间计算，重要的是选择
        for (auto& sub : subsets) {
            sub.insert(sub.end(), core_dets.begin(), core_dets.end());
        }
        std::vector<Determinant> selected;

        #pragma omp parallel
        {
            std::vector<Determinant> selected_private;
            #pragma omp for schedule(dynamic) nowait
            for (int i = 0; i < (int)subsets.size(); ++i) {
                auto& sub = subsets[i];
                if (sub.empty()) continue;

                auto start = std::chrono::high_resolution_clock::now();
                int max_iter = std::min<int>(100, (int)sub.size());
                auto [e, coeffs] = diagonalize_subspace_davidson(
                    sub, h1, eri, cache, quantization,
                    max_iter, 1e-3, false, n_orb
                );
                // 由于添加了core_dets，需要相应调整k值
                // auto topd = select_top_k_dets(sub, coeffs, k + core_dets.size());
                auto topd = select_top_k_dets(sub, coeffs, k, core_dets, false);
                selected_private.insert(
                    selected_private.end(), topd.begin(), topd.end());

                auto end = std::chrono::high_resolution_clock::now();
                double dur = std::chrono::duration<double>(end - start).count();

                #pragma omp critical
                {
                    if (verbose) {
                        std::cout << "[Trim] Round " << round + 1
                                  << ", Subspace " << i
                                  << ": E=" << std::fixed << std::setprecision(8) << e
                                  << ", time=" << std::setprecision(3) << dur << " s"
                                  << ", selected=" << topd.size() << std::endl << std::flush;;
                    }
                }
            }
            #pragma omp critical
            {
                selected.insert(selected.end(),
                                selected_private.begin(),
                                selected_private.end());
            }
        }
        // 合并selected和core_dets
        selected.insert(selected.end(), core_dets.begin(), core_dets.end());

        // Round summary
        std::cout << "[Trim] Round " << round + 1
                  << ": Selected total=" << selected.size() << std::endl;
        // Deduplicate
        std::unordered_set<Determinant> seen;
        std::vector<Determinant> uniq;
        uniq.reserve(selected.size());
        for (auto& d : selected) {
            if (seen.insert(d).second) uniq.push_back(d);
        }
        current_pool = std::move(uniq);
        std::cout << "[Trim] Round " << round + 1
                  << ": Unique total=" << current_pool.size() << std::endl;

        // Print round total time
        auto round_end = std::chrono::high_resolution_clock::now();
        double round_dur = std::chrono::duration<double>(round_end - round_start).count();
        std::cout << "[Trim] Round " << round + 1
                  << " total time: " << std::setprecision(3) << round_dur << " s" << std::endl;
    }

    // Final diagonalization
    std::cout << "\n--- [Trim] Final diagonalization (dim="
                << current_pool.size() << ") ---\n";
        if (current_pool.empty()) {
            std::cout << "[Trim] Final pool is empty. No final diagonalization.\n";
            return {0.0, {}, {}};
        }

        int final_max_iter = std::min<int>(200, (int)current_pool.size());
        auto diag_start = std::chrono::high_resolution_clock::now();
        auto [final_energy, final_coeffs] = diagonalize_subspace_davidson(
            current_pool, h1, eri, cache, quantization,
            final_max_iter, 1e-3, verbose, n_orb
        );
        auto diag_end = std::chrono::high_resolution_clock::now();
        double diag_time = std::chrono::duration<double>(diag_end - diag_start).count();

        std::cout << "[Trim] Final E=" << std::fixed << std::setprecision(12)
                << final_energy << "\n" << std::flush;
        std::cout << "[Trim] Final diagonalization time: "
                << diag_time << " s\n" << std::flush;

    // Save cache if requested
    if (save_cache) {
        std::ofstream ofs(cache_file, std::ios::binary);
        if (!ofs) {
            std::cerr << "Error: Could not open cache file: " << cache_file << std::endl;
        } else {
            uint64_t count = cache.size();
            ofs.write(reinterpret_cast<const char*>(&count), sizeof(count));
            for (auto const& [key, val] : cache) {
                const auto& [d1, d2] = key;
                ofs.write(reinterpret_cast<const char*>(&d1.alpha), sizeof(uint64_t));
                ofs.write(reinterpret_cast<const char*>(&d1.beta),  sizeof(uint64_t));
                ofs.write(reinterpret_cast<const char*>(&d2.alpha), sizeof(uint64_t));
                ofs.write(reinterpret_cast<const char*>(&d2.beta),  sizeof(uint64_t));
                ofs.write(reinterpret_cast<const char*>(&val),    sizeof(double));
            }
            std::cout << "[Trim] Cache saved (" << count << " entries)" << std::endl;
        }
    }

    // Return final results
    return {final_energy, current_pool, final_coeffs};
}


} // namespace trimci_core