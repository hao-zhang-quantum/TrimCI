// =================================================================================
// FILE: trim_scalable.cpp
// =================================================================================
#include "trim_scalable.hpp"
#include "hamiltonian.hpp"
#include "trim_excitation_enumerator.hpp"
#include <random>
#include <algorithm>
#include <unordered_set>
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
#include "bit_compat.hpp"
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

// =================================================================================
// 1) partition_pool_t : identical logic to the verified version
// =================================================================================
template<typename StorageType>
std::vector<std::vector<DeterminantT<StorageType>>>
partition_pool_t(const std::vector<DeterminantT<StorageType>>& pool, int m)
{
    size_t n = pool.size();
    if (m <= 0) throw std::invalid_argument("m must be positive");

    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);

    static thread_local std::mt19937_64 rng{ timestamp_seed() };
    std::shuffle(idx.begin(), idx.end(), rng);

    std::vector<std::vector<DeterminantT<StorageType>>> subsets(m);
    size_t base = (n + m - 1) / m;
    for (auto& sub : subsets) sub.reserve(base);

    int num_threads = omp_get_max_threads();
    std::vector<std::vector<std::vector<DeterminantT<StorageType>>>> thread_subsets(
        num_threads, std::vector<std::vector<DeterminantT<StorageType>>>(m)
    );
    for (int t = 0; t < num_threads; ++t)
        for (auto& sub : thread_subsets[t]) sub.reserve(base / num_threads + 1);

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

    std::cout << "[TrimT] Partition into " << m
              << " subsets, each with approx. size " << base << "\n";
    return subsets;
}

// =================================================================================
// 2) diagonalize_subspace_davidson_t : faithfully mirrored from your stable version
// =================================================================================
// =================================================================================
// diagonalize_subspace_davidson_t : 完整版，保留 trim.cpp 所有逻辑
// =================================================================================
template<typename StorageType>
std::tuple<double, std::vector<double>>
diagonalize_subspace_davidson_t(
    const std::vector<DeterminantT<StorageType>>& dets,
    const std::vector<std::vector<double>>& h1,
    const std::vector<std::vector<std::vector<std::vector<double>>>>& eri,
    HijCacheT<StorageType>& cache,
    bool quantization,
    int max_iter,
    double tol,
    bool verbose,
    int n_orb
)
{
    auto total_start = std::chrono::high_resolution_clock::now();
    
    int dim = static_cast<int>(dets.size());
    if (dim == 0) return {0.0, {}};
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
            if (h00 <= h11)
                return {h00, {1.0, 0.0}};
            else
                return {h11, {0.0, 1.0}};
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

    // ===================== Build Sparse Hamiltonian =====================
    auto matrix_build_start = std::chrono::high_resolution_clock::now();
    std::vector<Triplet> triplets;
    triplets.reserve(dim * 50);

    int skip = 0, keep = 0;
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j <= i; ++j) {
            int diff = detail::HamiltonianBitOps<StorageType>::count_differences(dets[i].alpha, dets[j].alpha)
                     + detail::HamiltonianBitOps<StorageType>::count_differences(dets[i].beta, dets[j].beta);
            if (diff > 4) {
                skip++;
                continue;
            }
            keep++;

            double Hij = compute_H_ij_t(dets[i], dets[j], h1, eri);
            if (std::abs(Hij) > 1e-12) {
                triplets.emplace_back(i, j, Hij);
                if (i != j) triplets.emplace_back(j, i, Hij);
            }
        }
    }
    if (verbose)
        std::cout << "[Debug] Keep " << keep << " pairs, skip " << skip << " pairs" << std::endl;

    auto matrix_build_end = std::chrono::high_resolution_clock::now();
    double matrix_build_time = std::chrono::duration<double>(matrix_build_end - matrix_build_start).count();
    if (verbose)
        std::cout << "  [DavidsonT] Matrix build completed: " << std::fixed << std::setprecision(6)
                  << matrix_build_time << " s" << std::endl;

    // ===================== Sparse Matrix Setup =====================
    auto sparse_matrix_start = std::chrono::high_resolution_clock::now();
    SpMat H(dim, dim);
    H.setFromTriplets(triplets.begin(), triplets.end());
    if (quantization) {
        Eigen::SparseMatrix<float> Hf = H.cast<float>();
        H = Hf.cast<double>();
    }
    auto sparse_matrix_end = std::chrono::high_resolution_clock::now();
    double sparse_matrix_time = std::chrono::duration<double>(sparse_matrix_end - sparse_matrix_start).count();
    if (verbose)
        std::cout << "  [DavidsonT] Sparse matrix setup completed: " << std::fixed << std::setprecision(6)
                  << sparse_matrix_time << " s" << std::endl;
    
    int nz_off = 0;
    for (auto& t : triplets) if (t.row() != t.col()) nz_off++;
    if (verbose)
        std::cout << "[Debug] Off-diagonal nonzero count = " << nz_off << std::endl;

    // ===================== Initialization =====================
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
    if (verbose)
        std::cout << "  [DavidsonT] Initialization completed: " << std::fixed << std::setprecision(6)
                  << davidson_init_time << " s" << std::endl;

    // ===================== Iterations =====================
    auto davidson_iter_start = std::chrono::high_resolution_clock::now();

#ifdef _OPENMP
    int num_threads = omp_get_max_threads();
    if (verbose)
        std::cout << "  [DavidsonT] Using OpenMP with " << num_threads << " threads" << std::endl;
#endif

    const int restart_size = std::min(max_iter / 2, 10);
    double prev_energy = 0.0;

    for (int iter = 0; iter < max_iter; ++iter) {
        Mat V_current = V.leftCols(current_subspace_size);
        Mat HV_current = H * V_current;
        Mat H_sub = V_current.transpose() * HV_current;

        Eigen::SelfAdjointEigenSolver<Mat> sub_solver(H_sub);
        current_energy = sub_solver.eigenvalues()(0);
        Vec sub_ritz_vec = sub_solver.eigenvectors().col(0);
        current_ritz_vec = V_current * sub_ritz_vec;

        Vec Hv = HV_current * sub_ritz_vec;
        Vec residual = Hv - current_energy * current_ritz_vec;
        double res_norm = residual.norm();
        double energy_change = std::abs(current_energy - prev_energy);
        prev_energy = current_energy;

        if (verbose) {
            std::cout << "  [DavidsonT] iter=" << std::setw(3) << iter
                      << ", subspace=" << std::setw(4) << current_subspace_size
                      << ", E=" << std::fixed << std::setprecision(12) << current_energy
                      << ", |r|=" << std::scientific << res_norm
                      << ", ΔE=" << energy_change << std::defaultfloat << std::endl;
        }

        // ✅ 收敛判据
        if (iter > 0 && res_norm < tol && energy_change < tol * 1e-2) {
            if (verbose)
                std::cout << "  [DavidsonT] Converged in " << iter + 1 << " iterations." << std::endl;
            break;
        }

        // ✅ 子空间重启机制
        if (current_subspace_size >= restart_size && current_subspace_size < max_iter && current_subspace_size < dim) {
            if (verbose)
                std::cout << "  [DavidsonT] Restarting subspace at iter " << iter << std::endl;
            V.col(0) = current_ritz_vec;
            current_subspace_size = 1;
        } else if (current_subspace_size >= max_iter || current_subspace_size >= dim) {
            if (verbose)
                std::cerr << "  [DavidsonT] Warning: Subspace limit reached." << std::endl;
            break;
        }

        // ✅ 改进预条件器
        const double shift = 0.1;
        Vec correction = Vec::Zero(dim);
        for (int i = 0; i < dim; ++i) {
            double denom = H_diag(i) - current_energy + shift;
            if (std::abs(denom) > 1e-12)
                correction(i) = -residual(i) / denom;
        }

        // Gram-Schmidt 正交化
        if (current_subspace_size > 0) {
            Mat V_proj = V.leftCols(current_subspace_size);
            correction -= V_proj * (V_proj.transpose() * correction);
        }

        double correction_norm = correction.norm();
        // ✅ 首轮塌缩处理
        if (iter == 0 && correction_norm < 1e-12 && res_norm < 1e-12) {
            if (verbose)
                std::cout << "  [DavidsonT] Warning: Initial vector collapsed, reinitializing..." << std::endl;
            V.col(0).setRandom();
            V.col(0).normalize();
            current_subspace_size = 1;
            iter = -1;
            continue;
        }

        if (correction_norm > 1e-12) {
            correction /= correction_norm;
            V.col(current_subspace_size++) = correction;
        } else {
            // fallback 使用残差
            if (iter == 0 && res_norm > 1e-12) {
                if (verbose)
                    std::cout << "  [DavidsonT] Using residual as correction vector." << std::endl;
                Vec alt = residual / res_norm;
                V.col(current_subspace_size++) = alt;
                continue;
            }
        }
    }

    auto davidson_iter_end = std::chrono::high_resolution_clock::now();
    double davidson_iter_time = std::chrono::duration<double>(davidson_iter_end - davidson_iter_start).count();
    if (verbose)
        std::cout << "  [DavidsonT] Iterations completed: " << std::fixed << std::setprecision(6)
                  << davidson_iter_time << " s" << std::endl;

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(total_end - total_start).count();
    if (verbose) {
        std::cout << "  [DavidsonT] Total diagonalization time: " << std::fixed << std::setprecision(6)
                  << total_time << " s" << std::endl;
        std::cout << "  [DavidsonT] Timing summary - Matrix build: " << matrix_build_time
                  << "s, Sparse setup: " << sparse_matrix_time
                  << "s, Init: " << davidson_init_time
                  << "s, Iterations: " << davidson_iter_time << "s" << std::endl;
    }

    std::vector<double> coeffs(dim);
    for (int i = 0; i < dim; ++i)
        coeffs[i] = current_ritz_vec(i);
    return {current_energy, coeffs};
}


// =================================================================================
// 3) select_top_k_dets_t : same logic as your core version
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
    top.reserve(k + (keep_core ? core_set.size() : 0));

    if (keep_core)
        for (auto& d : core_set) top.push_back(d);

    for (size_t i = 0; i < k; ++i)
        top.push_back(scored[i].second);

    return top;
}

// =================================================================================
// 4) run_trim_t : faithful replication of multi-round logic
// =================================================================================
template<typename StorageType>
std::tuple<double, std::vector<DeterminantT<StorageType>>, std::vector<double>>
run_trim_t(
    const std::vector<DeterminantT<StorageType>>& pool,
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
    const std::vector<DeterminantT<StorageType>>& external_core_dets
)
{
    if (group_sizes.size() != keep_sizes.size())
        throw std::runtime_error("[TrimT] group_sizes and keep_sizes mismatch");

    std::cout << "[TrimT] Start: pool=" << pool.size() << "\n";
    for (size_t i = 0; i < group_sizes.size(); ++i)
        std::cout << "[TrimT] Round " << i+1 << ": m=" << group_sizes[i]
                  << ", k=" << keep_sizes[i] << "\n";

    std::vector<DeterminantT<StorageType>> core_dets =
        external_core_dets.empty()
        ? std::vector<DeterminantT<StorageType>>{ generate_reference_det_t<StorageType>(
            (n_elec + 1)/2, n_elec/2 ) }
        : external_core_dets;

    HijCacheT<StorageType> cache;
    std::string cache_file;
    std::tie(cache, cache_file) = load_or_create_Hij_cache_t<StorageType>(mol_name, n_elec, n_orb);
    std::cout << "[TrimT] Cache file: " << cache_file
              << " (entries=" << cache.size() << ")\n";

    std::vector<DeterminantT<StorageType>> current_pool = pool;

    for (size_t round = 0; round < group_sizes.size(); ++round) {
        auto subsets = partition_pool_t(current_pool, group_sizes[round]);
        for (auto& sub : subsets)
            sub.insert(sub.end(), core_dets.begin(), core_dets.end());

        std::vector<DeterminantT<StorageType>> selected;

        #pragma omp parallel
        {
            std::vector<DeterminantT<StorageType>> local;
            #pragma omp for schedule(dynamic) nowait
            for (int i = 0; i < static_cast<int>(subsets.size()); ++i) {
                auto& sub = subsets[static_cast<size_t>(i)];
                 if (sub.empty()) continue;
                 auto [E, coeffs] = diagonalize_subspace_davidson_t(
                     sub, h1, eri, cache, quantization, 100, 1e-3, false, n_orb);
                 auto topd = select_top_k_dets_t(sub, coeffs, keep_sizes[round], core_dets, false);
                 local.insert(local.end(), topd.begin(), topd.end());
            }
            #pragma omp critical
            selected.insert(selected.end(), local.begin(), local.end());
        }

        selected.insert(selected.end(), core_dets.begin(), core_dets.end());
        std::unordered_set<DeterminantT<StorageType>> uniq(selected.begin(), selected.end());
        current_pool.assign(uniq.begin(), uniq.end());

        std::cout << "[TrimT] Round " << round+1 << ": pool=" << current_pool.size() << "\n";
    }

    auto [final_E, coeffs] =
        diagonalize_subspace_davidson_t(current_pool, h1, eri, cache, quantization, 200, 1e-3, verbose, n_orb);

    std::cout << "[TrimT] Final E=" << std::fixed << std::setprecision(12) << final_E << "\n";
    
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
                // Write determinant data based on StorageType
                if constexpr (std::is_same_v<StorageType, uint64_t>) {
                    ofs.write(reinterpret_cast<const char*>(&d1.alpha), sizeof(uint64_t));
                    ofs.write(reinterpret_cast<const char*>(&d1.beta),  sizeof(uint64_t));
                    ofs.write(reinterpret_cast<const char*>(&d2.alpha), sizeof(uint64_t));
                    ofs.write(reinterpret_cast<const char*>(&d2.beta),  sizeof(uint64_t));
                } else {
                    ofs.write(reinterpret_cast<const char*>(d1.alpha.data()), d1.alpha.size() * sizeof(uint64_t));
                    ofs.write(reinterpret_cast<const char*>(d1.beta.data()),  d1.beta.size() * sizeof(uint64_t));
                    ofs.write(reinterpret_cast<const char*>(d2.alpha.data()), d2.alpha.size() * sizeof(uint64_t));
                    ofs.write(reinterpret_cast<const char*>(d2.beta.data()),  d2.beta.size() * sizeof(uint64_t));
                }
                ofs.write(reinterpret_cast<const char*>(&val), sizeof(double));
            }
            std::cout << "[TrimT] Cache saved (" << count << " entries)" << std::endl;
        }
    }

    return {final_E, current_pool, coeffs};
}

// =================================================================================
// Explicit Instantiations
// =================================================================================
template std::vector<std::vector<DeterminantT<uint64_t>>> partition_pool_t<uint64_t>(const std::vector<DeterminantT<uint64_t>>&, int);
template std::vector<std::vector<DeterminantT<std::array<uint64_t,2>>>> partition_pool_t<std::array<uint64_t,2>>(const std::vector<DeterminantT<std::array<uint64_t,2>>>&, int);
template std::vector<std::vector<DeterminantT<std::array<uint64_t,3>>>> partition_pool_t<std::array<uint64_t,3>>(const std::vector<DeterminantT<std::array<uint64_t,3>>>&, int);

template std::tuple<double,std::vector<double>> diagonalize_subspace_davidson_t<uint64_t>(const std::vector<DeterminantT<uint64_t>>&,const std::vector<std::vector<double>>&,const std::vector<std::vector<std::vector<std::vector<double>>>>&,HijCacheT<uint64_t>&,bool,int,double,bool,int);
template std::tuple<double,std::vector<double>> diagonalize_subspace_davidson_t<std::array<uint64_t,2>>(const std::vector<DeterminantT<std::array<uint64_t,2>>>&,const std::vector<std::vector<double>>&,const std::vector<std::vector<std::vector<std::vector<double>>>>&,HijCacheT<std::array<uint64_t,2>>&,bool,int,double,bool,int);
template std::tuple<double,std::vector<double>> diagonalize_subspace_davidson_t<std::array<uint64_t,3>>(const std::vector<DeterminantT<std::array<uint64_t,3>>>&,const std::vector<std::vector<double>>&,const std::vector<std::vector<std::vector<std::vector<double>>>>&,HijCacheT<std::array<uint64_t,3>>&,bool,int,double,bool,int);

template std::vector<DeterminantT<uint64_t>> select_top_k_dets_t<uint64_t>(const std::vector<DeterminantT<uint64_t>>&,const std::vector<double>&,size_t,const std::vector<DeterminantT<uint64_t>>&,bool);
template std::vector<DeterminantT<std::array<uint64_t,2>>> select_top_k_dets_t<std::array<uint64_t,2>>(const std::vector<DeterminantT<std::array<uint64_t,2>>>&,const std::vector<double>&,size_t,const std::vector<DeterminantT<std::array<uint64_t,2>>>&,bool);
template std::vector<DeterminantT<std::array<uint64_t,3>>> select_top_k_dets_t<std::array<uint64_t,3>>(const std::vector<DeterminantT<std::array<uint64_t,3>>>&,const std::vector<double>&,size_t,const std::vector<DeterminantT<std::array<uint64_t,3>>>&,bool
);

template std::tuple<double,std::vector<DeterminantT<uint64_t>>,std::vector<double>> run_trim_t<uint64_t>(const std::vector<DeterminantT<uint64_t>>&,const std::vector<std::vector<double>>&,const std::vector<std::vector<std::vector<std::vector<double>>>>&,const std::string&,int,int,const std::vector<int>&,const std::vector<int>&,bool,bool,bool,const std::vector<DeterminantT<uint64_t>>&);
template std::tuple<double,std::vector<DeterminantT<std::array<uint64_t,2>>>,std::vector<double>> run_trim_t<std::array<uint64_t,2>>(const std::vector<DeterminantT<std::array<uint64_t,2>>>&,const std::vector<std::vector<double>>&,const std::vector<std::vector<std::vector<std::vector<double>>>>&,const std::string&,int,int,const std::vector<int>&,const std::vector<int>&,bool,bool,bool,const std::vector<DeterminantT<std::array<uint64_t,2>>>&);
template std::tuple<double,std::vector<DeterminantT<std::array<uint64_t,3>>>,std::vector<double>> run_trim_t<std::array<uint64_t,3>>(const std::vector<DeterminantT<std::array<uint64_t,3>>>&,const std::vector<std::vector<double>>&,const std::vector<std::vector<std::vector<std::vector<double>>>>&,const std::string&,int,int,const std::vector<int>&,const std::vector<int>&,bool,bool,bool,const std::vector<DeterminantT<std::array<uint64_t,3>>>&);

} // namespace trimci_core