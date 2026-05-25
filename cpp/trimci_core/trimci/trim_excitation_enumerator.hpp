#pragma once

#include <vector>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <Eigen/Sparse>
#ifdef _OPENMP
#  include <omp.h>
#endif
#include "bit_compat.hpp"

// 高性能哈希表选择
#ifdef USE_ABSL
#include <absl/container/flat_hash_map.h>
template<typename K, typename V>
using fast_hash_map = absl::flat_hash_map<K, V>;
#elif defined(USE_ROBIN_HOOD)
#include <robin_hood.h>
template<typename K, typename V>
using fast_hash_map = robin_hood::unordered_map<K, V>;
#else
#include <unordered_map>
template<typename K, typename V>
using fast_hash_map = std::unordered_map<K, V>;
#endif

namespace trimci_core {

using Triplet = Eigen::Triplet<double>;

// 前置声明：计算 H_ij
template<typename Det, typename H1, typename ERI>
double compute_H_ij(const Det& di, const Det& dj, const H1& h1, const ERI& eri);

/**
 * ExcitationGenerator: 并行优化的激发 triplet 枚举
 * - HalfDet ID 映射
 * - alpha↔beta 双向邻接表
 * - abm1 半行列式索引
 */
template<typename Det, typename H1, typename ERI>
class ExcitationGenerator {
public:
    ExcitationGenerator(int norb, double threshold = 1e-12)
        : norb_(norb), threshold_(threshold) {}

    // 初始化所有辅助数据结构
    void init(const std::vector<Det>& dets) {
        dets_ = &dets;
        buildHalfDetMapping(dets);
        buildAdjacency(dets);
        buildAbm1(dets);
    }

    // 生成 triplets
    void generateTriplets(
        const H1& h1,
        const ERI& eri,
        std::vector<Triplet>& triplets
    ) const {
        int dim = int(dets_->size());
        triplets.clear();
        triplets.reserve(dim * 50);

        int nthreads = 1;
#ifdef _OPENMP
        nthreads = omp_get_max_threads();
#endif
        std::vector<std::vector<Triplet>> local_lists(nthreads);
        for (auto& lst : local_lists) lst.reserve(dim * 50 / nthreads + 1);

        enumAlphaExcitations(local_lists, h1, eri);
        enumBetaExcitations(local_lists, h1, eri);
        enumOppositeSpinExcitations(local_lists, h1, eri);

        // 合并
        for (int t = 0; t < nthreads; ++t) {
            const auto& lst = local_lists[t];
            triplets.insert(triplets.end(), lst.begin(), lst.end());
        }
    }

private:
    int norb_;
    double threshold_;
    const std::vector<Det>* dets_ = nullptr;

    // HalfDet ID 映射 - 使用高性能哈希表
    fast_hash_map<uint64_t,int> alpha_to_id_, beta_to_id_;
    std::vector<uint64_t> unique_alphas_, unique_betas_;

    // alpha<->det, beta<->det 邻接表
    std::vector<std::vector<int>> alpha_id_to_det_ids_, beta_id_to_det_ids_;

    // abm1 半行列式索引：移除 1 电子后对应的 det 索引 - 使用高性能哈希表
    fast_hash_map<uint64_t,std::vector<int>> abm1_alpha_to_det_ids_, abm1_beta_to_det_ids_;

    // 构建 HalfDet->ID 映射
    void buildHalfDetMapping(const std::vector<Det>& dets) {
        alpha_to_id_.clear(); unique_alphas_.clear();
        beta_to_id_.clear();  unique_betas_.clear();
        for (int i = 0; i < int(dets.size()); ++i) {
            uint64_t a = dets[i].alpha;
            if (!alpha_to_id_.count(a)) {
                alpha_to_id_[a] = unique_alphas_.size();
                unique_alphas_.push_back(a);
            }
            uint64_t b = dets[i].beta;
            if (!beta_to_id_.count(b)) {
                beta_to_id_[b] = unique_betas_.size();
                unique_betas_.push_back(b);
            }
        }
    }

    // 构建 alpha/ beta 到 det 索引的邻接表
    void buildAdjacency(const std::vector<Det>& dets) {
        alpha_id_to_det_ids_.assign(unique_alphas_.size(), {});
        beta_id_to_det_ids_.assign(unique_betas_.size(), {});
        for (int i = 0; i < int(dets.size()); ++i) {
            alpha_id_to_det_ids_[alpha_to_id_[dets[i].alpha]].push_back(i);
            beta_id_to_det_ids_[beta_to_id_[dets[i].beta]].push_back(i);
        }
        for (auto& v : alpha_id_to_det_ids_) std::sort(v.begin(), v.end());
        for (auto& v : beta_id_to_det_ids_)  std::sort(v.begin(), v.end());
    }

    // 构建 abm1 映射：half mask -> 所有 det 索引
    void buildAbm1(const std::vector<Det>& dets) {
        abm1_alpha_to_det_ids_.clear();
        abm1_beta_to_det_ids_.clear();
        for (int i = 0; i < int(dets.size()); ++i) {
            uint64_t a = dets[i].alpha;
            uint64_t mask = a;
            while (mask) {
                int p = __builtin_ctzll(mask);
                mask &= mask - 1;
                uint64_t m1 = a ^ (1ULL << p);
                abm1_alpha_to_det_ids_[m1].push_back(i);
            }
            uint64_t b = dets[i].beta;
            mask = b;
            while (mask) {
                int p = __builtin_ctzll(mask);
                mask &= mask - 1;
                uint64_t m1 = b ^ (1ULL << p);
                abm1_beta_to_det_ids_[m1].push_back(i);
            }
        }
        for (auto& kv : abm1_alpha_to_det_ids_) std::sort(kv.second.begin(), kv.second.end());
        for (auto& kv : abm1_beta_to_det_ids_)  std::sort(kv.second.begin(), kv.second.end());
    }

    // 同自旋 α 激发：按 β 区分组
    void enumAlphaExcitations(
        std::vector<std::vector<Triplet>>& local_lists,
        const H1& h1, const ERI& eri) const {
#pragma omp parallel for schedule(dynamic)
        for (int bid = 0; bid < int(beta_id_to_det_ids_.size()); ++bid) {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            auto& local = local_lists[tid];
            const auto& group = beta_id_to_det_ids_[bid];
            int n = group.size();
            for (int ii = 0; ii < n; ++ii) {
                int i = group[ii];
                for (int jj = ii; jj < n; ++jj) {
                    int j = group[jj];
                    if (__builtin_popcountll((*dets_)[i].alpha ^ (*dets_)[j].alpha) > 4) continue;
                    double Hij = compute_H_ij((*dets_)[i], (*dets_)[j], h1, eri);
                    if (std::abs(Hij) <= threshold_) continue;
                    local.emplace_back(i, j, Hij);
                    if (i != j) local.emplace_back(j, i, Hij);
                }
            }
        }
    }

    // 同自旋 β 激发：按 α 区分组
    void enumBetaExcitations(
        std::vector<std::vector<Triplet>>& local_lists,
        const H1& h1, const ERI& eri) const {
#pragma omp parallel for schedule(dynamic)
        for (int aid = 0; aid < int(alpha_id_to_det_ids_.size()); ++aid) {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            auto& local = local_lists[tid];
            const auto& group = alpha_id_to_det_ids_[aid];
            int n = group.size();
            for (int ii = 0; ii < n; ++ii) {
                int i = group[ii];
                for (int jj = ii + 1; jj < n; ++jj) {
                    int j = group[jj];
                    if (__builtin_popcountll((*dets_)[i].beta ^ (*dets_)[j].beta) > 4) continue;
                    double Hij = compute_H_ij((*dets_)[i], (*dets_)[j], h1, eri);
                    if (std::abs(Hij) <= threshold_) continue;
                    local.emplace_back(i, j, Hij);
                    local.emplace_back(j, i, Hij);
                }
            }
        }
    }

    // 异自旋双激发：按 abm1 映射
    void enumOppositeSpinExcitations(
        std::vector<std::vector<Triplet>>& local_lists,
        const H1& h1, const ERI& eri) const {
        int dim = int(dets_->size());
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < dim; ++i) {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            auto& local = local_lists[tid];
            const auto& di = (*dets_)[i];
            uint64_t ai = di.alpha;
            uint64_t bi = di.beta;
            std::vector<int> visited(dim, 0);
            int tag = i + 1;
            std::vector<int> cand;
            cand.reserve(16);

            uint64_t maskA = ai;
            while (maskA) {
                int p = __builtin_ctzll(maskA);
                maskA &= maskA - 1;
                uint64_t m1a = ai ^ (1ULL << p);
                auto itA = abm1_alpha_to_det_ids_.find(m1a);
                if (itA == abm1_alpha_to_det_ids_.end()) continue;
                const auto& vA = itA->second;

                uint64_t maskB = bi;
                while (maskB) {
                    int q = __builtin_ctzll(maskB);
                    maskB &= maskB - 1;
                    uint64_t m1b = bi ^ (1ULL << q);
                    auto itB = abm1_beta_to_det_ids_.find(m1b);
                    if (itB == abm1_beta_to_det_ids_.end()) continue;
                    const auto& vB = itB->second;

                    size_t ia = 0, ib = 0;
                    while (ia < vA.size() && ib < vB.size()) {
                        int a = vA[ia], b = vB[ib];
                        if (a < b) ++ia;
                        else if (a > b) ++ib;
                        else {
                            if (a > i && visited[a] != tag) {
                                visited[a] = tag;
                                cand.push_back(a);
                            }
                            ++ia; ++ib;
                        }
                    }
                }
            }

            for (int j : cand) {
                double Hij = compute_H_ij(di, (*dets_)[j], h1, eri);
                if (std::abs(Hij) <= threshold_) continue;
                local.emplace_back(i, j, Hij);
                local.emplace_back(j, i, Hij);
            }
        }
    }
};

// 简洁封装接口
template<typename Det, typename H1, typename ERI>
void generate_triplets_by_excitations(
    const std::vector<Det>& dets,
    const H1& h1,
    const ERI& eri,
    std::vector<Triplet>& triplets,
    int norb,
    double threshold = 1e-12
) {
    ExcitationGenerator<Det,H1,ERI> gen(norb, threshold);
    gen.init(dets);
    gen.generateTriplets(h1, eri, triplets);
}

} // namespace trimci_core