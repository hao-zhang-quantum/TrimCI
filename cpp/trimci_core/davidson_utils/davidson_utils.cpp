#include "davidson_utils.hpp"
#include <algorithm>
#include <numeric>
#include <cstring>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <cstdio>
#include <cassert>
#include <climits>
#include <stdexcept>

#ifdef USE_ROBIN_HOOD
#include <robin_hood.h>
#endif

#include "../fast_expansion/fe_types.hpp"  // SplitMix64Hash, fe_map, fe_set

#ifdef _OPENMP
#include "omp_compat.hpp"
#endif
#include "../common/parallel_sort.hpp"

namespace trimci_core {
namespace davidson_utils {

// ============================================================================
// Timer helper
// ============================================================================
static double now_sec() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// ============================================================================
// 1. groupby_argsort + adjacency build
// ============================================================================

GroupBuildResult groupby_argsort(
    const uint64_t* keys, const uint64_t* partner_keys,
    int64_t N, bool verbose)
{
    auto t0 = now_sec();
    auto _log = [&](const char* msg) {
        if (verbose) {
            std::printf("  [groupby_cpp] %s (%.1fs)\n", msg, now_sec() - t0);
            std::fflush(stdout);
        }
    };

    // 1. argsort keys
    std::vector<int64_t> order(N);
    std::iota(order.begin(), order.end(), int64_t{0});
    _log("argsort start");
    trimci::parallel_sort(order.begin(), order.end(),
        [keys](int64_t a, int64_t b) { return keys[a] < keys[b]; });

    char buf[128];
    std::snprintf(buf, sizeof(buf), "argsort done (%lld dets)", (long long)N);
    _log(buf);

    // 2. Find group boundaries
    std::vector<int64_t> breaks;
    breaks.reserve(N / 100);  // rough estimate
    uint64_t prev = keys[order[0]];
    for (int64_t i = 1; i < N; ++i) {
        uint64_t cur = keys[order[i]];
        if (cur != prev) {
            breaks.push_back(i);
            prev = cur;
        }
    }
    int64_t n_groups = static_cast<int64_t>(breaks.size()) + 1;
    std::snprintf(buf, sizeof(buf), "%lld groups found", (long long)n_groups);
    _log(buf);

    // 3. Build groups
    GroupBuildResult result;
    result.groups.resize(n_groups);
    result.map_keys.resize(n_groups);
    result.map_gids.resize(n_groups);

    // Group boundaries: [start[g], start[g+1])
    // breaks contains the start of each group > 0
    auto group_start = [&](int64_t g) -> int64_t {
        return g == 0 ? 0 : breaks[g - 1];
    };
    auto group_end = [&](int64_t g) -> int64_t {
        return g < n_groups - 1 ? breaks[g] : N;
    };

    for (int64_t g = 0; g < n_groups; ++g) {
        int64_t gs = group_start(g);
        int64_t ge = group_end(g);
        int64_t sz = ge - gs;

        uint64_t key = keys[order[gs]];
        result.groups[g].key = key;
        result.map_keys[g] = key;
        result.map_gids[g] = g;

        // Within-group sort by partner key (stable)
        // Build temp array of (partner_key, original_index) for this group
        struct PK { uint64_t pk; int64_t idx; };
        std::vector<PK> tmp(sz);
        for (int64_t i = 0; i < sz; ++i) {
            int64_t oi = order[gs + i];
            tmp[i] = {partner_keys[oi], oi};
        }
        std::stable_sort(tmp.begin(), tmp.end(),
            [](const PK& a, const PK& b) { return a.pk < b.pk; });

        auto& grp = result.groups[g];
        grp.partner_keys.resize(sz);
        grp.global_indices.resize(sz);
        for (int64_t i = 0; i < sz; ++i) {
            grp.partner_keys[i] = tmp[i].pk;
            grp.global_indices[i] = tmp[i].idx;
        }

        if (verbose && g > 0 && g % 100000 == 0) {
            std::snprintf(buf, sizeof(buf), "groups: %lld/%lld",
                          (long long)g, (long long)n_groups);
            _log(buf);
        }
    }

    std::snprintf(buf, sizeof(buf), "done: %lld groups", (long long)n_groups);
    _log(buf);
    return result;
}

void build_adjacency(
    const uint64_t* alpha_strings, int64_t n_alpha, int n_orb,
    std::vector<int64_t>& adj_offsets,
    std::vector<int32_t>& adj_list,
    bool verbose)
{
    auto t0 = now_sec();
    auto _log = [&](const char* msg) {
        if (verbose) {
            std::printf("  [adjacency_cpp] %s (%.1fs)\n", msg, now_sec() - t0);
            std::fflush(stdout);
        }
    };

    // Build sorted strings + index for binary search
    std::vector<int64_t> sorted_idx(n_alpha);
    std::iota(sorted_idx.begin(), sorted_idx.end(), int64_t{0});
    std::sort(sorted_idx.begin(), sorted_idx.end(),
        [alpha_strings](int64_t a, int64_t b) {
            return alpha_strings[a] < alpha_strings[b];
        });
    std::vector<uint64_t> sorted_strings(n_alpha);
    for (int64_t i = 0; i < n_alpha; ++i) {
        sorted_strings[i] = alpha_strings[sorted_idx[i]];
    }

    // Build hash map for O(1) lookup: string -> gid
    fe::fe_map<uint64_t, int32_t, fe::SplitMix64Hash> string_to_gid;
    string_to_gid.reserve(n_alpha);
    for (int64_t i = 0; i < n_alpha; ++i) {
        string_to_gid[alpha_strings[i]] = static_cast<int32_t>(i);
    }
    _log("hash map built");

    // Per-group adjacency sets (use sorted vector, deduplicated later)
    std::vector<std::vector<int32_t>> adj_per_group(n_alpha);

    // Enumerate all single excitations: for each m->p where m occ, p virt
    int64_t n_candidates = 0;
    int64_t n_matches = 0;
    for (int m = 0; m < n_orb; ++m) {
        uint64_t mask_m = 1ULL << m;
        for (int p = 0; p < n_orb; ++p) {
            if (p == m) continue;
            uint64_t mask_p = 1ULL << p;
            for (int64_t g = 0; g < n_alpha; ++g) {
                uint64_t s = alpha_strings[g];
                if ((s & mask_m) == 0) continue;  // m not occupied
                if ((s & mask_p) != 0) continue;  // p already occupied
                uint64_t excited = (s ^ mask_m) ^ mask_p;
                n_candidates++;
                auto it = string_to_gid.find(excited);
                if (it != string_to_gid.end()) {
                    int32_t dst_gid = it->second;
                    adj_per_group[g].push_back(dst_gid);
                    n_matches++;
                }
            }
        }
    }

    char buf[256];
    std::snprintf(buf, sizeof(buf), "%lld candidates, %lld matches",
                  (long long)n_candidates, (long long)n_matches);
    _log(buf);

    // Deduplicate and sort each adjacency list
    for (int64_t g = 0; g < n_alpha; ++g) {
        auto& v = adj_per_group[g];
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
    }

    // Build CSR output
    adj_offsets.resize(n_alpha + 1);
    adj_offsets[0] = 0;
    int64_t total = 0;
    for (int64_t g = 0; g < n_alpha; ++g) {
        total += static_cast<int64_t>(adj_per_group[g].size());
        adj_offsets[g + 1] = total;
    }
    adj_list.resize(total);
    for (int64_t g = 0; g < n_alpha; ++g) {
        int64_t off = adj_offsets[g];
        for (size_t i = 0; i < adj_per_group[g].size(); ++i) {
            adj_list[off + i] = adj_per_group[g][i];
        }
    }

    std::snprintf(buf, sizeof(buf), "done: %lld total pairs",
                  (long long)total);
    _log(buf);
}

// ============================================================================
// 2. build_v_compact_indices
// ============================================================================

void build_v_compact_indices(
    const int64_t* original_offsets,
    const int64_t* group_ids,
    const int64_t* sizes,
    int64_t n_needed,
    bool use_int32,
    std::vector<int32_t>& result_i32,
    std::vector<int64_t>& result_i64)
{
    // Compute total compact size
    int64_t total = 0;
    for (int64_t i = 0; i < n_needed; ++i) {
        total += sizes[group_ids[i]];
    }

    if (use_int32) {
        result_i32.resize(total);
        int64_t pos = 0;
        for (int64_t i = 0; i < n_needed; ++i) {
            int64_t gid = group_ids[i];
            int64_t orig_start = original_offsets[gid];
            int64_t sz = sizes[gid];
            for (int64_t j = 0; j < sz; ++j) {
                result_i32[pos++] = static_cast<int32_t>(orig_start + j);
            }
        }
    } else {
        result_i64.resize(total);
        int64_t pos = 0;
        for (int64_t i = 0; i < n_needed; ++i) {
            int64_t gid = group_ids[i];
            int64_t orig_start = original_offsets[gid];
            int64_t sz = sizes[gid];
            for (int64_t j = 0; j < sz; ++j) {
                result_i64[pos++] = orig_start + j;
            }
        }
    }
}

// ============================================================================
// 3. build_gpu_adj_csr
// ============================================================================

void build_gpu_adj_csr(
    int64_t n_alpha_groups,
    const int64_t* assigned_gids, int64_t n_assigned,
    const int64_t* adj_sizes,
    const int64_t* full_adj_offsets,
    const int32_t* full_adj_list,
    std::vector<int64_t>& gpu_adj_offsets,
    std::vector<int32_t>& gpu_adj_list)
{
    // Build assigned set for O(1) lookup
    fe::fe_set<int64_t> assigned_set;
    assigned_set.reserve(n_assigned);
    for (int64_t i = 0; i < n_assigned; ++i) {
        assigned_set.insert(assigned_gids[i]);
    }

    gpu_adj_offsets.resize(n_alpha_groups + 1);
    std::vector<int32_t> parts;
    int64_t pos = 0;

    for (int64_t gid = 0; gid < n_alpha_groups; ++gid) {
        gpu_adj_offsets[gid] = pos;
        if (assigned_set.count(gid)) {
            int64_t n = adj_sizes[gid];
            if (n > 0) {
                int64_t start = full_adj_offsets[gid];
                parts.insert(parts.end(),
                             full_adj_list + start,
                             full_adj_list + start + n);
            }
            pos += n;
        }
    }
    gpu_adj_offsets[n_alpha_groups] = pos;
    gpu_adj_list = std::move(parts);
}

// ============================================================================
// 4. TailFloat compress / decompress
// ============================================================================

TailFloatData tailfloat_compress(
    const double* v, int64_t N,
    double tau, double norm_budget)
{
    // Single pass: compute v32, find head elements, compute norms
    TailFloatData result;
    result.N = N;
    result.v32.resize(N);

    // First pass: convert to float32, identify heads, accumulate norms
    double total_norm_sq = 0.0;
    double tail_norm_sq = 0.0;
    std::vector<int32_t> head_idx_tmp;
    head_idx_tmp.reserve(N / 100);  // ~1% heads typical

    for (int64_t i = 0; i < N; ++i) {
        double vi = v[i];
        float v32i = static_cast<float>(vi);
        result.v32[i] = v32i;
        total_norm_sq += vi * vi;
        double avi = std::fabs(vi);
        if (avi > tau) {
            head_idx_tmp.push_back(static_cast<int32_t>(i));
        } else {
            tail_norm_sq += vi * vi;
        }
    }

    // Check if adaptive threshold is needed
    if (total_norm_sq > 0.0 && tail_norm_sq / total_norm_sq > norm_budget) {
        // Binary search for threshold
        double lo = 0.0, hi = tau;
        double target = norm_budget * total_norm_sq;
        for (int iter = 0; iter < 30; ++iter) {
            double mid = (lo + hi) / 2.0;
            double tns = 0.0;
            for (int64_t i = 0; i < N; ++i) {
                double avi = std::fabs(v[i]);
                if (avi <= mid) tns += v[i] * v[i];
            }
            if (tns > target) hi = mid;
            else lo = mid;
        }
        tau = lo;
        // Rebuild head list with new threshold
        head_idx_tmp.clear();
        tail_norm_sq = 0.0;
        for (int64_t i = 0; i < N; ++i) {
            double avi = std::fabs(v[i]);
            if (avi > tau) {
                head_idx_tmp.push_back(static_cast<int32_t>(i));
            } else {
                tail_norm_sq += v[i] * v[i];
            }
        }
    }

    result.head_idx = std::move(head_idx_tmp);
    result.K = static_cast<int64_t>(result.head_idx.size());

    // Compute head corrections: v[i] - float32(v[i])
    result.head_corr.resize(result.K);
    for (int64_t j = 0; j < result.K; ++j) {
        int32_t idx = result.head_idx[j];
        result.head_corr[j] = v[idx] - static_cast<double>(result.v32[idx]);
    }

    return result;
}

void tailfloat_decompress(
    const float* v32, int64_t N,
    const int32_t* head_idx, const double* head_corr, int64_t K,
    double* out)
{
    // Convert float32 -> float64
    for (int64_t i = 0; i < N; ++i) {
        out[i] = static_cast<double>(v32[i]);
    }
    // Apply head corrections
    for (int64_t j = 0; j < K; ++j) {
        out[head_idx[j]] += head_corr[j];
    }
}

// ============================================================================
// 5. Task manifest (cost-balanced packing)
// ============================================================================

ManifestResult build_mini_task_manifest(
    const int64_t* alpha_sizes, int64_t n_alpha,
    const int64_t* beta_sizes, int64_t n_beta,
    const int64_t* adj_offsets, const int32_t* adj_list,
    int64_t C,  // mini_task_cost
    bool verbose)
{
    if (n_alpha > INT32_MAX || n_beta > INT32_MAX) {
        throw std::overflow_error(
            "n_alpha or n_beta exceeds INT32_MAX; ManifestItem.gid needs int64");
    }
    auto t0 = now_sec();
    ManifestResult result;
    result.total_cost = 0;
    result.n_atoms_ch2 = 0;

    std::vector<ManifestItem> buf;
    int64_t buf_cost = 0;

    auto flush = [&]() {
        if (!buf.empty()) {
            result.mini_tasks.push_back({std::move(buf), buf_cost});
            buf.clear();
            buf_cost = 0;
        }
    };

    auto add = [&](uint8_t ch, int32_t gid, int32_t adj, int64_t rs, int64_t re) {
        int64_t cost = re - rs;
        result.total_cost += cost;
        if (!buf.empty() && buf_cost + cost > C * 3 / 2) {
            flush();
        }
        buf.push_back({ch, gid, adj, rs, re});
        buf_cost += cost;
        if (buf_cost >= C) {
            flush();
        }
    };

    // Ch1: per alpha group, row-split large groups
    for (int64_t gid = 0; gid < n_alpha; ++gid) {
        int64_t s = alpha_sizes[gid];
        if (s == 0) continue;
        if (s <= C) {
            add(1, static_cast<int32_t>(gid), -1, 0, s);
        } else {
            for (int64_t start = 0; start < s; start += C) {
                int64_t end = std::min(start + C, s);
                add(1, static_cast<int32_t>(gid), -1, start, end);
            }
        }
    }
    flush();
    result.n_mt_ch1 = static_cast<int64_t>(result.mini_tasks.size());

    // Ch3: per beta group, row-split large groups
    for (int64_t gid = 0; gid < n_beta; ++gid) {
        int64_t s = beta_sizes[gid];
        if (s == 0) continue;
        if (s <= C) {
            add(3, static_cast<int32_t>(gid), -1, 0, s);
        } else {
            for (int64_t start = 0; start < s; start += C) {
                int64_t end = std::min(start + C, s);
                add(3, static_cast<int32_t>(gid), -1, start, end);
            }
        }
    }
    flush();
    result.n_mt_ch3 = static_cast<int64_t>(result.mini_tasks.size()) - result.n_mt_ch1;

    // Ch2: per (gid, adj) pair, row-split large groups
    for (int64_t gid = 0; gid < n_alpha; ++gid) {
        int64_t s = alpha_sizes[gid];
        if (s == 0) continue;
        int64_t adj_start = adj_offsets[gid];
        int64_t adj_end = adj_offsets[gid + 1];
        for (int64_t ai = adj_start; ai < adj_end; ++ai) {
            int32_t adj_gid = adj_list[ai];
            if (s <= C) {
                add(2, static_cast<int32_t>(gid), adj_gid, 0, s);
                result.n_atoms_ch2++;
            } else {
                for (int64_t start = 0; start < s; start += C) {
                    int64_t end = std::min(start + C, s);
                    add(2, static_cast<int32_t>(gid), adj_gid, start, end);
                    result.n_atoms_ch2++;
                }
            }
        }
        if (verbose && gid > 0 && gid % 200000 == 0) {
            std::printf("  [manifest_cpp] ch2 atoms: %lld/%lld alpha groups (%.1fs)\n",
                        (long long)gid, (long long)n_alpha, now_sec() - t0);
            std::fflush(stdout);
        }
    }
    flush();
    result.n_mt_ch2 = static_cast<int64_t>(result.mini_tasks.size())
                    - result.n_mt_ch1 - result.n_mt_ch3;

    if (verbose) {
        int64_t n_items = 0;
        for (auto& mt : result.mini_tasks) {
            n_items += static_cast<int64_t>(mt.items.size());
        }
        std::printf("  [manifest_cpp] %lld mini-tasks "
                    "(ch1=%lld, ch3=%lld, ch2=%lld), %lld items (%.1fs)\n",
                    (long long)result.mini_tasks.size(),
                    (long long)result.n_mt_ch1,
                    (long long)result.n_mt_ch3,
                    (long long)result.n_mt_ch2,
                    (long long)n_items,
                    now_sec() - t0);
        std::fflush(stdout);
    }

    return result;
}

// ============================================================================
// 6. Sorted merge-match for warmstart
// ============================================================================

MergeMatchResult sorted_merge_match(
    const uint64_t* ws_alpha,  const uint64_t* ws_beta,
    const double*   ws_coeffs, int64_t M,
    const uint64_t* main_alpha, const uint64_t* main_beta,
    int64_t N, bool verbose, bool presorted_main)
{
    auto t0 = now_sec();
    auto _log = [&](const char* msg) {
        if (verbose) {
            std::printf("  [merge_match_cpp] %s (%.1fs)\n", msg, now_sec() - t0);
            std::fflush(stdout);
        }
    };

    // 1. Argsort warmstart by (alpha, beta) — always needed (TCPT is unsorted)
    std::vector<int64_t> ws_order(M);
    std::iota(ws_order.begin(), ws_order.end(), int64_t{0});
    trimci::parallel_sort(ws_order.begin(), ws_order.end(),
        [ws_alpha, ws_beta](int64_t a, int64_t b) {
            if (ws_alpha[a] != ws_alpha[b]) return ws_alpha[a] < ws_alpha[b];
            return ws_beta[a] < ws_beta[b];
        });

    char buf[128];
    std::snprintf(buf, sizeof(buf), "ws argsort done (%lld dets)", (long long)M);
    _log(buf);

    // 2. Argsort main — skip if caller filled main in group order
    std::vector<int64_t> main_order_buf;
    if (!presorted_main) {
        main_order_buf.resize(N);
        std::iota(main_order_buf.begin(), main_order_buf.end(), int64_t{0});
        trimci::parallel_sort(main_order_buf.begin(), main_order_buf.end(),
            [main_alpha, main_beta](int64_t a, int64_t b) {
                if (main_alpha[a] != main_alpha[b]) return main_alpha[a] < main_alpha[b];
                return main_beta[a] < main_beta[b];
            });
        std::snprintf(buf, sizeof(buf), "main argsort done (%lld dets)", (long long)N);
        _log(buf);
    } else {
        _log("presorted_main: skipping main argsort");
    }

    // 3. Two-pointer merge
    //    ws: always indirect via ws_order
    //    main: direct (j) if presorted_main, indirect (main_order_buf[j]) otherwise
    MergeMatchResult result;
    result.x0.resize(N, 0.0);
    result.matched = 0;

    int64_t i = 0, j = 0;
    while (i < M && j < N) {
        int64_t wi = ws_order[i];
        int64_t mj = presorted_main ? j : main_order_buf[j];
        uint64_t wa = ws_alpha[wi], wb = ws_beta[wi];
        uint64_t ma = main_alpha[mj], mb = main_beta[mj];
        if (wa == ma && wb == mb) {
            result.x0[mj] = ws_coeffs[wi];
            result.matched++;
            i++; j++;
        } else if (wa < ma || (wa == ma && wb < mb)) {
            i++;
        } else {
            j++;
        }
    }

    std::snprintf(buf, sizeof(buf), "merge done: %lld/%lld matched",
                  (long long)result.matched, (long long)M);
    _log(buf);

    return result;
}

}  // namespace davidson_utils
}  // namespace trimci_core
