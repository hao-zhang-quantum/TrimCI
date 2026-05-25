#pragma once
/**
 * davidson_utils: C++ accelerated helpers for the GPU Davidson solver.
 *
 * Five features:
 *   1. groupby_argsort + adjacency build
 *   2. build_v_compact_indices
 *   3. build_gpu_adj_csr
 *   4. TailFloat compress / decompress
 *   5. task manifest construction (cost-balanced packing)
 */

#include <cstdint>
#include <cstddef>
#include <vector>
#include <cmath>

#ifdef _OPENMP
#include "omp_compat.hpp"
#endif

namespace trimci_core {
namespace davidson_utils {

// ============================================================================
// 1. groupby_argsort + adjacency build
// ============================================================================

struct GroupData {
    uint64_t key;                        // alpha or beta bitstring
    std::vector<uint64_t> partner_keys;  // sorted partner bitstrings
    std::vector<int64_t> global_indices; // corresponding det indices
};

struct GroupBuildResult {
    std::vector<GroupData> groups;
    // key_to_gid: maps bitstring -> group index (returned as parallel arrays)
    std::vector<uint64_t> map_keys;
    std::vector<int64_t>  map_gids;
};

/// Build groups from keys/partner_keys via radix-style argsort.
/// Equivalent to Python _groupby_argsort.
GroupBuildResult groupby_argsort(
    const uint64_t* keys, const uint64_t* partner_keys,
    int64_t N, bool verbose);

/// Build alpha adjacency lists by enumerating single excitations.
/// Returns CSR: adj_offsets[n_alpha+1], adj_list[total_pairs].
/// Equivalent to the adjacency block in Python build_groups.
void build_adjacency(
    const uint64_t* alpha_strings, int64_t n_alpha, int n_orb,
    std::vector<int64_t>& adj_offsets,
    std::vector<int32_t>& adj_list,
    bool verbose);

// ============================================================================
// 2. build_v_compact_indices
// ============================================================================

/// Build compaction indices for a subset of groups.
/// single-pass fill, zero temporaries.
/// Equivalent to Python _build_v_compact_indices_vectorized.
void build_v_compact_indices(
    const int64_t* original_offsets,  // [n_groups+1]
    const int64_t* group_ids,         // [n_needed], sorted
    const int64_t* sizes,             // [n_groups]
    int64_t n_needed,
    bool use_int32,
    // outputs:
    std::vector<int32_t>& result_i32,
    std::vector<int64_t>& result_i64);

// ============================================================================
// 3. build_gpu_adj_csr
// ============================================================================

/// Build per-GPU adjacency CSR for assigned alpha groups.
/// Equivalent to Python MultiGPUContext._build_gpu_adj_csr.
void build_gpu_adj_csr(
    int64_t n_alpha_groups,
    const int64_t* assigned_gids, int64_t n_assigned,
    const int64_t* adj_sizes,       // [n_alpha_groups]
    const int64_t* full_adj_offsets, // [n_alpha_groups+1]
    const int32_t* full_adj_list,    // [total_adj_pairs]
    // outputs:
    std::vector<int64_t>& gpu_adj_offsets,   // [n_alpha_groups+1]
    std::vector<int32_t>& gpu_adj_list);

// ============================================================================
// 4. TailFloat compress / decompress
// ============================================================================

struct TailFloatData {
    std::vector<float>   v32;       // float32[N]
    std::vector<int32_t> head_idx;  // int32[K]
    std::vector<double>  head_corr; // float64[K]
    int64_t N;
    int64_t K;
};

/// Single-pass compress: scan v once, output v32 + head_idx + head_corr.
TailFloatData tailfloat_compress(
    const double* v, int64_t N,
    double tau, double norm_budget);

/// Decompress: v32 -> float64, apply head corrections.
void tailfloat_decompress(
    const float* v32, int64_t N,
    const int32_t* head_idx, const double* head_corr, int64_t K,
    double* out);

// ============================================================================
// 5. Task manifest (cost-balanced packing)
// ============================================================================

struct ManifestItem {
    uint8_t channel;  // 1, 2, or 3
    int32_t gid;
    int32_t adj;      // -1 for ch1/ch3
    int64_t row_start;
    int64_t row_end;
};

struct MiniTask {
    std::vector<ManifestItem> items;
    int64_t cost;
};

struct ManifestResult {
    std::vector<MiniTask> mini_tasks;
    int64_t n_mt_ch1;
    int64_t n_mt_ch3;
    int64_t n_mt_ch2;
    int64_t total_cost;
    int64_t n_atoms_ch2;
};

/// Build mini-task manifest with cost-balanced packing.
/// Equivalent to Python create_mini_task_manifest (the atom generation part).
ManifestResult build_mini_task_manifest(
    const int64_t* alpha_sizes, int64_t n_alpha,
    const int64_t* beta_sizes, int64_t n_beta,
    const int64_t* adj_offsets, const int32_t* adj_list,  // adjacency CSR
    int64_t mini_task_cost,
    bool verbose);

// ============================================================================
// 6. Sorted merge-match for warmstart
// ============================================================================

struct MergeMatchResult {
    std::vector<double> x0;      // [N_main] scattered coefficients
    int64_t matched;             // number of matched determinants
};

/// Match warmstart determinants against main basis by argsort + two-pointer merge.
/// When presorted_main=true, main arrays are already sorted by (alpha, beta)
/// (filled in group order), skipping the expensive O(N log N) main argsort.
/// Warmstart is always argsorted (TCPT checkpoint stores original order).
MergeMatchResult sorted_merge_match(
    const uint64_t* ws_alpha,  const uint64_t* ws_beta,
    const double*   ws_coeffs, int64_t M,
    const uint64_t* main_alpha, const uint64_t* main_beta,
    int64_t N, bool verbose, bool presorted_main = false);

}  // namespace davidson_utils
}  // namespace trimci_core
