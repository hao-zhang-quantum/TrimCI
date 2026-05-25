#pragma once
/**
 * Fast Expansion: Alpha/Beta String Indexing
 *
 * Given a set of determinants, builds CSR-based index structures that enable
 * efficient enumeration of all non-zero Hamiltonian connections for each det.
 *
 * Key data structures:
 *   - CSR flat arrays: alpha_id -> list of det indices (and their beta_ids)
 *   - abm1 table: "alpha minus 1" -> alpha_ids sharing a common (N-1)-substring
 *   - alpha_singles CSR: alpha_id -> all alpha_ids differing by exactly 1 orbital
 *
 * Memory: ~1.7 GB for 10^8 dets (n_orb=36), vs ~60 GB with vector<vector<>>.
 */

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <functional>

#include "fe_types.hpp"
#include "determinant.hpp"
#include "simd_popcount.hpp"

namespace trimci_core {
namespace fe {

template<typename StorageType>
class ABIndex {
public:
    /// Build index from scratch for all dets.
    void build(const std::vector<DeterminantT<StorageType>>& dets, int n_orb);

    /// Incremental update: only process dets[old_size .. dets.size()).
    void update(const std::vector<DeterminantT<StorageType>>& dets, size_t old_size);

    /// Enumerate all non-zero connections of det_id.
    /// callback(connected_det_id, excitation_type, alpha_j, beta_j)
    template<typename Callback>
    void for_each_connection(size_t det_id, Callback&& callback) const;

    /// Iterate over alpha groups for cache-friendly matvec.
    /// callback(alpha_id, det_ids_begin, det_ids_end)
    template<typename Callback>
    void for_each_alpha_group(Callback&& callback) const;

    size_t n_dets() const { return n_dets_; }
    size_t n_unique_alpha() const { return unique_alphas_.size(); }
    size_t n_unique_beta() const { return unique_betas_.size(); }

    /// Per-det alpha/beta group accessors (for dressed matvec).
    uint32_t det_alpha_id(size_t i) const { return det_alpha_id_[i]; }
    uint32_t det_beta_id(size_t i) const { return det_beta_id_[i]; }
    const StorageType& unique_alpha(uint32_t a) const { return unique_alphas_[a]; }
    const StorageType& unique_beta(uint32_t b) const { return unique_betas_[b]; }

    /// Beta CSR accessors (for dual-order gather/scatter and beta pass).
    const uint32_t* beta_det_flat_data() const { return beta_det_flat_.data(); }
    const StorageType* beta_alpha_strings_data() const { return beta_alpha_strings_.data(); }
    size_t beta_group_begin(uint32_t b) const { return beta_det_ptr_[b]; }
    size_t beta_group_end(uint32_t b) const { return beta_det_ptr_[b + 1]; }

    /// Enumerate only Channel 1 (same-alpha) + Channel 2 (mixed) connections.
    /// Skips Channel 3 (same-beta) — used by dual-order alpha pass.
    template<typename Callback>
    void for_each_connection_alpha_mixed(size_t det_id, Callback&& callback) const;

    /// Iterate over beta groups (mirror of for_each_alpha_group).
    template<typename Callback>
    void for_each_beta_group(Callback&& callback) const;

    /// Save serialized index to a binary file.
    void save(const std::string& path) const;

    /// Load serialized index from a binary file.
    static ABIndex<StorageType> load(const std::string& path);

    size_t memory_bytes() const;
    void clear();

private:
    size_t n_dets_ = 0;
    int n_orb_ = 0;

    // === Unique string tables ===
    std::vector<StorageType> unique_alphas_;
    std::vector<StorageType> unique_betas_;
    fe_map<StorageType, uint32_t, SplitMix64Hash> alpha_to_id_;
    fe_map<StorageType, uint32_t, SplitMix64Hash> beta_to_id_;

    // === CSR: alpha_id -> det indices ===
    std::vector<uint32_t> alpha_det_flat_;    // [N_dets] det indices sorted by alpha_id
    std::vector<size_t>   alpha_det_ptr_;     // [n_unique_alpha + 1]
    std::vector<uint32_t> alpha_beta_ids_;    // [N_dets] corresponding beta_id for each entry

    // === CSR: beta_id -> det indices ===
    std::vector<uint32_t> beta_det_flat_;
    std::vector<size_t>   beta_det_ptr_;
    std::vector<uint32_t> beta_alpha_ids_;    // corresponding alpha_id for each entry

    // === abm1-derived singles neighbor table (CSR) ===
    std::vector<uint32_t> alpha_singles_flat_;
    std::vector<size_t>   alpha_singles_ptr_;
    std::vector<uint32_t> beta_singles_flat_;
    std::vector<size_t>   beta_singles_ptr_;

    // === Inline string storage (sequential access, no indirection) ===
    std::vector<StorageType> alpha_beta_strings_;  // [N_dets] beta string at each alpha CSR pos
    std::vector<StorageType> beta_alpha_strings_;  // [N_dets] alpha string at each beta CSR pos

    // === Per-det alpha_id and beta_id ===
    std::vector<uint32_t> det_alpha_id_;      // [N_dets]
    std::vector<uint32_t> det_beta_id_;       // [N_dets]

    // Internal helpers
    void build_unique_strings(const std::vector<DeterminantT<StorageType>>& dets);
    void build_csr(const std::vector<DeterminantT<StorageType>>& dets);
    void build_singles_table();
};

// ============================================================================
// Inline template implementations (must be in header for cross-TU linkage)
// ============================================================================

template<typename StorageType>
template<typename Callback>
void ABIndex<StorageType>::for_each_connection(size_t det_id, Callback&& callback) const
{
    const uint32_t a_i = det_alpha_id_[det_id];
    const uint32_t b_i = det_beta_id_[det_id];
    const StorageType& alpha_i = unique_alphas_[a_i];
    const StorageType& beta_i  = unique_betas_[b_i];
    const uint32_t det_id_u32 = static_cast<uint32_t>(det_id);

    // Channel 1: Same-alpha (Δα=0, Δβ≤2)
    // SIMD scan on inline beta strings (sequential access)
    {
        const size_t begin = alpha_det_ptr_[a_i];
        const size_t end   = alpha_det_ptr_[a_i + 1];
        const uint32_t* det_ids = alpha_det_flat_.data() + begin;
        const StorageType* beta_strings = alpha_beta_strings_.data() + begin;

        simd::scan_popcount_sd(
            beta_strings,
            end - begin,
            beta_i,
            [&](size_t pos) {
                uint32_t j = det_ids[pos];
                if (j != det_id_u32)
                    callback(static_cast<size_t>(j), SAME_ALPHA_S,
                             alpha_i, beta_strings[pos]);
            },
            [&](size_t pos) {
                uint32_t j = det_ids[pos];
                if (j != det_id_u32)
                    callback(static_cast<size_t>(j), SAME_ALPHA_D,
                             alpha_i, beta_strings[pos]);
            }
        );
    }

    // Channel 2: Mixed doubles (Δα=1, Δβ=1) — abm1+Scan
    for (size_t sp = alpha_singles_ptr_[a_i]; sp < alpha_singles_ptr_[a_i + 1]; ++sp) {
        uint32_t a_prime = alpha_singles_flat_[sp];
        const size_t begin = alpha_det_ptr_[a_prime];
        const size_t end   = alpha_det_ptr_[a_prime + 1];
        const uint32_t* det_ids = alpha_det_flat_.data() + begin;
        const StorageType* beta_strings = alpha_beta_strings_.data() + begin;
        const StorageType& alpha_prime = unique_alphas_[a_prime];

        simd::scan_popcount_eq(
            beta_strings,
            end - begin,
            beta_i,
            2,
            [&](size_t pos) {
                callback(static_cast<size_t>(det_ids[pos]), MIXED_D,
                         alpha_prime, beta_strings[pos]);
            }
        );
    }

    // Channel 3: Same-beta (Δα≤2, Δβ=0)
    // SIMD scan on inline alpha strings (sequential access)
    {
        const size_t begin = beta_det_ptr_[b_i];
        const size_t end   = beta_det_ptr_[b_i + 1];
        const uint32_t* det_ids = beta_det_flat_.data() + begin;
        const StorageType* alpha_strings = beta_alpha_strings_.data() + begin;

        simd::scan_popcount_sd(
            alpha_strings,
            end - begin,
            alpha_i,
            [&](size_t pos) {
                uint32_t j = det_ids[pos];
                if (j != det_id_u32)
                    callback(static_cast<size_t>(j), SAME_BETA_S,
                             alpha_strings[pos], beta_i);
            },
            [&](size_t pos) {
                uint32_t j = det_ids[pos];
                if (j != det_id_u32)
                    callback(static_cast<size_t>(j), SAME_BETA_D,
                             alpha_strings[pos], beta_i);
            }
        );
    }
}

template<typename StorageType>
template<typename Callback>
void ABIndex<StorageType>::for_each_alpha_group(Callback&& callback) const
{
    for (size_t a = 0; a < unique_alphas_.size(); ++a) {
        size_t begin = alpha_det_ptr_[a];
        size_t end   = alpha_det_ptr_[a + 1];
        if (begin < end) {
            callback(static_cast<uint32_t>(a),
                     alpha_det_flat_.data() + begin,
                     alpha_det_flat_.data() + end);
        }
    }
}

template<typename StorageType>
template<typename Callback>
void ABIndex<StorageType>::for_each_connection_alpha_mixed(size_t det_id, Callback&& callback) const
{
    const uint32_t a_i = det_alpha_id_[det_id];
    const uint32_t b_i = det_beta_id_[det_id];
    const StorageType& alpha_i = unique_alphas_[a_i];
    const StorageType& beta_i  = unique_betas_[b_i];
    const uint32_t det_id_u32 = static_cast<uint32_t>(det_id);

    // Channel 1: Same-alpha (Δα=0, Δβ≤2)
    {
        const size_t begin = alpha_det_ptr_[a_i];
        const size_t end   = alpha_det_ptr_[a_i + 1];
        const uint32_t* det_ids = alpha_det_flat_.data() + begin;
        const StorageType* beta_strings = alpha_beta_strings_.data() + begin;

        simd::scan_popcount_sd(
            beta_strings,
            end - begin,
            beta_i,
            [&](size_t pos) {
                uint32_t j = det_ids[pos];
                if (j != det_id_u32)
                    callback(static_cast<size_t>(j), SAME_ALPHA_S,
                             alpha_i, beta_strings[pos]);
            },
            [&](size_t pos) {
                uint32_t j = det_ids[pos];
                if (j != det_id_u32)
                    callback(static_cast<size_t>(j), SAME_ALPHA_D,
                             alpha_i, beta_strings[pos]);
            }
        );
    }

    // Channel 2: Mixed doubles (Δα=1, Δβ=1)
    for (size_t sp = alpha_singles_ptr_[a_i]; sp < alpha_singles_ptr_[a_i + 1]; ++sp) {
        uint32_t a_prime = alpha_singles_flat_[sp];
        const size_t begin = alpha_det_ptr_[a_prime];
        const size_t end   = alpha_det_ptr_[a_prime + 1];
        const uint32_t* det_ids = alpha_det_flat_.data() + begin;
        const StorageType* beta_strings = alpha_beta_strings_.data() + begin;
        const StorageType& alpha_prime = unique_alphas_[a_prime];

        simd::scan_popcount_eq(
            beta_strings,
            end - begin,
            beta_i,
            2,
            [&](size_t pos) {
                callback(static_cast<size_t>(det_ids[pos]), MIXED_D,
                         alpha_prime, beta_strings[pos]);
            }
        );
    }

    // Channel 3 (same-beta) is OMITTED — handled by the beta pass in dual-order matvec.
}

template<typename StorageType>
template<typename Callback>
void ABIndex<StorageType>::for_each_beta_group(Callback&& callback) const
{
    for (size_t b = 0; b < unique_betas_.size(); ++b) {
        size_t begin = beta_det_ptr_[b];
        size_t end   = beta_det_ptr_[b + 1];
        if (begin < end) {
            callback(static_cast<uint32_t>(b),
                     beta_det_flat_.data() + begin,
                     beta_det_flat_.data() + end);
        }
    }
}

}  // namespace fe
}  // namespace trimci_core
