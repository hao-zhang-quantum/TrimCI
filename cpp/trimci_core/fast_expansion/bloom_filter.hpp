#pragma once
/**
 * Fast Expansion: Bloom filter for O(1) membership check.
 *
 * Used in PT2 to check "is det a in the variational core?"
 *   - 10^8 core dets, FPR=0.1% → ~187 MB (vs hash set ~5 GB)
 *   - False positive → skip a valid external det → energy bias << μHa
 *
 * Header-only, ~80 lines.
 */

#include <vector>
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <algorithm>

#include "fe_types.hpp"
#include "determinant.hpp"

namespace trimci_core {
namespace fe {

template<typename StorageType>
class BloomFilter {
public:
    BloomFilter() = default;

    void init(size_t n_items, double false_positive_rate = 0.001) {
        double ln2 = std::log(2.0);
        n_bits_ = static_cast<size_t>(
            -double(n_items) * std::log(false_positive_rate) / (ln2 * ln2));
        n_bits_ = std::max<size_t>(n_bits_, 64);
        k_ = std::max<size_t>(1, static_cast<size_t>(
            std::llround(double(n_bits_) / double(n_items) * ln2)));
        bits_.assign((n_bits_ + 63) / 64, 0);
    }

    void add(const DeterminantT<StorageType>& det) {
        auto [h1, h2] = dual_hash(det);
        for (size_t i = 0; i < k_; ++i) {
            size_t idx = (h1 + i * h2) % n_bits_;
            bits_[idx >> 6] |= (1ULL << (idx & 63));
        }
    }

    bool maybe_has(const DeterminantT<StorageType>& det) const {
        auto [h1, h2] = dual_hash(det);
        for (size_t i = 0; i < k_; ++i) {
            size_t idx = (h1 + i * h2) % n_bits_;
            if ((bits_[idx >> 6] & (1ULL << (idx & 63))) == 0) return false;
        }
        return true;
    }

    size_t memory_bytes() const { return bits_.size() * 8; }
    size_t n_bits() const { return n_bits_; }
    size_t n_hash_functions() const { return k_; }

private:
    std::vector<uint64_t> bits_;
    size_t n_bits_ = 0;
    size_t k_ = 0;

    std::pair<uint64_t, uint64_t> dual_hash(const DeterminantT<StorageType>& det) const {
        SplitMix64Hash hasher;
        uint64_t h1 = hasher(det.alpha);
        uint64_t h2 = hasher(det.beta);
        // Combine differently for the two hash channels
        uint64_t a = h1 ^ (h2 * 0x9e3779b97f4a7c15ULL);
        uint64_t b = h2 ^ (h1 * 0x517cc1b727220a95ULL);
        return {a, b};
    }
};

}  // namespace fe
}  // namespace trimci_core
