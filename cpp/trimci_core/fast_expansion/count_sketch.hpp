#pragma once
/**
 * Streaming sketches for PT2 energy estimation.
 *
 * CountSketch: estimates ‖g‖² = Σ_a g_a² from streaming (key, value) updates.
 *   Memory: d * w * 8 bytes (d=5, w=200K → 8 MB)
 *   Error:  relative std ≈ √(2/w) per row, median-of-d for concentration
 *
 * HyperLogLog: estimates the number of unique keys (cardinality).
 *   Memory: m registers × 6 bits ≈ 0.75m bytes (m=16384 → 12 KB)
 *   Error:  σ ≈ 1.04/√m
 *   Range:  up to 2^64 ≈ 1.8×10^19
 *
 * References:
 *   Count Sketch: Charikar, Chen, Farach-Colton (2004)
 *   HyperLogLog: Flajolet, Fusy, Gandouet, Meunier (2007)
 */

#include <vector>
#include <cstdint>
#include <cmath>
#include <cstddef>
#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <string>

#include "fe_types.hpp"
#include "determinant.hpp"

namespace trimci_core {
namespace fe {

template<typename StorageType>
class CountSketch {
public:
    CountSketch() = default;

    CountSketch(size_t w, size_t d, uint64_t seed = 42)
        : w_(round_up_pow2(w)), d_(d), w_mask_(w_ - 1), data_(d * w_, 0.0)
    {
        init_hash(seed);
    }

    /// Reset accumulator to zeros (keep hash params).
    void clear_data() {
        std::fill(data_.begin(), data_.end(), 0.0);
        n_updates_ = 0;
    }

    /// Update sketch with (key=det, value).
    void update(const DeterminantT<StorageType>& key, double value) {
        uint64_t dk = make_det_key(key);
        for (size_t r = 0; r < d_; ++r) {
            size_t bucket = compute_bucket(r, dk);
            double sign = compute_sign(r, dk);
            data_[r * w_ + bucket] += sign * value;
        }
        ++n_updates_;
    }

    /// Return median-of-rows estimate of ‖g‖².
    double estimate_l2_norm_sq() const {
        std::vector<double> row_estimates(d_);
        for (size_t r = 0; r < d_; ++r) {
            double sum = 0.0;
            const double* row = data_.data() + r * w_;
            for (size_t j = 0; j < w_; ++j) {
                sum += row[j] * row[j];
            }
            row_estimates[r] = sum;
        }
        std::sort(row_estimates.begin(), row_estimates.end());
        return row_estimates[d_ / 2];  // median
    }

    /// Inner product with another sketch (must share same hash/sign functions).
    /// Returns median-of-rows estimate of ⟨u, v⟩.
    double inner_product(const CountSketch& other) const {
        std::vector<double> row_products(d_);
        for (size_t r = 0; r < d_; ++r) {
            double sum = 0.0;
            const double* this_row = data_.data() + r * w_;
            const double* other_row = other.data_.data() + r * w_;
            for (size_t j = 0; j < w_; ++j) {
                sum += this_row[j] * other_row[j];
            }
            row_products[r] = sum;
        }
        std::sort(row_products.begin(), row_products.end());
        return row_products[d_ / 2];  // median
    }

    /// Element-wise addition for parallel reduction.
    void merge(const CountSketch& other) {
        for (size_t i = 0; i < data_.size(); ++i) {
            data_[i] += other.data_[i];
        }
        n_updates_ += other.n_updates_;
    }

    /// Scaled merge: data_[i] += alpha * other.data_[i].
    /// Used in dressed CI to build S_v = Σ_j v_j · S_j.
    void add_scaled(const CountSketch& other, double alpha) {
        for (size_t i = 0; i < data_.size(); ++i) {
            data_[i] += alpha * other.data_[i];
        }
    }

    /// Streaming inner product: accumulate ⟨f, this_sketch⟩ one key at a time.
    /// row_accum[r] += value * ξ_r(key) * data[r][h_r(key)]
    /// After processing all keys, call median_estimate(row_accum) for result.
    /// Used in hybrid dressed CI for on-the-fly (K·v)_i computation.
    void accumulate_inner_product(std::vector<double>& row_accum,
                                   const DeterminantT<StorageType>& key,
                                   double value) const {
        uint64_t dk = make_det_key(key);
        for (size_t r = 0; r < d_; ++r) {
            size_t bucket = compute_bucket(r, dk);
            double sign = compute_sign(r, dk);
            row_accum[r] += value * sign * data_[r * w_ + bucket];
        }
    }

    /// Finalize streaming inner product: return median of per-row estimates.
    static double median_estimate(std::vector<double>& row_estimates) {
        std::sort(row_estimates.begin(), row_estimates.end());
        return row_estimates[row_estimates.size() / 2];
    }

    /// Direct data access for optimized operations.
    const double* data() const { return data_.data(); }
    double* data() { return data_.data(); }
    size_t data_size() const { return data_.size(); }

    size_t memory_bytes() const { return data_.size() * sizeof(double); }
    size_t n_updates() const { return n_updates_; }
    size_t width() const { return w_; }
    size_t depth() const { return d_; }

    /// Save sketch data to binary file.
    /// Format: "TCSK" (4B) | w uint64 | d uint64 | seed uint64 | n_updates uint64 |
    ///         hash_a[d] uint64 | hash_b[d] uint64 | sign_a[d] uint64 | sign_b[d] uint64 |
    ///         data[d*w] float64
    void save_to_file(const std::string& path) const {
        std::ofstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("CountSketch::save_to_file: cannot open " + path);
        f.write("TCSK", 4);
        auto write_u64 = [&](uint64_t v) { f.write(reinterpret_cast<const char*>(&v), 8); };
        write_u64(w_);
        write_u64(d_);
        write_u64(0);  // seed placeholder (hash params saved explicitly)
        write_u64(n_updates_);
        f.write(reinterpret_cast<const char*>(hash_a_.data()), d_ * 8);
        f.write(reinterpret_cast<const char*>(hash_b_.data()), d_ * 8);
        f.write(reinterpret_cast<const char*>(sign_a_.data()), d_ * 8);
        f.write(reinterpret_cast<const char*>(sign_b_.data()), d_ * 8);
        f.write(reinterpret_cast<const char*>(data_.data()), data_.size() * 8);
        if (!f) throw std::runtime_error("CountSketch::save_to_file: write failed for " + path);
    }

    /// Load sketch from binary file. Restores hash params and data.
    static CountSketch load_from_file(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("CountSketch::load_from_file: cannot open " + path);
        char magic[4];
        f.read(magic, 4);
        if (std::string(magic, 4) != "TCSK")
            throw std::runtime_error("CountSketch::load_from_file: bad magic");
        auto read_u64 = [&]() -> uint64_t {
            uint64_t v; f.read(reinterpret_cast<char*>(&v), 8); return v;
        };
        uint64_t w = read_u64();
        uint64_t d = read_u64();
        read_u64();  // seed placeholder
        uint64_t n_upd = read_u64();
        CountSketch sk;
        sk.w_ = w;
        sk.d_ = d;
        sk.w_mask_ = w - 1;
        sk.n_updates_ = n_upd;
        sk.hash_a_.resize(d); sk.hash_b_.resize(d);
        sk.sign_a_.resize(d); sk.sign_b_.resize(d);
        f.read(reinterpret_cast<char*>(sk.hash_a_.data()), d * 8);
        f.read(reinterpret_cast<char*>(sk.hash_b_.data()), d * 8);
        f.read(reinterpret_cast<char*>(sk.sign_a_.data()), d * 8);
        f.read(reinterpret_cast<char*>(sk.sign_b_.data()), d * 8);
        sk.data_.resize(d * w);
        f.read(reinterpret_cast<char*>(sk.data_.data()), d * w * 8);
        if (!f) throw std::runtime_error("CountSketch::load_from_file: read failed for " + path);
        return sk;
    }

    /// Public accessors for Precomputed Sketch Fingerprint (PSF).
    /// Expose hash/sign functions so fingerprint can be built externally.
    static uint64_t det_key(const DeterminantT<StorageType>& det) {
        return make_det_key(det);
    }
    size_t bucket_for(size_t r, uint64_t dk) const { return compute_bucket(r, dk); }
    double sign_for(size_t r, uint64_t dk) const { return compute_sign(r, dk); }

private:
    size_t w_ = 0;
    size_t d_ = 0;
    size_t w_mask_ = 0;  // w_ - 1 for fast modulo (w_ is power of 2)
    std::vector<double> data_;  // d_ × w_ flattened, row-major
    size_t n_updates_ = 0;

    /// Round up to next power of 2 (or return n if already power of 2).
    static size_t round_up_pow2(size_t n) {
        if (n == 0) return 1;
        --n;
        n |= n >> 1;
        n |= n >> 2;
        n |= n >> 4;
        n |= n >> 8;
        n |= n >> 16;
        n |= n >> 32;
        return n + 1;
    }

    // 2-universal hash family parameters: h(x) = (a*x + b) mod p mod w
    static constexpr uint64_t PRIME = (1ULL << 61) - 1;  // Mersenne prime
    std::vector<uint64_t> hash_a_, hash_b_;  // bucket hash
    std::vector<uint64_t> sign_a_, sign_b_;  // sign hash

    void init_hash(uint64_t seed) {
        hash_a_.resize(d_);
        hash_b_.resize(d_);
        sign_a_.resize(d_);
        sign_b_.resize(d_);

        // Use SplitMix64 to generate parameters from seed
        uint64_t s = seed;
        auto next = [&s]() -> uint64_t {
            s += 0x9e3779b97f4a7c15ULL;
            uint64_t z = s;
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
            return z ^ (z >> 31);
        };

        for (size_t r = 0; r < d_; ++r) {
            hash_a_[r] = (next() % (PRIME - 1)) + 1;  // a ∈ [1, p-1]
            hash_b_[r] = next() % PRIME;               // b ∈ [0, p-1]
            sign_a_[r] = (next() % (PRIME - 1)) + 1;
            sign_b_[r] = next() % PRIME;
        }
    }

    /// Combine alpha+beta bitstrings into a single 64-bit key.
    static uint64_t make_det_key(const DeterminantT<StorageType>& det) {
        SplitMix64Hash h;
        uint64_t h1 = h(det.alpha);
        uint64_t h2 = h(det.beta);
        return h1 ^ (h2 * 0x9e3779b97f4a7c15ULL);
    }

    /// Multiply two uint64_t and reduce mod Mersenne prime (2^61 - 1).
    /// Portable across GCC/Clang (native __uint128_t) and MSVC (_umul128).
    static uint64_t mul_mod_mersenne(uint64_t a, uint64_t b) {
#if defined(_MSC_VER)
        // MSVC: _umul128 returns low 64 bits and writes high 64 bits via pointer.
        // The 128-bit product is hi:lo where x = (hi << 64) | lo.
        // x >> 61 = (hi << 3) | (lo >> 61); only the upper 61 bits of hi
        // matter here because hash_a_/sign_a_ are < 2^61, so hi < 2^61
        // and (hi << 3) fits in uint64_t.
        uint64_t hi;
        uint64_t lo = _umul128(a, b, &hi);
        uint64_t lo_part = lo & PRIME;
        uint64_t hi_part = (lo >> 61) | (hi << 3);
        uint64_t result = lo_part + hi_part;
#else
        __uint128_t x = static_cast<__uint128_t>(a) * b;
        uint64_t lo_part = static_cast<uint64_t>(x) & PRIME;
        uint64_t hi_part = static_cast<uint64_t>(x >> 61);
        uint64_t result = lo_part + hi_part;
#endif
        // Second reduction (hi can be up to 2^67)
        uint64_t lo = result & PRIME;
        uint64_t hi = result >> 61;
        result = lo + hi;
        if (result >= PRIME) result -= PRIME;
        return result;
    }

    /// Compute bucket index for row r.
    size_t compute_bucket(size_t r, uint64_t det_key) const {
        uint64_t axb = mul_mod_mersenne(hash_a_[r], det_key);
        axb = (axb + hash_b_[r]) % PRIME;  // safe: both < 2^61
        return static_cast<size_t>(axb & w_mask_);  // fast: w_ is power of 2
    }

    /// Compute sign (+1.0 or -1.0) for row r.
    double compute_sign(size_t r, uint64_t det_key) const {
        uint64_t axb = mul_mod_mersenne(sign_a_[r], det_key);
        axb = (axb + sign_b_[r]) % PRIME;
        return (axb & 1) ? -1.0 : 1.0;
    }
};

// ============================================================================
// HyperLogLog: streaming cardinality (unique count) estimation
// ============================================================================

template<typename StorageType>
class HyperLogLog {
public:
    HyperLogLog() = default;

    /// @param p  log2(m), where m = number of registers. p=14 → m=16384, ~0.8% error
    explicit HyperLogLog(int p) : p_(p), m_(1u << p), registers_(m_, 0) {
        // Bias correction constant α_m
        switch (m_) {
            case 16:   alpha_m_ = 0.673;  break;
            case 32:   alpha_m_ = 0.697;  break;
            case 64:   alpha_m_ = 0.709;  break;
            default:   alpha_m_ = 0.7213 / (1.0 + 1.079 / m_);  break;
        }
    }

    /// Add a determinant to the counter.
    void add(const DeterminantT<StorageType>& det) {
        uint64_t h = make_hash(det);
        uint32_t idx = h >> (64 - p_);          // top p bits → register index
        uint64_t w = (h << p_) | (1ULL << (p_ - 1));  // remaining bits (set guard bit)
        uint8_t rho = static_cast<uint8_t>(clz64(w) + 1);  // leading zeros + 1
        if (rho > registers_[idx]) {
            registers_[idx] = rho;
        }
    }

    /// Merge another HLL into this one (for parallel reduction).
    void merge(const HyperLogLog& other) {
        for (uint32_t i = 0; i < m_; ++i) {
            if (other.registers_[i] > registers_[i]) {
                registers_[i] = other.registers_[i];
            }
        }
    }

    /// Estimate cardinality (number of unique elements seen).
    double estimate() const {
        // Harmonic mean of 2^(-reg[j])
        double Z = 0.0;
        uint32_t V = 0;  // count of zero registers (for small range correction)
        for (uint32_t i = 0; i < m_; ++i) {
            Z += 1.0 / static_cast<double>(1ULL << registers_[i]);
            if (registers_[i] == 0) ++V;
        }
        double E = alpha_m_ * m_ * m_ / Z;

        // Small range correction (linear counting)
        if (E <= 2.5 * m_ && V > 0) {
            E = m_ * std::log(static_cast<double>(m_) / V);
        }
        // Large range correction not needed for 64-bit hash

        return E;
    }

    size_t memory_bytes() const { return registers_.size(); }
    uint32_t n_registers() const { return m_; }
    double relative_error() const { return 1.04 / std::sqrt(static_cast<double>(m_)); }

private:
    int p_ = 14;
    uint32_t m_ = 16384;
    double alpha_m_ = 0.7213 / (1.0 + 1.079 / 16384.0);
    std::vector<uint8_t> registers_;

    static uint64_t make_hash(const DeterminantT<StorageType>& det) {
        SplitMix64Hash h;
        uint64_t h1 = h(det.alpha);
        uint64_t h2 = h(det.beta);
        // Use a different combiner than CountSketch to avoid correlation
        return SplitMix64Hash::mix(h1 ^ (h2 + 0x517cc1b727220a95ULL + (h1 << 6) + (h1 >> 2)));
    }

    static int clz64(uint64_t x) {
        if (x == 0) return 64;
#if defined(__GNUC__) || defined(__clang__)
        return __builtin_clzll(x);
#else
        // Portable fallback
        int n = 0;
        if (x <= 0x00000000FFFFFFFFULL) { n += 32; x <<= 32; }
        if (x <= 0x0000FFFFFFFFFFFFULL) { n += 16; x <<= 16; }
        if (x <= 0x00FFFFFFFFFFFFFFULL) { n +=  8; x <<=  8; }
        if (x <= 0x0FFFFFFFFFFFFFFFULL) { n +=  4; x <<=  4; }
        if (x <= 0x3FFFFFFFFFFFFFFFULL) { n +=  2; x <<=  2; }
        if (x <= 0x7FFFFFFFFFFFFFFFULL) { n +=  1; }
        return n;
#endif
    }
};

}  // namespace fe
}  // namespace trimci_core
