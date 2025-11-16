#pragma once

#include <cstdint>
#include <vector>
#include <ostream>
#include <sstream>
#include <bitset>
#include <array>
#include <type_traits>
#include <unordered_set>

namespace trimci_core {

// Bit manipulation utilities for different storage types
template<typename StorageType>
struct BitOps {
    static constexpr int bits_per_unit = sizeof(StorageType) * 8;
    
    static bool get_bit(const StorageType& storage, int pos) {
        return (storage >> pos) & 1;
    }
    
    static void set_bit(StorageType& storage, int pos) {
        storage |= (StorageType(1) << pos);
    }
    
    static void clear_bit(StorageType& storage, int pos) {
        storage &= ~(StorageType(1) << pos);
    }
    
    static void flip_bit(StorageType& storage, int pos) {
        storage ^= (StorageType(1) << pos);
    }
};

// Specialization for array-based storage (for >64 orbitals)
template<size_t N>
struct BitOps<std::array<uint64_t, N>> {
    using StorageType = std::array<uint64_t, N>;
    static constexpr int bits_per_unit = 64;
    static constexpr int total_bits = N * 64;
    
    static bool get_bit(const StorageType& storage, int pos) {
        int unit = pos / bits_per_unit;
        int offset = pos % bits_per_unit;
        return (unit >= 0 && unit < static_cast<int>(N)) ? ((storage[unit] >> offset) & 1) : false;
    }
    
    static void set_bit(StorageType& storage, int pos) {
        int unit = pos / bits_per_unit;
        int offset = pos % bits_per_unit;
        if (unit >= 0 && unit < static_cast<int>(N) && offset >= 0 && offset < bits_per_unit) {
            storage[unit] |= (uint64_t(1) << offset);
        }
    }
    
    static void clear_bit(StorageType& storage, int pos) {
        int unit = pos / bits_per_unit;
        int offset = pos % bits_per_unit;
        if (unit >= 0 && unit < static_cast<int>(N) && offset >= 0 && offset < bits_per_unit) {
            storage[unit] &= ~(uint64_t(1) << offset);
        }
    }
    
    static void flip_bit(StorageType& storage, int pos) {
        int unit = pos / bits_per_unit;
        int offset = pos % bits_per_unit;
        if (unit >= 0 && unit < static_cast<int>(N) && offset >= 0 && offset < bits_per_unit) {
            storage[unit] ^= (uint64_t(1) << offset);
        }
    }
};

// Template Determinant class
template<typename StorageType>
class DeterminantT {
public:
    StorageType alpha;
    StorageType beta;
    
    using BitOpsType = BitOps<StorageType>;

    DeterminantT() : alpha{}, beta{} {}
    
    DeterminantT(const StorageType& alpha_mask, const StorageType& beta_mask) noexcept
        : alpha(alpha_mask), beta(beta_mask) {}

    bool operator==(const DeterminantT& other) const noexcept {
        return alpha == other.alpha && beta == other.beta;
    }

    bool operator<(const DeterminantT& other) const noexcept {
        if (alpha < other.alpha) return true;
        if (alpha > other.alpha) return false;
        return beta < other.beta;
    }

    // Get occupied orbital indices
    std::vector<int> getOccupiedAlpha() const noexcept {
        std::vector<int> occ;
        if constexpr (std::is_same_v<StorageType, uint64_t>) {
            // Optimized version for uint64_t
            StorageType m = alpha;
            int idx = 0;
            while (m) {
                if (m & 1) occ.push_back(idx);
                m >>= 1; 
                ++idx;
            }
        } else {
            // General version for array storage
            constexpr int max_orbs = BitOpsType::total_bits;
            for (int i = 0; i < max_orbs; ++i) {
                if (BitOpsType::get_bit(alpha, i)) {
                    occ.push_back(i);
                }
            }
        }
        return occ;
    }

    std::vector<int> getOccupiedBeta() const noexcept {
        std::vector<int> occ;
        if constexpr (std::is_same_v<StorageType, uint64_t>) {
            // Optimized version for uint64_t
            StorageType m = beta;
            int idx = 0;
            while (m) {
                if (m & 1) occ.push_back(idx);
                m >>= 1; 
                ++idx;
            }
        } else {
            // General version for array storage
            constexpr int max_orbs = BitOpsType::total_bits;
            for (int i = 0; i < max_orbs; ++i) {
                if (BitOpsType::get_bit(beta, i)) {
                    occ.push_back(i);
                }
            }
        }
        return occ;
    }

    // Single excitation: i -> p
    DeterminantT singleExcite(int i, int p, bool isAlpha) const {
        if (isAlpha) {
            if (!BitOpsType::get_bit(alpha, i))
                throw std::runtime_error("Alpha orbital not occupied");
            if (BitOpsType::get_bit(alpha, p))
                throw std::runtime_error("Alpha target orbital already occupied");
            
            StorageType new_alpha = alpha;
            BitOpsType::clear_bit(new_alpha, i);
            BitOpsType::set_bit(new_alpha, p);
            return DeterminantT(new_alpha, beta);
        } else {
            if (!BitOpsType::get_bit(beta, i))
                throw std::runtime_error("Beta orbital not occupied");
            if (BitOpsType::get_bit(beta, p))
                throw std::runtime_error("Beta target orbital already occupied");
            
            StorageType new_beta = beta;
            BitOpsType::clear_bit(new_beta, i);
            BitOpsType::set_bit(new_beta, p);
            return DeterminantT(alpha, new_beta);
        }
    }

    // Double excitation: i,j -> p,q
    DeterminantT doubleExcite(int i, int j, int p, int q, bool isAlpha) const {
        if (isAlpha) {
            if (!BitOpsType::get_bit(alpha, i) || !BitOpsType::get_bit(alpha, j))
                throw std::runtime_error("Alpha orbitals not both occupied");
            if (BitOpsType::get_bit(alpha, p) || BitOpsType::get_bit(alpha, q))
                throw std::runtime_error("Alpha target orbitals already occupied");
            
            StorageType new_alpha = alpha;
            BitOpsType::clear_bit(new_alpha, i);
            BitOpsType::clear_bit(new_alpha, j);
            BitOpsType::set_bit(new_alpha, p);
            BitOpsType::set_bit(new_alpha, q);
            return DeterminantT(new_alpha, beta);
        } else {
            if (!BitOpsType::get_bit(beta, i) || !BitOpsType::get_bit(beta, j))
                throw std::runtime_error("Beta orbitals not both occupied");
            if (BitOpsType::get_bit(beta, p) || BitOpsType::get_bit(beta, q))
                throw std::runtime_error("Beta target orbitals already occupied");
            
            StorageType new_beta = beta;
            BitOpsType::clear_bit(new_beta, i);
            BitOpsType::clear_bit(new_beta, j);
            BitOpsType::set_bit(new_beta, p);
            BitOpsType::set_bit(new_beta, q);
            return DeterminantT(alpha, new_beta);
        }
    }

    // Streaming operator for logging
    template<typename T>
    friend std::ostream& operator<<(std::ostream& os, const DeterminantT<T>& d);
};

// Streaming operator implementation
template<typename StorageType>
std::ostream& operator<<(std::ostream& os, const DeterminantT<StorageType>& d) {
    os << "DeterminantT(alpha=";
    if constexpr (std::is_same_v<StorageType, uint64_t>) {
        os << std::bitset<64>(d.alpha);
    } else {
        // For array storage, print each unit
        os << "[";
        for (size_t i = 0; i < d.alpha.size(); ++i) {
            if (i > 0) os << ",";
            os << std::bitset<64>(d.alpha[i]);
        }
        os << "]";
    }
    os << ", beta=";
    if constexpr (std::is_same_v<StorageType, uint64_t>) {
        os << std::bitset<64>(d.beta);
    } else {
        os << "[";
        for (size_t i = 0; i < d.beta.size(); ++i) {
            if (i > 0) os << ",";
            os << std::bitset<64>(d.beta[i]);
        }
        os << "]";
    }
    os << ")";
    return os;
}

// Type aliases for different orbital counts
using Determinant64 = DeterminantT<uint64_t>;                    // Up to 64 orbitals
using Determinant128 = DeterminantT<std::array<uint64_t, 2>>;    // Up to 128 orbitals
using Determinant192 = DeterminantT<std::array<uint64_t, 3>>;    // Up to 192 orbitals

// Note: We don't redefine 'Determinant' here to avoid conflicts with the original class
// The original Determinant class remains the default for backward compatibility

// Template functions for generating reference determinants
template<typename StorageType>
DeterminantT<StorageType> generate_reference_det_t(int n_alpha, int n_beta) noexcept {
    using BitOpsType = BitOps<StorageType>;
    StorageType alpha_mask{}, beta_mask{};
    
    // Set the first n_alpha bits for alpha
    for (int i = 0; i < n_alpha; ++i) {
        BitOpsType::set_bit(alpha_mask, i);
    }
    
    // Set the first n_beta bits for beta
    for (int i = 0; i < n_beta; ++i) {
        BitOpsType::set_bit(beta_mask, i);
    }
    
    return DeterminantT<StorageType>(alpha_mask, beta_mask);
}

// Generate single, double, and mixed excitations (template version) 
// didn't use
template<typename StorageType>
std::vector<DeterminantT<StorageType>> generate_excitations_t(
    const DeterminantT<StorageType>& det, int n_orb) {
    using BitOpsType = BitOps<StorageType>;
    std::vector<DeterminantT<StorageType>> excitations;

    // 1. Collect occupied / virtual orbitals
    std::vector<int> occ_alpha, virt_alpha, occ_beta, virt_beta;
    for (int i = 0; i < n_orb; ++i) {
        if (BitOpsType::get_bit(det.alpha, i)) occ_alpha.push_back(i);
        else virt_alpha.push_back(i);

        if (BitOpsType::get_bit(det.beta, i)) occ_beta.push_back(i);
        else virt_beta.push_back(i);
    }

    // 2. α single excitations
    for (int p : occ_alpha)
    for (int a : virt_alpha) {
        StorageType new_alpha = det.alpha;
        BitOpsType::flip_bit(new_alpha, p);
        BitOpsType::flip_bit(new_alpha, a);
        excitations.emplace_back(new_alpha, det.beta);
    }

    // 3. α double excitations
    for (size_t i = 0; i < occ_alpha.size(); ++i)
    for (size_t j = i + 1; j < occ_alpha.size(); ++j)
    for (int a : virt_alpha)
    for (int b : virt_alpha) {
        if (a == b) continue;
        StorageType new_alpha = det.alpha;
        BitOpsType::flip_bit(new_alpha, occ_alpha[i]);
        BitOpsType::flip_bit(new_alpha, occ_alpha[j]);
        BitOpsType::flip_bit(new_alpha, a);
        BitOpsType::flip_bit(new_alpha, b);
        excitations.emplace_back(new_alpha, det.beta);
    }

    // 4. β single excitations
    for (int p : occ_beta)
    for (int a : virt_beta) {
        StorageType new_beta = det.beta;
        BitOpsType::flip_bit(new_beta, p);
        BitOpsType::flip_bit(new_beta, a);
        excitations.emplace_back(det.alpha, new_beta);
    }

    // 5. β double excitations
    for (size_t i = 0; i < occ_beta.size(); ++i)
    for (size_t j = i + 1; j < occ_beta.size(); ++j)
    for (int a : virt_beta)
    for (int b : virt_beta) {
        if (a == b) continue;
        StorageType new_beta = det.beta;
        BitOpsType::flip_bit(new_beta, occ_beta[i]);
        BitOpsType::flip_bit(new_beta, occ_beta[j]);
        BitOpsType::flip_bit(new_beta, a);
        BitOpsType::flip_bit(new_beta, b);
        excitations.emplace_back(det.alpha, new_beta);
    }

    // 6. mixed αβ doubles
    std::unordered_set<std::string> seen;
    for (int pa : occ_alpha)
    for (int va : virt_alpha) {
        StorageType am = det.alpha;
        BitOpsType::flip_bit(am, pa);
        BitOpsType::flip_bit(am, va);

        for (int pb : occ_beta)
        for (int vb : virt_beta) {
            StorageType bm = det.beta;
            BitOpsType::flip_bit(bm, pb);
            BitOpsType::flip_bit(bm, vb);

            // create a unique string key for deduplication
            // seems to be not neccessary?
            std::ostringstream oss;
            if constexpr (std::is_same_v<StorageType, uint64_t>) {
                oss << am << ":" << bm;
            } else {
                for (auto v : am) oss << v << ",";
                oss << ":";
                for (auto v : bm) oss << v << ",";
            }

            std::string key = oss.str();
            if (seen.insert(key).second) {
                excitations.emplace_back(am, bm);
            }
        }
    }

    return excitations;
}


} // namespace trimci_core

// Hash support for template determinants
namespace std {
template<typename StorageType>
struct hash<trimci_core::DeterminantT<StorageType>> {
    size_t operator()(const trimci_core::DeterminantT<StorageType>& d) const noexcept {
        if constexpr (std::is_same_v<StorageType, uint64_t>) {
            return std::hash<uint64_t>()(d.alpha) ^ (std::hash<uint64_t>()(d.beta) << 1);
        } else {
            // For array storage, combine hashes of all elements
            size_t h1 = 0, h2 = 0;
            for (size_t i = 0; i < d.alpha.size(); ++i) {
                h1 ^= std::hash<uint64_t>()(d.alpha[i]) + 0x9e3779b9 + (h1 << 6) + (h1 >> 2);
                h2 ^= std::hash<uint64_t>()(d.beta[i]) + 0x9e3779b9 + (h2 << 6) + (h2 >> 2);
            }
            return h1 ^ (h2 << 1);
        }
    }
};

// Note: We don't redefine hash<trimci_core::Determinant> here to avoid conflicts
// The original hash specialization in determinant.hpp will be used

}  // namespace std