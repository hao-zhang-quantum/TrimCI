#pragma once

#include <cstdint>
#include <vector>
#include <ostream>
#include <sstream>
#include <bitset>
#include <array>
#include <type_traits>
#include <unordered_set>

// Provides __builtin_ctzll / __builtin_popcountll mappings on MSVC.
#include "bit_compat.hpp"

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

    // Get occupied orbital indices using ctz (count trailing zeros) extraction.
    // For uint64_t: iterates only over set bits (~n_elec iterations).
    // For array<uint64_t, N>: word-by-word ctz walk, same cost as uint64_t path
    // but across N words. Replaces the old bit-by-bit scan (128 iterations for
    // Determinant128 even with only ~20 occupied orbitals).
    std::vector<int> getOccupiedAlpha() const noexcept {
        std::vector<int> occ;
        if constexpr (std::is_same_v<StorageType, uint64_t>) {
            occ.reserve(__builtin_popcountll(alpha));
            uint64_t m = alpha;
            while (m) {
                occ.push_back(__builtin_ctzll(m));
                m &= m - 1;  // clear lowest set bit
            }
        } else {
            constexpr size_t N = std::tuple_size_v<StorageType>;
            int total = 0;
            for (size_t w = 0; w < N; ++w) total += __builtin_popcountll(alpha[w]);
            occ.reserve(total);
            for (size_t w = 0; w < N; ++w) {
                uint64_t m = alpha[w];
                int base = static_cast<int>(w * 64);
                while (m) {
                    occ.push_back(base + __builtin_ctzll(m));
                    m &= m - 1;
                }
            }
        }
        return occ;
    }

    std::vector<int> getOccupiedBeta() const noexcept {
        std::vector<int> occ;
        if constexpr (std::is_same_v<StorageType, uint64_t>) {
            occ.reserve(__builtin_popcountll(beta));
            uint64_t m = beta;
            while (m) {
                occ.push_back(__builtin_ctzll(m));
                m &= m - 1;
            }
        } else {
            constexpr size_t N = std::tuple_size_v<StorageType>;
            int total = 0;
            for (size_t w = 0; w < N; ++w) total += __builtin_popcountll(beta[w]);
            occ.reserve(total);
            for (size_t w = 0; w < N; ++w) {
                uint64_t m = beta[w];
                int base = static_cast<int>(w * 64);
                while (m) {
                    occ.push_back(base + __builtin_ctzll(m));
                    m &= m - 1;
                }
            }
        }
        return occ;
    }

    // Single excitation: i -> p
    DeterminantT singleExcite(int i, int p, bool isAlpha) const {
        if (isAlpha) {
            // Safety checks (optional, kept for robustness)
            
            StorageType new_alpha = alpha;
            BitOpsType::clear_bit(new_alpha, i);
            BitOpsType::set_bit(new_alpha, p);
            return DeterminantT(new_alpha, beta);
        } else {
            StorageType new_beta = beta;
            BitOpsType::clear_bit(new_beta, i);
            BitOpsType::set_bit(new_beta, p);
            return DeterminantT(alpha, new_beta);
        }
    }

    // Double excitation: i,j -> p,q
    DeterminantT doubleExcite(int i, int j, int p, int q, bool isAlpha) const {
        if (isAlpha) {
            StorageType new_alpha = alpha;
            BitOpsType::clear_bit(new_alpha, i);
            BitOpsType::clear_bit(new_alpha, j);
            BitOpsType::set_bit(new_alpha, p);
            BitOpsType::set_bit(new_alpha, q);
            return DeterminantT(new_alpha, beta);
        } else {
            StorageType new_beta = beta;
            BitOpsType::clear_bit(new_beta, i);
            BitOpsType::clear_bit(new_beta, j);
            BitOpsType::set_bit(new_beta, p);
            BitOpsType::set_bit(new_beta, q);
            return DeterminantT(alpha, new_beta);
        }
    }
};

// Streaming operator implementation
template<typename StorageType>
std::ostream& operator<<(std::ostream& os, const DeterminantT<StorageType>& d) {
    os << "Determinant(alpha=";
    if constexpr (std::is_same_v<StorageType, uint64_t>) {
        os << std::bitset<64>(d.alpha);
    } else {
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

// Type aliases
using Determinant64 = DeterminantT<uint64_t>;
using Determinant128 = DeterminantT<std::array<uint64_t, 2>>;
using Determinant192 = DeterminantT<std::array<uint64_t, 3>>;
using Determinant256 = DeterminantT<std::array<uint64_t, 4>>;
using Determinant320 = DeterminantT<std::array<uint64_t, 5>>;
using Determinant384 = DeterminantT<std::array<uint64_t, 6>>;
using Determinant448 = DeterminantT<std::array<uint64_t, 7>>;
using Determinant512 = DeterminantT<std::array<uint64_t, 8>>;

// Backward compatibility
using Determinant = Determinant64;

// Template functions for generating reference determinants
template<typename StorageType>
DeterminantT<StorageType> generate_reference_det_t(int n_alpha, int n_beta) noexcept {
    using BitOpsType = BitOps<StorageType>;
    StorageType alpha_mask{}, beta_mask{};
    
    for (int i = 0; i < n_alpha; ++i) BitOpsType::set_bit(alpha_mask, i);
    for (int i = 0; i < n_beta; ++i)  BitOpsType::set_bit(beta_mask, i);
    
    return DeterminantT<StorageType>(alpha_mask, beta_mask);
}

// Wrapper for backward compatibility
inline Determinant generate_reference_det(int n_alpha, int n_beta) noexcept {
    return generate_reference_det_t<uint64_t>(n_alpha, n_beta);
}

// Template for generating excitations
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

    // 3. α double excitations (a < b to avoid duplicates)
    for (size_t i = 0; i < occ_alpha.size(); ++i)
    for (size_t j = i + 1; j < occ_alpha.size(); ++j)
    for (int a : virt_alpha)
    for (int b : virt_alpha) {
        if (b <= a) continue;
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

    // 5. β double excitations (a < b to avoid duplicates)
    for (size_t i = 0; i < occ_beta.size(); ++i)
    for (size_t j = i + 1; j < occ_beta.size(); ++j)
    for (int a : virt_beta)
    for (int b : virt_beta) {
        if (b <= a) continue;
        StorageType new_beta = det.beta;
        BitOpsType::flip_bit(new_beta, occ_beta[i]);
        BitOpsType::flip_bit(new_beta, occ_beta[j]);
        BitOpsType::flip_bit(new_beta, a);
        BitOpsType::flip_bit(new_beta, b);
        excitations.emplace_back(det.alpha, new_beta);
    }

    // 6. mixed αβ doubles
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

            // Mixed excitations are unique by construction
            excitations.emplace_back(am, bm);
        }
    }

    return excitations;
}

/// Callback version: calls cb(Det) for each single/double excitation without allocation.
template<typename StorageType, typename Callback>
void for_each_excitation_t(const DeterminantT<StorageType>& det, int n_orb, Callback&& cb) {
    using BitOpsType = BitOps<StorageType>;

    std::vector<int> occ_alpha, virt_alpha, occ_beta, virt_beta;
    for (int i = 0; i < n_orb; ++i) {
        if (BitOpsType::get_bit(det.alpha, i)) occ_alpha.push_back(i);
        else virt_alpha.push_back(i);
        if (BitOpsType::get_bit(det.beta, i)) occ_beta.push_back(i);
        else virt_beta.push_back(i);
    }

    // α singles
    for (int p : occ_alpha)
    for (int a : virt_alpha) {
        StorageType na = det.alpha;
        BitOpsType::flip_bit(na, p);
        BitOpsType::flip_bit(na, a);
        cb(DeterminantT<StorageType>(na, det.beta));
    }
    // α doubles
    for (size_t i = 0; i < occ_alpha.size(); ++i)
    for (size_t j = i + 1; j < occ_alpha.size(); ++j)
    for (int a : virt_alpha)
    for (int b : virt_alpha) {
        if (b <= a) continue;
        StorageType na = det.alpha;
        BitOpsType::flip_bit(na, occ_alpha[i]);
        BitOpsType::flip_bit(na, occ_alpha[j]);
        BitOpsType::flip_bit(na, a);
        BitOpsType::flip_bit(na, b);
        cb(DeterminantT<StorageType>(na, det.beta));
    }
    // β singles
    for (int p : occ_beta)
    for (int a : virt_beta) {
        StorageType nb = det.beta;
        BitOpsType::flip_bit(nb, p);
        BitOpsType::flip_bit(nb, a);
        cb(DeterminantT<StorageType>(det.alpha, nb));
    }
    // β doubles
    for (size_t i = 0; i < occ_beta.size(); ++i)
    for (size_t j = i + 1; j < occ_beta.size(); ++j)
    for (int a : virt_beta)
    for (int b : virt_beta) {
        if (b <= a) continue;
        StorageType nb = det.beta;
        BitOpsType::flip_bit(nb, occ_beta[i]);
        BitOpsType::flip_bit(nb, occ_beta[j]);
        BitOpsType::flip_bit(nb, a);
        BitOpsType::flip_bit(nb, b);
        cb(DeterminantT<StorageType>(det.alpha, nb));
    }
    // αβ mixed doubles
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
            cb(DeterminantT<StorageType>(am, bm));
        }
    }
}

/// Callback version that also computes |H_{DI}| inline for each excitation.
/// Avoids the overhead of compute_H_ij_t (no XOR/popcount detection needed).
/// cb(const DeterminantT<StorageType>& D, double absH) is called for each excitation.
/// Requires get_eri(eri, n_orb, i, j, k, l) to be available.
template<typename StorageType, typename Callback>
void for_each_excitation_with_absH_t(
    const DeterminantT<StorageType>& det,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    int n_orb,
    Callback&& cb)
{
    using BitOpsType = BitOps<StorageType>;

    std::vector<int> occ_alpha, virt_alpha, occ_beta, virt_beta;
    occ_alpha.reserve(n_orb);
    virt_alpha.reserve(n_orb);
    occ_beta.reserve(n_orb);
    virt_beta.reserve(n_orb);
    for (int i = 0; i < n_orb; ++i) {
        if (BitOpsType::get_bit(det.alpha, i)) occ_alpha.push_back(i);
        else virt_alpha.push_back(i);
        if (BitOpsType::get_bit(det.beta, i)) occ_beta.push_back(i);
        else virt_beta.push_back(i);
    }

    auto eri_idx = [&](int i, int j, int k, int l) -> double {
        return eri[((size_t(i) * n_orb + j) * n_orb + k) * n_orb + l];
    };

    // α single excitations: m → p
    for (int m : occ_alpha)
    for (int p : virt_alpha) {
        double H = h1[m][p];
        for (int n : occ_alpha)
            if (n != m) H += eri_idx(m, p, n, n) - eri_idx(m, n, n, p);
        for (int n : occ_beta)
            H += eri_idx(m, p, n, n);

        StorageType na = det.alpha;
        BitOpsType::flip_bit(na, m);
        BitOpsType::flip_bit(na, p);
        cb(DeterminantT<StorageType>(na, det.beta), std::abs(H));
    }

    // β single excitations: m → p
    for (int m : occ_beta)
    for (int p : virt_beta) {
        double H = h1[m][p];
        for (int n : occ_beta)
            if (n != m) H += eri_idx(m, p, n, n) - eri_idx(m, n, n, p);
        for (int n : occ_alpha)
            H += eri_idx(m, p, n, n);

        StorageType nb = det.beta;
        BitOpsType::flip_bit(nb, m);
        BitOpsType::flip_bit(nb, p);
        cb(DeterminantT<StorageType>(det.alpha, nb), std::abs(H));
    }

    // αα double excitations: (m, n) → (p, q) with m<n, p<q
    for (size_t ii = 0; ii < occ_alpha.size(); ++ii)
    for (size_t jj = ii + 1; jj < occ_alpha.size(); ++jj) {
        int m = occ_alpha[ii], n = occ_alpha[jj];
        for (int p : virt_alpha)
        for (int q : virt_alpha) {
            if (q <= p) continue;
            double H = eri_idx(m, p, n, q) - eri_idx(m, q, n, p);

            StorageType na = det.alpha;
            BitOpsType::flip_bit(na, m);
            BitOpsType::flip_bit(na, n);
            BitOpsType::flip_bit(na, p);
            BitOpsType::flip_bit(na, q);
            cb(DeterminantT<StorageType>(na, det.beta), std::abs(H));
        }
    }

    // ββ double excitations: (m, n) → (p, q) with m<n, p<q
    for (size_t ii = 0; ii < occ_beta.size(); ++ii)
    for (size_t jj = ii + 1; jj < occ_beta.size(); ++jj) {
        int m = occ_beta[ii], n = occ_beta[jj];
        for (int p : virt_beta)
        for (int q : virt_beta) {
            if (q <= p) continue;
            double H = eri_idx(m, p, n, q) - eri_idx(m, q, n, p);

            StorageType nb = det.beta;
            BitOpsType::flip_bit(nb, m);
            BitOpsType::flip_bit(nb, n);
            BitOpsType::flip_bit(nb, p);
            BitOpsType::flip_bit(nb, q);
            cb(DeterminantT<StorageType>(det.alpha, nb), std::abs(H));
        }
    }

    // αβ mixed double excitations: (m_α, m_β) → (p_α, p_β)
    for (int ma : occ_alpha)
    for (int pa : virt_alpha) {
        StorageType am = det.alpha;
        BitOpsType::flip_bit(am, ma);
        BitOpsType::flip_bit(am, pa);
        for (int mb : occ_beta)
        for (int pb : virt_beta) {
            double H = eri_idx(ma, pa, mb, pb);

            StorageType bm = det.beta;
            BitOpsType::flip_bit(bm, mb);
            BitOpsType::flip_bit(bm, pb);
            cb(DeterminantT<StorageType>(am, bm), std::abs(H));
        }
    }
}

// Wrapper for backward compatibility
std::vector<Determinant> generate_excitations(const Determinant& det, int n_orb);

} // namespace trimci_core

// Hash support
namespace std {
template<typename StorageType>
struct hash<trimci_core::DeterminantT<StorageType>> {
    size_t operator()(const trimci_core::DeterminantT<StorageType>& d) const noexcept {
        if constexpr (std::is_same_v<StorageType, uint64_t>) {
            return std::hash<uint64_t>()(d.alpha) ^ (std::hash<uint64_t>()(d.beta) << 1);
        } else {
            size_t h1 = 0, h2 = 0;
            for (size_t i = 0; i < d.alpha.size(); ++i) {
                h1 ^= std::hash<uint64_t>()(d.alpha[i]) + 0x9e3779b9 + (h1 << 6) + (h1 >> 2);
                h2 ^= std::hash<uint64_t>()(d.beta[i]) + 0x9e3779b9 + (h2 << 6) + (h2 >> 2);
            }
            return h1 ^ (h2 << 1);
        }
    }
};
}
