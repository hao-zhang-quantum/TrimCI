#pragma once

#include <cstdint>
#include <vector>
#include <ostream>
#include <bitset>

namespace trimci_core {

class Determinant {
public:
    uint64_t alpha;
    uint64_t beta;

    Determinant(uint64_t alpha_mask, uint64_t beta_mask) noexcept;
    bool operator==(Determinant const& other) const noexcept;
    bool operator<(Determinant const& other) const noexcept;

    // 获取占据轨道索引
    std::vector<int> getOccupiedAlpha() const noexcept;
    std::vector<int> getOccupiedBeta()  const noexcept;

    // 激发操作：isAlpha=true 表示α自旋，否则β自旋
    Determinant singleExcite(int i, int p, bool isAlpha) const;
    Determinant doubleExcite(int i, int j, int p, int q, bool isAlpha) const;

    // 方便日志输出
    friend std::ostream& operator<<(std::ostream& os, Determinant const& d);
};

// Generate the Hartree-Fock reference determinant
Determinant generate_reference_det(int n_alpha, int n_beta) noexcept;

// Generate single, double, and mixed excitations
std::vector<Determinant> generate_excitations(Determinant const& det, int n_orb);

} // namespace trimci_core

// Hash support so Determinant can go in unordered_set/map
namespace std {
template<>
struct hash<trimci_core::Determinant> {
    size_t operator()(trimci_core::Determinant const& d) const noexcept {
        return std::hash<uint64_t>()(d.alpha)
             ^ (std::hash<uint64_t>()(d.beta) << 1);
    }
};
}
