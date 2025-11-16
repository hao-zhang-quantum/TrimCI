#include "determinant.hpp"
#include <iostream>     // for std::ostream, std::cout, etc.
#include <bitset>       // for std::bitset
#include <unordered_set>

namespace trimci_core {

Determinant::Determinant(uint64_t alpha_mask, uint64_t beta_mask) noexcept
    : alpha(alpha_mask), beta(beta_mask) {}

bool Determinant::operator==(Determinant const& other) const noexcept {
    return alpha == other.alpha && beta == other.beta;
}

bool Determinant::operator<(Determinant const& other) const noexcept {
    if (alpha < other.alpha) return true;
    if (alpha > other.alpha) return false;
    return beta < other.beta;
}

Determinant generate_reference_det(int n_alpha, int n_beta) noexcept {
    uint64_t a_mask = (uint64_t(1) << n_alpha) - 1;
    uint64_t b_mask = (uint64_t(1) << n_beta)  - 1;
    return Determinant(a_mask, b_mask);
}

// early version, didn't use
std::vector<Determinant> generate_excitations(Determinant const& det, int n_orb) {
    uint64_t alpha_mask = det.alpha;
    uint64_t beta_mask  = det.beta;

    // Occupied & virtual lists
    std::vector<int> occ_alpha, virt_alpha, occ_beta, virt_beta;
    for (int i = 0; i < n_orb; ++i) {
        if ((alpha_mask >> i) & 1) occ_alpha.push_back(i);
        else                       virt_alpha.push_back(i);
        if ((beta_mask  >> i) & 1) occ_beta.push_back(i);
        else                       virt_beta.push_back(i);
    }

    std::vector<Determinant> excitations;

    // α single
    for (auto p : occ_alpha)
    for (auto a : virt_alpha) {
        uint64_t mask = (uint64_t(1)<<p)|(uint64_t(1)<<a);
        excitations.emplace_back(alpha_mask ^ mask, beta_mask);
    }

    // α double
    for (size_t i = 0; i < occ_alpha.size(); ++i)
    for (size_t j = i+1; j < occ_alpha.size(); ++j) {
        int p = occ_alpha[i], q = occ_alpha[j];
        for (auto a : virt_alpha)
        for (auto b : virt_alpha) {
            if (a == b) continue;
            uint64_t mask = (uint64_t(1)<<p)|(uint64_t(1)<<a)
                          |(uint64_t(1)<<q)|(uint64_t(1)<<b);
            excitations.emplace_back(alpha_mask ^ mask, beta_mask);
        }
    }

    // β single
    for (auto p : occ_beta)
    for (auto a : virt_beta) {
        uint64_t mask = (uint64_t(1)<<p)|(uint64_t(1)<<a);
        excitations.emplace_back(alpha_mask, beta_mask ^ mask);
    }

    // β double
    for (size_t i = 0; i < occ_beta.size(); ++i)
    for (size_t j = i+1; j < occ_beta.size(); ++j) {
        int p = occ_beta[i], q = occ_beta[j];
        for (auto a : virt_beta)
        for (auto b : virt_beta) {
            if (a == b) continue;
            uint64_t mask = (uint64_t(1)<<p)|(uint64_t(1)<<a)
                          |(uint64_t(1)<<q)|(uint64_t(1)<<b);
            excitations.emplace_back(alpha_mask, beta_mask ^ mask);
        }
    }

    // mixed αβ double (deduplicate)
    std::unordered_set<uint64_t> seen;
    for (auto p : occ_alpha)
    for (auto a : virt_alpha) {
        uint64_t am = alpha_mask ^ ((uint64_t(1)<<p)|(uint64_t(1)<<a));
        for (auto pb : occ_beta)
        for (auto b  : virt_beta) {
            uint64_t bm  = beta_mask ^ ((uint64_t(1)<<pb)|(uint64_t(1)<<b));
            uint64_t key = (am << 32) | bm;
            if (seen.insert(key).second) {
                excitations.emplace_back(am, bm);
            }
        }
    }

    return excitations;
}

std::vector<int> Determinant::getOccupiedAlpha() const noexcept {
    std::vector<int> occ;
    uint64_t m = alpha;
    int idx = 0;
    while (m) {
        if (m & 1) occ.push_back(idx);
        m >>= 1; ++idx;
    }
    return occ;
}

std::vector<int> Determinant::getOccupiedBeta() const noexcept {
    std::vector<int> occ;
    uint64_t m = beta;
    int idx = 0;
    while (m) {
        if (m & 1) occ.push_back(idx);
        m >>= 1; ++idx;
    }
    return occ;
}

Determinant Determinant::singleExcite(int i, int p, bool isAlpha) const {
    if (isAlpha) {
        if (!((alpha >> i) & 1))
            throw std::runtime_error("Alpha orbital not occupied");
        if ((alpha >> p) & 1)
            throw std::runtime_error("Alpha target orbital already occupied");
        uint64_t new_a = (alpha & ~(uint64_t(1)<<i)) | (uint64_t(1)<<p);
        return Determinant(new_a, beta);
    } else {
        if (!((beta >> i) & 1))
            throw std::runtime_error("Beta orbital not occupied");
        if ((beta >> p) & 1)
            throw std::runtime_error("Beta target orbital already occupied");
        uint64_t new_b = (beta & ~(uint64_t(1)<<i)) | (uint64_t(1)<<p);
        return Determinant(alpha, new_b);
    }
}

Determinant Determinant::doubleExcite(int i, int j, int p, int q, bool isAlpha) const {
    if (isAlpha) {
        if (!(((alpha>>i)&1) && ((alpha>>j)&1)))
            throw std::runtime_error("Alpha orbitals not both occupied");
        if (((alpha>>p)&1) || ((alpha>>q)&1))
            throw std::runtime_error("Alpha target orbitals already occupied");
        uint64_t new_a = (alpha & ~(uint64_t(1)<<i) & ~(uint64_t(1)<<j))
                       | (uint64_t(1)<<p) | (uint64_t(1)<<q);
        return Determinant(new_a, beta);
    } else {
        if (!(((beta>>i)&1) && ((beta>>j)&1)))
            throw std::runtime_error("Beta orbitals not both occupied");
        if (((beta>>p)&1) || ((beta>>q)&1))
            throw std::runtime_error("Beta target orbitals already occupied");
        uint64_t new_b = (beta & ~(uint64_t(1)<<i) & ~(uint64_t(1)<<j))
                       | (uint64_t(1)<<p) | (uint64_t(1)<<q);
        return Determinant(alpha, new_b);
    }
}

// streaming operator for logging
std::ostream& operator<<(std::ostream& os, Determinant const& d) {
    os << "Determinant(alpha="
       << std::bitset<64>(d.alpha)
       << ", beta="
       << std::bitset<64>(d.beta)
       << ")";
    return os;
}

} // namespace trimci_core
