#pragma once

#include <vector>
#include <tuple>
#include "determinant.hpp"

namespace trimci_core {

// Compute Orbital Gradient and Diagonal of Fock Matrix
// Returns {grad_mat, fock_diagonal}
// grad_mat is antisymmetric: g_pq = 2(F_pq - F_qp)
//
// ATTENTIVE MODE: If attentive_orbitals is non-empty, only computes gradient
// for pairs (p,q) where BOTH p and q are in attentive_orbitals.
// The Fock matrix is still computed for all orbitals in attentive set.
// Other entries in grad_mat will be zero.
// This provides speedup for Fock Phase 3 from O(N^5) to O(k^2 * N^3).
std::tuple<std::vector<std::vector<double>>, std::vector<double>>
compute_orbital_gradient(
    const std::vector<Determinant>& dets,
    const std::vector<double>& coeffs,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    int n_orb,
    int n_elec,
    const std::vector<int>& attentive_orbitals = {}  // Empty = full gradient
);

// Templated version with attentive support
template<typename StorageType>
std::tuple<std::vector<std::vector<double>>, std::vector<double>>
compute_orbital_gradient_t(
    const std::vector<DeterminantT<StorageType>>& dets,
    const std::vector<double>& coeffs,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    int n_orb,
    int n_elec,
    const std::vector<int>& attentive_orbitals = {}  // Empty = full gradient
);

} // namespace trimci_core


