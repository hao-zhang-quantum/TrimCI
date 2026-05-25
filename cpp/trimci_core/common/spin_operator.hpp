#pragma once

// S² operator evaluation for spinful fermion determinants.
//
// S² = S_z(S_z+1) + S_-S_+
//
// Diagonal:     ⟨D|S²|D⟩  = Sz(Sz+1) + N_β_singly(D)
// Off-diagonal: ⟨D'|S²|D⟩ = −phase_α · phase_β  iff D' is obtained from D
//                           by a "spin exchange" at two orbitals (p,q):
//                             D:  p is β-singly, q is α-singly
//                             D': p is α-singly, q is β-singly
//                           The leading "−" comes from reordering
//                           a†_{qβ}a_{qα}·a†_{pα}a_{pβ} into canonical
//                           α-first form (one fermionic anticommutation).
//                           phase_α / phase_β are the usual Slater–Condon
//                           single-excitation signs in each channel.

#include <vector>
#include <cstdint>
#include <array>
#include "determinant.hpp"

namespace trimci_core {

// Single matrix element ⟨det_i|S²|det_j⟩
template<typename StorageType>
double compute_S2_ij_t(const DeterminantT<StorageType>& det_i,
                       const DeterminantT<StorageType>& det_j);

// 64-bit wrapper for Python bindings
double compute_S2_ij(const Determinant& det_i, const Determinant& det_j);

// ⟨Ψ|S²|Ψ⟩ for a wavefunction given as (dets, coeffs).
// Uses a hash index of the determinants so each det only does an
// O(N_α_singly · N_β_singly) scan over its spin-exchange partners.
// Returns S2 = ⟨S²⟩. Caller can invert via S = (-1 + sqrt(1+4·S2))/2.
template<typename StorageType>
double evaluate_S2_t(
    const std::vector<DeterminantT<StorageType>>& dets,
    const std::vector<double>& coeffs);

// 64-bit convenience (takes alpha/beta bitstring vectors, matches
// evaluate_ci_energy signature).
double evaluate_S2(
    const std::vector<uint64_t>& dets_alpha,
    const std::vector<uint64_t>& dets_beta,
    const std::vector<double>& coeffs);

// 128-bit version (each det is two uint64_t per spin).
double evaluate_S2_128(
    const std::vector<std::array<uint64_t, 2>>& dets_alpha,
    const std::vector<std::array<uint64_t, 2>>& dets_beta,
    const std::vector<double>& coeffs);

} // namespace trimci_core
