#include "spin_operator.hpp"
#include "hamiltonian.hpp"   // brings detail::cre_des_sign_t
#include "bit_compat.hpp"
#include "../fast_expansion/fe_types.hpp"   // SplitMix64Hash / DetHash

#include <unordered_map>
#include <cmath>

namespace trimci_core {

// ─────────────────────────────────────────────────────────────────────────
// Single matrix element ⟨det_i | S² | det_j⟩
// ─────────────────────────────────────────────────────────────────────────
template<typename StorageType>
double compute_S2_ij_t(const DeterminantT<StorageType>& det_i,
                       const DeterminantT<StorageType>& det_j) {
    using BitOpsType = detail::HamiltonianBitOps<StorageType>;

    const auto& ai = det_i.alpha;
    const auto& bi = det_i.beta;
    const auto& aj = det_j.alpha;
    const auto& bj = det_j.beta;

    const int toggled_alpha = BitOpsType::count_differences(ai, aj);
    const int toggled_beta  = BitOpsType::count_differences(bi, bj);

    // ── Diagonal: Sz(Sz+1) + N_β_singly ─────────────────────────────
    if (toggled_alpha == 0 && toggled_beta == 0) {
        // N_α, N_β: total electron counts in each spin channel
        int n_a, n_b;
        if constexpr (std::is_same_v<StorageType, uint64_t>) {
            n_a = __builtin_popcountll(ai);
            n_b = __builtin_popcountll(bi);
        } else {
            n_a = 0; n_b = 0;
            for (size_t w = 0; w < ai.size(); ++w) {
                n_a += __builtin_popcountll(ai[w]);
                n_b += __builtin_popcountll(bi[w]);
            }
        }
        // β-singly occupied = β bits with α=0  → popcount(β & ~α)
        auto beta_only = BitOpsType::and_not(bi, ai);
        int n_b_singly;
        if constexpr (std::is_same_v<StorageType, uint64_t>) {
            n_b_singly = __builtin_popcountll(beta_only);
        } else {
            n_b_singly = 0;
            for (size_t w = 0; w < beta_only.size(); ++w)
                n_b_singly += __builtin_popcountll(beta_only[w]);
        }

        const double Sz = 0.5 * (n_a - n_b);
        return Sz * (Sz + 1.0) + static_cast<double>(n_b_singly);
    }

    // ── Off-diagonal: spin-exchange pattern ─────────────────────────
    // Must have exactly one α-bit flip at each of two orbitals
    // AND one β-bit flip at the same two orbitals.
    if (toggled_alpha != 2 || toggled_beta != 2) return 0.0;

    int da_rem[2], da_add[2], db_rem[2], db_add[2];
    int da_rem_cnt = BitOpsType::storage_to_indices_inline(
        BitOpsType::and_not(ai, aj), da_rem, 2);
    int da_add_cnt = BitOpsType::storage_to_indices_inline(
        BitOpsType::and_not(aj, ai), da_add, 2);
    int db_rem_cnt = BitOpsType::storage_to_indices_inline(
        BitOpsType::and_not(bi, bj), db_rem, 2);
    int db_add_cnt = BitOpsType::storage_to_indices_inline(
        BitOpsType::and_not(bj, bi), db_add, 2);

    if (da_rem_cnt != 1 || da_add_cnt != 1 ||
        db_rem_cnt != 1 || db_add_cnt != 1) {
        return 0.0;
    }

    const int m = da_rem[0];   // det_i has α here, det_j does not
    const int p = da_add[0];   // det_j has α here, det_i does not
    const int n = db_rem[0];   // det_i has β here, det_j does not
    const int q = db_add[0];   // det_j has β here, det_i does not

    // Spin-exchange pattern requires the two flipped orbitals to be the
    // *same* pair in both channels, with opposite roles.
    //   det_j:   α at p, β at m  (so at m: β only; at p: α only)
    //   det_i:   α at m, β at p  (so at m: α only; at p: β only)
    // Hence  {m,p} as unordered pair must equal {q,n}  with m==q and p==n.
    if (m != q || p != n) return 0.0;

    // Phase derivation. The (p,q) term of S_-S_+ is
    //     a†_{qβ}·a_{qα}·a†_{pα}·a_{pβ}
    // Reorder to "all α first, then all β" (canonical ordering) by
    // anticommuting a_{qα} past a†_{pα} — this flips one sign:
    //     = -a†_{pα}·a_{qα} · a†_{qβ}·a_{pβ}
    // So the matrix element is (-1) × [α single-excitation phase]
    //                              × [β single-excitation phase].
    // Each single-excitation phase is cre_des_sign_t over the pair of
    // orbitals in the α / β bitstring respectively.
    const int phase_alpha = detail::cre_des_sign_t(m, p, aj);
    const int phase_beta  = detail::cre_des_sign_t(n, q, bj);
    return -static_cast<double>(phase_alpha * phase_beta);
}

double compute_S2_ij(const Determinant& det_i, const Determinant& det_j) {
    return compute_S2_ij_t<uint64_t>(det_i, det_j);
}

// Explicit instantiations for the widths we expose to Python.
template double compute_S2_ij_t<uint64_t>(
    const Determinant&, const Determinant&);
template double compute_S2_ij_t<std::array<uint64_t, 2>>(
    const DeterminantT<std::array<uint64_t, 2>>&,
    const DeterminantT<std::array<uint64_t, 2>>&);
template double compute_S2_ij_t<std::array<uint64_t, 3>>(
    const DeterminantT<std::array<uint64_t, 3>>&,
    const DeterminantT<std::array<uint64_t, 3>>&);
template double compute_S2_ij_t<std::array<uint64_t, 4>>(
    const DeterminantT<std::array<uint64_t, 4>>&,
    const DeterminantT<std::array<uint64_t, 4>>&);


// ─────────────────────────────────────────────────────────────────────────
// ⟨Ψ|S²|Ψ⟩ via hash-indexed spin-exchange enumeration
// ─────────────────────────────────────────────────────────────────────────
// For each det D_i:
//   diag  += c_i² · ⟨D_i|S²|D_i⟩
//   off   += c_i · c_j · ⟨D_j|S²|D_i⟩   (both (i,j) and (j,i) are enumerated
//                                        because each det independently
//                                        produces partners — no factor of 2)
//
// The off-diagonal partners are enumerated by listing α-singly and β-singly
// orbitals and swapping them. This is tiny for realistic dets (few unpaired
// electrons), so the total cost is O(N · N_α_singly · N_β_singly).
template<typename StorageType>
double evaluate_S2_t(
    const std::vector<DeterminantT<StorageType>>& dets,
    const std::vector<double>& coeffs)
{
    using BitT = BitOps<StorageType>;
    using BitOpsType = detail::HamiltonianBitOps<StorageType>;

    const int N = static_cast<int>(coeffs.size());
    if (N == 0) return 0.0;

    // Index dets → position
    std::unordered_map<DeterminantT<StorageType>, int,
                       fe::DetHash<StorageType>> index;
    index.reserve(static_cast<size_t>(N) * 2);
    for (int i = 0; i < N; ++i) index.emplace(dets[i], i);

    double S2 = 0.0;

    #pragma omp parallel for reduction(+:S2) schedule(dynamic, 64)
    for (int i = 0; i < N; ++i) {
        const auto& d = dets[i];
        const double ci = coeffs[i];

        // ── diagonal ────────────────────────────────────────────────
        const double s2_ii = compute_S2_ij_t(d, d);
        S2 += ci * ci * s2_ii;

        // ── off-diagonal: enumerate spin-exchange partners ──────────
        //   α-only orbitals in d: α=1, β=0
        //   β-only orbitals in d: β=1, α=0
        auto alpha_only = BitOpsType::and_not(d.alpha, d.beta);
        auto beta_only  = BitOpsType::and_not(d.beta,  d.alpha);

        int alpha_sites[512];
        int beta_sites[512];
        int n_as = BitOpsType::storage_to_indices_inline(alpha_only, alpha_sites, 512);
        int n_bs = BitOpsType::storage_to_indices_inline(beta_only,  beta_sites,  512);

        // Swap β@p with α@q: produces partner d'
        //   d':  α at p, β at q  (p was β-only, q was α-only in d)
        for (int pi = 0; pi < n_bs; ++pi) {
            const int p = beta_sites[pi];
            for (int qi = 0; qi < n_as; ++qi) {
                const int q = alpha_sites[qi];
                // Build d'
                auto new_alpha = d.alpha;
                auto new_beta  = d.beta;
                BitT::set_bit  (new_alpha, p);   // add α at p
                BitT::clear_bit(new_alpha, q);   // remove α at q
                BitT::clear_bit(new_beta,  p);   // remove β at p
                BitT::set_bit  (new_beta,  q);   // add β at q

                DeterminantT<StorageType> d_prime(new_alpha, new_beta);
                auto it = index.find(d_prime);
                if (it == index.end()) continue;

                const int j = it->second;
                const double cj = coeffs[j];
                // ⟨D_j | S² | D_i⟩  with D_j = d' and D_i = d
                const double s2_ji = compute_S2_ij_t(d_prime, d);
                S2 += cj * ci * s2_ji;
            }
        }
    }

    return S2;
}

template double evaluate_S2_t<uint64_t>(
    const std::vector<Determinant>&, const std::vector<double>&);
template double evaluate_S2_t<std::array<uint64_t, 2>>(
    const std::vector<DeterminantT<std::array<uint64_t, 2>>>&,
    const std::vector<double>&);
template double evaluate_S2_t<std::array<uint64_t, 3>>(
    const std::vector<DeterminantT<std::array<uint64_t, 3>>>&,
    const std::vector<double>&);
template double evaluate_S2_t<std::array<uint64_t, 4>>(
    const std::vector<DeterminantT<std::array<uint64_t, 4>>>&,
    const std::vector<double>&);

// ── Convenience wrappers taking raw (α, β) bitstring arrays ─────────
double evaluate_S2(
    const std::vector<uint64_t>& dets_alpha,
    const std::vector<uint64_t>& dets_beta,
    const std::vector<double>& coeffs)
{
    const size_t N = coeffs.size();
    std::vector<Determinant> dets;
    dets.reserve(N);
    for (size_t i = 0; i < N; ++i)
        dets.emplace_back(dets_alpha[i], dets_beta[i]);
    return evaluate_S2_t<uint64_t>(dets, coeffs);
}

double evaluate_S2_128(
    const std::vector<std::array<uint64_t, 2>>& dets_alpha,
    const std::vector<std::array<uint64_t, 2>>& dets_beta,
    const std::vector<double>& coeffs)
{
    const size_t N = coeffs.size();
    using Det128 = DeterminantT<std::array<uint64_t, 2>>;
    std::vector<Det128> dets;
    dets.reserve(N);
    for (size_t i = 0; i < N; ++i)
        dets.emplace_back(dets_alpha[i], dets_beta[i]);
    return evaluate_S2_t<std::array<uint64_t, 2>>(dets, coeffs);
}

} // namespace trimci_core
