#include "hamiltonian.hpp"
#include <algorithm>
#include <cstring>

namespace trimci_core {

// Extract molecular formula from atom string
std::string extract_mol_name(const std::string& atom_str) {
    // Simplified implementation: return a safe default name
    if (atom_str.empty()) return "unknown";
    return "molecule";
}

// Template implementation for Hij cache loading
template<typename StorageType>
std::tuple<HijCacheT<StorageType>, std::string>
load_or_create_Hij_cache_t(const std::string& mol_name,
                           int n_elec, int n_orb,
                           const std::string& cache_dir) {
    std::string filename = cache_dir + "/" + mol_name + "_" + 
                           std::to_string(n_elec) + "_" + 
                           std::to_string(n_orb) + ".bin";
    
    HijCacheT<StorageType> cache;

    // Placeholder: Return empty cache. 
    // Real implementation would deserialize from binary file here.
    
    return {cache, filename};
}

// Wrapper for backward compatibility
std::tuple<HijCache, std::string>
load_or_create_Hij_cache(const std::string& mol_name,
                         int n_elec, int n_orb,
                         const std::string& cache_dir) {
    return load_or_create_Hij_cache_t<uint64_t>(mol_name, n_elec, n_orb, cache_dir);
}


// Template implementation for pair key
template<typename StorageType>
std::pair<DeterminantT<StorageType>, DeterminantT<StorageType>>
pair_key_t(const DeterminantT<StorageType>& d1, const DeterminantT<StorageType>& d2) {
    if (d1 < d2) return {d1, d2};
    return {d2, d1};
}

// Wrapper
std::pair<Determinant, Determinant>
pair_key(const Determinant& d1, const Determinant& d2) {
    return pair_key_t<uint64_t>(d1, d2);
}

// Helper namespace for Hamiltonians
namespace detail {

// Template-based bit manipulation utilities


// Template-based creation/destruction sign calculation


} // namespace detail


// Template implementation for Slater-Condon matrix elements
template<typename StorageType>
double compute_H_ij_t(const DeterminantT<StorageType>& det_i,
                      const DeterminantT<StorageType>& det_j,
                      const std::vector<std::vector<double>>& h1,
                      const std::vector<double>& eri) {
    
    using BitOpsType = detail::HamiltonianBitOps<StorageType>;
    int n_orb = (int)h1.size();
    
    const auto& ai = det_i.alpha;
    const auto& bi = det_i.beta;
    const auto& aj = det_j.alpha;
    const auto& bj = det_j.beta;
    
    // Count total differences
    int toggled_alpha = BitOpsType::count_differences(ai, aj);  // popcount of XOR on alpha
    int toggled_beta  = BitOpsType::count_differences(bi, bj);  // popcount of XOR on beta
    int n_toggled     = toggled_alpha + toggled_beta;           // total toggled bits (remove+add)
    
    // Same determinant
    if (n_toggled == 0) {
        double Hij = 0.0;
        
        // One-electron terms
        int occ_a[512], occ_b[512];
        int n_a = BitOpsType::storage_to_indices_inline(ai, occ_a, 512);
        int n_b = BitOpsType::storage_to_indices_inline(bi, occ_b, 512);
        
        for (int k = 0; k < n_a; ++k) { int i = occ_a[k]; Hij += h1[i][i]; }
        for (int k = 0; k < n_b; ++k) { int i = occ_b[k]; Hij += h1[i][i]; }
        
        // Two-electron terms
        for (int k = 0; k < n_a; ++k) {
            int i = occ_a[k];
            for (int l = 0; l < n_a; ++l) {
                int j = occ_a[l];
                if (j > i) Hij += get_eri(eri, n_orb, i, i, j, j) - get_eri(eri, n_orb, i, j, j, i);
            }
            for (int l = 0; l < n_b; ++l) {
                 int j = occ_b[l];
                 Hij += get_eri(eri, n_orb, i, i, j, j);
            }
        }
        for (int k = 0; k < n_b; ++k) {
            int i = occ_b[k];
            for (int l = 0; l < n_b; ++l) {
                int j = occ_b[l];
                if (j > i) Hij += get_eri(eri, n_orb, i, i, j, j) - get_eri(eri, n_orb, i, j, j, i);
            }
        }
        
        return Hij;
    }
    
    // Single excitation
    if (n_toggled == 2) {
        int da_rem[2], da_add[2], db_rem[2], db_add[2];
        int da_rem_cnt = BitOpsType::storage_to_indices_inline(BitOpsType::and_not(ai, aj), da_rem, 2);
        int da_add_cnt = BitOpsType::storage_to_indices_inline(BitOpsType::and_not(aj, ai), da_add, 2);
        int db_rem_cnt = BitOpsType::storage_to_indices_inline(BitOpsType::and_not(bi, bj), db_rem, 2);
        int db_add_cnt = BitOpsType::storage_to_indices_inline(BitOpsType::and_not(bj, bi), db_add, 2);
        
        if (da_rem_cnt == 1 && db_rem_cnt == 0) {
            int m = da_rem[0], p = da_add[0];
            int phase = detail::cre_des_sign_t(m, p, aj);
            double Hij = h1[m][p];
            
            int occ_a[512];
            int n_a = BitOpsType::storage_to_indices_inline(ai, occ_a, 512);
            for (int k = 0; k < n_a; ++k) {
                int n = occ_a[k];
                if (n != m) Hij += get_eri(eri, n_orb, m, p, n, n) - get_eri(eri, n_orb, m, n, n, p);
            }
            
            int occ_b[512];
            int n_b = BitOpsType::storage_to_indices_inline(bi, occ_b, 512);
            for (int k = 0; k < n_b; ++k) {
                int n = occ_b[k];
                Hij += get_eri(eri, n_orb, m, p, n, n);
            }
            
            return Hij * phase;
        } else if (db_rem_cnt == 1 && da_rem_cnt == 0) {
            int m = db_rem[0], p = db_add[0];
            int phase = detail::cre_des_sign_t(m, p, bj);
            double Hij = h1[m][p];
            
            int occ_b[512];
            int n_b = BitOpsType::storage_to_indices_inline(bi, occ_b, 512);
            for (int k = 0; k < n_b; ++k) {
                int n = occ_b[k];
                if (n != m) Hij += get_eri(eri, n_orb, m, p, n, n) - get_eri(eri, n_orb, m, n, n, p);
            }
            
            int occ_a[512];
            int n_a = BitOpsType::storage_to_indices_inline(ai, occ_a, 512);
            for (int k = 0; k < n_a; ++k) {
                int n = occ_a[k];
                Hij += get_eri(eri, n_orb, m, p, n, n);
            }
            
            return Hij * phase;
        }
    }
    
    // Double excitation
    if (n_toggled == 4) {
        int da_rem[2], da_add[2], db_rem[2], db_add[2];
        int da_rem_cnt = BitOpsType::storage_to_indices_inline(BitOpsType::and_not(ai, aj), da_rem, 2);
        int da_add_cnt = BitOpsType::storage_to_indices_inline(BitOpsType::and_not(aj, ai), da_add, 2);
        int db_rem_cnt = BitOpsType::storage_to_indices_inline(BitOpsType::and_not(bi, bj), db_rem, 2);
        int db_add_cnt = BitOpsType::storage_to_indices_inline(BitOpsType::and_not(bj, bi), db_add, 2);
        
        if (da_rem_cnt == 2 && db_rem_cnt == 0) {
            int m = std::min(da_rem[0], da_rem[1]);
            int n = std::max(da_rem[0], da_rem[1]);
            int p = std::min(da_add[0], da_add[1]);
            int q = std::max(da_add[0], da_add[1]);
            
            int phase1 = detail::cre_des_sign_t(m, p, aj);
            auto new_a = aj;
            BitOps<StorageType>::set_bit(new_a, m);
            BitOps<StorageType>::clear_bit(new_a, p);
            int phase2 = detail::cre_des_sign_t(n, q, new_a);
            
            return phase1 * phase2 * (get_eri(eri, n_orb, m, p, n, q) - get_eri(eri, n_orb, m, q, n, p));
        } else if (db_rem_cnt == 2 && da_rem_cnt == 0) {
            int m = std::min(db_rem[0], db_rem[1]);
            int n = std::max(db_rem[0], db_rem[1]);
            int p = std::min(db_add[0], db_add[1]);
            int q = std::max(db_add[0], db_add[1]);
            
            int phase1 = detail::cre_des_sign_t(m, p, bj);
            auto new_b = bj;
            BitOps<StorageType>::set_bit(new_b, m);
            BitOps<StorageType>::clear_bit(new_b, p);
            int phase2 = detail::cre_des_sign_t(n, q, new_b);
            
            return phase1 * phase2 * (get_eri(eri, n_orb, m, p, n, q) - get_eri(eri, n_orb, m, q, n, p));
        } else if (da_rem_cnt == 1 && db_rem_cnt == 1) {
            int m = da_rem[0], p = da_add[0];
            int n = db_rem[0], q = db_add[0];
            int phase = detail::cre_des_sign_t(m, p, aj) * detail::cre_des_sign_t(n, q, bj);
            return phase * get_eri(eri, n_orb, m, p, n, q);
        }
    }
    
    // Higher excitations
    return 0.0;
}

// Wrapper for backward compatibility
double compute_H_ij(const Determinant& det_i,
                    const Determinant& det_j,
                    const std::vector<std::vector<double>>& h1,
                    const std::vector<double>& eri) {
    return compute_H_ij_t<uint64_t>(det_i, det_j, h1, eri);
}

// =============================================================================
// Integral Sparsity Analysis
// =============================================================================
IntegralSparsityInfo build_sparsity_info(
    int n_orb,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    double threshold
) {
    IntegralSparsityInfo info;

    // --- h1 neighbor list ---
    info.h1_neighbors.resize(n_orb);
    int h1_nonzero = 0;
    int h1_offdiag_total = n_orb * (n_orb - 1);
    for (int i = 0; i < n_orb; ++i) {
        for (int j = 0; j < n_orb; ++j) {
            if (i != j && std::abs(h1[i][j]) > threshold) {
                info.h1_neighbors[i].push_back(j);
                h1_nonzero++;
            }
        }
        std::sort(info.h1_neighbors[i].begin(), info.h1_neighbors[i].end());
    }
    info.h1_sparsity = h1_offdiag_total > 0
        ? static_cast<double>(h1_nonzero) / h1_offdiag_total : 1.0;

    // --- ERI analysis ---
    size_t eri_nonzero = 0;
    size_t eri_total = static_cast<size_t>(n_orb) * n_orb * n_orb * n_orb;
    bool all_diagonal = true;

    for (int i = 0; i < n_orb; ++i) {
        for (int j = 0; j < n_orb; ++j) {
            for (int k = 0; k < n_orb; ++k) {
                for (int l = 0; l < n_orb; ++l) {
                    double val = get_eri(eri, n_orb, i, j, k, l);
                    if (std::abs(val) > threshold) {
                        eri_nonzero++;
                        if (!(i == j && j == k && k == l)) {
                            all_diagonal = false;
                        }
                    }
                }
            }
        }
    }
    info.eri_is_diagonal = all_diagonal;
    info.eri_sparsity = eri_total > 0
        ? static_cast<double>(eri_nonzero) / eri_total : 1.0;

    // --- αβ double excitation table ---
    // ab_exc_table[i] stores all (j, p, q, eri(i,p,j,q)) where the value is non-zero
    info.ab_exc_table.resize(n_orb);
    for (int i = 0; i < n_orb; ++i) {
        for (int p = 0; p < n_orb; ++p) {
            if (p == i) continue; // excitation requires p != i
            for (int j = 0; j < n_orb; ++j) {
                for (int q = 0; q < n_orb; ++q) {
                    if (q == j) continue; // excitation requires q != j
                    double val = get_eri(eri, n_orb, i, p, j, q);
                    if (std::abs(val) > threshold) {
                        info.ab_exc_table[i].emplace_back(j, p, q, val);
                    }
                }
            }
        }
        // Sort by |val| descending for potential early termination
        std::sort(info.ab_exc_table[i].begin(), info.ab_exc_table[i].end(),
                  [](const auto& a, const auto& b) {
                      return std::abs(std::get<3>(a)) > std::abs(std::get<3>(b));
                  });
    }

    info.is_sparse = (info.eri_sparsity < 0.1) || (info.h1_sparsity < 0.3);
    return info;
}

// =============================================================================
// compute_H_ij_t with sparsity fast paths
// =============================================================================
template<typename StorageType>
double compute_H_ij_t(const DeterminantT<StorageType>& det_i,
                      const DeterminantT<StorageType>& det_j,
                      const std::vector<std::vector<double>>& h1,
                      const std::vector<double>& eri,
                      const IntegralSparsityInfo* sparsity) {
    // If no sparsity or not diagonal ERI, fall back to original
    if (!sparsity || !sparsity->eri_is_diagonal) {
        return compute_H_ij_t(det_i, det_j, h1, eri);
    }

    using BitOpsType = detail::HamiltonianBitOps<StorageType>;
    int n_orb = (int)h1.size();

    const auto& ai = det_i.alpha;
    const auto& bi = det_i.beta;
    const auto& aj = det_j.alpha;
    const auto& bj = det_j.beta;

    int toggled_alpha = BitOpsType::count_differences(ai, aj);
    int toggled_beta  = BitOpsType::count_differences(bi, bj);
    int n_toggled     = toggled_alpha + toggled_beta;

    // Same determinant: H_ii = Σ h1[i][i] + U × n_doubly_occupied
    if (n_toggled == 0) {
        double Hij = 0.0;
        int occ_a[512], occ_b[512];
        int n_a = BitOpsType::storage_to_indices_inline(ai, occ_a, 512);
        int n_b = BitOpsType::storage_to_indices_inline(bi, occ_b, 512);

        for (int k = 0; k < n_a; ++k) Hij += h1[occ_a[k]][occ_a[k]];
        for (int k = 0; k < n_b; ++k) Hij += h1[occ_b[k]][occ_b[k]];

        // Only doubly-occupied orbitals contribute: eri(i,i,i,i)
        for (int k = 0; k < n_a; ++k) {
            int ia = occ_a[k];
            if (BitOps<StorageType>::get_bit(bi, ia)) {
                Hij += get_eri(eri, n_orb, ia, ia, ia, ia);
            }
        }
        return Hij;
    }

    // Single excitation: H_ij = sign * h1[m][p] (all ERI terms vanish)
    if (n_toggled == 2) {
        int da_rem[2], da_add[2], db_rem[2], db_add[2];
        int da_rem_cnt = BitOpsType::storage_to_indices_inline(BitOpsType::and_not(ai, aj), da_rem, 2);
        BitOpsType::storage_to_indices_inline(BitOpsType::and_not(aj, ai), da_add, 2);
        int db_rem_cnt = BitOpsType::storage_to_indices_inline(BitOpsType::and_not(bi, bj), db_rem, 2);
        BitOpsType::storage_to_indices_inline(BitOpsType::and_not(bj, bi), db_add, 2);

        if (da_rem_cnt == 1 && db_rem_cnt == 0) {
            int m = da_rem[0], p = da_add[0];
            if (std::abs(h1[m][p]) < 1e-14) return 0.0;  // h1 zero early exit
            int phase = detail::cre_des_sign_t(m, p, aj);
            return h1[m][p] * phase;
        } else if (db_rem_cnt == 1 && da_rem_cnt == 0) {
            int m = db_rem[0], p = db_add[0];
            if (std::abs(h1[m][p]) < 1e-14) return 0.0;  // h1 zero early exit
            int phase = detail::cre_des_sign_t(m, p, bj);
            return h1[m][p] * phase;
        }
    }

    // Double excitation: fall back (already O(1) in original)
    return compute_H_ij_t(det_i, det_j, h1, eri);
}

// Explicit instantiations
template std::tuple<HijCacheT<uint64_t>, std::string>
load_or_create_Hij_cache_t<uint64_t>(const std::string&, int, int, const std::string&);
template std::tuple<HijCacheT<std::array<uint64_t, 2>>, std::string>
load_or_create_Hij_cache_t<std::array<uint64_t, 2>>(const std::string&, int, int, const std::string&);
template std::tuple<HijCacheT<std::array<uint64_t, 3>>, std::string>
load_or_create_Hij_cache_t<std::array<uint64_t, 3>>(const std::string&, int, int, const std::string&);

template std::pair<DeterminantT<uint64_t>, DeterminantT<uint64_t>>
pair_key_t<uint64_t>(const DeterminantT<uint64_t>&, const DeterminantT<uint64_t>&);
template std::pair<DeterminantT<std::array<uint64_t, 2>>, DeterminantT<std::array<uint64_t, 2>>>
pair_key_t<std::array<uint64_t, 2>>(const DeterminantT<std::array<uint64_t, 2>>&, const DeterminantT<std::array<uint64_t, 2>>&);
template std::pair<DeterminantT<std::array<uint64_t, 3>>, DeterminantT<std::array<uint64_t, 3>>>
pair_key_t<std::array<uint64_t, 3>>(const DeterminantT<std::array<uint64_t, 3>>&, const DeterminantT<std::array<uint64_t, 3>>&);

template double compute_H_ij_t<uint64_t>(const DeterminantT<uint64_t>&, const DeterminantT<uint64_t>&,
                                         const std::vector<std::vector<double>>&,
                                         const std::vector<double>&);
template double compute_H_ij_t<std::array<uint64_t, 2>>(const DeterminantT<std::array<uint64_t, 2>>&, const DeterminantT<std::array<uint64_t, 2>>&,
                                                        const std::vector<std::vector<double>>&,
                                                        const std::vector<double>&);
template double compute_H_ij_t<std::array<uint64_t, 3>>(const DeterminantT<std::array<uint64_t, 3>>&, const DeterminantT<std::array<uint64_t, 3>>&,
                                                     const std::vector<std::vector<double>>&,
                                                     const std::vector<double>&);

template std::tuple<HijCacheT<std::array<uint64_t, 4>>, std::string>
load_or_create_Hij_cache_t<std::array<uint64_t, 4>>(const std::string&, int, int, const std::string&);
template std::tuple<HijCacheT<std::array<uint64_t, 5>>, std::string>
load_or_create_Hij_cache_t<std::array<uint64_t, 5>>(const std::string&, int, int, const std::string&);
template std::tuple<HijCacheT<std::array<uint64_t, 6>>, std::string>
load_or_create_Hij_cache_t<std::array<uint64_t, 6>>(const std::string&, int, int, const std::string&);
template std::tuple<HijCacheT<std::array<uint64_t, 7>>, std::string>
load_or_create_Hij_cache_t<std::array<uint64_t, 7>>(const std::string&, int, int, const std::string&);
template std::tuple<HijCacheT<std::array<uint64_t, 8>>, std::string>
load_or_create_Hij_cache_t<std::array<uint64_t, 8>>(const std::string&, int, int, const std::string&);

template std::pair<DeterminantT<std::array<uint64_t, 4>>, DeterminantT<std::array<uint64_t, 4>>>
pair_key_t<std::array<uint64_t, 4>>(const DeterminantT<std::array<uint64_t, 4>>&, const DeterminantT<std::array<uint64_t, 4>>&);
template std::pair<DeterminantT<std::array<uint64_t, 5>>, DeterminantT<std::array<uint64_t, 5>>>
pair_key_t<std::array<uint64_t, 5>>(const DeterminantT<std::array<uint64_t, 5>>&, const DeterminantT<std::array<uint64_t, 5>>&);
template std::pair<DeterminantT<std::array<uint64_t, 6>>, DeterminantT<std::array<uint64_t, 6>>>
pair_key_t<std::array<uint64_t, 6>>(const DeterminantT<std::array<uint64_t, 6>>&, const DeterminantT<std::array<uint64_t, 6>>&);
template std::pair<DeterminantT<std::array<uint64_t, 7>>, DeterminantT<std::array<uint64_t, 7>>>
pair_key_t<std::array<uint64_t, 7>>(const DeterminantT<std::array<uint64_t, 7>>&, const DeterminantT<std::array<uint64_t, 7>>&);
template std::pair<DeterminantT<std::array<uint64_t, 8>>, DeterminantT<std::array<uint64_t, 8>>>
pair_key_t<std::array<uint64_t, 8>>(const DeterminantT<std::array<uint64_t, 8>>&, const DeterminantT<std::array<uint64_t, 8>>&);

template double compute_H_ij_t<std::array<uint64_t, 4>>(const DeterminantT<std::array<uint64_t, 4>>&, const DeterminantT<std::array<uint64_t, 4>>&,
                                                        const std::vector<std::vector<double>>&,
                                                        const std::vector<double>&);
template double compute_H_ij_t<std::array<uint64_t, 5>>(const DeterminantT<std::array<uint64_t, 5>>&, const DeterminantT<std::array<uint64_t, 5>>&,
                                                        const std::vector<std::vector<double>>&,
                                                        const std::vector<double>&);
template double compute_H_ij_t<std::array<uint64_t, 6>>(const DeterminantT<std::array<uint64_t, 6>>&, const DeterminantT<std::array<uint64_t, 6>>&,
                                                        const std::vector<std::vector<double>>&,
                                                        const std::vector<double>&);
template double compute_H_ij_t<std::array<uint64_t, 7>>(const DeterminantT<std::array<uint64_t, 7>>&, const DeterminantT<std::array<uint64_t, 7>>&,
                                                        const std::vector<std::vector<double>>&,
                                                        const std::vector<double>&);
template double compute_H_ij_t<std::array<uint64_t, 8>>(const DeterminantT<std::array<uint64_t, 8>>&, const DeterminantT<std::array<uint64_t, 8>>&,
                                                        const std::vector<std::vector<double>>&,
                                                        const std::vector<double>&);

// Explicit instantiations for sparsity overload
template double compute_H_ij_t<uint64_t>(const DeterminantT<uint64_t>&, const DeterminantT<uint64_t>&,
                                         const std::vector<std::vector<double>>&,
                                         const std::vector<double>&,
                                         const IntegralSparsityInfo*);
template double compute_H_ij_t<std::array<uint64_t, 2>>(const DeterminantT<std::array<uint64_t, 2>>&, const DeterminantT<std::array<uint64_t, 2>>&,
                                                        const std::vector<std::vector<double>>&,
                                                        const std::vector<double>&,
                                                        const IntegralSparsityInfo*);
template double compute_H_ij_t<std::array<uint64_t, 3>>(const DeterminantT<std::array<uint64_t, 3>>&, const DeterminantT<std::array<uint64_t, 3>>&,
                                                        const std::vector<std::vector<double>>&,
                                                        const std::vector<double>&,
                                                        const IntegralSparsityInfo*);
template double compute_H_ij_t<std::array<uint64_t, 4>>(const DeterminantT<std::array<uint64_t, 4>>&, const DeterminantT<std::array<uint64_t, 4>>&,
                                                        const std::vector<std::vector<double>>&,
                                                        const std::vector<double>&,
                                                        const IntegralSparsityInfo*);
template double compute_H_ij_t<std::array<uint64_t, 5>>(const DeterminantT<std::array<uint64_t, 5>>&, const DeterminantT<std::array<uint64_t, 5>>&,
                                                        const std::vector<std::vector<double>>&,
                                                        const std::vector<double>&,
                                                        const IntegralSparsityInfo*);
template double compute_H_ij_t<std::array<uint64_t, 6>>(const DeterminantT<std::array<uint64_t, 6>>&, const DeterminantT<std::array<uint64_t, 6>>&,
                                                        const std::vector<std::vector<double>>&,
                                                        const std::vector<double>&,
                                                        const IntegralSparsityInfo*);
template double compute_H_ij_t<std::array<uint64_t, 7>>(const DeterminantT<std::array<uint64_t, 7>>&, const DeterminantT<std::array<uint64_t, 7>>&,
                                                        const std::vector<std::vector<double>>&,
                                                        const std::vector<double>&,
                                                        const IntegralSparsityInfo*);
template double compute_H_ij_t<std::array<uint64_t, 8>>(const DeterminantT<std::array<uint64_t, 8>>&, const DeterminantT<std::array<uint64_t, 8>>&,
                                                        const std::vector<std::vector<double>>&,
                                                        const std::vector<double>&,
                                                        const IntegralSparsityInfo*);

// =============================================================================
// CI Energy Evaluation Implementation
// =============================================================================

double evaluate_ci_energy(
    const std::vector<uint64_t>& dets_alpha,
    const std::vector<uint64_t>& dets_beta,
    const std::vector<double>& coeffs,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    int n_orb
) {
    int n_det = coeffs.size();
    
    // Parallel reduction for energy
    double E = 0.0;
    
    #pragma omp parallel for reduction(+:E) schedule(dynamic)
    for (int i = 0; i < n_det; ++i) {
        double c_i = coeffs[i];
        Determinant det_i(dets_alpha[i], dets_beta[i]);
        
        // Diagonal contribution: c_i^2 * H_ii
        double H_ii = compute_H_ij(det_i, det_i, h1, eri);
        E += c_i * c_i * H_ii;
        
        // Off-diagonal: 2 * c_i * c_j * H_ij (symmetry exploited)
        for (int j = i + 1; j < n_det; ++j) {
            double c_j = coeffs[j];
            Determinant det_j(dets_alpha[j], dets_beta[j]);
            
            double H_ij = compute_H_ij(det_i, det_j, h1, eri);
            E += 2.0 * c_i * c_j * H_ij;
        }
    }
    
    return E;
}

template<typename StorageType>
double evaluate_ci_energy_t(
    const std::vector<DeterminantT<StorageType>>& dets,
    const std::vector<double>& coeffs,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri
) {
    int n_det = coeffs.size();
    
    double E = 0.0;
    
    #pragma omp parallel for reduction(+:E) schedule(dynamic)
    for (int i = 0; i < n_det; ++i) {
        double c_i = coeffs[i];
        
        // Diagonal
        double H_ii = compute_H_ij_t(dets[i], dets[i], h1, eri);
        E += c_i * c_i * H_ii;
        
        // Off-diagonal
        for (int j = i + 1; j < n_det; ++j) {
            double c_j = coeffs[j];
            double H_ij = compute_H_ij_t(dets[i], dets[j], h1, eri);
            E += 2.0 * c_i * c_j * H_ij;
        }
    }
    
    return E;
}

// Explicit instantiations for evaluate_ci_energy_t
template double evaluate_ci_energy_t<uint64_t>(
    const std::vector<Determinant>&, const std::vector<double>&,
    const std::vector<std::vector<double>>&, const std::vector<double>&);

template double evaluate_ci_energy_t<std::array<uint64_t, 2>>(
    const std::vector<DeterminantT<std::array<uint64_t, 2>>>&, const std::vector<double>&,
    const std::vector<std::vector<double>>&, const std::vector<double>&);

template double evaluate_ci_energy_t<std::array<uint64_t, 3>>(
    const std::vector<DeterminantT<std::array<uint64_t, 3>>>&, const std::vector<double>&,
    const std::vector<std::vector<double>>&, const std::vector<double>&);

template double evaluate_ci_energy_t<std::array<uint64_t, 4>>(
    const std::vector<DeterminantT<std::array<uint64_t, 4>>>&, const std::vector<double>&,
    const std::vector<std::vector<double>>&, const std::vector<double>&);

} // namespace trimci_core

