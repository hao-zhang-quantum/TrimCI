#include "hamiltonian_scalable.hpp"
#include "hamiltonian.hpp"
#include <algorithm>
#include <cstring>

namespace trimci_core {

// Template implementation for Hij cache loading
template<typename StorageType>
std::tuple<HijCacheT<StorageType>, std::string>
load_or_create_Hij_cache_t(const std::string& mol_name,
                           int n_elec, int n_orb,
                           const std::string& cache_dir) {
    // For now, delegate to the original implementation and convert
    // In a full implementation, this would be template-specialized
    auto [cache, path] = load_or_create_Hij_cache(mol_name, n_elec, n_orb, cache_dir);
    
    HijCacheT<StorageType> template_cache;
    for (const auto& [key, value] : cache) {
        // Convert uint64_t to StorageType for template compatibility
        StorageType alpha1, beta1, alpha2, beta2;
        if constexpr (std::is_same_v<StorageType, uint64_t>) {
            alpha1 = key.first.alpha;
            beta1 = key.first.beta;
            alpha2 = key.second.alpha;
            beta2 = key.second.beta;
        } else {
            // For array types, initialize first element with uint64_t value
            alpha1 = {};
            beta1 = {};
            alpha2 = {};
            beta2 = {};
            alpha1[0] = key.first.alpha;
            beta1[0] = key.first.beta;
            alpha2[0] = key.second.alpha;
            beta2[0] = key.second.beta;
        }
        
        DeterminantT<StorageType> det1(alpha1, beta1);
        DeterminantT<StorageType> det2(alpha2, beta2);
        template_cache[{det1, det2}] = value;
    }
    
    return {template_cache, path};
}

// Explicit instantiations for common types
template std::tuple<HijCacheT<uint64_t>, std::string>
load_or_create_Hij_cache_t<uint64_t>(const std::string&, int, int, const std::string&);

template std::tuple<HijCacheT<std::array<uint64_t, 2>>, std::string>
load_or_create_Hij_cache_t<std::array<uint64_t, 2>>(const std::string&, int, int, const std::string&);

template std::tuple<HijCacheT<std::array<uint64_t, 3>>, std::string>
load_or_create_Hij_cache_t<std::array<uint64_t, 3>>(const std::string&, int, int, const std::string&);

// Backward compatible version (renamed to avoid conflict)
std::tuple<HijCacheT<uint64_t>, std::string>
load_or_create_Hij_cache_compat(const std::string& mol_name,
                                 int n_elec, int n_orb,
                                 const std::string& cache_dir) {
    // Delegate to template implementation
    return load_or_create_Hij_cache_t<uint64_t>(mol_name, n_elec, n_orb, cache_dir);
}

// Template implementation for pair key
template<typename StorageType>
std::pair<DeterminantT<StorageType>, DeterminantT<StorageType>>
pair_key_t(const DeterminantT<StorageType>& d1, const DeterminantT<StorageType>& d2) {
    if (d1 < d2) return {d1, d2};
    return {d2, d1};
}

// Explicit instantiations
template std::pair<DeterminantT<uint64_t>, DeterminantT<uint64_t>>
pair_key_t<uint64_t>(const DeterminantT<uint64_t>&, const DeterminantT<uint64_t>&);

template std::pair<DeterminantT<std::array<uint64_t, 2>>, DeterminantT<std::array<uint64_t, 2>>>
pair_key_t<std::array<uint64_t, 2>>(const DeterminantT<std::array<uint64_t, 2>>&, const DeterminantT<std::array<uint64_t, 2>>&);

template
std::pair<DeterminantT<std::array<uint64_t, 3>>, DeterminantT<std::array<uint64_t, 3>>>
pair_key_t<std::array<uint64_t, 3>>(const DeterminantT<std::array<uint64_t, 3>>&, const DeterminantT<std::array<uint64_t, 3>>&);

// Template implementation for Slater-Condon matrix elements
template<typename StorageType>
double compute_H_ij_t(const DeterminantT<StorageType>& det_i,
                      const DeterminantT<StorageType>& det_j,
                      const std::vector<std::vector<double>>& h1,
                      const std::vector<std::vector<std::vector<std::vector<double>>>>& eri) {
    
    using BitOpsType = detail::HamiltonianBitOps<StorageType>;
    
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
        auto occ_a = BitOpsType::storage_to_indices(ai);
        auto occ_b = BitOpsType::storage_to_indices(bi);
        
        for (int i : occ_a) Hij += h1[i][i];
        for (int i : occ_b) Hij += h1[i][i];
        
        // Two-electron terms
        for (int i : occ_a) {
            for (int j : occ_a) if (j > i)
                Hij += eri[i][i][j][j] - eri[i][j][j][i];
            for (int j : occ_b)
                Hij += eri[i][i][j][j];
        }
        for (int i : occ_b) {
            for (int j : occ_b) if (j > i)
                Hij += eri[i][i][j][j] - eri[i][j][j][i];
        }
        
        return Hij;
    }
    
    // Single excitation
    if (n_toggled == 2) {
        auto da_xor = BitOpsType::xor_storage(ai, aj);
        auto db_xor = BitOpsType::xor_storage(bi, bj);
        
        int da_rem[2], da_add[2], db_rem[2], db_add[2];
        int da_rem_cnt = BitOpsType::storage_to_indices_inline(BitOpsType::and_not(ai, aj), da_rem, 2);
        int da_add_cnt = BitOpsType::storage_to_indices_inline(BitOpsType::and_not(aj, ai), da_add, 2);
        int db_rem_cnt = BitOpsType::storage_to_indices_inline(BitOpsType::and_not(bi, bj), db_rem, 2);
        int db_add_cnt = BitOpsType::storage_to_indices_inline(BitOpsType::and_not(bj, bi), db_add, 2);
        
        if (da_rem_cnt == 1 && db_rem_cnt == 0) {
            int m = da_rem[0], p = da_add[0];
            int phase = detail::cre_des_sign_t(m, p, aj);
            double Hij = h1[m][p];
            
            auto occ_a = BitOpsType::storage_to_indices(ai);
            for (int n : occ_a) if (n != m)
                Hij += eri[m][p][n][n] - eri[m][n][n][p];
            
            auto occ_b = BitOpsType::storage_to_indices(bi);
            for (int n : occ_b)
                Hij += eri[m][p][n][n];
            
            return Hij * phase;
        } else if (db_rem_cnt == 1 && da_rem_cnt == 0) {
            int m = db_rem[0], p = db_add[0];
            int phase = detail::cre_des_sign_t(m, p, bj);
            double Hij = h1[m][p];
            
            auto occ_b = BitOpsType::storage_to_indices(bi);
            for (int n : occ_b) if (n != m)
                Hij += eri[m][p][n][n] - eri[m][n][n][p];
            
            auto occ_a = BitOpsType::storage_to_indices(ai);
            for (int n : occ_a)
                Hij += eri[m][p][n][n];
            
            return Hij * phase;
        }
    }
    
    // Double excitation
    if (n_toggled == 4) {
        auto da_xor = BitOpsType::xor_storage(ai, aj);
        auto db_xor = BitOpsType::xor_storage(bi, bj);
        
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
            
            return phase1 * phase2 * (eri[m][p][n][q] - eri[m][q][n][p]);
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
            
            return phase1 * phase2 * (eri[m][p][n][q] - eri[m][q][n][p]);
        } else if (da_rem_cnt == 1 && db_rem_cnt == 1) {
            int m = da_rem[0], p = da_add[0];
            int n = db_rem[0], q = db_add[0];
            int phase = detail::cre_des_sign_t(m, p, aj) * detail::cre_des_sign_t(n, q, bj);
            return phase * eri[m][p][n][q];
        }
    }
    
    // Higher excitations
    return 0.0;
}

// Explicit instantiations
template double compute_H_ij_t<uint64_t>(const DeterminantT<uint64_t>&, const DeterminantT<uint64_t>&,
                                         const std::vector<std::vector<double>>&,
                                         const std::vector<std::vector<std::vector<std::vector<double>>>>&);

template double compute_H_ij_t<std::array<uint64_t, 2>>(const DeterminantT<std::array<uint64_t, 2>>&, const DeterminantT<std::array<uint64_t, 2>>&,
                                                        const std::vector<std::vector<double>>&,
                                                        const std::vector<std::vector<std::vector<std::vector<double>>>>&);

template double compute_H_ij_t<std::array<uint64_t, 3>>(const DeterminantT<std::array<uint64_t, 3>>&, const DeterminantT<std::array<uint64_t, 3>>&,
                                                     const std::vector<std::vector<double>>&,
                                                     const std::vector<std::vector<std::vector<std::vector<double>>>>&);

} // namespace trimci_core