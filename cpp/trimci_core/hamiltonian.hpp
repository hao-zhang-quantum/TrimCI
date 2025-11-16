#ifndef TRIMCI_CORE_HAMILTONIAN_HPP
#define TRIMCI_CORE_HAMILTONIAN_HPP

#include <string>
#include <tuple>
#include <map>
#include <vector>
#include <cstdint>
#include "determinant.hpp"

// Include scalable hamiltonian for template-based functionality
#include "hamiltonian_scalable.hpp"

namespace trimci_core {

// 把 "X x y z; Y ..." 转成分子式比如 "N2O"
std::string extract_mol_name(const std::string& atom_str);

// Hij 缓存（key 用一对 Determinant）
using HijCache = std::map<std::pair<Determinant, Determinant>, double>;

// 加载或创建磁盘缓存
std::tuple<HijCache, std::string>
load_or_create_Hij_cache(const std::string& mol_name,
                         int n_elec, int n_orb,
                         const std::string& cache_dir = "cache");

// 对称化 key 保证 (i,j) 与 (j,i) 同一 entry
std::pair<Determinant, Determinant>
pair_key(const Determinant& d1, const Determinant& d2);

// 计算 Slater–Condon 矩阵元素 ⟨D_i|H|D_j⟩
double compute_H_ij(const Determinant& det_i,
                    const Determinant& det_j,
                    const std::vector<std::vector<double>>& h1,
                    const std::vector<std::vector<std::vector<std::vector<double>>>>& eri);

} // namespace trimci_core

#endif // TRIMCI_CORE_HAMILTONIAN_HPP
