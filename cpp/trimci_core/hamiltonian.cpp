#include "determinant.hpp"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <tuple>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include "bit_compat.hpp"

namespace trimci_core {
namespace fs = std::filesystem;

// —— 最大激发差异数（单/双激发索引上限，用于栈上数组）
constexpr int MAX_DIFF = 4;

// —— 辅助：计算激发相位 ——
static inline int cre_des_sign(int p, int a, uint64_t bitstring) {
    if (p == a) return 1;
    int low  = std::min(p, a) + 1;
    int high = std::max(p, a);
    uint64_t mask = ((uint64_t(1) << high) - 1)
                  ^ ((uint64_t(1) << low)  - 1);
    int cnt = __builtin_popcountll(bitstring & mask);
    return (cnt % 2 == 0) ? 1 : -1;
}

// —— 内联：将位掩码转为索引列表（栈分配版）—— avoids heap allocations
static inline int bits_to_indices_inline(uint64_t mask, int *out) {
    int cnt = 0;
    while (mask) {
        int idx = __builtin_ctzll(mask);
        out[cnt++] = idx;
        mask &= mask - 1;
    }
    return cnt;
}

// 原 dynamic bits_to_indices，仅用于获取占据列表
static std::vector<int> bits_to_indices(uint64_t mask) {
    std::vector<int> idx;
    while (mask) {
        uint64_t lsb = mask & -mask;
        int i = __builtin_ctzll(lsb);
        idx.push_back(i);
        mask &= mask - 1;
    }
    return idx;
}

// 1) 分子式提取
std::string extract_mol_name(const std::string& atom_str) {
    std::istringstream iss(atom_str);
    std::string token;
    std::map<std::string,int> counter;
    while (std::getline(iss, token, ';')) {
        std::istringstream line(token);
        std::string sym;
        line >> sym;
        if (!sym.empty()) counter[sym]++;
    }
    std::ostringstream oss;
    for (auto const& [el, cnt] : counter) {
        oss << el << (cnt > 1 ? std::to_string(cnt) : "");
    }
    return oss.str();
}

// 2) Hij 缓存类型定义
using HijCache = std::map<std::pair<Determinant,Determinant>, double>;

// 3) 磁盘缓存加载或创建
std::tuple<HijCache, std::string>
load_or_create_Hij_cache(const std::string& mol_name,
                         int n_elec, int n_orb,
                         const std::string& cache_dir = "cache") {
    fs::create_directories(cache_dir);
    std::ostringstream fname;
    fname << cache_dir << "/" << mol_name
          << "__" << n_elec << "e_" << n_orb << "o.bin";
    std::string path = fname.str();

    if (!fs::exists(path)) {
        std::ofstream ofs(path, std::ios::binary);
        uint64_t zero = 0;
        ofs.write(reinterpret_cast<const char*>(&zero), sizeof(zero));
    }

    HijCache cache;
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) throw std::runtime_error("Cannot open cache file " + path);
    uint64_t count = 0;
    ifs.read(reinterpret_cast<char*>(&count), sizeof(count));
    for (uint64_t i = 0; i < count; ++i) {
        uint64_t a1,b1,a2,b2;
        double val;
        ifs.read(reinterpret_cast<char*>(&a1), sizeof(a1));
        ifs.read(reinterpret_cast<char*>(&b1), sizeof(b1));
        ifs.read(reinterpret_cast<char*>(&a2), sizeof(a2));
        ifs.read(reinterpret_cast<char*>(&b2), sizeof(b2));
        ifs.read(reinterpret_cast<char*>(&val), sizeof(val));
        Determinant d1(a1,b1), d2(a2,b2);
        cache[{d1,d2}] = val;
    }
    return {cache, path};
}

// 4) 对称化 key 确保 (i,j) 与 (j,i) 同一条目
std::pair<Determinant,Determinant>
pair_key(const Determinant& d1, const Determinant& d2) {
    if (std::tie(d1.alpha,d1.beta) <= std::tie(d2.alpha,d2.beta))
        return {d1,d2};
    else
        return {d2,d1};
}

// 5) Slater–Condon 矩阵元计算（使用内联差异提取）
double compute_H_ij(
    const Determinant& di,
    const Determinant& dj,
    const std::vector<std::vector<double>>& h1,
    const std::vector<std::vector<std::vector<std::vector<double>>>>& eri
) {
    uint64_t ai = di.alpha, bi = di.beta;
    uint64_t aj = dj.alpha, bj = dj.beta;


    int diff = __builtin_popcountll(ai ^ aj) + __builtin_popcountll(bi ^ bj);
    // 如果差异超过双激发（4个轨道不同），则矩阵元必为0
    if (diff > 4) {
        return 0.0;
    }

    uint64_t da_rem_mask = ai & ~aj;
    uint64_t da_add_mask = aj & ~ai;
    uint64_t db_rem_mask = bi & ~bj;
    uint64_t db_add_mask = bj & ~bi;

    int da_rem_cnt, da_add_cnt, db_rem_cnt, db_add_cnt;
    int da_rem[MAX_DIFF], da_add[MAX_DIFF], db_rem[MAX_DIFF], db_add[MAX_DIFF];

    da_rem_cnt  = bits_to_indices_inline(da_rem_mask, da_rem);
    da_add_cnt  = bits_to_indices_inline(da_add_mask, da_add);
    db_rem_cnt  = bits_to_indices_inline(db_rem_mask, db_rem);
    db_add_cnt  = bits_to_indices_inline(db_add_mask, db_add);

    int n_diff = da_rem_cnt + db_rem_cnt;

    // ——— 0) 对角项 ———
    if (n_diff == 0) {
        double Hij = 0.0;
        auto occ_a = bits_to_indices(ai);
        auto occ_b = bits_to_indices(bi);

        for (int p : occ_a) Hij += h1[p][p];
        for (int p : occ_b) Hij += h1[p][p];

        for (int p : occ_a)
            for (int q : occ_a)
                Hij += 0.5 * (eri[p][p][q][q] - eri[p][q][q][p]);
        for (int p : occ_b)
            for (int q : occ_b)
                Hij += 0.5 * (eri[p][p][q][q] - eri[p][q][q][p]);
        for (int p : occ_b)
            for (int q : occ_a)
                Hij += eri[p][p][q][q];
        
        return Hij;
    }

    // ——— 1) 单激发 ———
    if (n_diff == 1) {
        if  (da_rem_cnt == 1 && db_rem_cnt == 0) {
            int m = da_rem[0], p = da_add[0];
            int phase = cre_des_sign(m, p, aj);
            double Hij = h1[m][p];
            auto occ_a = bits_to_indices(ai);
            for (int n : occ_a) if (n != m)
                Hij += eri[m][p][n][n] - eri[m][n][n][p];
            auto occ_b = bits_to_indices(bi);
            for (int n : occ_b)
                Hij += eri[m][p][n][n];
            return Hij * phase;
        } else if (db_rem_cnt == 1 && da_rem_cnt == 0) {
            int m = db_rem[0], p = db_add[0];
            int phase = cre_des_sign(m, p, bj);
            double Hij = h1[m][p];
            auto occ_b = bits_to_indices(bi);
            for (int n : occ_b) if (n != m)
                Hij += eri[m][p][n][n] - eri[m][n][n][p];
            auto occ_a = bits_to_indices(ai);
            for (int n : occ_a)
                Hij += eri[m][p][n][n];
            return Hij * phase;
        }
    }

    // ——— 2) 双激发 ———
    if (n_diff == 2) {
        if      (da_rem_cnt == 2 && db_rem_cnt == 0) {
            int m = std::min(da_rem[0], da_rem[1]);
            int n = std::max(da_rem[0], da_rem[1]);
            int p = std::min(da_add[0], da_add[1]);
            int q = std::max(da_add[0], da_add[1]);
            int phase1 = cre_des_sign(m, p, aj);
            uint64_t new_a = (aj | (uint64_t(1) << m)) & ~(uint64_t(1) << p);
            int phase2 = cre_des_sign(n, q, new_a);
            return phase1 * phase2 * (eri[m][p][n][q] - eri[m][q][n][p]);
        } else if (db_rem_cnt == 2 && da_rem_cnt == 0) {
            int m = std::min(db_rem[0], db_rem[1]);
            int n = std::max(db_rem[0], db_rem[1]);
            int p = std::min(db_add[0], db_add[1]);
            int q = std::max(db_add[0], db_add[1]);
            int phase1 = cre_des_sign(m, p, bj);
            uint64_t new_b = (bj | (uint64_t(1) << m)) & ~(uint64_t(1) << p);
            int phase2 = cre_des_sign(n, q, new_b);
            return phase1 * phase2 * (eri[m][p][n][q] - eri[m][q][n][p]);
        } else {
            int m = da_rem[0], p = da_add[0];
            int n = db_rem[0], q = db_add[0];
            int phase = cre_des_sign(m, p, aj) * cre_des_sign(n, q, bj);
            return phase * eri[m][p][n][q];
        }
    }

    // 其余激发 → 0
    return 0.0;
}

} // namespace trimci_core
