#include "excitation_generator.hpp"

#include <vector>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <unordered_map>

#include "bit_compat.hpp"

namespace trimci_core {

// Excitation type IDs (matching Python convention)
enum ExcTypeGen : int {
    GEN_S_ALPHA = 0,
    GEN_S_BETA  = 1,
    GEN_D_AA    = 2,
    GEN_D_AB    = 3,
    GEN_D_BB    = 4
};

// Excitation key: packed into a single uint64_t for fast hashing.
// Layout: [type:3][i:8][j:8][a:8][b:8] = 35 bits total
// For singles: j=0, b=0 (unused)
inline uint64_t pack_key(int type, int i, int j, int a, int b) {
    return (static_cast<uint64_t>(type) << 32)
         | (static_cast<uint64_t>(i & 0xFF) << 24)
         | (static_cast<uint64_t>(j & 0xFF) << 16)
         | (static_cast<uint64_t>(a & 0xFF) << 8)
         | (static_cast<uint64_t>(b & 0xFF));
}

inline void unpack_key(uint64_t key, int& type, int& i, int& j, int& a, int& b) {
    type = static_cast<int>(key >> 32);
    i = static_cast<int>((key >> 24) & 0xFF);
    j = static_cast<int>((key >> 16) & 0xFF);
    a = static_cast<int>((key >> 8) & 0xFF);
    b = static_cast<int>(key & 0xFF);
}

// Extract occupied orbital indices from a bitstring
inline void get_occupied(uint64_t bits, int n_orb, std::vector<int>& occ) {
    occ.clear();
    uint64_t b = bits;
    while (b) {
        int idx = __builtin_ctzll(b);
        if (idx < n_orb) occ.push_back(idx);
        b &= b - 1;  // clear lowest set bit
    }
}

// Extract virtual orbital indices from a bitstring
inline void get_virtual(uint64_t bits, int n_orb, std::vector<int>& vir) {
    vir.clear();
    for (int p = 0; p < n_orb; ++p) {
        if (!(bits & (1ULL << p))) {
            vir.push_back(p);
        }
    }
}

int generate_excitations_heatbath(
    const uint64_t* ref_alpha,
    const uint64_t* ref_beta,
    const double* ref_coeffs,
    int n_ref,
    int n_orb,
    const double* F_aa,
    const double* F_bb,
    const double* eri,
    int max_excitations,
    std::vector<int>& out_types,
    std::vector<int>& out_indices,
    int scoring_mode)
{
    // Phase 1: Enumerate all excitations and accumulate weights
    // key -> accumulated |c_I|^2
    std::unordered_map<uint64_t, double> weight_map;
    weight_map.reserve(1 << 20);  // pre-allocate for ~1M entries

    std::vector<int> occ_a, occ_b, vir_a, vir_b;

    for (int I = 0; I < n_ref; ++I) {
        double w = ref_coeffs[I] * ref_coeffs[I];
        uint64_t alpha = ref_alpha[I];
        uint64_t beta  = ref_beta[I];

        get_occupied(alpha, n_orb, occ_a);
        get_occupied(beta,  n_orb, occ_b);
        get_virtual(alpha, n_orb, vir_a);
        get_virtual(beta,  n_orb, vir_b);

        int n_oa = static_cast<int>(occ_a.size());
        int n_ob = static_cast<int>(occ_b.size());
        int n_va = static_cast<int>(vir_a.size());
        int n_vb = static_cast<int>(vir_b.size());

        // Single alpha: i_a -> a_a
        for (int ii = 0; ii < n_oa; ++ii) {
            for (int aa = 0; aa < n_va; ++aa) {
                uint64_t key = pack_key(GEN_S_ALPHA, occ_a[ii], 0, vir_a[aa], 0);
                weight_map[key] += w;
            }
        }

        // Single beta: i_b -> a_b
        for (int ii = 0; ii < n_ob; ++ii) {
            for (int aa = 0; aa < n_vb; ++aa) {
                uint64_t key = pack_key(GEN_S_BETA, occ_b[ii], 0, vir_b[aa], 0);
                weight_map[key] += w;
            }
        }

        // Double alpha-alpha: (i,j) -> (a,b) with i<j, a<b
        for (int ii = 0; ii < n_oa; ++ii) {
            for (int jj = ii + 1; jj < n_oa; ++jj) {
                for (int aa = 0; aa < n_va; ++aa) {
                    for (int bb = aa + 1; bb < n_va; ++bb) {
                        uint64_t key = pack_key(GEN_D_AA,
                            occ_a[ii], occ_a[jj], vir_a[aa], vir_a[bb]);
                        weight_map[key] += w;
                    }
                }
            }
        }

        // Double beta-beta: (i,j) -> (a,b) with i<j, a<b
        for (int ii = 0; ii < n_ob; ++ii) {
            for (int jj = ii + 1; jj < n_ob; ++jj) {
                for (int aa = 0; aa < n_vb; ++aa) {
                    for (int bb = aa + 1; bb < n_vb; ++bb) {
                        uint64_t key = pack_key(GEN_D_BB,
                            occ_b[ii], occ_b[jj], vir_b[aa], vir_b[bb]);
                        weight_map[key] += w;
                    }
                }
            }
        }

        // Double alpha-beta: (i_a, i_b) -> (a_a, a_b)
        for (int ii = 0; ii < n_oa; ++ii) {
            for (int jj = 0; jj < n_ob; ++jj) {
                for (int aa = 0; aa < n_va; ++aa) {
                    for (int bb = 0; bb < n_vb; ++bb) {
                        uint64_t key = pack_key(GEN_D_AB,
                            occ_a[ii], occ_b[jj], vir_a[aa], vir_b[bb]);
                        weight_map[key] += w;
                    }
                }
            }
        }
    }

    int n_total = static_cast<int>(weight_map.size());

    // Phase 2: If no filtering needed, output all
    if (max_excitations < 0 || n_total <= max_excitations) {
        out_types.resize(n_total);
        out_indices.resize(n_total * 4, 0);
        int idx = 0;
        for (auto& kv : weight_map) {
            int type, i, j, a, b;
            unpack_key(kv.first, type, i, j, a, b);
            out_types[idx] = type;
            out_indices[idx * 4 + 0] = i;
            out_indices[idx * 4 + 1] = (type >= GEN_D_AA) ? j : a;  // singles: [i,a,0,0]
            out_indices[idx * 4 + 2] = (type >= GEN_D_AA) ? a : 0;
            out_indices[idx * 4 + 3] = (type >= GEN_D_AA) ? b : 0;
            idx++;
        }
        fprintf(stderr, "[excgen] Total %d excitations (no filtering)\n", n_total);
        return n_total;
    }

    // Phase 3: Scoring and top-K selection
    // scoring_mode 0 (heat_bath): score = |coupling| × weight
    // scoring_mode 1 (enpt2):     score = (coupling × weight)² / max(Δε, 1e-3)
    struct ScoredExc {
        uint64_t key;
        double score;
    };

    std::vector<ScoredExc> scored;
    scored.reserve(n_total);

    int N = n_orb;  // shorthand for indexing
    constexpr double ENPT2_REG = 1e-3;  // regularization for Δε

    for (auto& kv : weight_map) {
        int type, i, j, a, b;
        unpack_key(kv.first, type, i, j, a, b);
        double weight = kv.second;
        double coupling = 0.0;
        double delta_eps = 1.0;

        switch (type) {
        case GEN_S_ALPHA:
            coupling = std::abs(F_aa[a * N + i]);
            if (scoring_mode == 1)
                delta_eps = std::abs(F_aa[a * N + a] - F_aa[i * N + i]);
            break;
        case GEN_S_BETA:
            coupling = std::abs(F_bb[a * N + i]);
            if (scoring_mode == 1)
                delta_eps = std::abs(F_bb[a * N + a] - F_bb[i * N + i]);
            break;
        case GEN_D_AA:
            coupling = std::abs(
                eri[((a * N + i) * N + b) * N + j] -
                eri[((a * N + j) * N + b) * N + i]);
            if (scoring_mode == 1)
                delta_eps = std::abs(F_aa[a*N+a] + F_aa[b*N+b]
                                   - F_aa[i*N+i] - F_aa[j*N+j]);
            break;
        case GEN_D_AB:
            coupling = std::abs(
                eri[((a * N + i) * N + b) * N + j]);
            if (scoring_mode == 1)
                delta_eps = std::abs(F_aa[a*N+a] + F_bb[b*N+b]
                                   - F_aa[i*N+i] - F_bb[j*N+j]);
            break;
        case GEN_D_BB:
            coupling = std::abs(
                eri[((a * N + i) * N + b) * N + j] -
                eri[((a * N + j) * N + b) * N + i]);
            if (scoring_mode == 1)
                delta_eps = std::abs(F_bb[a*N+a] + F_bb[b*N+b]
                                   - F_bb[i*N+i] - F_bb[j*N+j]);
            break;
        }

        double h0mu = coupling * weight;
        double score;
        if (scoring_mode == 1) {
            delta_eps = std::max(delta_eps, ENPT2_REG);
            score = h0mu * h0mu / delta_eps;
        } else {
            score = h0mu;
        }

        scored.push_back({kv.first, score});
    }

    // Partial sort: top max_excitations in descending score order
    std::partial_sort(scored.begin(), scored.begin() + max_excitations,
                      scored.end(),
                      [](const ScoredExc& a, const ScoredExc& b) {
                          return a.score > b.score;
                      });

    const char* mode_str = (scoring_mode == 1) ? "enpt2" : "heat-bath";
    fprintf(stderr, "[excgen] Filtered %d -> %d excitations (%s)\n",
            n_total, max_excitations, mode_str);

    // Phase 4: Output
    int n_out = max_excitations;
    out_types.resize(n_out);
    out_indices.resize(n_out * 4, 0);

    for (int idx = 0; idx < n_out; ++idx) {
        int type, i, j, a, b;
        unpack_key(scored[idx].key, type, i, j, a, b);
        out_types[idx] = type;
        // Pack indices: singles [i,a,0,0], doubles [i,j,a,b]
        if (type < GEN_D_AA) {
            // Single: i, a
            out_indices[idx * 4 + 0] = i;
            out_indices[idx * 4 + 1] = a;
            out_indices[idx * 4 + 2] = 0;
            out_indices[idx * 4 + 3] = 0;
        } else {
            // Double: i, j, a, b
            out_indices[idx * 4 + 0] = i;
            out_indices[idx * 4 + 1] = j;
            out_indices[idx * 4 + 2] = a;
            out_indices[idx * 4 + 3] = b;
        }
    }

    return n_out;
}

} // namespace trimci_core
