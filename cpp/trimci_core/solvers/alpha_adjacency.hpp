#pragma once
// Build adjacency list for alpha string groups.
// Two groups are adjacent if their alpha strings differ by exactly one
// single excitation, i.e. popcount(alpha[i] XOR alpha[j]) == 2.

#include <vector>
#include <cstdint>
#include <unordered_map>

namespace trimci_core {

/// Build adjacency list for alpha string groups (CSR-friendly).
/// Input:  sorted array of unique alpha strings (uint64_t), length n_groups.
/// Output: for each group i, a vector of adjacent group indices j.
///
/// Algorithm: for each group, enumerate all single excitations (occ -> virt)
/// and look up the resulting string in a hash map.  Complexity is
/// O(n_groups * n_occ * n_virt) hash lookups, which is much faster than the
/// O(n_groups^2) brute-force approach when n_groups is large.
inline std::vector<std::vector<int32_t>> build_alpha_adjacency(
    const uint64_t* alpha_strings,
    int n_groups,
    int n_orb)
{
    // Step 1: Build hash map  string -> index
    std::unordered_map<uint64_t, int32_t> str_to_idx;
    str_to_idx.reserve(n_groups);
    for (int i = 0; i < n_groups; i++) {
        str_to_idx[alpha_strings[i]] = i;
    }

    // Step 2: For each group, enumerate all single excitations and look up
    std::vector<std::vector<int32_t>> adjacency(n_groups);

    for (int g = 0; g < n_groups; g++) {
        uint64_t alpha = alpha_strings[g];

        // For each occupied orbital m, for each virtual orbital p:
        //   excited = alpha ^ (1ULL << m) ^ (1ULL << p)
        // This flips m off and p on, producing a single excitation.
        for (int m = 0; m < n_orb; m++) {
            if (!((alpha >> m) & 1)) continue;  // m must be occupied
            for (int p = 0; p < n_orb; p++) {
                if ((alpha >> p) & 1) continue;  // p must be virtual
                uint64_t excited = alpha ^ (1ULL << m) ^ (1ULL << p);
                auto it = str_to_idx.find(excited);
                if (it != str_to_idx.end()) {
                    adjacency[g].push_back(it->second);
                }
            }
        }
    }

    return adjacency;
}

}  // namespace trimci_core
