#include "morton_sort.hpp"
#include <algorithm>
#include <numeric>

namespace trimci_core {
namespace fe {

template<typename StorageType>
void morton_sort_dets(std::vector<DeterminantT<StorageType>>& dets, int n_orb)
{
    const size_t N = dets.size();
    if (N <= 1) return;

    // Step 1: Assign unique alpha/beta IDs via hash map
    fe_map<StorageType, uint32_t, SplitMix64Hash> alpha_map, beta_map;
    std::vector<uint32_t> a_ids(N), b_ids(N);
    uint32_t next_a = 0, next_b = 0;

    for (size_t i = 0; i < N; ++i) {
        auto [it_a, ins_a] = alpha_map.emplace(dets[i].alpha, next_a);
        if (ins_a) ++next_a;
        a_ids[i] = it_a->second;

        auto [it_b, ins_b] = beta_map.emplace(dets[i].beta, next_b);
        if (ins_b) ++next_b;
        b_ids[i] = it_b->second;
    }

    // Step 2: Compute Morton keys
    std::vector<uint64_t> keys(N);
    for (size_t i = 0; i < N; ++i) {
        keys[i] = morton_interleave(a_ids[i], b_ids[i]);
    }

    // Step 3: Argsort by Morton key
    std::vector<size_t> perm(N);
    std::iota(perm.begin(), perm.end(), size_t(0));
    std::sort(perm.begin(), perm.end(), [&](size_t x, size_t y) {
        return keys[x] < keys[y];
    });

    // Step 4: Apply permutation in-place using cycle-leader algorithm
    std::vector<bool> visited(N, false);
    for (size_t i = 0; i < N; ++i) {
        if (visited[i] || perm[i] == i) continue;
        DeterminantT<StorageType> tmp = std::move(dets[i]);
        size_t j = i;
        while (!visited[j]) {
            visited[j] = true;
            size_t next = perm[j];
            if (next == i) {
                dets[j] = std::move(tmp);
            } else {
                dets[j] = std::move(dets[next]);
            }
            j = next;
        }
    }
}

// Explicit template instantiations
template void morton_sort_dets<uint64_t>(
    std::vector<DeterminantT<uint64_t>>& dets, int n_orb);
template void morton_sort_dets<std::array<uint64_t, 2>>(
    std::vector<DeterminantT<std::array<uint64_t, 2>>>& dets, int n_orb);
template void morton_sort_dets<std::array<uint64_t, 4>>(
    std::vector<DeterminantT<std::array<uint64_t, 4>>>& dets, int n_orb);

}  // namespace fe
}  // namespace trimci_core
