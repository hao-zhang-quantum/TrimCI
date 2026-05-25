#pragma once
// Portable parallel sort using OpenMP (no GNU extensions needed).
// Falls back to std::sort when OpenMP is unavailable or N is small.

#include <algorithm>
#include <iterator>
#include "omp_compat.hpp"

namespace trimci {

template<typename RandomIt, typename Compare>
void parallel_sort(RandomIt first, RandomIt last, Compare comp) {
    using diff_t = typename std::iterator_traits<RandomIt>::difference_type;
    const diff_t n = std::distance(first, last);
    const int P = omp_get_max_threads();

    if (n < 100000 || P <= 1) {
        std::sort(first, last, comp);
        return;
    }

    const diff_t chunk = (n + P - 1) / P;

    // Phase 1: sort each chunk in parallel
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < P; t++) {
        diff_t lo = std::min(static_cast<diff_t>(t) * chunk, n);
        diff_t hi = std::min(lo + chunk, n);
        if (lo < hi) {
            std::sort(first + lo, first + hi, comp);
        }
    }

    // Phase 2: pairwise merge (log2(P) rounds)
    for (diff_t width = chunk; width < n; width *= 2) {
        const diff_t stride = 2 * width;
        const diff_t n_pairs = (n + stride - 1) / stride;
        #pragma omp parallel for schedule(dynamic)
        for (diff_t i = 0; i < n_pairs; i++) {
            auto lo  = first + std::min(i * stride, n);
            auto mid = first + std::min(i * stride + width, n);
            auto hi  = first + std::min((i + 1) * stride, n);
            if (mid < hi) {
                std::inplace_merge(lo, mid, hi, comp);
            }
        }
    }
}

template<typename RandomIt>
void parallel_sort(RandomIt first, RandomIt last) {
    using T = typename std::iterator_traits<RandomIt>::value_type;
    parallel_sort(first, last, std::less<T>{});
}

} // namespace trimci
