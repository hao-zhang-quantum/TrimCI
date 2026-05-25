#pragma once
/**
 * Fast Expansion: Threshold self-adaptation for growth control.
 *
 * Given current dets/coeffs and a target_n_new_dets, estimates the
 * screening threshold that will produce approximately that many new dets.
 *
 * Algorithm: pilot screening on 10% of dets → fit log-log relationship
 *            → interpolate to find target threshold.
 *
 * Header-only.
 */

#include <vector>
#include <cstddef>

#include "determinant.hpp"

namespace trimci_core {
namespace fe {

/// Estimate screening threshold to produce approximately target_n_new_dets.
template<typename StorageType>
double estimate_threshold(
    const std::vector<DeterminantT<StorageType>>& dets,
    const std::vector<double>& coeffs,
    size_t target_n_new_dets,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    int n_orb);

/// Compute growth schedule: sequence of target sizes from initial to max.
std::vector<size_t> make_growth_schedule(
    size_t initial_size,
    size_t max_size,
    double growth_factor = 5.0);

}  // namespace fe
}  // namespace trimci_core
