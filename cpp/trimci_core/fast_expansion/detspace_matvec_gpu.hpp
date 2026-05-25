#pragma once
/**
 * Fast Expansion: GPU matvec session for Davidson.
 *
 * Reuses Phase 0 GPU infrastructure (GpuContext, kernel_ch1/ch2/ch3) to
 * run H*v for Phase 1/2 Davidson on the GPU. One session per expansion
 * round: integrals + dets + AB groups uploaded once, then `apply()` is
 * called for each Davidson iteration.
 *
 * Build gate: only available when TRIMCI_HAS_GPU is defined (i.e. the
 * trimci_gpu target was built). Guarded by #ifdef so CPU-only builds
 * don't need CUDA headers.
 *
 * Cost model:
 *   - Construct: O(N_dets) H2D + AB-groups build (one-time per round)
 *   - apply():  H2D v (N doubles) → kernel launches → D2H sigma (N doubles)
 */

#ifdef TRIMCI_HAS_GPU

#include <memory>
#include <vector>
#include "ab_index.hpp"

namespace trimci_core {
namespace fe {

/// GPU matvec session. Non-copyable, non-movable — owns CUDA resources
/// through a pimpl that holds GpuContext + device buffers.
template<typename StorageType>
class GpuMatvecSession {
public:
    GpuMatvecSession(
        const ABIndex<StorageType>& ab_index,
        const std::vector<DeterminantT<StorageType>>& dets,
        const std::vector<std::vector<double>>& h1,
        const std::vector<double>& eri,
        int n_orb,
        int verbose = 0);

    ~GpuMatvecSession();

    GpuMatvecSession(const GpuMatvecSession&) = delete;
    GpuMatvecSession& operator=(const GpuMatvecSession&) = delete;
    GpuMatvecSession(GpuMatvecSession&&) = delete;
    GpuMatvecSession& operator=(GpuMatvecSession&&) = delete;

    /// Compute sigma = H * v. Both host buffers, length N.
    /// Expects sigma already zero-filled by the caller (the Davidson loop
    /// does `std::fill(sigma, sigma+N, 0.0)` before each matvec).
    /// GPU kernel_ch1 zeros d_sigma at the start internally, so host zeroing
    /// is redundant for this path but kept to match the Davidson contract.
    void apply(const double* v, double* sigma, std::size_t N);

    /// Number of determinants this session is built for.
    std::size_t n_dets() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace fe
}  // namespace trimci_core

#endif  // TRIMCI_HAS_GPU
