#ifdef TRIMCI_HAS_GPU
/**
 * GPU matvec session for Phase 1/2 Davidson.
 *
 * Bridges fast_expansion's ABIndex / DeterminantT representation to the
 * Phase 0 GpuContext / GpuDet / AB-factored CSR representation, then
 * launches kernel_ch1 + kernel_ch2 + kernel_ch3 per Davidson iteration.
 *
 * Only Davidson matvec goes to GPU. PT2, dressed CI, pool-build screening,
 * orbital optimization all stay on CPU.
 */

#include "detspace_matvec_gpu.hpp"

#include <array>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <type_traits>

#include <cuda_runtime.h>

#include "../gpu/gpu_context.cuh"
#include "../gpu/gpu_matvec.cuh"
#include "../gpu/gpu_common.cuh"
#include "../gpu/gpu_types.cuh"

namespace trimci_core {
namespace fe {

// ============================================================================
// Impl (pimpl) — owns the GpuContext and device v/sigma buffers
// ============================================================================
template<typename StorageType>
struct GpuMatvecSession<StorageType>::Impl {
    trimci_gpu::GpuContext ctx;
    double* d_v     = nullptr;
    double* d_sigma = nullptr;
    int n_dets_ = 0;
    int n_orb_  = 0;
    int verbose_ = 0;

    Impl(int n_orb, int n_words, int verbose)
        : ctx(n_orb, n_words), n_orb_(n_orb), verbose_(verbose) {}

    ~Impl() {
        if (d_v)     cudaFree(d_v);
        if (d_sigma) cudaFree(d_sigma);
    }
};

// ============================================================================
// n_words helper — GpuDet uses 2 uint64 lanes always; StorageType is
// uint64_t (1 lane) or std::array<uint64_t,2> (2 lanes).
// ============================================================================
template<typename StorageType>
static inline int storage_n_words() {
    if constexpr (std::is_same_v<StorageType, uint64_t>) {
        return 1;
    } else {
        return 2;
    }
}

// ============================================================================
// Constructor: upload integrals, convert dets, build AB groups, alloc buffers
// ============================================================================
template<typename StorageType>
GpuMatvecSession<StorageType>::GpuMatvecSession(
    const ABIndex<StorageType>& /*ab_index*/,
    const std::vector<DeterminantT<StorageType>>& dets,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    int n_orb,
    int verbose
) {
    const int n_words = storage_n_words<StorageType>();
    impl_ = std::make_unique<Impl>(n_orb, n_words, verbose);
    auto& ctx = impl_->ctx;

    // 0. Defensive size check: GPU path silently would produce wrong energies
    //    if eri is sized wrong (e.g. from a symmetry-reduced buffer). CPU path
    //    would trip a different bound-check later; GPU would just read
    //    garbage. Catch it at the boundary.
    const std::size_t expected_eri = (std::size_t)n_orb * n_orb * n_orb * n_orb;
    if (eri.size() != expected_eri) {
        throw std::runtime_error(
            "GpuMatvecSession: eri.size()=" + std::to_string(eri.size()) +
            " != n_orb^4=" + std::to_string(expected_eri) +
            " (expected flat [n_orb,n_orb,n_orb,n_orb] row-major)");
    }

    // 1. Flatten h1 (ExpansionConfig passes as vector<vector<double>>) and
    //    upload integrals. eri is already a flat n_orb^4 vector.
    const std::size_t h1_size = (std::size_t)n_orb * n_orb;
    std::vector<double> h1_flat(h1_size);
    for (int i = 0; i < n_orb; ++i)
        for (int j = 0; j < n_orb; ++j)
            h1_flat[(std::size_t)i * n_orb + j] = h1[i][j];
    ctx.upload_integrals(h1_flat.data(), h1_flat.size(), eri.data(), eri.size());

    // 2. Convert DeterminantT<StorageType> → GpuDet (fixed 2-lane alpha/beta).
    //    For uint64_t (1-word) storage, the high lane is zero.
    std::vector<trimci_gpu::GpuDet> h_dets(dets.size());
    for (std::size_t i = 0; i < dets.size(); ++i) {
        std::memset(&h_dets[i], 0, sizeof(trimci_gpu::GpuDet));
        if constexpr (std::is_same_v<StorageType, uint64_t>) {
            h_dets[i].alpha[0] = dets[i].alpha;
            h_dets[i].beta[0]  = dets[i].beta;
        } else {
            for (int w = 0; w < n_words; ++w) {
                h_dets[i].alpha[w] = dets[i].alpha[w];
                h_dets[i].beta[w]  = dets[i].beta[w];
            }
        }
    }

    // 3. Build AB groups (alpha CSR, beta CSR, adjacency) directly from the
    //    host det array. The matvec path only reads from the CSR structures
    //    (kernel_ch1/ch2/ch3 input), NOT from ctx.d_dets_, so skip the
    //    separate upload_dets call that would duplicate data on GPU.
    //    (compute_diagonal uses d_dets_, but that's only called in Phase 0
    //    pool-build path, not in Davidson matvec.)
    const int n_dets = (int)h_dets.size();
    ctx.build_ab_groups(h_dets.data(), n_dets);

    // 4. Allocate v/sigma buffers (persist across Davidson iters).
    GPU_CHECK(cudaMalloc(&impl_->d_v,     (std::size_t)n_dets * sizeof(double)));
    GPU_CHECK(cudaMalloc(&impl_->d_sigma, (std::size_t)n_dets * sizeof(double)));
    impl_->n_dets_ = n_dets;

    if (verbose >= 1) {
        std::cout << "[GpuMatvecSession] uploaded " << n_dets
                  << " dets, n_orb=" << n_orb
                  << ", n_words=" << n_words
                  << std::endl;
    }
}

// ============================================================================
template<typename StorageType>
GpuMatvecSession<StorageType>::~GpuMatvecSession() = default;

// ============================================================================
template<typename StorageType>
void GpuMatvecSession<StorageType>::apply(const double* v, double* sigma, std::size_t N) {
    auto& impl = *impl_;
    if ((int)N != impl.n_dets_) {
        throw std::runtime_error(
            "GpuMatvecSession::apply: N mismatch (expected " +
            std::to_string(impl.n_dets_) + ", got " + std::to_string(N) + ")");
    }
    GPU_CHECK(cudaMemcpy(impl.d_v, v, N * sizeof(double),
                         cudaMemcpyHostToDevice));
    // gpu_matvec internally zeros d_sigma then accumulates ch1+ch2+ch3.
    trimci_gpu::gpu_matvec(impl.ctx, impl.d_v, impl.d_sigma, (int)N);
    GPU_CHECK(cudaMemcpy(sigma, impl.d_sigma, N * sizeof(double),
                         cudaMemcpyDeviceToHost));
}

// ============================================================================
template<typename StorageType>
std::size_t GpuMatvecSession<StorageType>::n_dets() const {
    return impl_ ? (std::size_t)impl_->n_dets_ : 0;
}

// ============================================================================
// Explicit instantiation: 64-bit and 128-bit determinants.
// ============================================================================
template class GpuMatvecSession<uint64_t>;
template class GpuMatvecSession<std::array<uint64_t, 2>>;

}  // namespace fe
}  // namespace trimci_core

#endif  // TRIMCI_HAS_GPU
