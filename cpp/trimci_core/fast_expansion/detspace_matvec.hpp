#pragma once
/**
 * Fast Expansion: Matrix-free H*v in determinant space.
 *
 * Computes sigma = H * v without storing H, using ABIndex for connection
 * enumeration and FCIDUMP integrals for H_ij evaluation.
 *
 * Performance features:
 *   - Alpha-group traversal for ERI cache locality
 *   - Spin-aware H_ij dispatch (skips XOR+popcount classification)
 *   - Thread-local buffers + reduce (avoids atomic contention)
 *   - Upper-triangle symmetry (halves computation)
 */

#include <vector>
#include "ab_index.hpp"
#include "fe_types.hpp"
#include "determinant.hpp"

namespace trimci_core {
namespace fe {

template<typename StorageType>
struct MatvecContext {
    const std::vector<DeterminantT<StorageType>>& dets;
    const ABIndex<StorageType>& ab_index;
    const std::vector<std::vector<double>>& h1;
    const std::vector<double>& eri;
    int n_orb;
    const std::vector<double>& diag;  // precomputed H_ii
};

/// Matrix-free matvec: sigma = H * v.
template<typename StorageType>
void matvec(const MatvecContext<StorageType>& ctx,
            const double* v,
            double* sigma,
            size_t N);

/// Compute H_ij from individual alpha/beta strings (avoids dets[] random access).
template<typename StorageType>
double compute_H_ij_strings(
    const StorageType& alpha_i, const StorageType& beta_i,
    const StorageType& alpha_j, const StorageType& beta_j,
    ExcType exc_type,
    const std::vector<std::vector<double>>& h1,
    const double* eri_data,
    int n_orb);

/// Compute H_ij given known excitation type (skip XOR classification).
template<typename StorageType>
double compute_H_ij_by_type(
    const DeterminantT<StorageType>& di,
    const DeterminantT<StorageType>& dj,
    ExcType exc_type,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    int n_orb);

/// Batch-compute diagonal elements H_ii.
template<typename StorageType>
std::vector<double> compute_diagonals(
    const std::vector<DeterminantT<StorageType>>& dets,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    int n_orb);

// ============================================================================
// Connection cache: precomputed (j, H_ij) CSR for cached matvec
// ============================================================================
template<typename StorageType>
struct ConnectionCache {
    std::vector<size_t>   ptr;   // [N+1] CSR row pointers
    std::vector<uint32_t> col;   // [nnz]  connected det index j (upper triangle: j > i)
    std::vector<double>   val;   // [nnz]  H_ij value
    size_t n_dets = 0;

    size_t memory_bytes() const {
        return ptr.capacity() * sizeof(size_t)
             + col.capacity() * sizeof(uint32_t)
             + val.capacity() * sizeof(double);
    }
    void clear() { ptr.clear(); col.clear(); val.clear(); n_dets = 0; }
};

/// Build connection cache: precompute all upper-triangle (j, H_ij) pairs.
template<typename StorageType>
ConnectionCache<StorageType> build_connection_cache(
    const ABIndex<StorageType>& ab_index,
    const std::vector<DeterminantT<StorageType>>& dets,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    int n_orb);

/// Cached matvec: sigma = H * v using precomputed connection cache.
template<typename StorageType>
void matvec_cached(const ConnectionCache<StorageType>& cache,
                   const std::vector<double>& diag,
                   const double* v,
                   double* sigma,
                   size_t N);

/// Dressed on-the-fly matvec: sigma = H * v with alpha-group Coulomb sharing.
/// Precomputes J_alpha[m][p] per alpha group to accelerate SAME_ALPHA_S connections.
/// Best performance with alpha-sorted dets (contiguous alpha groups).
template<typename StorageType>
void matvec_dressed(const MatvecContext<StorageType>& ctx,
                    const double* v,
                    double* sigma,
                    size_t N);

/// Full-matrix dressed matvec: row-only accumulation, no per-thread sigma buffers.
/// Trades 2× H_ij computation for O(N) memory instead of O(N×T).
/// Critical for large N where per-thread buffers exceed physical memory.
template<typename StorageType>
void matvec_dressed_full(const MatvecContext<StorageType>& ctx,
                         const double* v,
                         double* sigma,
                         size_t N);

/// Row-sliced matvec for distributed computation.
/// Computes sigma[0..chunk_size-1] = H[row_start:row_end, :] @ v[0..N-1].
/// Each row is independent → no communication between workers.
/// Output array sigma must have size (row_end - row_start).
template<typename StorageType>
void matvec_row_sliced(const MatvecContext<StorageType>& ctx,
                       const double* v,          // full v[0..N-1]
                       double* sigma,             // partial sigma[0..chunk-1]
                       size_t N,
                       size_t row_start,
                       size_t row_end);

/// Block matvec: compute sigma_b = H * v_b for b = 0..n_vecs-1 in one pass.
/// Shares ABIndex traversal and H_ij computation across all vectors.
/// Reduces effective matvec cost by factor ~n_vecs when H_ij dominates.
template<typename StorageType>
void matvec_dressed_full_block(const MatvecContext<StorageType>& ctx,
                               const double* const* v_batch,
                               double** sigma_batch,
                               size_t N,
                               int n_vecs);

/// Dual-order matvec: splits sigma computation into alpha pass + beta pass.
///
/// Alpha pass: Channel 1 (same-alpha) + Channel 2 (mixed doubles).
///   v[j] access benefits from Morton ordering (O(sqrt(N)) locality).
///
/// Beta pass: Channel 3 (same-beta).
///   Gathers v into beta-sorted order (v_beta), then scans within beta groups.
///   v_beta[pos] is strictly sequential within each group → perfect cache behavior.
///
/// Memory: 2 × N × 8 bytes (v_beta + sigma_beta), much less than O(N×T).
template<typename StorageType>
void matvec_dual_order(const MatvecContext<StorageType>& ctx,
                       const double* v,
                       double* sigma,
                       size_t N);

/// Precompute J_alpha[m*n_orb+p] = Σ_{n∈occ_α} eri[m,p,n,n] for a given alpha string.
/// Result array must be pre-allocated to n_orb*n_orb doubles (row-major).
/// Constant for all dets sharing the same alpha string; fits in L1 cache (~10 KB).
template<typename StorageType>
void precompute_J_alpha(
    const StorageType& alpha,
    const double* eri_data,
    int n_orb,
    double* J_alpha);

/// Compute H_ij for SAME_ALPHA_S (beta single excitation) using precomputed J_alpha table.
/// Replaces the O(n_occ_alpha) alpha Coulomb loop with a single J_alpha[m*n_orb+p] lookup.
/// Only call when exc_type == SAME_ALPHA_S.
template<typename StorageType>
double compute_H_ij_same_alpha_s_dressed(
    const StorageType& bi, const StorageType& bj,
    const std::vector<std::vector<double>>& h1,
    const double* eri_data,
    const double* J_alpha,
    int n_orb);

}  // namespace fe
}  // namespace trimci_core
