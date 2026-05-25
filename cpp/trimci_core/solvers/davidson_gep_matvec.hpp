#pragma once
/**
 * @file davidson_gep_matvec.hpp
 * @brief Matrix-free Davidson solver for Generalized Eigenvalue Problem.
 *
 * Solves Hc = E Sc for the lowest eigenvalue using only matvec operations.
 * No full matrix storage required — O(M * max_subspace) memory instead of O(M²).
 *
 * For M=100K:
 *   - Dense matrix: 100K² × 8 bytes = 80 GB (impossible)
 *   - This solver: 100K × 40 × 8 bytes = 32 MB (trivial)
 *
 * Each matvec is O(M² × cost_per_element), parallelized with OpenMP.
 */

#include <vector>
#include <cmath>
#include <functional>
#include <algorithm>
#include <stdexcept>
#include <cstdio>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace trimci_core {

struct DavidsonGEPMVResult {
    double eigenvalue;
    std::vector<double> eigenvector;
    bool converged;
    int iterations;
    double residual_norm;
};

/**
 * @brief Matrix-free Davidson for Hc = ESc.
 *
 * @param matvec_H  Function: sigma = H * v (writes into sigma, length n)
 * @param matvec_S  Function: sigma = S * v
 * @param diag_H    Diagonal elements H[i][i], length n
 * @param diag_S    Diagonal elements S[i][i], length n (usually all 1.0 for PCS)
 * @param n         Problem dimension
 * @param max_iter  Maximum iterations
 * @param tol       Convergence tolerance on energy change
 * @param max_sub   Maximum subspace size before restart
 * @param verbose   0=silent, 1=summary, 2=per-iteration
 * @param v_init    Optional initial vector (length n). If empty, use unit vector at best diagonal.
 */
DavidsonGEPMVResult davidson_gep_matvec(
    std::function<void(const double*, double*, int)> matvec_H,
    std::function<void(const double*, double*, int)> matvec_S,
    const double* diag_H,
    const double* diag_S,
    int n,
    int max_iter = 200,
    double tol = 1e-6,
    int max_sub = 40,
    int verbose = 0,
    const double* v_init = nullptr)
{
    DavidsonGEPMVResult result;
    result.converged = false;
    result.iterations = 0;

    // Preconditioner: H_ii / S_ii
    std::vector<double> precond(n);
    int best_idx = 0;
    double best_val = 1e30;
    for (int i = 0; i < n; ++i) {
        double si = (std::abs(diag_S[i]) > 1e-14) ? diag_S[i] : 1.0;
        precond[i] = diag_H[i] / si;
        if (precond[i] < best_val) {
            best_val = precond[i];
            best_idx = i;
        }
    }

    // Subspace storage: V[k] = column vectors (S-orthonormal)
    // Store as flat array: V[col * n + row]
    int max_cols = max_sub + 2;
    std::vector<double> V(static_cast<size_t>(n) * max_cols, 0.0);
    std::vector<double> HV(static_cast<size_t>(n) * max_cols, 0.0);
    std::vector<double> SV(static_cast<size_t>(n) * max_cols, 0.0);

    // Initial guess: warm-start from v_init if provided, else unit vector at best diagonal
    if (v_init != nullptr) {
        // Check v_init is nonzero (otherwise fall back)
        double v_init_norm = 0.0;
        for (int i = 0; i < n; ++i) v_init_norm += v_init[i] * v_init[i];
        if (v_init_norm > 1e-20) {
            std::copy(v_init, v_init + n, V.begin());
        } else {
            V[best_idx] = 1.0;
        }
    } else {
        V[best_idx] = 1.0;  // V[:, 0]
    }

    // S-normalize
    std::vector<double> Sv_tmp(n);
    matvec_S(&V[0], Sv_tmp.data(), n);
    double snorm = 0.0;
    for (int i = 0; i < n; ++i) snorm += V[i] * Sv_tmp[i];
    snorm = std::sqrt(std::abs(snorm));
    if (snorm > 1e-14) {
        for (int i = 0; i < n; ++i) V[i] /= snorm;
    }

    // Precompute HV[:, 0] and SV[:, 0]
    matvec_H(&V[0], &HV[0], n);
    matvec_S(&V[0], &SV[0], n);

    // Small projected matrices (subspace)
    std::vector<double> H_sub(max_cols * max_cols, 0.0);
    std::vector<double> S_sub(max_cols * max_cols, 0.0);

    H_sub[0] = 0.0;
    S_sub[0] = 0.0;
    for (int i = 0; i < n; ++i) {
        H_sub[0] += V[i] * HV[i];
        S_sub[0] += V[i] * SV[i];
    }

    double E_old = H_sub[0] / S_sub[0];
    int k = 1;  // current subspace size

    // Work arrays
    std::vector<double> Hx(n), Sx(n), r(n), t(n);

    for (int iter = 0; iter < max_iter; ++iter) {
        result.iterations = iter + 1;

        // Solve small GEP in subspace
        // For small k, use dense solve
        // H_sub alpha = E S_sub alpha

        // Simple approach: canonical orthogonalization of S_sub
        // For k < 60, this is cheap
        double E_ritz = 0.0;
        std::vector<double> alpha(k, 0.0);

        if (k == 1) {
            E_ritz = H_sub[0] / S_sub[0];
            alpha[0] = 1.0;
        } else {
            // Use Eigen for small GEP (subspace is tiny)
            // Inline simple implementation to avoid Eigen dependency in this header
            // Cholesky-like: find lowest eigenvalue of S^{-1/2} H S^{-1/2}

            // Build dense subspace matrices
            std::vector<double> Hs(k * k), Ss(k * k);
            for (int i = 0; i < k; ++i) {
                for (int j = 0; j < k; ++j) {
                    Hs[i * k + j] = H_sub[i * max_cols + j];
                    Ss[i * k + j] = S_sub[i * max_cols + j];
                }
            }

            // Eigendecompose S_sub (Jacobi, fine for k < 60)
            // Direct solve with a simple implementation

            // Direct approach: iterate to find lowest eigenpair of (S^-1 H)
            // Start with current alpha, do a few inverse iterations
            // This is good enough for the subspace problem

            // Simpler: construct H_eff = S^{-1} H using LU
            // For k < 60, direct Gaussian elimination is fine

            // Even simpler for our case: use the canonical orthogonalization
            // approach already implemented in solve_generalized_eigenproblem

            // Since k is small (<60), just do it:
            // 1. Eigendecompose S_sub
            // 2. Canonical basis X = U D^{-1/2}
            // 3. Diag X^T H X

            // Jacobi eigendecomposition of S_sub
            std::vector<double> S_eig(k), S_vec(k * k, 0.0);
            std::vector<double> S_work(k * k);
            std::copy(Ss.begin(), Ss.end(), S_work.begin());

            // Initialize eigenvectors to identity
            for (int i = 0; i < k; ++i) S_vec[i * k + i] = 1.0;

            // Jacobi rotations
            for (int sweep = 0; sweep < 100; ++sweep) {
                double off = 0.0;
                for (int i = 0; i < k; ++i)
                    for (int j = i+1; j < k; ++j)
                        off += S_work[i*k+j] * S_work[i*k+j];
                if (off < 1e-24) break;

                for (int p = 0; p < k; ++p) {
                    for (int q = p+1; q < k; ++q) {
                        double app = S_work[p*k+p], aqq = S_work[q*k+q], apq = S_work[p*k+q];
                        if (std::abs(apq) < 1e-15) continue;
                        double tau = (aqq - app) / (2.0 * apq);
                        double t_val = (tau >= 0) ? 1.0/(tau + std::sqrt(1+tau*tau))
                                                  : -1.0/(-tau + std::sqrt(1+tau*tau));
                        double c = 1.0 / std::sqrt(1 + t_val*t_val);
                        double s = t_val * c;
                        // Update S_work
                        for (int i = 0; i < k; ++i) {
                            double sip = S_work[i*k+p], siq = S_work[i*k+q];
                            S_work[i*k+p] = c*sip - s*siq;
                            S_work[i*k+q] = s*sip + c*siq;
                        }
                        for (int j = 0; j < k; ++j) {
                            double spj = S_work[p*k+j], sqj = S_work[q*k+j];
                            S_work[p*k+j] = c*spj - s*sqj;
                            S_work[q*k+j] = s*spj + c*sqj;
                        }
                        // Update eigenvectors
                        for (int i = 0; i < k; ++i) {
                            double vip = S_vec[i*k+p], viq = S_vec[i*k+q];
                            S_vec[i*k+p] = c*vip - s*viq;
                            S_vec[i*k+q] = s*vip + c*viq;
                        }
                    }
                }
            }
            for (int i = 0; i < k; ++i) S_eig[i] = S_work[i*k+i];

            // Canonical orthogonalization: X = U * D^{-1/2} for valid modes
            double s_thresh = 1e-10;
            int n_valid = 0;
            for (int i = 0; i < k; ++i)
                if (S_eig[i] > s_thresh) n_valid++;

            if (n_valid == 0) { n_valid = 1; S_eig[0] = 1.0; }

            std::vector<double> X(k * n_valid);
            int col = 0;
            for (int i = 0; i < k; ++i) {
                if (S_eig[i] > s_thresh) {
                    double inv_sqrt = 1.0 / std::sqrt(S_eig[i]);
                    for (int j = 0; j < k; ++j)
                        X[j * n_valid + col] = S_vec[j * k + i] * inv_sqrt;
                    col++;
                }
            }

            // H_orth = X^T H X
            std::vector<double> H_orth(n_valid * n_valid, 0.0);
            for (int i = 0; i < n_valid; ++i) {
                for (int j = 0; j < n_valid; ++j) {
                    double sum = 0.0;
                    for (int a = 0; a < k; ++a)
                        for (int b = 0; b < k; ++b)
                            sum += X[a*n_valid+i] * Hs[a*k+b] * X[b*n_valid+j];
                    H_orth[i*n_valid+j] = sum;
                }
            }

            // Diag H_orth (small, use Jacobi again)
            std::vector<double> H_eig(n_valid), H_vec(n_valid * n_valid, 0.0);
            for (int i = 0; i < n_valid; ++i) H_vec[i*n_valid+i] = 1.0;
            std::vector<double> H_work(H_orth);

            for (int sweep = 0; sweep < 100; ++sweep) {
                double off = 0.0;
                for (int i = 0; i < n_valid; ++i)
                    for (int j = i+1; j < n_valid; ++j)
                        off += H_work[i*n_valid+j] * H_work[i*n_valid+j];
                if (off < 1e-24) break;
                for (int p = 0; p < n_valid; ++p) {
                    for (int q = p+1; q < n_valid; ++q) {
                        double hpp = H_work[p*n_valid+p], hqq = H_work[q*n_valid+q];
                        double hpq = H_work[p*n_valid+q];
                        if (std::abs(hpq) < 1e-15) continue;
                        double tau = (hqq - hpp) / (2.0 * hpq);
                        double t_val = (tau >= 0) ? 1.0/(tau + std::sqrt(1+tau*tau))
                                                  : -1.0/(-tau + std::sqrt(1+tau*tau));
                        double c = 1.0 / std::sqrt(1 + t_val*t_val), s = t_val * c;
                        for (int i = 0; i < n_valid; ++i) {
                            double hip = H_work[i*n_valid+p], hiq = H_work[i*n_valid+q];
                            H_work[i*n_valid+p] = c*hip - s*hiq;
                            H_work[i*n_valid+q] = s*hip + c*hiq;
                        }
                        for (int j = 0; j < n_valid; ++j) {
                            double hpj = H_work[p*n_valid+j], hqj = H_work[q*n_valid+j];
                            H_work[p*n_valid+j] = c*hpj - s*hqj;
                            H_work[q*n_valid+j] = s*hpj + c*hqj;
                        }
                        for (int i = 0; i < n_valid; ++i) {
                            double vip = H_vec[i*n_valid+p], viq = H_vec[i*n_valid+q];
                            H_vec[i*n_valid+p] = c*vip - s*viq;
                            H_vec[i*n_valid+q] = s*vip + c*viq;
                        }
                    }
                }
            }
            for (int i = 0; i < n_valid; ++i) H_eig[i] = H_work[i*n_valid+i];

            // Find lowest eigenvalue
            int min_idx = 0;
            for (int i = 1; i < n_valid; ++i)
                if (H_eig[i] < H_eig[min_idx]) min_idx = i;

            E_ritz = H_eig[min_idx];

            // alpha in original subspace: alpha = X * H_vec[:, min_idx]
            std::fill(alpha.begin(), alpha.end(), 0.0);
            for (int a = 0; a < k; ++a) {
                for (int i = 0; i < n_valid; ++i) {
                    alpha[a] += X[a*n_valid+i] * H_vec[i*n_valid+min_idx];
                }
            }
        }

        // Ritz vector: x = V * alpha
        std::vector<double> x(n, 0.0);
        for (int j = 0; j < k; ++j) {
            double aj = alpha[j];
            const double* Vj = &V[static_cast<size_t>(j) * n];
            for (int i = 0; i < n; ++i) x[i] += aj * Vj[i];
        }

        // Hx = HV * alpha, Sx = SV * alpha
        std::fill(Hx.begin(), Hx.end(), 0.0);
        std::fill(Sx.begin(), Sx.end(), 0.0);
        for (int j = 0; j < k; ++j) {
            double aj = alpha[j];
            const double* HVj = &HV[static_cast<size_t>(j) * n];
            const double* SVj = &SV[static_cast<size_t>(j) * n];
            for (int i = 0; i < n; ++i) {
                Hx[i] += aj * HVj[i];
                Sx[i] += aj * SVj[i];
            }
        }

        // Residual: r = Hx - E*Sx
        double r_norm = 0.0;
        for (int i = 0; i < n; ++i) {
            r[i] = Hx[i] - E_ritz * Sx[i];
            r_norm += r[i] * r[i];
        }
        r_norm = std::sqrt(r_norm);

        if (verbose >= 2) {
            fprintf(stderr, "  DavidsonMV iter %3d: E = %.10f, |r| = %.2e, k = %d\n",
                    iter + 1, E_ritz, r_norm, k);
        }

        // Convergence
        if (std::abs(E_ritz - E_old) < tol && r_norm < tol * 100) {
            result.converged = true;
            result.eigenvalue = E_ritz;
            result.eigenvector = x;
            result.residual_norm = r_norm;
            if (verbose >= 1) {
                fprintf(stderr, "  DavidsonMV converged in %d iters, E = %.10f\n",
                        iter + 1, E_ritz);
            }
            return result;
        }
        E_old = E_ritz;

        // Preconditioned correction
        for (int i = 0; i < n; ++i) {
            double denom = precond[i] - E_ritz;
            t[i] = (std::abs(denom) > 1e-12) ? -r[i] / denom : -r[i];
        }

        // S-orthogonalize t against V (double Gram-Schmidt)
        for (int pass = 0; pass < 2; ++pass) {
            // St = S * t
            std::vector<double> St(n);
            matvec_S(t.data(), St.data(), n);
            // coeffs = V^T * St
            for (int j = 0; j < k; ++j) {
                double dot = 0.0;
                const double* Vj = &V[static_cast<size_t>(j) * n];
                for (int i = 0; i < n; ++i) dot += Vj[i] * St[i];
                for (int i = 0; i < n; ++i) t[i] -= dot * Vj[i];
            }
        }

        // S-normalize t
        {
            std::vector<double> St(n);
            matvec_S(t.data(), St.data(), n);
            double t_snorm = 0.0;
            for (int i = 0; i < n; ++i) t_snorm += t[i] * St[i];
            t_snorm = std::sqrt(std::abs(t_snorm));
            if (t_snorm < 1e-14) continue;  // skip
            for (int i = 0; i < n; ++i) t[i] /= t_snorm;
        }

        // Restart if subspace too large
        if (k >= max_sub) {
            // Keep only the Ritz vector
            std::copy(x.begin(), x.end(), &V[0]);
            matvec_H(&V[0], &HV[0], n);
            matvec_S(&V[0], &SV[0], n);
            H_sub[0] = 0.0; S_sub[0] = 0.0;
            for (int i = 0; i < n; ++i) {
                H_sub[0] += V[i] * HV[i];
                S_sub[0] += V[i] * SV[i];
            }
            k = 1;
            if (verbose >= 2) {
                fprintf(stderr, "  DavidsonMV: restart, k -> 1\n");
            }
        }

        // Append t as new column
        double* Vk = &V[static_cast<size_t>(k) * n];
        std::copy(t.begin(), t.end(), Vk);

        // Compute HV[:, k] and SV[:, k]
        matvec_H(Vk, &HV[static_cast<size_t>(k) * n], n);
        matvec_S(Vk, &SV[static_cast<size_t>(k) * n], n);

        // Update projected matrices
        for (int j = 0; j <= k; ++j) {
            double h_jk = 0.0, s_jk = 0.0;
            const double* Vj = &V[static_cast<size_t>(j) * n];
            const double* HVk = &HV[static_cast<size_t>(k) * n];
            const double* SVk = &SV[static_cast<size_t>(k) * n];
            for (int i = 0; i < n; ++i) {
                h_jk += Vj[i] * HVk[i];
                s_jk += Vj[i] * SVk[i];
            }
            H_sub[j * max_cols + k] = h_jk;
            H_sub[k * max_cols + j] = h_jk;
            S_sub[j * max_cols + k] = s_jk;
            S_sub[k * max_cols + j] = s_jk;
        }

        k++;
    }

    // Not converged
    result.eigenvalue = E_old;
    result.eigenvector.assign(n, 0.0);
    result.eigenvector[best_idx] = 1.0;
    result.residual_norm = 1.0;

    if (verbose >= 1) {
        fprintf(stderr, "  DavidsonMV: NOT converged after %d iters\n", max_iter);
    }
    return result;
}

} // namespace trimci_core
