#include "detspace_davidson.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <iostream>
#include <iomanip>

#include "omp_compat.hpp"
namespace trimci_core {
namespace fe {

// ============================================================================
// Helper: dot product
// ============================================================================
static double dot(const double* a, const double* b, size_t N) {
    double s = 0.0;
    #pragma omp parallel for reduction(+:s) schedule(static)
    for (int i = 0; i < static_cast<int>(N); ++i) s += a[i] * b[i];
    return s;
}

// ============================================================================
// Helper: axpy  y += alpha * x
// ============================================================================
static void axpy(double alpha, const double* x, double* y, size_t N) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(N); ++i) y[i] += alpha * x[i];
}

// ============================================================================
// Helper: norm
// ============================================================================
static double norm(const double* x, size_t N) {
    return std::sqrt(dot(x, x, N));
}

// ============================================================================
// Helper: normalize in-place, return original norm
// ============================================================================
static double normalize(double* x, size_t N) {
    double n = norm(x, N);
    if (n > 1e-15) {
        double inv = 1.0 / n;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < static_cast<int>(N); ++i) x[i] *= inv;
    }
    return n;
}

// ============================================================================
// Two-pass Modified Gram-Schmidt: orthogonalize t against V columns
// ============================================================================
static void orthogonalize_2pass(double* t, const std::vector<std::vector<double>>& V,
                                 int n_vecs, size_t N) {
    for (int pass = 0; pass < 2; ++pass) {
        for (int k = 0; k < n_vecs; ++k) {
            double proj = dot(t, V[k].data(), N);
            axpy(-proj, V[k].data(), t, N);
        }
    }
}

// ============================================================================
// Davidson solver
// ============================================================================
DavidsonResult davidson_solve(
    std::function<void(const double* v, double* sigma, size_t N)> matvec_func,
    const std::vector<double>& diag,
    size_t N,
    const DavidsonParams& params,
    const std::vector<std::vector<double>>& initial_guess,
    BlockMatvecFunc block_matvec)
{
    DavidsonResult result;
    result.eigenvalues.resize(params.n_states, 0.0);
    result.eigenvectors.resize(params.n_states, std::vector<double>(N, 0.0));

    // Subspace vectors V and H*V
    std::vector<std::vector<double>> V;
    std::vector<std::vector<double>> HV;

    // Initialize with guess or diagonal-based unit vectors
    if (!initial_guess.empty() && (int)initial_guess.size() >= params.n_states) {
        for (int s = 0; s < params.n_states; ++s) {
            V.push_back(initial_guess[s]);
            normalize(V.back().data(), N);
        }
    } else {
        // Find indices of smallest diagonal elements
        std::vector<size_t> idx(N);
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + params.n_states, idx.end(),
            [&diag](size_t a, size_t b) { return diag[a] < diag[b]; });
        for (int s = 0; s < params.n_states; ++s) {
            V.emplace_back(N, 0.0);
            V.back()[idx[s]] = 1.0;
        }
    }

    // Orthogonalize initial vectors
    for (int s = 0; s < params.n_states; ++s) {
        orthogonalize_2pass(V[s].data(), V, s, N);
        normalize(V[s].data(), N);
    }

    // Compute initial H*V
    for (int s = 0; s < params.n_states; ++s) {
        HV.emplace_back(N);
        matvec_func(V[s].data(), HV.back().data(), N);
    }

    // Small projected matrices
    int k = params.n_states;  // current subspace size
    double prev_E_dav = 0.0;  // track energy for energy_tol convergence

    // Block expansion: precompute mean diagonal for multi-shift Jacobi
    const int bs = (block_matvec && params.block_size > 1) ? params.block_size : 1;
    double diag_mean = 0.0;
    if (bs > 1) {
        for (size_t i = 0; i < N; ++i) diag_mean += diag[i];
        diag_mean /= N;
    }

    // Persist across iterations so non-convergence path can use the last Ritz vectors
    std::vector<std::vector<double>> ritz_vecs(params.n_states, std::vector<double>(N, 0.0));
    std::vector<std::vector<double>> ritz_Hvecs(params.n_states, std::vector<double>(N, 0.0));

    for (int iter = 0; iter < params.max_iter; ++iter) {
        // Build projected H_small = V^T * HV  (k × k)
        std::vector<double> H_small(k * k, 0.0);
        for (int i = 0; i < k; ++i) {
            for (int j = i; j < k; ++j) {
                double val = dot(V[i].data(), HV[j].data(), N);
                H_small[i * k + j] = val;
                H_small[j * k + i] = val;
            }
        }

        // Solve small eigenvalue problem (Jacobi or direct for small k)
        // Using simple Jacobi rotation method for k <= max_subspace
        std::vector<double> evals(k);
        std::vector<double> evecs(k * k);
        {
            // Initialize evecs to identity
            std::fill(evecs.begin(), evecs.end(), 0.0);
            for (int i = 0; i < k; ++i) evecs[i * k + i] = 1.0;

            // Copy H_small for in-place diagonalization
            std::vector<double> A = H_small;

            // Jacobi eigenvalue algorithm
            const int max_sweeps = 100;
            for (int sweep = 0; sweep < max_sweeps; ++sweep) {
                double off_diag = 0.0;
                for (int i = 0; i < k; ++i)
                    for (int j = i + 1; j < k; ++j)
                        off_diag += A[i * k + j] * A[i * k + j];
                if (off_diag < 1e-30) break;

                for (int p = 0; p < k; ++p) {
                    for (int q = p + 1; q < k; ++q) {
                        double apq = A[p * k + q];
                        if (std::abs(apq) < 1e-15) continue;

                        double d = A[q * k + q] - A[p * k + p];
                        double t;
                        if (std::abs(d) < 1e-15) {
                            t = 1.0;
                        } else {
                            double tau = d / (2.0 * apq);
                            t = 1.0 / (std::abs(tau) + std::sqrt(1.0 + tau * tau));
                            if (tau < 0) t = -t;
                        }
                        double c = 1.0 / std::sqrt(1.0 + t * t);
                        double s = t * c;

                        // Rotate A
                        double app = A[p * k + p], aqq = A[q * k + q];
                        A[p * k + p] = app - t * apq;
                        A[q * k + q] = aqq + t * apq;
                        A[p * k + q] = 0.0;
                        A[q * k + p] = 0.0;
                        for (int r = 0; r < k; ++r) {
                            if (r == p || r == q) continue;
                            double arp = A[r * k + p], arq = A[r * k + q];
                            A[r * k + p] = c * arp - s * arq;
                            A[p * k + r] = A[r * k + p];
                            A[r * k + q] = s * arp + c * arq;
                            A[q * k + r] = A[r * k + q];
                        }
                        // Rotate evecs
                        for (int r = 0; r < k; ++r) {
                            double erp = evecs[r * k + p], erq = evecs[r * k + q];
                            evecs[r * k + p] = c * erp - s * erq;
                            evecs[r * k + q] = s * erp + c * erq;
                        }
                    }
                }
            }
            for (int i = 0; i < k; ++i) evals[i] = A[i * k + i];
        }

        // Sort eigenvalues
        std::vector<int> order(k);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
            [&evals](int a, int b) { return evals[a] < evals[b]; });

        // Compute Ritz vectors and residuals (updates persistent ritz_vecs/ritz_Hvecs)
        double max_residual = 0.0;
        for (int s = 0; s < params.n_states; ++s) {
            int idx = order[s];
            result.eigenvalues[s] = evals[idx];

            // Ritz vector: x = V * y
            std::fill(ritz_vecs[s].begin(), ritz_vecs[s].end(), 0.0);
            std::fill(ritz_Hvecs[s].begin(), ritz_Hvecs[s].end(), 0.0);
            for (int j = 0; j < k; ++j) {
                double coeff = evecs[j * k + idx];
                axpy(coeff, V[j].data(), ritz_vecs[s].data(), N);
                axpy(coeff, HV[j].data(), ritz_Hvecs[s].data(), N);
            }

            // Residual: r = H*x - λ*x
            std::vector<double> residual(N);
            {
                const double eval_s = evals[idx];
                const double* rv = ritz_vecs[s].data();
                const double* rhv = ritz_Hvecs[s].data();
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < static_cast<int>(N); ++i) {
                    residual[i] = rhv[i] - eval_s * rv[i];
                }
            }
            double rnorm = norm(residual.data(), N);
            if (rnorm > max_residual) max_residual = rnorm;
        }

        result.n_iters = iter + 1;
        result.residual_norm = max_residual;

        if (params.verbose >= 2) {
            std::cout << "  Davidson iter " << std::setw(3) << iter + 1
                      << "  E = " << std::fixed << std::setprecision(10) << result.eigenvalues[0]
                      << "  |r| = " << std::scientific << std::setprecision(2) << max_residual
                      << "  k = " << k << std::endl;
        }

        // Check convergence: residual OR energy tolerance
        bool resid_conv = (max_residual < params.residual_tol);
        bool energy_conv = false;
        double dE = 0.0;
        if (params.energy_tol > 0.0 && iter > 0) {
            dE = std::abs(result.eigenvalues[0] - prev_E_dav);
            energy_conv = (dE < params.energy_tol);
        }
        prev_E_dav = result.eigenvalues[0];

        if (resid_conv || energy_conv) {
            result.converged = true;
            for (int s = 0; s < params.n_states; ++s) {
                result.eigenvectors[s] = ritz_vecs[s];
            }
            if (params.verbose >= 1) {
                std::cout << "  Davidson converged in " << iter + 1 << " iterations"
                          << "  E = " << std::fixed << std::setprecision(10) << result.eigenvalues[0]
                          << "  |r| = " << std::scientific << std::setprecision(2) << max_residual;
                if (energy_conv && !resid_conv) {
                    std::cout << "  (energy_tol: |dE|=" << dE << ")";
                }
                std::cout << std::endl;
            }
            return result;
        }

        // Restart if subspace is full
        if (k >= params.max_subspace) {
            V.resize(params.n_states);
            HV.resize(params.n_states);
            for (int s = 0; s < params.n_states; ++s) {
                V[s] = ritz_vecs[s];
                HV[s] = ritz_Hvecs[s];
            }
            k = params.n_states;

            if (params.verbose >= 2) {
                std::cout << "  Davidson restart at iter " << iter + 1 << std::endl;
            }
            continue;
        }

        // Expand subspace with preconditioned residuals
        int max_new = params.max_subspace - k;
        if (max_new <= 0) continue;

        // Generate trial vectors (block_size per state via multi-shift Jacobi)
        std::vector<std::vector<double>> new_vecs;
        for (int s = 0; s < params.n_states && (int)new_vecs.size() < max_new; ++s) {
            double lambda = result.eigenvalues[s];
            const double* rv = ritz_vecs[s].data();
            const double* rhv = ritz_Hvecs[s].data();

            // Multi-shift Jacobi: shift_step spreads preconditioner across spectrum
            double shift_step = 0.0;
            if (bs > 1) {
                double gap = diag_mean - lambda;
                if (gap < 0.1) gap = 0.1;
                shift_step = gap / (2.0 * bs);
            }

            for (int b = 0; b < bs && (int)new_vecs.size() < max_new; ++b) {
                std::vector<double> t(N);
                double shift = b * shift_step;

                #pragma omp parallel for schedule(static)
                for (int i = 0; i < static_cast<int>(N); ++i) {
                    double r_i = rhv[i] - lambda * rv[i];
                    double denom = diag[i] - lambda - shift;
                    if (std::abs(denom) < 1e-4) {
                        denom = (denom >= 0) ? 1e-4 : -1e-4;
                    }
                    t[i] = -r_i / denom;
                }

                // Orthogonalize against V and already-generated trial vectors
                for (int pass = 0; pass < 2; ++pass) {
                    for (int kk = 0; kk < k; ++kk) {
                        double proj = dot(t.data(), V[kk].data(), N);
                        axpy(-proj, V[kk].data(), t.data(), N);
                    }
                    for (size_t prev = 0; prev < new_vecs.size(); ++prev) {
                        double proj = dot(t.data(), new_vecs[prev].data(), N);
                        axpy(-proj, new_vecs[prev].data(), t.data(), N);
                    }
                }
                double tnorm = normalize(t.data(), N);
                if (tnorm < 1e-14) continue;  // linearly dependent, skip

                new_vecs.push_back(std::move(t));
            }
        }

        if (new_vecs.empty()) continue;

        // Compute H*t for all new vectors
        int n_new = (int)new_vecs.size();
        if (block_matvec && n_new > 1) {
            // Block matvec: share H_ij computation across vectors
            std::vector<const double*> v_ptrs(n_new);
            std::vector<std::vector<double>> Ht_vecs(n_new, std::vector<double>(N));
            std::vector<double*> sigma_ptrs(n_new);
            for (int b = 0; b < n_new; ++b) {
                v_ptrs[b] = new_vecs[b].data();
                sigma_ptrs[b] = Ht_vecs[b].data();
            }
            block_matvec(v_ptrs.data(), sigma_ptrs.data(), N, n_new);

            for (int b = 0; b < n_new; ++b) {
                V.push_back(std::move(new_vecs[b]));
                HV.push_back(std::move(Ht_vecs[b]));
                k++;
            }
        } else {
            // Sequential matvec (standard path or single vector)
            for (int b = 0; b < n_new; ++b) {
                std::vector<double> Ht(N);
                matvec_func(new_vecs[b].data(), Ht.data(), N);
                V.push_back(std::move(new_vecs[b]));
                HV.push_back(std::move(Ht));
                k++;
            }
        }
    }

    // Did not converge — return best Ritz vectors anyway
    result.converged = false;
    for (int s = 0; s < params.n_states; ++s) {
        result.eigenvectors[s] = ritz_vecs[s];
    }

    if (params.verbose >= 1) {
        std::cout << "  Davidson did NOT converge after " << params.max_iter << " iterations"
                  << "  |r| = " << std::scientific << std::setprecision(2) << result.residual_norm
                  << std::endl;
    }

    return result;
}

}  // namespace fe
}  // namespace trimci_core
