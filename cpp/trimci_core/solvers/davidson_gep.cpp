#include "davidson_gep.hpp"
#include <iostream>
#include <cstdio>

namespace trimci_core {

DavidsonGEPResult davidson_gep(
    const Eigen::MatrixXd& H,
    const Eigen::MatrixXd& S,
    int max_iter,
    double tol,
    int max_subspace,
    int verbose,
    const Eigen::VectorXd& v_init)
{
    const int n = H.rows();
    if (n != H.cols() || n != S.rows() || n != S.cols()) {
        throw std::invalid_argument("davidson_gep: H and S must be square and same size");
    }

    DavidsonGEPResult result;
    result.converged = false;
    result.iterations = 0;
    result.residual_norm = 0.0;

    // Diagonal preconditioner: diag(H) / diag(S)
    Eigen::VectorXd precond(n);
    for (int i = 0; i < n; ++i) {
        precond(i) = (std::abs(S(i, i)) > 1e-14) ? H(i, i) / S(i, i) : H(i, i);
    }

    // Subspace basis V (columns are S-orthonormal)
    Eigen::MatrixXd V(n, 1);
    V.col(0).setZero();

    // Warm-start from v_init if provided, else unit vector at best diagonal
    if (v_init.size() == n && v_init.norm() > 1e-14) {
        V.col(0) = v_init;
    } else {
        // Initial guess: unit vector at index of smallest precond element
        int idx = 0;
        precond.minCoeff(&idx);
        V(idx, 0) = 1.0;
    }

    // S-normalize
    double snorm = std::sqrt(std::abs(V.col(0).dot(S * V.col(0))));
    if (snorm > 1e-14) V.col(0) /= snorm;

    double E_old = 0.0;

    for (int iter = 0; iter < max_iter; ++iter) {
        int k = V.cols();

        // Projected matrices: H_sub = V^T H V, S_sub = V^T S V
        Eigen::MatrixXd HV = H * V;  // O(n^2 * k)
        Eigen::MatrixXd SV = S * V;  // O(n^2 * k)
        Eigen::MatrixXd H_sub = V.transpose() * HV;
        Eigen::MatrixXd S_sub = V.transpose() * SV;

        // Symmetrize (numerical stability)
        H_sub = 0.5 * (H_sub + H_sub.transpose());
        S_sub = 0.5 * (S_sub + S_sub.transpose());

        // Solve small GEP via canonical orthogonalization
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> S_solver(S_sub);
        Eigen::VectorXd s_eigs = S_solver.eigenvalues();
        Eigen::MatrixXd s_vecs = S_solver.eigenvectors();

        // Keep valid modes (positive eigenvalues)
        double s_thresh = 1e-10;
        int n_valid = 0;
        for (int i = 0; i < k; ++i) {
            if (s_eigs(i) > s_thresh) ++n_valid;
        }

        if (n_valid == 0) {
            if (verbose >= 1) {
                fprintf(stderr, "  Davidson: no valid subspace modes, aborting\n");
            }
            break;
        }

        // Build X for canonical orthogonalization of subspace
        Eigen::MatrixXd X(k, n_valid);
        int col = 0;
        for (int i = 0; i < k; ++i) {
            if (s_eigs(i) > s_thresh) {
                X.col(col) = s_vecs.col(i) / std::sqrt(s_eigs(i));
                ++col;
            }
        }

        Eigen::MatrixXd H_orth = X.transpose() * H_sub * X;
        H_orth = 0.5 * (H_orth + H_orth.transpose());

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> H_solver(H_orth);
        double E = H_solver.eigenvalues()(0);
        Eigen::VectorXd alpha_orth = H_solver.eigenvectors().col(0);
        Eigen::VectorXd alpha = X * alpha_orth;  // subspace coefficients

        // Ritz vector in full space
        Eigen::VectorXd x = V * alpha;

        // Residual: r = Hx - E*Sx
        Eigen::VectorXd Hx = HV * alpha;
        Eigen::VectorXd Sx = SV * alpha;
        Eigen::VectorXd r = Hx - E * Sx;
        double r_norm = r.norm();

        if (verbose >= 2) {
            fprintf(stderr, "  Davidson iter %3d: E = %.10f, |r| = %.2e, k = %d\n",
                    iter + 1, E, r_norm, k);
        }

        result.iterations = iter + 1;
        result.residual_norm = r_norm;

        // Convergence check
        if (std::abs(E - E_old) < tol && r_norm < tol * 100) {
            result.converged = true;
            result.eigenvalue = E;
            // S-normalize eigenvector
            double norm_sq = x.dot(S * x);
            if (norm_sq > 0) x /= std::sqrt(norm_sq);
            if (x(0) < 0) x = -x;
            result.eigenvector = x;

            if (verbose >= 1) {
                fprintf(stderr, "  Davidson converged in %d iterations, E = %.10f, |r| = %.2e\n",
                        iter + 1, E, r_norm);
            }
            return result;
        }
        E_old = E;

        // Preconditioned correction: t_i = -r_i / (precond_i - E)
        Eigen::VectorXd t(n);
        for (int i = 0; i < n; ++i) {
            double denom = precond(i) - E;
            t(i) = (std::abs(denom) > 1e-12) ? -r(i) / denom : -r(i);
        }

        // S-orthogonalize t against V (double Gram-Schmidt)
        for (int pass = 0; pass < 2; ++pass) {
            Eigen::VectorXd St = S * t;
            Eigen::VectorXd coeffs = V.transpose() * St;
            t -= V * coeffs;
        }

        // S-normalize t
        double t_snorm = std::sqrt(std::abs(t.dot(S * t)));
        if (t_snorm < 1e-14) {
            // Expansion vector too small, try random perturbation
            if (verbose >= 2) {
                fprintf(stderr, "  Davidson: expansion vector too small at iter %d\n", iter + 1);
            }
            continue;
        }
        t /= t_snorm;

        // Restart if subspace too large
        if (k >= max_subspace) {
            // Keep best few Ritz vectors
            int n_keep = std::min(3, k);
            Eigen::MatrixXd best(n, n_keep);
            for (int j = 0; j < n_keep; ++j) {
                Eigen::VectorXd sub_alpha = X * H_solver.eigenvectors().col(j);
                best.col(j) = V * sub_alpha;
            }

            // Re-S-orthonormalize
            for (int j = 0; j < n_keep; ++j) {
                for (int i = 0; i < j; ++i) {
                    double overlap = best.col(i).dot(S * best.col(j));
                    best.col(j) -= overlap * best.col(i);
                }
                double sn = std::sqrt(std::abs(best.col(j).dot(S * best.col(j))));
                if (sn > 1e-14) best.col(j) /= sn;
            }

            // New V = [best | t]
            V.resize(n, n_keep + 1);
            V.leftCols(n_keep) = best;
            V.col(n_keep) = t;

            if (verbose >= 2) {
                fprintf(stderr, "  Davidson: subspace restart %d -> %d\n", k, n_keep + 1);
            }
        } else {
            // Append t to V
            V.conservativeResize(n, k + 1);
            V.col(k) = t;
        }
    }

    // Did not converge: return best estimate
    result.eigenvalue = E_old;
    // Recompute eigenvector from last iteration
    int k = V.cols();
    Eigen::MatrixXd HV = H * V;
    Eigen::MatrixXd SV = S * V;
    Eigen::MatrixXd H_sub = V.transpose() * HV;
    Eigen::MatrixXd S_sub = V.transpose() * SV;
    H_sub = 0.5 * (H_sub + H_sub.transpose());
    S_sub = 0.5 * (S_sub + S_sub.transpose());

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> S_final(S_sub);
    Eigen::VectorXd s_eigs = S_final.eigenvalues();
    Eigen::MatrixXd s_vecs = S_final.eigenvectors();
    int n_valid = 0;
    for (int i = 0; i < k; ++i) {
        if (s_eigs(i) > 1e-10) ++n_valid;
    }
    if (n_valid > 0) {
        Eigen::MatrixXd X(k, n_valid);
        int col = 0;
        for (int i = 0; i < k; ++i) {
            if (s_eigs(i) > 1e-10) {
                X.col(col) = s_vecs.col(i) / std::sqrt(s_eigs(i));
                ++col;
            }
        }
        Eigen::MatrixXd H_orth = X.transpose() * H_sub * X;
        H_orth = 0.5 * (H_orth + H_orth.transpose());
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> H_final(H_orth);
        result.eigenvalue = H_final.eigenvalues()(0);
        Eigen::VectorXd alpha = X * H_final.eigenvectors().col(0);
        Eigen::VectorXd x = V * alpha;
        double norm_sq = x.dot(S * x);
        if (norm_sq > 0) x /= std::sqrt(norm_sq);
        if (x(0) < 0) x = -x;
        result.eigenvector = x;
    } else {
        result.eigenvector = Eigen::VectorXd::Zero(n);
        result.eigenvector(0) = 1.0;
    }

    if (verbose >= 1) {
        fprintf(stderr, "  Davidson: NOT converged after %d iterations, E = %.10f, |r| = %.2e\n",
                max_iter, result.eigenvalue, result.residual_norm);
    }

    return result;
}

} // namespace trimci_core
