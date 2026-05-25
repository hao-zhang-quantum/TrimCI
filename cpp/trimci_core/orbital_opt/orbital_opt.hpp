#pragma once

#include <array>
#include <functional>
#include <tuple>
#include <vector>

#include <Eigen/Dense>

#include "determinant.hpp"

namespace trimci_core {

/// Optional Davidson callback: (h1, eri, initial_guess) -> (energy, coeffs)
/// When provided, replaces internal diagonalize_subspace_davidson_t.
template<typename StorageType>
using DiagFunc = std::function<
    std::tuple<double, std::vector<double>>(
        const std::vector<std::vector<double>>& h1,
        const std::vector<double>& eri,
        const std::vector<double>& initial_guess)>;

template<typename StorageType>
Eigen::MatrixXd orbital_gradient_t(
    const std::vector<DeterminantT<StorageType>>& dets,
    const std::vector<double>& coeffs,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    int n_orb,
    int n_alpha = -1,
    int n_beta = -1);

template<typename StorageType>
std::tuple<Eigen::MatrixXd,
           std::vector<std::vector<double>>,
           std::vector<double>,
           double,
           double>
orbital_approx_newton_step_t(
    const std::vector<DeterminantT<StorageType>>& dets,
    const std::vector<double>& coeffs,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    int n_orb,
    int n_alpha = -1,
    int n_beta = -1,
    bool rotate_integrals = true,
    double step_scale = 1.0,
    double min_hess = 1e-5,
    double max_param = 0.2);

template<typename StorageType>
std::tuple<Eigen::MatrixXd,
           std::vector<std::vector<double>>,
           std::vector<double>,
           double,
           double,
           bool,
           int,
           std::vector<std::array<double, 2>>>
orbital_approx_newton_optimize_t(
    const std::vector<DeterminantT<StorageType>>& dets,
    const std::vector<double>& coeffs,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    int n_orb,
    int n_alpha = -1,
    int n_beta = -1,
    int max_iter = 50,
    double grad_tol = 1e-6,
    double energy_tol = 1e-8,
    double step_scale = 1.0,
    double min_hess = 1e-5,
    double max_param = 0.2,
    bool line_search = true,
    int ls_max_iter = 6,
    double ls_shrink = 0.5,
    double ls_energy_tol = 0.0,
    double davidson_tol = 1e-3,
    int davidson_max_iter = 200,
    bool rotate_integrals = true,
    bool record_trace = false);

template<typename StorageType>
std::tuple<Eigen::MatrixXd,
           std::vector<std::vector<double>>,
           std::vector<double>,
           double,
           double,
           bool,
           int,
           std::vector<std::array<double, 2>>>
orbital_bfgs_optimize_t(
    const std::vector<DeterminantT<StorageType>>& dets,
    const std::vector<double>& coeffs,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    int n_orb,
    int n_alpha = -1,
    int n_beta = -1,
    int max_iter = 50,
    double grad_tol = 1e-6,
    double energy_tol = 1e-8,
    double step_scale = 1.0,
    double min_hess = 1e-5,
    double max_param = 0.2,
    double bfgs_damp = 1e-3,
    double bfgs_update_tol = 1e-6,
    double bfgs_step_factor = 5.0,
    double bfgs_step_decay = 0.97,
    double bfgs_step_min = 0.25,
    double bfgs_init_div = 4.0,
    bool line_search = true,
    int ls_max_iter = 6,
    double ls_shrink = 0.5,
    double ls_energy_tol = 0.0,
    double davidson_tol = 1e-3,
    int davidson_max_iter = 200,
    bool rotate_integrals = true,
    bool record_trace = false,
    DiagFunc<StorageType> diag_func = nullptr);

} // namespace trimci_core
