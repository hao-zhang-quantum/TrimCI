// Bindings for Fast Expansion module
#include "bind_common.hpp"
#include "fast_expansion/fe_types.hpp"
#include "fast_expansion/ab_group_matvec.hpp"
#include "fast_expansion/ab_index.hpp"
#include "fast_expansion/detspace_matvec.hpp"
#include "fast_expansion/detspace_davidson.hpp"
#include "fast_expansion/expansion_loop.hpp"
#include "fast_expansion/bloom_filter.hpp"
#include "fast_expansion/streaming_pt2.hpp"
#include "fast_expansion/enpt3.hpp"
#include "fast_expansion/dressed_ci.hpp"
#include "fast_expansion/ab_group_matvec.hpp"

using namespace trimci_core;
using namespace trimci_core::fe;

void bind_fast_expansion(py::module& m) {
    auto fe_mod = m.def_submodule("fast_expansion", "Fast Expansion: matrix-free Selected CI");

    // ========================================================================
    // compute_diagonals
    // ========================================================================
    fe_mod.def("compute_diagonals", [](
        py::array_t<uint64_t, py::array::c_style> alpha_arr,
        py::array_t<uint64_t, py::array::c_style> beta_arr,
        py::array_t<double, py::array::c_style> h1_arr,
        py::array_t<double, py::array::c_style> eri_arr,
        int n_orb) -> py::array_t<double>
    {
        auto a = alpha_arr.unchecked<1>();
        auto b = beta_arr.unchecked<1>();
        size_t N = a.shape(0);

        // Build det list
        std::vector<DeterminantT<uint64_t>> dets(N);
        for (size_t i = 0; i < N; ++i) {
            dets[i] = DeterminantT<uint64_t>(a(i), b(i));
        }

        // Build h1 as vector<vector<double>>
        auto h1_2d = h1_arr.unchecked<2>();
        std::vector<std::vector<double>> h1(n_orb, std::vector<double>(n_orb));
        for (int i = 0; i < n_orb; ++i)
            for (int j = 0; j < n_orb; ++j)
                h1[i][j] = h1_2d(i, j);

        // ERI as flat vector
        auto eri_buf = eri_arr.request();
        std::vector<double> eri(static_cast<double*>(eri_buf.ptr),
                                static_cast<double*>(eri_buf.ptr) + eri_buf.size);

        std::vector<double> diag;
        {
            py::gil_scoped_release release;
            diag = compute_diagonals<uint64_t>(dets, h1, eri, n_orb);
        }

        py::array_t<double> result(N);
        auto r = result.mutable_unchecked<1>();
        for (size_t i = 0; i < N; ++i) r(i) = diag[i];
        return result;
    },
    py::arg("alpha"), py::arg("beta"), py::arg("h1"), py::arg("eri"), py::arg("n_orb"),
    "Compute diagonal H_ii elements for a set of determinants.");

    // ========================================================================
    // ABIndex (persistent object for distributed matvec)
    // ========================================================================
    py::class_<ABIndex<uint64_t>>(fe_mod, "ABIndex")
        .def(py::init<>())
        .def("build", [](ABIndex<uint64_t>& self,
                         py::array_t<uint64_t, py::array::c_style> alpha_arr,
                         py::array_t<uint64_t, py::array::c_style> beta_arr,
                         int n_orb) {
            auto a = alpha_arr.unchecked<1>();
            auto b = beta_arr.unchecked<1>();
            size_t N = a.shape(0);
            std::vector<DeterminantT<uint64_t>> dets(N);
            for (size_t i = 0; i < N; ++i) {
                dets[i] = DeterminantT<uint64_t>(a(i), b(i));
            }
            py::gil_scoped_release release;
            self.build(dets, n_orb);
        }, py::arg("alpha"), py::arg("beta"), py::arg("n_orb"),
        "Build ABIndex from alpha/beta arrays. Call once, reuse for many matvecs.")
        .def("save", [](const ABIndex<uint64_t>& self, const std::string& path) {
            py::gil_scoped_release release;
            self.save(path);
        }, py::arg("path"),
        "Save ABIndex to a binary file for later reuse by workers.")
        .def_static("load", [](const std::string& path) -> ABIndex<uint64_t> {
            py::gil_scoped_release release;
            return ABIndex<uint64_t>::load(path);
        }, py::arg("path"),
        "Load a pre-built ABIndex from a binary file.");

    // ========================================================================
    // matvec_row_sliced (for distributed computation)
    // ========================================================================
    fe_mod.def("matvec_row_sliced", [](
        ABIndex<uint64_t>& ab_index,
        py::array_t<uint64_t, py::array::c_style> alpha_arr,
        py::array_t<uint64_t, py::array::c_style> beta_arr,
        py::array_t<double, py::array::c_style> v_arr,
        py::array_t<double, py::array::c_style> diag_arr,
        py::array_t<double, py::array::c_style> h1_arr,
        py::array_t<double, py::array::c_style> eri_arr,
        int n_orb,
        size_t row_start,
        size_t row_end) -> py::array_t<double>
    {
        auto a = alpha_arr.unchecked<1>();
        auto b = beta_arr.unchecked<1>();
        auto v_in = v_arr.unchecked<1>();
        auto d_in = diag_arr.unchecked<1>();
        size_t N = a.shape(0);

        std::vector<DeterminantT<uint64_t>> dets(N);
        for (size_t i = 0; i < N; ++i) {
            dets[i] = DeterminantT<uint64_t>(a(i), b(i));
        }

        auto h1_2d = h1_arr.unchecked<2>();
        std::vector<std::vector<double>> h1(n_orb, std::vector<double>(n_orb));
        for (int i = 0; i < n_orb; ++i)
            for (int j = 0; j < n_orb; ++j)
                h1[i][j] = h1_2d(i, j);

        auto eri_buf = eri_arr.request();
        std::vector<double> eri(static_cast<double*>(eri_buf.ptr),
                                static_cast<double*>(eri_buf.ptr) + eri_buf.size);

        std::vector<double> v(N);
        for (size_t i = 0; i < N; ++i) v[i] = v_in(i);

        std::vector<double> diag(N);
        for (size_t i = 0; i < N; ++i) diag[i] = d_in(i);

        if (row_end > N) row_end = N;
        size_t chunk_size = (row_start < row_end) ? (row_end - row_start) : 0;
        std::vector<double> sigma(chunk_size, 0.0);

        {
            py::gil_scoped_release release;
            MatvecContext<uint64_t> ctx{dets, ab_index, h1, eri, n_orb, diag};
            matvec_row_sliced<uint64_t>(ctx, v.data(), sigma.data(), N,
                                        row_start, row_end);
        }

        py::array_t<double> result(chunk_size);
        auto r = result.mutable_unchecked<1>();
        for (size_t i = 0; i < chunk_size; ++i) r(i) = sigma[i];
        return result;
    },
    py::arg("ab_index"), py::arg("alpha"), py::arg("beta"),
    py::arg("v"), py::arg("diag"),
    py::arg("h1"), py::arg("eri"), py::arg("n_orb"),
    py::arg("row_start"), py::arg("row_end"),
    "Row-sliced matvec: sigma[0..chunk-1] = H[row_start:row_end, :] @ v. "
    "For distributed Davidson: each worker computes its row slice.");

    // ========================================================================
    // DavidsonParams & DavidsonResult
    // ========================================================================
    py::class_<DavidsonParams>(fe_mod, "DavidsonParams")
        .def(py::init<>())
        .def_readwrite("max_subspace", &DavidsonParams::max_subspace)
        .def_readwrite("max_iter", &DavidsonParams::max_iter)
        .def_readwrite("residual_tol", &DavidsonParams::residual_tol)
        .def_readwrite("energy_tol", &DavidsonParams::energy_tol)
        .def_readwrite("n_states", &DavidsonParams::n_states)
        .def_readwrite("verbose", &DavidsonParams::verbose)
        .def_readwrite("block_size", &DavidsonParams::block_size);

    py::class_<DavidsonResult>(fe_mod, "DavidsonResult")
        .def_readonly("eigenvalues", &DavidsonResult::eigenvalues)
        .def_readonly("eigenvectors", &DavidsonResult::eigenvectors)
        .def_readonly("n_iters", &DavidsonResult::n_iters)
        .def_readonly("converged", &DavidsonResult::converged)
        .def_readonly("residual_norm", &DavidsonResult::residual_norm)
        .def_readonly("ffm_wall_time", &DavidsonResult::ffm_wall_time);

    // ========================================================================
    // davidson_solve (dense H matrix version for testing)
    // ========================================================================
    fe_mod.def("davidson_solve_dense", [](
        py::array_t<double, py::array::c_style> H_arr,
        py::array_t<double, py::array::c_style> diag_arr,
        DavidsonParams params,
        py::array_t<double, py::array::c_style> initial_guess_arr) -> DavidsonResult
    {
        auto H = H_arr.unchecked<2>();
        auto d = diag_arr.unchecked<1>();
        size_t N = d.shape(0);

        std::vector<double> diag(N);
        for (size_t i = 0; i < N; ++i) diag[i] = d(i);

        // Dense matvec lambda
        const double* H_ptr = H_arr.data();
        auto matvec_fn = [H_ptr, N](const double* v, double* sigma, size_t n) {
            for (size_t i = 0; i < n; ++i) {
                double s = 0.0;
                for (size_t j = 0; j < n; ++j) {
                    s += H_ptr[i * n + j] * v[j];
                }
                sigma[i] = s;
            }
        };

        // Initial guess
        std::vector<std::vector<double>> guess;
        if (initial_guess_arr.size() > 0) {
            auto ig = initial_guess_arr.unchecked<2>();
            int n_guess = ig.shape(0);
            for (int s = 0; s < n_guess; ++s) {
                guess.emplace_back(N);
                for (size_t i = 0; i < N; ++i) guess.back()[i] = ig(s, i);
            }
        }

        DavidsonResult result;
        {
            py::gil_scoped_release release;
            result = davidson_solve(matvec_fn, diag, N, params, guess);
        }
        return result;
    },
    py::arg("H"), py::arg("diag"), py::arg("params") = DavidsonParams{},
    py::arg("initial_guess") = py::array_t<double>(),
    "Solve H x = λ x using Davidson with a dense H matrix (for testing).");

    // ========================================================================
    // ABIndex + matrix-free matvec (combined test endpoint)
    // ========================================================================
    fe_mod.def("matvec_matfree", [](
        py::array_t<uint64_t, py::array::c_style> alpha_arr,
        py::array_t<uint64_t, py::array::c_style> beta_arr,
        py::array_t<double, py::array::c_style> v_arr,
        py::array_t<double, py::array::c_style> h1_arr,
        py::array_t<double, py::array::c_style> eri_arr,
        int n_orb) -> py::array_t<double>
    {
        auto a = alpha_arr.unchecked<1>();
        auto b = beta_arr.unchecked<1>();
        auto v_in = v_arr.unchecked<1>();
        size_t N = a.shape(0);

        std::vector<DeterminantT<uint64_t>> dets(N);
        for (size_t i = 0; i < N; ++i) {
            dets[i] = DeterminantT<uint64_t>(a(i), b(i));
        }

        auto h1_2d = h1_arr.unchecked<2>();
        std::vector<std::vector<double>> h1(n_orb, std::vector<double>(n_orb));
        for (int i = 0; i < n_orb; ++i)
            for (int j = 0; j < n_orb; ++j)
                h1[i][j] = h1_2d(i, j);

        auto eri_buf = eri_arr.request();
        std::vector<double> eri(static_cast<double*>(eri_buf.ptr),
                                static_cast<double*>(eri_buf.ptr) + eri_buf.size);

        std::vector<double> v(N);
        for (size_t i = 0; i < N; ++i) v[i] = v_in(i);

        std::vector<double> sigma(N, 0.0);

        {
            py::gil_scoped_release release;

            // Build AB index
            ABIndex<uint64_t> ab_index;
            ab_index.build(dets, n_orb);

            // Compute diagonals
            auto diag = compute_diagonals<uint64_t>(dets, h1, eri, n_orb);

            // Matrix-free matvec
            MatvecContext<uint64_t> ctx{dets, ab_index, h1, eri, n_orb, diag};
            matvec<uint64_t>(ctx, v.data(), sigma.data(), N);
        }

        py::array_t<double> result(N);
        auto r = result.mutable_unchecked<1>();
        for (size_t i = 0; i < N; ++i) r(i) = sigma[i];
        return result;
    },
    py::arg("alpha"), py::arg("beta"), py::arg("v"),
    py::arg("h1"), py::arg("eri"), py::arg("n_orb"),
    "Matrix-free H*v using ABIndex + abm1+Scan enumeration.");

    // ========================================================================
    // davidson_solve_matfree (ABIndex-based, for end-to-end testing)
    // ========================================================================
    fe_mod.def("davidson_solve_matfree", [](
        py::array_t<uint64_t, py::array::c_style> alpha_arr,
        py::array_t<uint64_t, py::array::c_style> beta_arr,
        py::array_t<double, py::array::c_style> h1_arr,
        py::array_t<double, py::array::c_style> eri_arr,
        int n_orb,
        DavidsonParams params,
        py::array_t<double, py::array::c_style> initial_guess_arr) -> DavidsonResult
    {
        auto a = alpha_arr.unchecked<1>();
        auto b = beta_arr.unchecked<1>();
        size_t N = a.shape(0);

        std::vector<DeterminantT<uint64_t>> dets(N);
        for (size_t i = 0; i < N; ++i) {
            dets[i] = DeterminantT<uint64_t>(a(i), b(i));
        }

        auto h1_2d = h1_arr.unchecked<2>();
        std::vector<std::vector<double>> h1(n_orb, std::vector<double>(n_orb));
        for (int i = 0; i < n_orb; ++i)
            for (int j = 0; j < n_orb; ++j)
                h1[i][j] = h1_2d(i, j);

        auto eri_buf = eri_arr.request();
        std::vector<double> eri(static_cast<double*>(eri_buf.ptr),
                                static_cast<double*>(eri_buf.ptr) + eri_buf.size);

        std::vector<std::vector<double>> guess;
        if (initial_guess_arr.size() > 0) {
            auto ig = initial_guess_arr.unchecked<2>();
            int n_guess = ig.shape(0);
            for (int s = 0; s < n_guess; ++s) {
                guess.emplace_back(N);
                for (size_t i = 0; i < N; ++i) guess.back()[i] = ig(s, i);
            }
        }

        DavidsonResult result;
        {
            py::gil_scoped_release release;

            ABIndex<uint64_t> ab_index;
            ab_index.build(dets, n_orb);
            auto diag = compute_diagonals<uint64_t>(dets, h1, eri, n_orb);

            MatvecContext<uint64_t> ctx{dets, ab_index, h1, eri, n_orb, diag};
            auto matvec_fn = [&ctx, N](const double* v, double* sigma, size_t n) {
                std::fill(sigma, sigma + n, 0.0);
                matvec<uint64_t>(ctx, v, sigma, n);
            };

            result = davidson_solve(matvec_fn, diag, N, params, guess);
        }
        return result;
    },
    py::arg("alpha"), py::arg("beta"),
    py::arg("h1"), py::arg("eri"), py::arg("n_orb"),
    py::arg("params") = DavidsonParams{},
    py::arg("initial_guess") = py::array_t<double>(),
    "Solve H x = λ x using matrix-free Davidson with ABIndex.");

    // ========================================================================
    // ExpansionConfig & ExpansionResult
    // ========================================================================
    py::class_<ExpansionConfig<uint64_t>>(fe_mod, "ExpansionConfig")
        .def(py::init<>())
        .def_readwrite("growth_factor", &ExpansionConfig<uint64_t>::growth_factor)
        .def_readwrite("max_n_dets", &ExpansionConfig<uint64_t>::max_n_dets)
        .def_readwrite("max_expansion_rounds", &ExpansionConfig<uint64_t>::max_expansion_rounds)
        .def_readwrite("strict_target_size", &ExpansionConfig<uint64_t>::strict_target_size)
        .def_readwrite("screening_mode", &ExpansionConfig<uint64_t>::screening_mode)
        .def_readwrite("threshold", &ExpansionConfig<uint64_t>::threshold)
        .def_readwrite("threshold_decay", &ExpansionConfig<uint64_t>::threshold_decay)
        .def_readwrite("davidson_params", &ExpansionConfig<uint64_t>::davidson_params)
        .def_readwrite("expansion_energy_tol", &ExpansionConfig<uint64_t>::expansion_energy_tol)
        .def_readwrite("dets_conv_ratio", &ExpansionConfig<uint64_t>::dets_conv_ratio)
        .def_readwrite("use_connection_cache", &ExpansionConfig<uint64_t>::use_connection_cache)
        .def_readwrite("symmetric_sigma", &ExpansionConfig<uint64_t>::symmetric_sigma)
        .def_readwrite("davidson_block_size", &ExpansionConfig<uint64_t>::davidson_block_size)
        .def_readwrite("backend", &ExpansionConfig<uint64_t>::backend)
        .def_readwrite("use_morton_sort", &ExpansionConfig<uint64_t>::use_morton_sort)
        .def_readwrite("use_dual_order", &ExpansionConfig<uint64_t>::use_dual_order)
        .def_readwrite("use_sparse_update", &ExpansionConfig<uint64_t>::use_sparse_update)
        .def_readwrite("sparse_truncation_ratio", &ExpansionConfig<uint64_t>::sparse_truncation_ratio)
        .def_readwrite("sparse_min_active", &ExpansionConfig<uint64_t>::sparse_min_active)
        .def_readwrite("sparse_n_warm_iters", &ExpansionConfig<uint64_t>::sparse_n_warm_iters)
        .def_readwrite("sparse_capture_threshold", &ExpansionConfig<uint64_t>::sparse_capture_threshold)
        .def_readwrite("sparse_adaptive_K", &ExpansionConfig<uint64_t>::sparse_adaptive_K)
        .def_readwrite("sparse_truncation_mode", &ExpansionConfig<uint64_t>::sparse_truncation_mode)
        .def_readwrite("sparse_warm_ratio", &ExpansionConfig<uint64_t>::sparse_warm_ratio)
        .def_readwrite("sparse_random_sample_ratio", &ExpansionConfig<uint64_t>::sparse_random_sample_ratio)
        .def_readwrite("sparse_ucb_kappa", &ExpansionConfig<uint64_t>::sparse_ucb_kappa)
        .def_readwrite("sparse_olsen_correction", &ExpansionConfig<uint64_t>::sparse_olsen_correction)
        .def_readwrite("sparse_thick_restart_size", &ExpansionConfig<uint64_t>::sparse_thick_restart_size)
        .def_readwrite("sparse_momentum_beta", &ExpansionConfig<uint64_t>::sparse_momentum_beta)
        .def_readwrite("sparse_K_schedule", &ExpansionConfig<uint64_t>::sparse_K_schedule)
        .def_readwrite("sparse_final_full_matvec", &ExpansionConfig<uint64_t>::sparse_final_full_matvec)
        .def_readwrite("perturbative_warmstart", &ExpansionConfig<uint64_t>::perturbative_warmstart)
        .def_readwrite("no_warm_start", &ExpansionConfig<uint64_t>::no_warm_start)
        .def_readwrite("orbital_optimization", &ExpansionConfig<uint64_t>::orbital_optimization)
        .def_readwrite("orbital_opt_max_iter", &ExpansionConfig<uint64_t>::orbital_opt_max_iter)
        .def_readwrite("orbital_opt_grad_tol", &ExpansionConfig<uint64_t>::orbital_opt_grad_tol)
        .def_readwrite("orbital_opt_energy_tol", &ExpansionConfig<uint64_t>::orbital_opt_energy_tol)
        .def_readwrite("dressed_energy", &ExpansionConfig<uint64_t>::dressed_energy)
        .def_readwrite("dressed_sc_max_iter", &ExpansionConfig<uint64_t>::dressed_sc_max_iter)
        .def_readwrite("dressed_sc_energy_tol", &ExpansionConfig<uint64_t>::dressed_sc_energy_tol)
        .def_readwrite("evaluate_only", &ExpansionConfig<uint64_t>::evaluate_only)
        .def_readwrite("pool_build_only", &ExpansionConfig<uint64_t>::pool_build_only)
        .def_readwrite("verbose", &ExpansionConfig<uint64_t>::verbose)
        .def_readwrite("log_file", &ExpansionConfig<uint64_t>::log_file)
        .def_readwrite("checkpoint_dir", &ExpansionConfig<uint64_t>::checkpoint_dir)
        .def_readwrite("checkpoint_round_offset", &ExpansionConfig<uint64_t>::checkpoint_round_offset)
        .def_property("pt2_config",
            [](const ExpansionConfig<uint64_t>& c) -> py::object {
                if (c.pt2_config.has_value()) return py::cast(c.pt2_config.value());
                return py::none();
            },
            [](ExpansionConfig<uint64_t>& c, py::object val) {
                if (val.is_none()) c.pt2_config = std::nullopt;
                else c.pt2_config = val.cast<PT2Config<uint64_t>>();
            })
        .def_property("dressed_ci_config",
            [](const ExpansionConfig<uint64_t>& c) -> py::object {
                if (c.dressed_ci_config.has_value()) return py::cast(c.dressed_ci_config.value());
                return py::none();
            },
            [](ExpansionConfig<uint64_t>& c, py::object val) {
                if (val.is_none()) c.dressed_ci_config = std::nullopt;
                else c.dressed_ci_config = val.cast<DressedCIConfig>();
            });

    py::class_<ExpansionResult<uint64_t>>(fe_mod, "ExpansionResult")
        .def_readonly("energy_var", &ExpansionResult<uint64_t>::energy_var)
        .def_readonly("ffm_wall_time", &ExpansionResult<uint64_t>::ffm_wall_time)
        .def_readonly("n_rounds", &ExpansionResult<uint64_t>::n_rounds)
        .def_readonly("energy_history", &ExpansionResult<uint64_t>::energy_history)
        .def_readonly("ndets_history", &ExpansionResult<uint64_t>::ndets_history)
        .def_readonly("pt2_history", &ExpansionResult<uint64_t>::pt2_history)
        .def_readonly("variance_ext_history", &ExpansionResult<uint64_t>::variance_ext_history)
        .def_readonly("screening_error_history", &ExpansionResult<uint64_t>::screening_error_history)
        .def_readonly("dressed_energy_history", &ExpansionResult<uint64_t>::dressed_energy_history)
        .def_readonly("dressed_pt2_energy_history", &ExpansionResult<uint64_t>::dressed_pt2_energy_history)
        .def("get_alpha_beta", [](const ExpansionResult<uint64_t>& res)
            -> std::pair<py::array_t<uint64_t>, py::array_t<uint64_t>>
        {
            size_t N = res.dets.size();
            py::array_t<uint64_t> alpha(N), beta(N);
            auto a = alpha.mutable_unchecked<1>();
            auto b = beta.mutable_unchecked<1>();
            for (size_t i = 0; i < N; ++i) {
                a(i) = res.dets[i].alpha;
                b(i) = res.dets[i].beta;
            }
            return {alpha, beta};
        }, "Get alpha and beta arrays from final determinants.")
        .def("get_coefficients", [](const ExpansionResult<uint64_t>& res)
            -> py::array_t<double>
        {
            size_t N = res.coefficients.size();
            py::array_t<double> arr(N);
            auto r = arr.mutable_unchecked<1>();
            for (size_t i = 0; i < N; ++i) r(i) = res.coefficients[i];
            return arr;
        }, "Get coefficient array.")
        .def("get_U_total", [](const ExpansionResult<uint64_t>& res)
            -> py::array_t<double>
        {
            if (res.U_total.empty()) return py::array_t<double>();
            size_t n = (size_t)std::sqrt((double)res.U_total.size());
            py::array_t<double> arr({n, n});
            std::memcpy(arr.mutable_data(), res.U_total.data(), n * n * sizeof(double));
            return arr;
        }, "Get cumulative orbital rotation matrix (n_orb x n_orb).")
        .def("get_h1_optimized", [](const ExpansionResult<uint64_t>& res)
            -> py::array_t<double>
        {
            if (res.h1_optimized.empty()) return py::array_t<double>();
            size_t n = res.h1_optimized.size();
            py::array_t<double> arr({n, n});
            auto r = arr.mutable_unchecked<2>();
            for (size_t i = 0; i < n; ++i)
                for (size_t j = 0; j < n; ++j)
                    r(i, j) = res.h1_optimized[i][j];
            return arr;
        }, "Get optimized one-electron integrals (n_orb x n_orb).")
        .def("get_eri_optimized", [](const ExpansionResult<uint64_t>& res)
            -> py::array_t<double>
        {
            if (res.eri_optimized.empty()) return py::array_t<double>();
            size_t n4 = res.eri_optimized.size();
            py::array_t<double> arr(n4);
            std::memcpy(arr.mutable_data(), res.eri_optimized.data(), n4 * sizeof(double));
            return arr;
        }, "Get optimized two-electron integrals (flat n_orb^4).");

    // ========================================================================
    // run_expansion
    // ========================================================================
    fe_mod.def("run_expansion", [](
        py::array_t<uint64_t, py::array::c_style> alpha_arr,
        py::array_t<uint64_t, py::array::c_style> beta_arr,
        py::array_t<double, py::array::c_style> coeffs_arr,
        py::array_t<double, py::array::c_style> h1_arr,
        py::array_t<double, py::array::c_style> eri_arr,
        int n_orb,
        ExpansionConfig<uint64_t> config) -> ExpansionResult<uint64_t>
    {
        auto a = alpha_arr.unchecked<1>();
        auto b = beta_arr.unchecked<1>();
        auto c = coeffs_arr.unchecked<1>();
        size_t N = a.shape(0);

        std::vector<DeterminantT<uint64_t>> dets(N);
        for (size_t i = 0; i < N; ++i) {
            dets[i] = DeterminantT<uint64_t>(a(i), b(i));
        }

        std::vector<double> coeffs(N);
        for (size_t i = 0; i < N; ++i) coeffs[i] = c(i);

        auto h1_2d = h1_arr.unchecked<2>();
        std::vector<std::vector<double>> h1(n_orb, std::vector<double>(n_orb));
        for (int i = 0; i < n_orb; ++i)
            for (int j = 0; j < n_orb; ++j)
                h1[i][j] = h1_2d(i, j);

        auto eri_buf = eri_arr.request();
        std::vector<double> eri(static_cast<double*>(eri_buf.ptr),
                                static_cast<double*>(eri_buf.ptr) + eri_buf.size);

        ExpansionResult<uint64_t> result;
        {
            py::gil_scoped_release release;
            result = run_expansion<uint64_t>(dets, coeffs, h1, eri, n_orb, config);
        }
        return result;
    },
    py::arg("alpha"), py::arg("beta"), py::arg("coefficients"),
    py::arg("h1"), py::arg("eri"), py::arg("n_orb"),
    py::arg("config") = ExpansionConfig<uint64_t>{},
    "Run iterative expansion: HF -> variational space via heat-bath screening + Davidson.");

    // ========================================================================
    // run_expansion_phased (multi-phase wrapper)
    // ========================================================================
    fe_mod.def("run_expansion_phased", [](
        py::array_t<uint64_t, py::array::c_style> alpha_arr,
        py::array_t<uint64_t, py::array::c_style> beta_arr,
        py::array_t<double, py::array::c_style> coeffs_arr,
        py::array_t<double, py::array::c_style> h1_arr,
        py::array_t<double, py::array::c_style> eri_arr,
        int n_orb,
        std::vector<ExpansionConfig<uint64_t>> phase_configs) -> ExpansionResult<uint64_t>
    {
        auto a = alpha_arr.unchecked<1>();
        auto b = beta_arr.unchecked<1>();
        auto c = coeffs_arr.unchecked<1>();
        size_t N = a.shape(0);

        std::vector<DeterminantT<uint64_t>> dets(N);
        for (size_t i = 0; i < N; ++i) {
            dets[i] = DeterminantT<uint64_t>(a(i), b(i));
        }

        std::vector<double> coeffs(N);
        for (size_t i = 0; i < N; ++i) coeffs[i] = c(i);

        auto h1_2d = h1_arr.unchecked<2>();
        std::vector<std::vector<double>> h1(n_orb, std::vector<double>(n_orb));
        for (int i = 0; i < n_orb; ++i)
            for (int j = 0; j < n_orb; ++j)
                h1[i][j] = h1_2d(i, j);

        auto eri_buf = eri_arr.request();
        std::vector<double> eri(static_cast<double*>(eri_buf.ptr),
                                static_cast<double*>(eri_buf.ptr) + eri_buf.size);

        ExpansionResult<uint64_t> result;
        {
            py::gil_scoped_release release;
            result = run_expansion_phased<uint64_t>(dets, coeffs, h1, eri, n_orb, phase_configs);
        }
        return result;
    },
    py::arg("alpha"), py::arg("beta"), py::arg("coefficients"),
    py::arg("h1"), py::arg("eri"), py::arg("n_orb"),
    py::arg("phase_configs"),
    "Run multi-phase expansion: each config is a fully resolved ExpansionConfig.");

    // ========================================================================
    // 128-bit Determinant support (n_orb > 64)
    // ========================================================================
    using Det128 = DeterminantT<std::array<uint64_t, 2>>;
    using Cfg128 = ExpansionConfig<std::array<uint64_t, 2>>;
    using Res128 = ExpansionResult<std::array<uint64_t, 2>>;

    py::class_<Cfg128>(fe_mod, "ExpansionConfig128")
        .def(py::init<>())
        .def_readwrite("growth_factor", &Cfg128::growth_factor)
        .def_readwrite("max_n_dets", &Cfg128::max_n_dets)
        .def_readwrite("max_expansion_rounds", &Cfg128::max_expansion_rounds)
        .def_readwrite("strict_target_size", &Cfg128::strict_target_size)
        .def_readwrite("screening_mode", &Cfg128::screening_mode)
        .def_readwrite("threshold", &Cfg128::threshold)
        .def_readwrite("threshold_decay", &Cfg128::threshold_decay)
        .def_readwrite("davidson_params", &Cfg128::davidson_params)
        .def_readwrite("expansion_energy_tol", &Cfg128::expansion_energy_tol)
        .def_readwrite("dets_conv_ratio", &Cfg128::dets_conv_ratio)
        .def_readwrite("use_connection_cache", &Cfg128::use_connection_cache)
        .def_readwrite("symmetric_sigma", &Cfg128::symmetric_sigma)
        .def_readwrite("davidson_block_size", &Cfg128::davidson_block_size)
        .def_readwrite("backend", &Cfg128::backend)
        .def_readwrite("use_morton_sort", &Cfg128::use_morton_sort)
        .def_readwrite("use_dual_order", &Cfg128::use_dual_order)
        .def_readwrite("use_sparse_update", &Cfg128::use_sparse_update)
        .def_readwrite("sparse_truncation_ratio", &Cfg128::sparse_truncation_ratio)
        .def_readwrite("sparse_min_active", &Cfg128::sparse_min_active)
        .def_readwrite("sparse_n_warm_iters", &Cfg128::sparse_n_warm_iters)
        .def_readwrite("sparse_capture_threshold", &Cfg128::sparse_capture_threshold)
        .def_readwrite("sparse_adaptive_K", &Cfg128::sparse_adaptive_K)
        .def_readwrite("sparse_truncation_mode", &Cfg128::sparse_truncation_mode)
        .def_readwrite("sparse_warm_ratio", &Cfg128::sparse_warm_ratio)
        .def_readwrite("sparse_random_sample_ratio", &Cfg128::sparse_random_sample_ratio)
        .def_readwrite("sparse_ucb_kappa", &Cfg128::sparse_ucb_kappa)
        .def_readwrite("sparse_olsen_correction", &Cfg128::sparse_olsen_correction)
        .def_readwrite("sparse_thick_restart_size", &Cfg128::sparse_thick_restart_size)
        .def_readwrite("sparse_momentum_beta", &Cfg128::sparse_momentum_beta)
        .def_readwrite("sparse_K_schedule", &Cfg128::sparse_K_schedule)
        .def_readwrite("sparse_final_full_matvec", &Cfg128::sparse_final_full_matvec)
        .def_readwrite("perturbative_warmstart", &Cfg128::perturbative_warmstart)
        .def_readwrite("no_warm_start", &Cfg128::no_warm_start)
        .def_readwrite("orbital_optimization", &Cfg128::orbital_optimization)
        .def_readwrite("orbital_opt_max_iter", &Cfg128::orbital_opt_max_iter)
        .def_readwrite("orbital_opt_grad_tol", &Cfg128::orbital_opt_grad_tol)
        .def_readwrite("orbital_opt_energy_tol", &Cfg128::orbital_opt_energy_tol)
        .def_readwrite("dressed_energy", &Cfg128::dressed_energy)
        .def_readwrite("dressed_sc_max_iter", &Cfg128::dressed_sc_max_iter)
        .def_readwrite("dressed_sc_energy_tol", &Cfg128::dressed_sc_energy_tol)
        .def_readwrite("evaluate_only", &Cfg128::evaluate_only)
        .def_readwrite("pool_build_only", &Cfg128::pool_build_only)
        .def_readwrite("verbose", &Cfg128::verbose)
        .def_readwrite("log_file", &Cfg128::log_file)
        .def_readwrite("checkpoint_dir", &Cfg128::checkpoint_dir)
        .def_readwrite("checkpoint_round_offset", &Cfg128::checkpoint_round_offset)
        .def_property("pt2_config",
            [](const Cfg128& c) -> py::object {
                if (c.pt2_config.has_value()) return py::cast(c.pt2_config.value());
                return py::none();
            },
            [](Cfg128& c, py::object val) {
                if (val.is_none()) c.pt2_config = std::nullopt;
                else c.pt2_config = val.cast<PT2Config<std::array<uint64_t, 2>>>();
            })
        .def_property("dressed_ci_config",
            [](const Cfg128& c) -> py::object {
                if (c.dressed_ci_config.has_value()) return py::cast(c.dressed_ci_config.value());
                return py::none();
            },
            [](Cfg128& c, py::object val) {
                if (val.is_none()) c.dressed_ci_config = std::nullopt;
                else c.dressed_ci_config = val.cast<DressedCIConfig>();
            });

    py::class_<Res128>(fe_mod, "ExpansionResult128")
        .def_readonly("energy_var", &Res128::energy_var)
        .def_readonly("ffm_wall_time", &Res128::ffm_wall_time)
        .def_readonly("n_rounds", &Res128::n_rounds)
        .def_readonly("energy_history", &Res128::energy_history)
        .def_readonly("ndets_history", &Res128::ndets_history)
        .def_readonly("pt2_history", &Res128::pt2_history)
        .def_readonly("variance_ext_history", &Res128::variance_ext_history)
        .def_readonly("screening_error_history", &Res128::screening_error_history)
        .def_readonly("dressed_energy_history", &Res128::dressed_energy_history)
        .def_readonly("dressed_pt2_energy_history", &Res128::dressed_pt2_energy_history)
        .def("get_alpha_beta", [](const Res128& res)
            -> std::pair<py::array_t<uint64_t>, py::array_t<uint64_t>>
        {
            size_t N = res.dets.size();
            py::array_t<uint64_t> alpha({(int)N, 2}), beta({(int)N, 2});
            auto a = alpha.mutable_unchecked<2>();
            auto b = beta.mutable_unchecked<2>();
            for (size_t i = 0; i < N; ++i) {
                a(i, 0) = res.dets[i].alpha[0];
                a(i, 1) = res.dets[i].alpha[1];
                b(i, 0) = res.dets[i].beta[0];
                b(i, 1) = res.dets[i].beta[1];
            }
            return {alpha, beta};
        }, "Get alpha and beta arrays (N x 2) from final 128-bit determinants.")
        .def("get_coefficients", [](const Res128& res)
            -> py::array_t<double>
        {
            size_t N = res.coefficients.size();
            py::array_t<double> arr(N);
            auto r = arr.mutable_unchecked<1>();
            for (size_t i = 0; i < N; ++i) r(i) = res.coefficients[i];
            return arr;
        }, "Get coefficient array.")
        .def("get_U_total", [](const Res128& res)
            -> py::array_t<double>
        {
            if (res.U_total.empty()) return py::array_t<double>();
            size_t n = (size_t)std::sqrt((double)res.U_total.size());
            py::array_t<double> arr({n, n});
            std::memcpy(arr.mutable_data(), res.U_total.data(), n * n * sizeof(double));
            return arr;
        }, "Get cumulative orbital rotation matrix.")
        .def("get_h1_optimized", [](const Res128& res)
            -> py::array_t<double>
        {
            if (res.h1_optimized.empty()) return py::array_t<double>();
            size_t n = res.h1_optimized.size();
            py::array_t<double> arr({n, n});
            auto r = arr.mutable_unchecked<2>();
            for (size_t i = 0; i < n; ++i)
                for (size_t j = 0; j < n; ++j)
                    r(i, j) = res.h1_optimized[i][j];
            return arr;
        }, "Get optimized one-electron integrals.")
        .def("get_eri_optimized", [](const Res128& res)
            -> py::array_t<double>
        {
            if (res.eri_optimized.empty()) return py::array_t<double>();
            size_t n4 = res.eri_optimized.size();
            py::array_t<double> arr(n4);
            std::memcpy(arr.mutable_data(), res.eri_optimized.data(), n4 * sizeof(double));
            return arr;
        }, "Get optimized two-electron integrals.");

    // PT2Config128 (layout-identical to PT2Config<uint64_t>, but separate C++ type)
    using PT2Cfg128 = PT2Config<std::array<uint64_t, 2>>;
    py::class_<PT2Cfg128>(fe_mod, "PT2Config128")
        .def(py::init<>())
        .def_readwrite("sketch_width", &PT2Cfg128::sketch_width)
        .def_readwrite("sketch_depth", &PT2Cfg128::sketch_depth)
        .def_readwrite("sketch_seed", &PT2Cfg128::sketch_seed)
        .def_readwrite("bloom_fp_rate", &PT2Cfg128::bloom_fp_rate)
        .def_readwrite("hll_precision", &PT2Cfg128::hll_precision)
        .def_readwrite("intruder_threshold", &PT2Cfg128::intruder_threshold)
        .def_readwrite("n_deterministic", &PT2Cfg128::n_deterministic)
        .def_readwrite("deterministic_fraction", &PT2Cfg128::deterministic_fraction)
        .def_readwrite("norm_target", &PT2Cfg128::norm_target)
        .def_readwrite("stochastic_fraction", &PT2Cfg128::stochastic_fraction)
        .def_readwrite("n_stochastic_samples", &PT2Cfg128::n_stochastic_samples)
        .def_readwrite("stochastic_seed", &PT2Cfg128::stochastic_seed)
        .def_readwrite("eps_hc_filter", &PT2Cfg128::eps_hc_filter)
        .def_readwrite("screening_error_target", &PT2Cfg128::screening_error_target)
        .def_readwrite("eps_pt_shci", &PT2Cfg128::eps_pt_shci)
        .def_readwrite("dressed_sketch_width", &PT2Cfg128::dressed_sketch_width)
        .def_readwrite("dressed_build_norm_target", &PT2Cfg128::dressed_build_norm_target)
        .def_readwrite("dressed_query_norm_target", &PT2Cfg128::dressed_query_norm_target)
        .def_readwrite("dressed_davidson_energy_tol", &PT2Cfg128::dressed_davidson_energy_tol)
        .def_readwrite("psf_max_memory_gb", &PT2Cfg128::psf_max_memory_gb)
        .def_readwrite("verbose", &PT2Cfg128::verbose)
        .def_readwrite("sketch_save_path", &PT2Cfg128::sketch_save_path);

    // run_expansion 128-bit
    fe_mod.def("run_expansion_128", [](
        py::array_t<uint64_t, py::array::c_style> alpha_arr,
        py::array_t<uint64_t, py::array::c_style> beta_arr,
        py::array_t<double, py::array::c_style> coeffs_arr,
        py::array_t<double, py::array::c_style> h1_arr,
        py::array_t<double, py::array::c_style> eri_arr,
        int n_orb,
        Cfg128 config) -> Res128
    {
        auto a = alpha_arr.unchecked<2>();  // (N, 2)
        auto b = beta_arr.unchecked<2>();   // (N, 2)
        auto c = coeffs_arr.unchecked<1>();
        size_t N = a.shape(0);

        std::vector<Det128> dets(N);
        for (size_t i = 0; i < N; ++i) {
            dets[i] = Det128({a(i, 0), a(i, 1)}, {b(i, 0), b(i, 1)});
        }

        std::vector<double> coeffs(N);
        for (size_t i = 0; i < N; ++i) coeffs[i] = c(i);

        auto h1_2d = h1_arr.unchecked<2>();
        std::vector<std::vector<double>> h1(n_orb, std::vector<double>(n_orb));
        for (int i = 0; i < n_orb; ++i)
            for (int j = 0; j < n_orb; ++j)
                h1[i][j] = h1_2d(i, j);

        auto eri_buf = eri_arr.request();
        std::vector<double> eri(static_cast<double*>(eri_buf.ptr),
                                static_cast<double*>(eri_buf.ptr) + eri_buf.size);

        Res128 result;
        {
            py::gil_scoped_release release;
            result = run_expansion<std::array<uint64_t, 2>>(dets, coeffs, h1, eri, n_orb, config);
        }
        return result;
    },
    py::arg("alpha"), py::arg("beta"), py::arg("coefficients"),
    py::arg("h1"), py::arg("eri"), py::arg("n_orb"),
    py::arg("config") = Cfg128{},
    "Run iterative expansion with 128-bit determinants (n_orb > 64).");

    // run_expansion_phased 128-bit
    fe_mod.def("run_expansion_phased_128", [](
        py::array_t<uint64_t, py::array::c_style> alpha_arr,
        py::array_t<uint64_t, py::array::c_style> beta_arr,
        py::array_t<double, py::array::c_style> coeffs_arr,
        py::array_t<double, py::array::c_style> h1_arr,
        py::array_t<double, py::array::c_style> eri_arr,
        int n_orb,
        std::vector<Cfg128> phase_configs) -> Res128
    {
        auto a = alpha_arr.unchecked<2>();  // (N, 2)
        auto b = beta_arr.unchecked<2>();   // (N, 2)
        auto c = coeffs_arr.unchecked<1>();
        size_t N = a.shape(0);

        std::vector<Det128> dets(N);
        for (size_t i = 0; i < N; ++i) {
            dets[i] = Det128({a(i, 0), a(i, 1)}, {b(i, 0), b(i, 1)});
        }

        std::vector<double> coeffs(N);
        for (size_t i = 0; i < N; ++i) coeffs[i] = c(i);

        auto h1_2d = h1_arr.unchecked<2>();
        std::vector<std::vector<double>> h1(n_orb, std::vector<double>(n_orb));
        for (int i = 0; i < n_orb; ++i)
            for (int j = 0; j < n_orb; ++j)
                h1[i][j] = h1_2d(i, j);

        auto eri_buf = eri_arr.request();
        std::vector<double> eri(static_cast<double*>(eri_buf.ptr),
                                static_cast<double*>(eri_buf.ptr) + eri_buf.size);

        Res128 result;
        {
            py::gil_scoped_release release;
            result = run_expansion_phased<std::array<uint64_t, 2>>(dets, coeffs, h1, eri, n_orb, phase_configs);
        }
        return result;
    },
    py::arg("alpha"), py::arg("beta"), py::arg("coefficients"),
    py::arg("h1"), py::arg("eri"), py::arg("n_orb"),
    py::arg("phase_configs"),
    "Run multi-phase expansion with 128-bit determinants (n_orb > 64).");

    // ========================================================================
    // BloomFilter
    // ========================================================================
    py::class_<BloomFilter<uint64_t>>(fe_mod, "BloomFilter")
        .def(py::init<>())
        .def("init", &BloomFilter<uint64_t>::init,
             py::arg("n_items"), py::arg("false_positive_rate") = 0.001)
        .def("memory_bytes", &BloomFilter<uint64_t>::memory_bytes);

    // ========================================================================
    // PT2Config & PT2Result
    // ========================================================================
    py::class_<PT2Config<uint64_t>>(fe_mod, "PT2Config")
        .def(py::init<>())
        .def_readwrite("sketch_width", &PT2Config<uint64_t>::sketch_width)
        .def_readwrite("sketch_depth", &PT2Config<uint64_t>::sketch_depth)
        .def_readwrite("sketch_seed", &PT2Config<uint64_t>::sketch_seed)
        .def_readwrite("bloom_fp_rate", &PT2Config<uint64_t>::bloom_fp_rate)
        .def_readwrite("hll_precision", &PT2Config<uint64_t>::hll_precision)
        .def_readwrite("intruder_threshold", &PT2Config<uint64_t>::intruder_threshold)
        .def_readwrite("n_deterministic", &PT2Config<uint64_t>::n_deterministic)
        .def_readwrite("deterministic_fraction", &PT2Config<uint64_t>::deterministic_fraction)
        .def_readwrite("norm_target", &PT2Config<uint64_t>::norm_target)
        .def_readwrite("stochastic_fraction", &PT2Config<uint64_t>::stochastic_fraction)
        .def_readwrite("n_stochastic_samples", &PT2Config<uint64_t>::n_stochastic_samples)
        .def_readwrite("stochastic_seed", &PT2Config<uint64_t>::stochastic_seed)
        .def_readwrite("eps_hc_filter", &PT2Config<uint64_t>::eps_hc_filter)
        .def_readwrite("screening_error_target", &PT2Config<uint64_t>::screening_error_target)
        .def_readwrite("eps_pt_shci", &PT2Config<uint64_t>::eps_pt_shci)
        .def_readwrite("dressed_sketch_width", &PT2Config<uint64_t>::dressed_sketch_width)
        .def_readwrite("dressed_build_norm_target", &PT2Config<uint64_t>::dressed_build_norm_target)
        .def_readwrite("dressed_query_norm_target", &PT2Config<uint64_t>::dressed_query_norm_target)
        .def_readwrite("dressed_davidson_energy_tol", &PT2Config<uint64_t>::dressed_davidson_energy_tol)
        .def_readwrite("psf_max_memory_gb", &PT2Config<uint64_t>::psf_max_memory_gb)
        .def_readwrite("verbose", &PT2Config<uint64_t>::verbose)
        .def_readwrite("sketch_save_path", &PT2Config<uint64_t>::sketch_save_path);

    py::class_<PT2Result<uint64_t>>(fe_mod, "PT2Result")
        .def_readonly("energy_pt2", &PT2Result<uint64_t>::energy_pt2)
        .def_readonly("delta_pt2", &PT2Result<uint64_t>::delta_pt2)
        .def_readonly("l2_norm_sq", &PT2Result<uint64_t>::l2_norm_sq)
        .def_readonly("variance_ext", &PT2Result<uint64_t>::variance_ext)
        .def_readonly("sum_pt1_sq", &PT2Result<uint64_t>::sum_pt1_sq)
        .def_readonly("n_pairs_processed", &PT2Result<uint64_t>::n_pairs_processed)
        .def_readonly("n_bloom_filtered", &PT2Result<uint64_t>::n_bloom_filtered)
        .def_readonly("n_intruder_skipped", &PT2Result<uint64_t>::n_intruder_skipped)
        .def_readonly("n_integral_screened", &PT2Result<uint64_t>::n_integral_screened)
        .def_readonly("screening_error_estimate", &PT2Result<uint64_t>::screening_error_estimate)
        .def_readonly("n_unique_external_est", &PT2Result<uint64_t>::n_unique_external_est)
        .def_readonly("avg_overlap_k", &PT2Result<uint64_t>::avg_overlap_k)
        .def_readonly("sketch_memory_mb", &PT2Result<uint64_t>::sketch_memory_mb)
        .def_readonly("bloom_memory_mb", &PT2Result<uint64_t>::bloom_memory_mb)
        .def_readonly("time_seconds", &PT2Result<uint64_t>::time_seconds);

    // ========================================================================
    // compute_pt2 (Main: dual-sketch semistochastic PT2)
    // ========================================================================
    fe_mod.def("compute_pt2", [](
        py::array_t<uint64_t, py::array::c_style> alpha_arr,
        py::array_t<uint64_t, py::array::c_style> beta_arr,
        py::array_t<double, py::array::c_style> coeffs_arr,
        double energy_var,
        py::array_t<double, py::array::c_style> h1_arr,
        py::array_t<double, py::array::c_style> eri_arr,
        int n_orb,
        PT2Config<uint64_t> config) -> PT2Result<uint64_t>
    {
        auto a = alpha_arr.unchecked<1>();
        auto b = beta_arr.unchecked<1>();
        auto c = coeffs_arr.unchecked<1>();
        size_t N = a.shape(0);

        std::vector<DeterminantT<uint64_t>> dets(N);
        for (size_t i = 0; i < N; ++i) {
            dets[i] = DeterminantT<uint64_t>(a(i), b(i));
        }

        std::vector<double> coeffs(N);
        for (size_t i = 0; i < N; ++i) coeffs[i] = c(i);

        auto h1_2d = h1_arr.unchecked<2>();
        std::vector<std::vector<double>> h1(n_orb, std::vector<double>(n_orb));
        for (int i = 0; i < n_orb; ++i)
            for (int j = 0; j < n_orb; ++j)
                h1[i][j] = h1_2d(i, j);

        auto eri_buf = eri_arr.request();
        std::vector<double> eri(static_cast<double*>(eri_buf.ptr),
                                static_cast<double*>(eri_buf.ptr) + eri_buf.size);

        PT2Result<uint64_t> result;
        {
            py::gil_scoped_release release;
            result = compute_pt2<uint64_t>(dets, coeffs, energy_var,
                                           h1, eri, n_orb, config);
        }
        return result;
    },
    py::arg("alpha"), py::arg("beta"), py::arg("coefficients"),
    py::arg("energy_var"),
    py::arg("h1"), py::arg("eri"), py::arg("n_orb"),
    py::arg("config") = PT2Config<uint64_t>{},
    "Main PT2: dual-sketch semistochastic PT2 with importance sampling. "
    "Set deterministic_fraction=0 for fully deterministic mode.");

    // ========================================================================
    // compute_pt2_exact_for_experiment (hash-table PT2 for validation)
    // ========================================================================
    fe_mod.def("compute_pt2_exact_for_experiment", [](
        py::array_t<uint64_t, py::array::c_style> alpha_arr,
        py::array_t<uint64_t, py::array::c_style> beta_arr,
        py::array_t<double, py::array::c_style> coeffs_arr,
        double energy_var,
        py::array_t<double, py::array::c_style> h1_arr,
        py::array_t<double, py::array::c_style> eri_arr,
        int n_orb,
        PT2Config<uint64_t> config) -> PT2Result<uint64_t>
    {
        auto a = alpha_arr.unchecked<1>();
        auto b = beta_arr.unchecked<1>();
        auto c = coeffs_arr.unchecked<1>();
        size_t N = a.shape(0);

        std::vector<DeterminantT<uint64_t>> dets(N);
        for (size_t i = 0; i < N; ++i) {
            dets[i] = DeterminantT<uint64_t>(a(i), b(i));
        }

        std::vector<double> coeffs(N);
        for (size_t i = 0; i < N; ++i) coeffs[i] = c(i);

        auto h1_2d = h1_arr.unchecked<2>();
        std::vector<std::vector<double>> h1(n_orb, std::vector<double>(n_orb));
        for (int i = 0; i < n_orb; ++i)
            for (int j = 0; j < n_orb; ++j)
                h1[i][j] = h1_2d(i, j);

        auto eri_buf = eri_arr.request();
        std::vector<double> eri(static_cast<double*>(eri_buf.ptr),
                                static_cast<double*>(eri_buf.ptr) + eri_buf.size);

        PT2Result<uint64_t> result;
        {
            py::gil_scoped_release release;
            result = compute_pt2_exact_for_experiment<uint64_t>(dets, coeffs, energy_var,
                                                  h1, eri, n_orb, config);
        }
        return result;
    },
    py::arg("alpha"), py::arg("beta"), py::arg("coefficients"),
    py::arg("energy_var"),
    py::arg("h1"), py::arg("eri"), py::arg("n_orb"),
    py::arg("config") = PT2Config<uint64_t>{},
    "[Experiment] Exact (hash-table) PT2 energy correction for validation.");

    // ========================================================================
    // compute_pt2_shci_for_experiment (SHCI integral screening for ablation)
    // ========================================================================
    fe_mod.def("compute_pt2_shci_for_experiment", [](
        py::array_t<uint64_t, py::array::c_style> alpha_arr,
        py::array_t<uint64_t, py::array::c_style> beta_arr,
        py::array_t<double, py::array::c_style> coeffs_arr,
        double energy_var,
        py::array_t<double, py::array::c_style> h1_arr,
        py::array_t<double, py::array::c_style> eri_arr,
        int n_orb,
        PT2Config<uint64_t> config) -> PT2Result<uint64_t>
    {
        auto a = alpha_arr.unchecked<1>();
        auto b = beta_arr.unchecked<1>();
        auto c = coeffs_arr.unchecked<1>();
        size_t N = a.shape(0);

        std::vector<DeterminantT<uint64_t>> dets(N);
        for (size_t i = 0; i < N; ++i) {
            dets[i] = DeterminantT<uint64_t>(a(i), b(i));
        }

        std::vector<double> coeffs(N);
        for (size_t i = 0; i < N; ++i) coeffs[i] = c(i);

        auto h1_2d = h1_arr.unchecked<2>();
        std::vector<std::vector<double>> h1(n_orb, std::vector<double>(n_orb));
        for (int i = 0; i < n_orb; ++i)
            for (int j = 0; j < n_orb; ++j)
                h1[i][j] = h1_2d(i, j);

        auto eri_buf = eri_arr.request();
        std::vector<double> eri(static_cast<double*>(eri_buf.ptr),
                                static_cast<double*>(eri_buf.ptr) + eri_buf.size);

        PT2Result<uint64_t> result;
        {
            py::gil_scoped_release release;
            result = compute_pt2_shci_for_experiment<uint64_t>(
                dets, coeffs, energy_var, h1, eri, n_orb, config);
        }
        return result;
    },
    py::arg("alpha"), py::arg("beta"), py::arg("coefficients"),
    py::arg("energy_var"),
    py::arg("h1"), py::arg("eri"), py::arg("n_orb"),
    py::arg("config") = PT2Config<uint64_t>{},
    "[Experiment] SHCI-style PT2 with integral-level screening for ablation.");

    // ========================================================================
    // PT3Config & PT3Result  [EXPERIMENTAL - series diverges, see enpt3.hpp]
    // ========================================================================
    py::class_<PT3Config>(fe_mod, "PT3Config")
        .def(py::init<>())
        .def_readwrite("intruder_threshold", &PT3Config::intruder_threshold)
        .def_readwrite("verbose", &PT3Config::verbose);

    py::class_<PT3Result<uint64_t>>(fe_mod, "PT3Result")
        .def_readonly("delta_pt2", &PT3Result<uint64_t>::delta_pt2)
        .def_readonly("delta_pt2_formal", &PT3Result<uint64_t>::delta_pt2_formal)
        .def_readonly("delta_pt3", &PT3Result<uint64_t>::delta_pt3)
        .def_readonly("e_first_order", &PT3Result<uint64_t>::e_first_order)
        .def_readonly("cross_term", &PT3Result<uint64_t>::cross_term)
        .def_readonly("diag_correction", &PT3Result<uint64_t>::diag_correction)
        .def_readonly("sum_f_sq", &PT3Result<uint64_t>::sum_f_sq)
        .def_readonly("n_external", &PT3Result<uint64_t>::n_external)
        .def_readonly("n_connected_pairs", &PT3Result<uint64_t>::n_connected_pairs)
        .def_readonly("n_intruder_skipped", &PT3Result<uint64_t>::n_intruder_skipped)
        .def_readonly("time_pass1", &PT3Result<uint64_t>::time_pass1)
        .def_readonly("time_pass2", &PT3Result<uint64_t>::time_pass2);

    // ========================================================================
    // compute_enpt3  [EXPERIMENTAL - series diverges, see enpt3.hpp]
    // ========================================================================
    fe_mod.def("compute_enpt3", [](
        py::array_t<uint64_t, py::array::c_style> alpha_arr,
        py::array_t<uint64_t, py::array::c_style> beta_arr,
        py::array_t<double, py::array::c_style> coeffs_arr,
        double energy_var,
        py::array_t<double, py::array::c_style> h1_arr,
        py::array_t<double, py::array::c_style> eri_arr,
        int n_orb,
        PT3Config config) -> PT3Result<uint64_t>
    {
        auto a = alpha_arr.unchecked<1>();
        auto b = beta_arr.unchecked<1>();
        auto c = coeffs_arr.unchecked<1>();
        size_t N = a.shape(0);

        std::vector<DeterminantT<uint64_t>> dets(N);
        for (size_t i = 0; i < N; ++i) {
            dets[i] = DeterminantT<uint64_t>(a(i), b(i));
        }

        std::vector<double> coeffs(N);
        for (size_t i = 0; i < N; ++i) coeffs[i] = c(i);

        auto h1_2d = h1_arr.unchecked<2>();
        std::vector<std::vector<double>> h1(n_orb, std::vector<double>(n_orb));
        for (int i = 0; i < n_orb; ++i)
            for (int j = 0; j < n_orb; ++j)
                h1[i][j] = h1_2d(i, j);

        auto eri_buf = eri_arr.request();
        std::vector<double> eri(static_cast<double*>(eri_buf.ptr),
                                static_cast<double*>(eri_buf.ptr) + eri_buf.size);

        PT3Result<uint64_t> result;
        {
            py::gil_scoped_release release;
            result = compute_enpt3<uint64_t>(dets, coeffs, energy_var,
                                              h1, eri, n_orb, config);
        }
        return result;
    },
    py::arg("alpha"), py::arg("beta"), py::arg("coefficients"),
    py::arg("energy_var"),
    py::arg("h1"), py::arg("eri"), py::arg("n_orb"),
    py::arg("config") = PT3Config{},
    "Compute EN-PT3 (third-order Epstein-Nesbet) energy correction.");

    // ========================================================================
    // DressedCIConfig & DressedCIResult  [Sketch-based Dressed CI]
    // ========================================================================
    py::class_<DressedCIConfig>(fe_mod, "DressedCIConfig")
        .def(py::init<>())
        .def_readwrite("sketch_width", &DressedCIConfig::sketch_width)
        .def_readwrite("sketch_depth", &DressedCIConfig::sketch_depth)
        .def_readwrite("sketch_seed", &DressedCIConfig::sketch_seed)
        .def_readwrite("intruder_threshold", &DressedCIConfig::intruder_threshold)
        .def_readwrite("verbose", &DressedCIConfig::verbose)
        .def_readwrite("davidson_params", &DressedCIConfig::davidson_params)
        .def_readwrite("compute_exact_K", &DressedCIConfig::compute_exact_K)
        .def_readwrite("compute_pt2_dressed", &DressedCIConfig::compute_pt2_dressed)
        .def_readwrite("memory_budget_mb", &DressedCIConfig::memory_budget_mb)
        .def_readwrite("auto_tune_w_min", &DressedCIConfig::auto_tune_w_min)
        .def_readwrite("auto_tune_w_max", &DressedCIConfig::auto_tune_w_max)
        .def_readwrite("export_external_data", &DressedCIConfig::export_external_data);

    py::class_<DressedCIResult<uint64_t>>(fe_mod, "DressedCIResult")
        .def_readonly("cKc_sketch", &DressedCIResult<uint64_t>::cKc_sketch)
        .def_readonly("delta_pt2", &DressedCIResult<uint64_t>::delta_pt2)
        .def_readonly("variance_ext", &DressedCIResult<uint64_t>::variance_ext)
        .def_readonly("n_external", &DressedCIResult<uint64_t>::n_external)
        .def_readonly("n_intruder_skipped", &DressedCIResult<uint64_t>::n_intruder_skipped)
        .def_readonly("sketch_memory_mb", &DressedCIResult<uint64_t>::sketch_memory_mb)
        .def_readonly("cKc_exact", &DressedCIResult<uint64_t>::cKc_exact)
        .def_readonly("K_sketch_rel_error", &DressedCIResult<uint64_t>::K_sketch_rel_error)
        .def_property_readonly("K_exact_matrix", [](const DressedCIResult<uint64_t>& r) {
            if (r.K_exact_matrix.empty()) return py::array_t<double>();
            size_t n = static_cast<size_t>(std::sqrt(r.K_exact_matrix.size()));
            return py::array_t<double>({n, n}, r.K_exact_matrix.data());
        })
        .def_property_readonly("K_sketch_matrix", [](const DressedCIResult<uint64_t>& r) {
            if (r.K_sketch_matrix.empty()) return py::array_t<double>();
            size_t n = static_cast<size_t>(std::sqrt(r.K_sketch_matrix.size()));
            return py::array_t<double>({n, n}, r.K_sketch_matrix.data());
        })
        .def_readonly("energy_dressed", &DressedCIResult<uint64_t>::energy_dressed)
        .def_readonly("energy_var", &DressedCIResult<uint64_t>::energy_var)
        .def_readonly("energy_improvement", &DressedCIResult<uint64_t>::energy_improvement)
        .def_readonly("coefficients_dressed", &DressedCIResult<uint64_t>::coefficients_dressed)
        .def_readonly("davidson_iters", &DressedCIResult<uint64_t>::davidson_iters)
        .def_readonly("davidson_converged", &DressedCIResult<uint64_t>::davidson_converged)
        .def_readonly("cKc_dressed", &DressedCIResult<uint64_t>::cKc_dressed)
        .def_readonly("energy_var_dressed", &DressedCIResult<uint64_t>::energy_var_dressed)
        .def_readonly("delta_pt2_dressed", &DressedCIResult<uint64_t>::delta_pt2_dressed)
        .def_property_readonly("external_H_aa", [](const DressedCIResult<uint64_t>& r) {
            if (r.external_H_aa.empty()) return py::array_t<double>();
            return py::array_t<double>(r.external_H_aa.size(), r.external_H_aa.data());
        })
        .def_property_readonly("coupling_ext_idx", [](const DressedCIResult<uint64_t>& r) {
            if (r.coupling_ext_idx.empty()) return py::array_t<int32_t>();
            return py::array_t<int32_t>(r.coupling_ext_idx.size(), r.coupling_ext_idx.data());
        })
        .def_property_readonly("coupling_core_idx", [](const DressedCIResult<uint64_t>& r) {
            if (r.coupling_core_idx.empty()) return py::array_t<int32_t>();
            return py::array_t<int32_t>(r.coupling_core_idx.size(), r.coupling_core_idx.data());
        })
        .def_property_readonly("coupling_H_ia", [](const DressedCIResult<uint64_t>& r) {
            if (r.coupling_H_ia.empty()) return py::array_t<double>();
            return py::array_t<double>(r.coupling_H_ia.size(), r.coupling_H_ia.data());
        })
        .def_readonly("time_sketches", &DressedCIResult<uint64_t>::time_sketches)
        .def_readonly("time_exact_K", &DressedCIResult<uint64_t>::time_exact_K)
        .def_readonly("time_davidson", &DressedCIResult<uint64_t>::time_davidson)
        .def_readonly("time_pt2_dressed", &DressedCIResult<uint64_t>::time_pt2_dressed)
        .def_readonly("time_total", &DressedCIResult<uint64_t>::time_total);

    // ========================================================================
    // compute_dressed_ci
    // ========================================================================
    fe_mod.def("compute_dressed_ci", [](
        py::array_t<uint64_t, py::array::c_style> alpha_arr,
        py::array_t<uint64_t, py::array::c_style> beta_arr,
        py::array_t<double, py::array::c_style> coeffs_arr,
        double energy_var,
        py::array_t<double, py::array::c_style> h1_arr,
        py::array_t<double, py::array::c_style> eri_arr,
        int n_orb,
        DressedCIConfig config) -> DressedCIResult<uint64_t>
    {
        auto a = alpha_arr.unchecked<1>();
        auto b = beta_arr.unchecked<1>();
        auto c = coeffs_arr.unchecked<1>();
        size_t N = a.shape(0);

        std::vector<DeterminantT<uint64_t>> dets(N);
        for (size_t i = 0; i < N; ++i) {
            dets[i] = DeterminantT<uint64_t>(a(i), b(i));
        }

        std::vector<double> coeffs(N);
        for (size_t i = 0; i < N; ++i) coeffs[i] = c(i);

        auto h1_2d = h1_arr.unchecked<2>();
        std::vector<std::vector<double>> h1(n_orb, std::vector<double>(n_orb));
        for (int i = 0; i < n_orb; ++i)
            for (int j = 0; j < n_orb; ++j)
                h1[i][j] = h1_2d(i, j);

        auto eri_buf = eri_arr.request();
        std::vector<double> eri(static_cast<double*>(eri_buf.ptr),
                                static_cast<double*>(eri_buf.ptr) + eri_buf.size);

        DressedCIResult<uint64_t> result;
        {
            py::gil_scoped_release release;
            result = compute_dressed_ci<uint64_t>(dets, coeffs, energy_var,
                                                   h1, eri, n_orb, config);
        }
        return result;
    },
    py::arg("alpha"), py::arg("beta"), py::arg("coefficients"),
    py::arg("energy_var"),
    py::arg("h1"), py::arg("eri"), py::arg("n_orb"),
    py::arg("config") = DressedCIConfig{},
    "Sketch-based Dressed CI: per-det sketches + dressed Davidson.");

    // ========================================================================
    // AB group-level matvec (for distributed Davidson)
    // ========================================================================

    // Ch1: Same-α (Sβ + Dββ)
    fe_mod.def("matvec_same_alpha_group", [](
        uint64_t alpha,
        py::array_t<uint64_t, py::array::c_style> betas_arr,
        py::array_t<double, py::array::c_style> v_group_arr,
        py::array_t<double, py::array::c_style> h1_arr,
        py::array_t<double, py::array::c_style> eri_arr,
        int n_orb) -> py::array_t<double>
    {
        auto betas = betas_arr.unchecked<1>();
        auto v_in = v_group_arr.unchecked<1>();
        size_t group_size = betas.shape(0);

        auto h1_2d = h1_arr.unchecked<2>();
        std::vector<std::vector<double>> h1(n_orb, std::vector<double>(n_orb));
        for (int i = 0; i < n_orb; ++i)
            for (int j = 0; j < n_orb; ++j)
                h1[i][j] = h1_2d(i, j);

        const double* eri_data = eri_arr.data();

        py::array_t<double> result(group_size);
        auto r = result.mutable_unchecked<1>();
        for (size_t i = 0; i < group_size; ++i) r(i) = 0.0;

        {
            py::gil_scoped_release release;
            matvec_same_alpha_group(
                alpha, betas.data(0), group_size,
                v_in.data(0), r.mutable_data(0),
                h1, eri_data, n_orb);
        }
        return result;
    },
    py::arg("alpha"), py::arg("sorted_betas"), py::arg("v_group"),
    py::arg("h1"), py::arg("eri"), py::arg("n_orb"),
    "Ch1: Same-α group matvec (Sβ + Dββ). Returns sigma_group.");

    // Ch3: Same-β (Sα + Dαα)
    fe_mod.def("matvec_same_beta_group", [](
        uint64_t beta,
        py::array_t<uint64_t, py::array::c_style> alphas_arr,
        py::array_t<double, py::array::c_style> v_group_arr,
        py::array_t<double, py::array::c_style> h1_arr,
        py::array_t<double, py::array::c_style> eri_arr,
        int n_orb) -> py::array_t<double>
    {
        auto alphas = alphas_arr.unchecked<1>();
        auto v_in = v_group_arr.unchecked<1>();
        size_t group_size = alphas.shape(0);

        auto h1_2d = h1_arr.unchecked<2>();
        std::vector<std::vector<double>> h1(n_orb, std::vector<double>(n_orb));
        for (int i = 0; i < n_orb; ++i)
            for (int j = 0; j < n_orb; ++j)
                h1[i][j] = h1_2d(i, j);

        const double* eri_data = eri_arr.data();

        py::array_t<double> result(group_size);
        auto r = result.mutable_unchecked<1>();
        for (size_t i = 0; i < group_size; ++i) r(i) = 0.0;

        {
            py::gil_scoped_release release;
            matvec_same_beta_group(
                beta, alphas.data(0), group_size,
                v_in.data(0), r.mutable_data(0),
                h1, eri_data, n_orb);
        }
        return result;
    },
    py::arg("beta"), py::arg("sorted_alphas"), py::arg("v_group"),
    py::arg("h1"), py::arg("eri"), py::arg("n_orb"),
    "Ch3: Same-β group matvec (Sα + Dαα). Returns sigma_group.");

    // Ch2: Mixed Dαβ (one pair)
    fe_mod.def("matvec_mixed_pair", [](
        uint64_t alpha_self,
        py::array_t<uint64_t, py::array::c_style> betas_self_arr,
        uint64_t alpha_partner,
        py::array_t<uint64_t, py::array::c_style> betas_partner_arr,
        py::array_t<double, py::array::c_style> v_partner_arr,
        py::array_t<double, py::array::c_style> h1_arr,
        py::array_t<double, py::array::c_style> eri_arr,
        int n_orb) -> py::array_t<double>
    {
        auto betas_self = betas_self_arr.unchecked<1>();
        auto betas_partner = betas_partner_arr.unchecked<1>();
        auto v_partner = v_partner_arr.unchecked<1>();
        size_t self_size = betas_self.shape(0);
        size_t partner_size = betas_partner.shape(0);

        auto h1_2d = h1_arr.unchecked<2>();
        std::vector<std::vector<double>> h1(n_orb, std::vector<double>(n_orb));
        for (int i = 0; i < n_orb; ++i)
            for (int j = 0; j < n_orb; ++j)
                h1[i][j] = h1_2d(i, j);

        const double* eri_data = eri_arr.data();

        py::array_t<double> result(self_size);
        auto r = result.mutable_unchecked<1>();
        for (size_t i = 0; i < self_size; ++i) r(i) = 0.0;

        {
            py::gil_scoped_release release;
            matvec_mixed_pair(
                alpha_self, betas_self.data(0), self_size,
                r.mutable_data(0),
                alpha_partner, betas_partner.data(0), partner_size,
                v_partner.data(0),
                h1, eri_data, n_orb);
        }
        return result;
    },
    py::arg("alpha_self"), py::arg("betas_self"),
    py::arg("alpha_partner"), py::arg("betas_partner"),
    py::arg("v_partner"),
    py::arg("h1"), py::arg("eri"), py::arg("n_orb"),
    "Ch2: Mixed Dαβ pair matvec. Returns sigma contribution for self group.");
}
