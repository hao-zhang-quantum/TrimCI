// Bindings for direct builders and Davidson solvers
#include "bind_common.hpp"
#include <cstring>

#include "direct_s_builder.hpp"
#include "direct_h_builder.hpp"
#include "direct_h_detspace.hpp"
#include "excitation_generator.hpp"
#include "davidson_gep.hpp"
#include "matfree_davidson.hpp"
#include "alpha_adjacency.hpp"

using namespace trimci_core;

void bind_direct_builders(py::module& m) {
    // build_S_direct
    m.def("build_S_direct", [](
        py::array_t<uint64_t, py::array::c_style | py::array::forcecast> ref_alpha,
        py::array_t<uint64_t, py::array::c_style | py::array::forcecast> ref_beta,
        py::array_t<double, py::array::c_style | py::array::forcecast> ref_coeffs,
        py::array_t<int, py::array::c_style | py::array::forcecast> exc_types,
        py::array_t<int, py::array::c_style | py::array::forcecast> exc_indices,
        int n_basis) -> py::array_t<double> {

        auto ra = ref_alpha.unchecked<1>();
        auto rb = ref_beta.unchecked<1>();
        auto rc = ref_coeffs.unchecked<1>();
        auto et = exc_types.unchecked<1>();
        int n_ref = static_cast<int>(ra.shape(0));
        int n_exc = static_cast<int>(et.shape(0));

        py::array_t<double> result({n_basis, n_basis});
        double* S_ptr = result.mutable_data();
        std::memset(S_ptr, 0, static_cast<size_t>(n_basis) * n_basis * sizeof(double));
        {
            py::gil_scoped_release release;
            build_S_direct(ra.data(0), rb.data(0), rc.data(0), n_ref,
                et.data(0), exc_indices.data(), n_exc, n_basis, S_ptr);
        }
        return result;
    },
    py::arg("ref_alpha"), py::arg("ref_beta"), py::arg("ref_coeffs"),
    py::arg("exc_types"), py::arg("exc_indices"), py::arg("n_basis"),
    "Build S matrix using direct determinant expansion (C++).");

    // build_H_direct_excspace
    m.def("build_H_direct_excspace", [](
        py::array_t<uint64_t, py::array::c_style | py::array::forcecast> ref_alpha,
        py::array_t<uint64_t, py::array::c_style | py::array::forcecast> ref_beta,
        py::array_t<double, py::array::c_style | py::array::forcecast> ref_coeffs,
        py::array_t<int, py::array::c_style | py::array::forcecast> exc_types,
        py::array_t<int, py::array::c_style | py::array::forcecast> exc_indices,
        py::array_t<double, py::array::c_style | py::array::forcecast> h1,
        py::array_t<double, py::array::c_style | py::array::forcecast> eri,
        int n_orb, int n_basis) -> py::array_t<double> {

        auto ra = ref_alpha.unchecked<1>();
        auto rb = ref_beta.unchecked<1>();
        auto rc = ref_coeffs.unchecked<1>();
        auto et = exc_types.unchecked<1>();
        int n_ref = static_cast<int>(ra.shape(0));
        int n_exc = static_cast<int>(et.shape(0));

        py::array_t<double> result({n_basis, n_basis});
        double* H_ptr = result.mutable_data();
        std::memset(H_ptr, 0, static_cast<size_t>(n_basis) * n_basis * sizeof(double));
        {
            py::gil_scoped_release release;
            build_H_direct_excspace(ra.data(0), rb.data(0), rc.data(0), n_ref,
                et.data(0), exc_indices.data(), n_exc, n_basis,
                h1.data(), eri.data(), n_orb, H_ptr);
        }
        return result;
    },
    py::arg("ref_alpha"), py::arg("ref_beta"), py::arg("ref_coeffs"),
    py::arg("exc_types"), py::arg("exc_indices"),
    py::arg("h1"), py::arg("eri"), py::arg("n_orb"), py::arg("n_basis"),
    "Build H matrix using excitation-space algorithm (legacy, O(K^2 M^2)).");

    // build_H_detspace
    m.def("build_H_detspace", [](
        py::array_t<uint64_t, py::array::c_style | py::array::forcecast> ref_alpha,
        py::array_t<uint64_t, py::array::c_style | py::array::forcecast> ref_beta,
        py::array_t<double, py::array::c_style | py::array::forcecast> ref_coeffs,
        py::array_t<int, py::array::c_style | py::array::forcecast> exc_types,
        py::array_t<int, py::array::c_style | py::array::forcecast> exc_indices,
        py::array_t<double, py::array::c_style | py::array::forcecast> h1,
        py::array_t<double, py::array::c_style | py::array::forcecast> eri,
        int n_orb, int n_basis) -> py::array_t<double> {

        auto ra = ref_alpha.unchecked<1>();
        auto rb = ref_beta.unchecked<1>();
        auto rc = ref_coeffs.unchecked<1>();
        auto et = exc_types.unchecked<1>();
        int n_ref = static_cast<int>(ra.shape(0));
        int n_exc = static_cast<int>(et.shape(0));

        py::array_t<double> result({n_basis, n_basis});
        double* H_ptr = result.mutable_data();
        std::memset(H_ptr, 0, static_cast<size_t>(n_basis) * n_basis * sizeof(double));
        {
            py::gil_scoped_release release;
            build_H_detspace(ra.data(0), rb.data(0), rc.data(0), n_ref,
                et.data(0), exc_indices.data(), n_exc, n_basis,
                h1.data(), eri.data(), n_orb, H_ptr);
        }
        return result;
    },
    py::arg("ref_alpha"), py::arg("ref_beta"), py::arg("ref_coeffs"),
    py::arg("exc_types"), py::arg("exc_indices"),
    py::arg("h1"), py::arg("eri"), py::arg("n_orb"), py::arg("n_basis"),
    "Build H matrix using det-space algorithm with inverse map (C++).");

    // generate_excitations_heatbath
    m.def("generate_excitations_heatbath", [](
        py::array_t<uint64_t, py::array::c_style | py::array::forcecast> ref_alpha,
        py::array_t<uint64_t, py::array::c_style | py::array::forcecast> ref_beta,
        py::array_t<double, py::array::c_style | py::array::forcecast> ref_coeffs,
        int n_orb,
        py::array_t<double, py::array::c_style | py::array::forcecast> F_aa,
        py::array_t<double, py::array::c_style | py::array::forcecast> F_bb,
        py::array_t<double, py::array::c_style | py::array::forcecast> eri,
        int max_excitations,
        int scoring_mode) -> py::tuple {

        int n_ref = static_cast<int>(ref_alpha.shape(0));
        std::vector<int> out_types, out_indices;
        int n_exc;
        {
            py::gil_scoped_release release;
            n_exc = generate_excitations_heatbath(
                ref_alpha.data(), ref_beta.data(), ref_coeffs.data(),
                n_ref, n_orb, F_aa.data(), F_bb.data(), eri.data(),
                max_excitations, out_types, out_indices, scoring_mode);
        }
        py::array_t<int> types_arr(n_exc);
        py::array_t<int> indices_arr(n_exc * 4);
        std::memcpy(types_arr.mutable_data(), out_types.data(), n_exc * sizeof(int));
        std::memcpy(indices_arr.mutable_data(), out_indices.data(), n_exc * 4 * sizeof(int));
        return py::make_tuple(types_arr, indices_arr.reshape({n_exc, 4}), n_exc);
    },
    py::arg("ref_alpha"), py::arg("ref_beta"), py::arg("ref_coeffs"),
    py::arg("n_orb"), py::arg("F_aa"), py::arg("F_bb"), py::arg("eri"),
    py::arg("max_excitations"), py::arg("scoring_mode") = 0,
    "Generate excitations with heat-bath or ENPT2 selection (C++).");

    // build_alpha_adjacency
    m.def("build_alpha_adjacency", [](
        py::array_t<uint64_t, py::array::c_style | py::array::forcecast> alpha_strings,
        int n_orb) -> py::tuple {

        auto buf = alpha_strings.request();
        if (buf.ndim != 1)
            throw std::runtime_error("alpha_strings must be a 1-D array");
        int n_groups = static_cast<int>(buf.shape[0]);
        const uint64_t* data = static_cast<const uint64_t*>(buf.ptr);

        std::vector<std::vector<int32_t>> adj;
        {
            py::gil_scoped_release release;
            adj = build_alpha_adjacency(data, n_groups, n_orb);
        }

        // Convert to CSR format: (offsets, indices)
        std::vector<int32_t> offsets(n_groups + 1);
        offsets[0] = 0;
        size_t total = 0;
        for (int i = 0; i < n_groups; i++) {
            total += adj[i].size();
            offsets[i + 1] = static_cast<int32_t>(total);
        }

        std::vector<int32_t> flat_indices(total);
        for (int i = 0; i < n_groups; i++) {
            std::copy(adj[i].begin(), adj[i].end(),
                      flat_indices.begin() + offsets[i]);
        }

        py::array_t<int32_t> py_offsets(n_groups + 1);
        py::array_t<int32_t> py_indices(static_cast<py::ssize_t>(total));
        std::memcpy(py_offsets.mutable_data(), offsets.data(),
                    (n_groups + 1) * sizeof(int32_t));
        if (total > 0) {
            std::memcpy(py_indices.mutable_data(), flat_indices.data(),
                        total * sizeof(int32_t));
        }

        return py::make_tuple(py_offsets, py_indices);
    },
    py::arg("alpha_strings"), py::arg("n_orb"),
    "Build alpha-group adjacency list in CSR format (offsets, indices).\n\n"
    "Two groups are adjacent if popcount(alpha[i] XOR alpha[j]) == 2.\n"
    "Returns (offsets, indices) where adj[i] = indices[offsets[i]:offsets[i+1]].");
}

void bind_davidson_gep(py::module& m) {
    m.def("davidson_gep", [](
        Eigen::Ref<const Eigen::MatrixXd> H,
        Eigen::Ref<const Eigen::MatrixXd> S,
        int max_iter, double tol, int max_subspace, int verbose) {
            DavidsonGEPResult result;
            {
                py::gil_scoped_release release;
                result = davidson_gep(H, S, max_iter, tol, max_subspace, verbose);
            }
            py::dict info;
            info["converged"] = result.converged;
            info["iterations"] = result.iterations;
            info["residual_norm"] = result.residual_norm;
            return py::make_tuple(result.eigenvalue, result.eigenvector, info);
        },
        py::arg("H"), py::arg("S"),
        py::arg("max_iter") = 200, py::arg("tol") = 1e-8,
        py::arg("max_subspace") = 40, py::arg("verbose") = 0,
        "Davidson solver for generalized eigenvalue problem Hc = ESc");
}

void bind_matfree_davidson(py::module& m) {
    m.def("matfree_davidson_gep", [](
        py::array_t<uint64_t, py::array::c_style> ref_alpha,
        py::array_t<uint64_t, py::array::c_style> ref_beta,
        py::array_t<double, py::array::c_style> ref_coeffs,
        py::array_t<int, py::array::c_style> exc_types,
        py::array_t<int, py::array::c_style> exc_indices,
        py::array_t<double, py::array::c_style> h1,
        py::array_t<double, py::array::c_style> eri,
        int n_orb, int n_basis,
        int max_iter, double tol, int max_subspace, int verbose) {
            int n_ref = static_cast<int>(ref_alpha.size());
            int n_exc = n_basis - 1;
            MatfreeDavidsonResult result;
            {
                py::gil_scoped_release release;
                result = matfree_davidson_gep(
                    ref_alpha.data(), ref_beta.data(), ref_coeffs.data(), n_ref,
                    exc_types.data(), exc_indices.data(), n_exc, n_basis,
                    h1.data(), eri.data(), n_orb,
                    max_iter, tol, max_subspace, verbose);
            }
            py::dict info;
            info["converged"] = result.converged;
            info["iterations"] = result.iterations;
            info["residual_norm"] = result.residual_norm;
            info["h_diag_0"] = result.h_diag_0;
            return py::make_tuple(result.eigenvalue, result.eigenvector, info);
        },
        py::arg("ref_alpha"), py::arg("ref_beta"), py::arg("ref_coeffs"),
        py::arg("exc_types"), py::arg("exc_indices"),
        py::arg("h1"), py::arg("eri"),
        py::arg("n_orb"), py::arg("n_basis"),
        py::arg("max_iter") = 100, py::arg("tol") = 1e-6,
        py::arg("max_subspace") = 30, py::arg("verbose") = 0,
        "Matrix-free Davidson solver for LVCC generalized eigenvalue problem.");
}
