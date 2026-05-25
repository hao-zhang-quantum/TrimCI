// Bindings for davidson_utils: C++ accelerated helpers for GPU Davidson solver.
#include "bind_common.hpp"
#include "davidson_utils/davidson_utils.hpp"

using namespace trimci_core::davidson_utils;

void bind_davidson_utils(py::module& m) {
    auto du = m.def_submodule("davidson_utils",
        "C++ accelerated helpers for GPU Davidson solver");

    // ========================================================================
    // 1. groupby_argsort
    // ========================================================================
    du.def("groupby_argsort", [](
        py::array_t<uint64_t, py::array::c_style> keys_arr,
        py::array_t<uint64_t, py::array::c_style> partner_keys_arr,
        bool verbose) -> py::dict
    {
        auto keys = keys_arr.unchecked<1>();
        auto pkeys = partner_keys_arr.unchecked<1>();
        int64_t N = keys.shape(0);

        GroupBuildResult res;
        {
            py::gil_scoped_release release;
            res = groupby_argsort(keys.data(0), pkeys.data(0), N, verbose);
        }

        int64_t n_groups = static_cast<int64_t>(res.groups.size());

        // Convert to Python-friendly format: list of dicts
        py::list groups_list(n_groups);
        for (int64_t g = 0; g < n_groups; ++g) {
            auto& grp = res.groups[g];
            int64_t sz = static_cast<int64_t>(grp.partner_keys.size());

            py::array_t<uint64_t> pk(sz);
            py::array_t<int64_t> gi(sz);
            auto pk_mut = pk.mutable_unchecked<1>();
            auto gi_mut = gi.mutable_unchecked<1>();
            for (int64_t i = 0; i < sz; ++i) {
                pk_mut(i) = grp.partner_keys[i];
                gi_mut(i) = grp.global_indices[i];
            }

            py::dict d;
            d["key"] = grp.key;
            d["partner_keys"] = pk;
            d["global_indices"] = gi;
            groups_list[g] = d;
        }

        // key_to_gid as dict
        py::dict k2g;
        for (int64_t i = 0; i < n_groups; ++i) {
            k2g[py::int_(res.map_keys[i])] = res.map_gids[i];
        }

        py::dict out;
        out["groups"] = groups_list;
        out["key_to_gid"] = k2g;
        out["n_groups"] = n_groups;
        return out;
    },
    py::arg("keys"), py::arg("partner_keys"), py::arg("verbose") = true,
    "Group determinants by key with argsort (C++ accelerated).");

    // ========================================================================
    // 2. build_adjacency
    // ========================================================================
    du.def("build_adjacency", [](
        py::array_t<uint64_t, py::array::c_style> alpha_strings_arr,
        int n_orb,
        bool verbose) -> py::tuple
    {
        auto a = alpha_strings_arr.unchecked<1>();
        int64_t n_alpha = a.shape(0);

        std::vector<int64_t> adj_offsets;
        std::vector<int32_t> adj_list;
        {
            py::gil_scoped_release release;
            build_adjacency(a.data(0), n_alpha, n_orb,
                            adj_offsets, adj_list, verbose);
        }

        // Return as numpy arrays
        py::array_t<int64_t> py_offsets(adj_offsets.size());
        std::memcpy(py_offsets.mutable_data(), adj_offsets.data(),
                    adj_offsets.size() * sizeof(int64_t));

        py::array_t<int32_t> py_list(adj_list.size());
        std::memcpy(py_list.mutable_data(), adj_list.data(),
                    adj_list.size() * sizeof(int32_t));

        return py::make_tuple(py_offsets, py_list);
    },
    py::arg("alpha_strings"), py::arg("n_orb"), py::arg("verbose") = true,
    "Build adjacency CSR from alpha bitstrings (C++ accelerated).");

    // ========================================================================
    // 3. build_v_compact_indices
    // ========================================================================
    du.def("build_v_compact_indices", [](
        py::array_t<int64_t, py::array::c_style> original_offsets_arr,
        py::array_t<int64_t, py::array::c_style> group_ids_arr,
        py::array_t<int64_t, py::array::c_style> sizes_arr,
        bool use_int32) -> py::object
    {
        auto off = original_offsets_arr.unchecked<1>();
        auto gids = group_ids_arr.unchecked<1>();
        auto szs = sizes_arr.unchecked<1>();
        int64_t n_needed = gids.shape(0);

        std::vector<int32_t> res_i32;
        std::vector<int64_t> res_i64;
        {
            py::gil_scoped_release release;
            build_v_compact_indices(off.data(0), gids.data(0), szs.data(0),
                                    n_needed, use_int32, res_i32, res_i64);
        }

        if (use_int32) {
            py::array_t<int32_t> out(res_i32.size());
            std::memcpy(out.mutable_data(), res_i32.data(),
                        res_i32.size() * sizeof(int32_t));
            return out;
        } else {
            py::array_t<int64_t> out(res_i64.size());
            std::memcpy(out.mutable_data(), res_i64.data(),
                        res_i64.size() * sizeof(int64_t));
            return out;
        }
    },
    py::arg("original_offsets"), py::arg("group_ids"),
    py::arg("sizes"), py::arg("use_int32") = false,
    "Build compact index array for v gather/scatter (C++ accelerated).");

    // ========================================================================
    // 4. build_gpu_adj_csr
    // ========================================================================
    du.def("build_gpu_adj_csr", [](
        int64_t n_alpha_groups,
        py::array_t<int64_t, py::array::c_style> assigned_gids_arr,
        py::array_t<int64_t, py::array::c_style> adj_sizes_arr,
        py::array_t<int64_t, py::array::c_style> full_adj_offsets_arr,
        py::array_t<int32_t, py::array::c_style> full_adj_list_arr) -> py::tuple
    {
        auto ag = assigned_gids_arr.unchecked<1>();
        auto as = adj_sizes_arr.unchecked<1>();
        auto fao = full_adj_offsets_arr.unchecked<1>();
        auto fal = full_adj_list_arr.unchecked<1>();

        std::vector<int64_t> gpu_off;
        std::vector<int32_t> gpu_list;
        {
            py::gil_scoped_release release;
            build_gpu_adj_csr(n_alpha_groups,
                              ag.data(0), ag.shape(0),
                              as.data(0), fao.data(0), fal.data(0),
                              gpu_off, gpu_list);
        }

        py::array_t<int64_t> py_off(gpu_off.size());
        std::memcpy(py_off.mutable_data(), gpu_off.data(),
                    gpu_off.size() * sizeof(int64_t));

        py::array_t<int32_t> py_list(gpu_list.size());
        std::memcpy(py_list.mutable_data(), gpu_list.data(),
                    gpu_list.size() * sizeof(int32_t));

        int64_t total = gpu_off.back();
        return py::make_tuple(py_off, py_list, total);
    },
    py::arg("n_alpha_groups"), py::arg("assigned_gids"),
    py::arg("adj_sizes"), py::arg("full_adj_offsets"),
    py::arg("full_adj_list"),
    "Build per-GPU adjacency CSR (C++ accelerated).");

    // ========================================================================
    // 5. TailFloat compress
    // ========================================================================
    du.def("tailfloat_compress", [](
        py::array_t<double, py::array::c_style> v_arr,
        double tau, double norm_budget) -> py::dict
    {
        auto v = v_arr.unchecked<1>();
        int64_t N = v.shape(0);

        TailFloatData data;
        {
            py::gil_scoped_release release;
            data = tailfloat_compress(v.data(0), N, tau, norm_budget);
        }

        // Build output arrays (zero-copy where possible)
        py::array_t<float> py_v32(data.N);
        std::memcpy(py_v32.mutable_data(), data.v32.data(), data.N * sizeof(float));

        py::array_t<int32_t> py_idx(data.K);
        if (data.K > 0) {
            std::memcpy(py_idx.mutable_data(), data.head_idx.data(),
                        data.K * sizeof(int32_t));
        }

        py::array_t<double> py_corr(data.K);
        if (data.K > 0) {
            std::memcpy(py_corr.mutable_data(), data.head_corr.data(),
                        data.K * sizeof(double));
        }

        py::dict out;
        out["v32"] = py_v32;
        out["head_idx"] = py_idx;
        out["head_corr"] = py_corr;
        out["N"] = data.N;
        out["K"] = data.K;
        return out;
    },
    py::arg("v"), py::arg("tau") = 1e-3, py::arg("norm_budget") = 0.01,
    "TailFloat compress: single-pass float64->float32+corrections (C++ accelerated).");

    // ========================================================================
    // 6. TailFloat decompress
    // ========================================================================
    du.def("tailfloat_decompress", [](
        py::array_t<float, py::array::c_style> v32_arr,
        py::array_t<int32_t, py::array::c_style> head_idx_arr,
        py::array_t<double, py::array::c_style> head_corr_arr) -> py::array_t<double>
    {
        auto v32 = v32_arr.unchecked<1>();
        auto idx = head_idx_arr.unchecked<1>();
        auto corr = head_corr_arr.unchecked<1>();
        int64_t N = v32.shape(0);
        int64_t K = idx.shape(0);

        py::array_t<double> out(N);
        {
            py::gil_scoped_release release;
            tailfloat_decompress(v32.data(0), N,
                                 idx.data(0), corr.data(0), K,
                                 out.mutable_data());
        }
        return out;
    },
    py::arg("v32"), py::arg("head_idx"), py::arg("head_corr"),
    "TailFloat decompress: float32+corrections->float64 (C++ accelerated).");

    // ========================================================================
    // 7. build_mini_task_manifest
    // ========================================================================
    du.def("build_mini_task_manifest", [](
        py::array_t<int64_t, py::array::c_style> alpha_sizes_arr,
        py::array_t<int64_t, py::array::c_style> beta_sizes_arr,
        py::array_t<int64_t, py::array::c_style> adj_offsets_arr,
        py::array_t<int32_t, py::array::c_style> adj_list_arr,
        int64_t mini_task_cost,
        bool verbose) -> py::dict
    {
        auto as = alpha_sizes_arr.unchecked<1>();
        auto bs = beta_sizes_arr.unchecked<1>();
        auto ao = adj_offsets_arr.unchecked<1>();
        auto al = adj_list_arr.unchecked<1>();

        ManifestResult res;
        {
            py::gil_scoped_release release;
            res = build_mini_task_manifest(
                as.data(0), as.shape(0),
                bs.data(0), bs.shape(0),
                ao.data(0), al.data(0),
                mini_task_cost, verbose);
        }

        // Convert mini_tasks to list of (items_ndarray, cost)
        int64_t n_mt = static_cast<int64_t>(res.mini_tasks.size());
        py::list mt_list(n_mt);
        for (int64_t i = 0; i < n_mt; ++i) {
            auto& mt = res.mini_tasks[i];
            int64_t n_items = static_cast<int64_t>(mt.items.size());
            // Items as structured array: (ch, gid, adj, rs, re) per row
            // Use 5 separate arrays for simplicity
            py::array_t<uint8_t> ch(n_items);
            py::array_t<int32_t> gid(n_items);
            py::array_t<int32_t> adj(n_items);
            py::array_t<int64_t> rs(n_items);
            py::array_t<int64_t> re(n_items);
            auto ch_m = ch.mutable_unchecked<1>();
            auto gid_m = gid.mutable_unchecked<1>();
            auto adj_m = adj.mutable_unchecked<1>();
            auto rs_m = rs.mutable_unchecked<1>();
            auto re_m = re.mutable_unchecked<1>();
            for (int64_t j = 0; j < n_items; ++j) {
                ch_m(j) = mt.items[j].channel;
                gid_m(j) = mt.items[j].gid;
                adj_m(j) = mt.items[j].adj;
                rs_m(j) = mt.items[j].row_start;
                re_m(j) = mt.items[j].row_end;
            }
            py::dict item_dict;
            item_dict["ch"] = ch;
            item_dict["gid"] = gid;
            item_dict["adj"] = adj;
            item_dict["rs"] = rs;
            item_dict["re"] = re;
            item_dict["cost"] = mt.cost;
            mt_list[i] = item_dict;
        }

        py::dict out;
        out["mini_tasks"] = mt_list;
        out["n_mini_tasks"] = n_mt;
        out["n_mt_ch1"] = res.n_mt_ch1;
        out["n_mt_ch3"] = res.n_mt_ch3;
        out["n_mt_ch2"] = res.n_mt_ch2;
        out["total_cost"] = res.total_cost;
        out["n_atoms_ch2"] = res.n_atoms_ch2;
        return out;
    },
    py::arg("alpha_sizes"), py::arg("beta_sizes"),
    py::arg("adj_offsets"), py::arg("adj_list"),
    py::arg("mini_task_cost") = 100000,
    py::arg("verbose") = true,
    "Build mini-task manifest with cost-balanced packing (C++ accelerated).");

    // ========================================================================
    // 8. sorted_merge_match (warmstart)
    // ========================================================================
    du.def("sorted_merge_match", [](
        py::array_t<uint64_t, py::array::c_style> ws_alpha_arr,
        py::array_t<uint64_t, py::array::c_style> ws_beta_arr,
        py::array_t<double,   py::array::c_style> ws_coeffs_arr,
        py::array_t<uint64_t, py::array::c_style> main_alpha_arr,
        py::array_t<uint64_t, py::array::c_style> main_beta_arr,
        bool verbose, bool presorted_main) -> py::tuple
    {
        auto wa = ws_alpha_arr.unchecked<1>();
        auto wb = ws_beta_arr.unchecked<1>();
        auto wc = ws_coeffs_arr.unchecked<1>();
        auto ma = main_alpha_arr.unchecked<1>();
        auto mb = main_beta_arr.unchecked<1>();
        int64_t M = wa.shape(0);
        int64_t N = ma.shape(0);

        MergeMatchResult res;
        {
            py::gil_scoped_release release;
            res = sorted_merge_match(
                wa.data(0), wb.data(0), wc.data(0), M,
                ma.data(0), mb.data(0), N, verbose, presorted_main);
        }

        py::array_t<double> py_x0(N);
        std::memcpy(py_x0.mutable_data(), res.x0.data(), N * sizeof(double));

        return py::make_tuple(py_x0, res.matched);
    },
    py::arg("ws_alpha"), py::arg("ws_beta"), py::arg("ws_coeffs"),
    py::arg("main_alpha"), py::arg("main_beta"),
    py::arg("verbose") = true,
    py::arg("presorted_main") = false,
    "Warmstart merge-match: argsort + two-pointer merge (C++ accelerated).\n"
    "When presorted_main=True, main arrays are already sorted by (alpha, beta).");
}
