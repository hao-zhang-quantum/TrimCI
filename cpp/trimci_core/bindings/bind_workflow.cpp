// Bindings for iterative workflow
#include "bind_common.hpp"

#include "determinant.hpp"
#include "iterative_workflow.hpp"

#ifdef TRIMCI_HAS_GPU
#include "gpu_workflow.cuh"
#include "gpu_orbital_opt.cuh"
#endif

using namespace trimci_core;

template<size_t N>
void bind_scalable_workflow(py::module& m, const std::string& suffix) {
    using DetType = DeterminantT<std::array<uint64_t, N>>;
    using ResultType = IterativeWorkflowResult<std::array<uint64_t, N>>;

    std::string resultClassName = "IterativeWorkflowResult" + suffix;
    py::class_<ResultType>(m, resultClassName.c_str())
        .def(py::init<>())
        .def_readonly("final_energy", &ResultType::final_energy)
        .def_readonly("final_dets", &ResultType::final_dets)
        .def_readonly("final_coeffs", &ResultType::final_coeffs)
        .def_readonly("iteration_history", &ResultType::iteration_history)
        .def_readonly("total_time", &ResultType::total_time)
        .def_readonly("total_iterations", &ResultType::total_iterations)
        .def_readonly("success", &ResultType::success)
        .def_readonly("error_message", &ResultType::error_message);

    std::string fnName = "iterative_workflow_cpp_" + suffix;
    m.def(fnName.c_str(),
          &iterative_workflow_t<std::array<uint64_t, N>>,
          py::arg("h1"), py::arg("eri"),
          py::arg("n_alpha"), py::arg("n_beta"), py::arg("n_orb"),
          py::arg("system_name"),
          py::arg("initial_dets"), py::arg("initial_coeffs"),
          py::arg("nuclear_repulsion"), py::arg("params"),
          py::call_guard<py::gil_scoped_release>());

    std::string fnNameNp = "iterative_workflow_cpp_np_" + suffix;
    m.def(fnNameNp.c_str(),
          [](py::array_t<double, py::array::c_style | py::array::forcecast> h1_np,
             py::array_t<double, py::array::c_style | py::array::forcecast> eri_np,
             int n_alpha, int n_beta, int n_orb,
             const std::string& system_name,
             const std::vector<DetType>& initial_dets,
             const std::vector<double>& initial_coeffs,
             double nuclear_repulsion,
             const IterativeWorkflowParams& params) {
              auto h1_buf = h1_np.request();
              if (h1_buf.ndim != 2) throw std::runtime_error("h1 must be 2D array");
              size_t h1_rows = h1_buf.shape[0], h1_cols = h1_buf.shape[1];
              double* h1_ptr = static_cast<double*>(h1_buf.ptr);
              std::vector<std::vector<double>> h1(h1_rows, std::vector<double>(h1_cols));
              for (size_t i = 0; i < h1_rows; ++i)
                  for (size_t j = 0; j < h1_cols; ++j)
                      h1[i][j] = h1_ptr[i * h1_cols + j];
              auto eri_buf = eri_np.request();
              std::vector<double> eri(static_cast<double*>(eri_buf.ptr),
                                     static_cast<double*>(eri_buf.ptr) + eri_buf.size);
              return iterative_workflow_t<std::array<uint64_t, N>>(
                  h1, eri, n_alpha, n_beta, n_orb,
                  system_name, initial_dets, initial_coeffs,
                  nuclear_repulsion, params);
          },
          py::arg("h1"), py::arg("eri"),
          py::arg("n_alpha"), py::arg("n_beta"), py::arg("n_orb"),
          py::arg("system_name"),
          py::arg("initial_dets"), py::arg("initial_coeffs"),
          py::arg("nuclear_repulsion"), py::arg("params"));
}

void bind_iterative_workflow(py::module& m) {
    py::class_<IterativeWorkflowParams>(m, "IterativeWorkflowParams")
        .def(py::init<>())
        .def_readwrite("max_iterations", &IterativeWorkflowParams::max_iterations)
        .def_readwrite("energy_threshold", &IterativeWorkflowParams::energy_threshold)
        .def_readwrite("max_final_dets", &IterativeWorkflowParams::max_final_dets)
        .def_readwrite("core_set_ratio", &IterativeWorkflowParams::core_set_ratio)
        .def_readwrite("initial_pool_size", &IterativeWorkflowParams::initial_pool_size)
        .def_readwrite("core_set_schedule", &IterativeWorkflowParams::core_set_schedule)
        .def_readwrite("first_cycle_keep_size", &IterativeWorkflowParams::first_cycle_keep_size)
        .def_readwrite("pool_core_ratio", &IterativeWorkflowParams::pool_core_ratio)
        .def_readwrite("pool_build_strategy", &IterativeWorkflowParams::pool_build_strategy)
        .def_readwrite("threshold", &IterativeWorkflowParams::threshold)
        .def_readwrite("first_cycle_threshold", &IterativeWorkflowParams::first_cycle_threshold)
        .def_readwrite("threshold_decay", &IterativeWorkflowParams::threshold_decay)
        .def_readwrite("max_rounds", &IterativeWorkflowParams::max_rounds)
        .def_readwrite("attentive_orbitals", &IterativeWorkflowParams::attentive_orbitals)
        .def_readwrite("strategy_factor", &IterativeWorkflowParams::strategy_factor)
        .def_readwrite("e0", &IterativeWorkflowParams::e0)
        .def_readwrite("pool_strict_target_size", &IterativeWorkflowParams::pool_strict_target_size)
        .def_readwrite("max_pool_size", &IterativeWorkflowParams::max_pool_size)
        .def_readwrite("overshoot_factor", &IterativeWorkflowParams::overshoot_factor)
        .def_readwrite("stagnation_limit", &IterativeWorkflowParams::stagnation_limit)
        .def_readwrite("noise_strength", &IterativeWorkflowParams::noise_strength)
        .def_readwrite("noise_decay", &IterativeWorkflowParams::noise_decay)
        .def_readwrite("random_seed", &IterativeWorkflowParams::random_seed)
        .def_readwrite("num_groups", &IterativeWorkflowParams::num_groups)
        .def_readwrite("num_groups_ratio", &IterativeWorkflowParams::num_groups_ratio)
        .def_readwrite("local_trim_keep_ratio", &IterativeWorkflowParams::local_trim_keep_ratio)
        .def_readwrite("keep_ratio", &IterativeWorkflowParams::keep_ratio)
        .def_readwrite("verbosity", &IterativeWorkflowParams::verbosity)
        .def_readwrite("davidson_init", &IterativeWorkflowParams::davidson_init)
        .def_readwrite("save_period", &IterativeWorkflowParams::save_period)
        .def_readwrite("save_pool", &IterativeWorkflowParams::save_pool)
        .def_readwrite("save_initial", &IterativeWorkflowParams::save_initial)
        .def_readwrite("output_dir", &IterativeWorkflowParams::output_dir);

    py::class_<IterationInfo>(m, "IterationInfo")
        .def(py::init<>())
        .def_readonly("iteration", &IterationInfo::iteration)
        .def_readonly("core_set_size_before", &IterationInfo::core_set_size_before)
        .def_readonly("target_pool_size", &IterationInfo::target_pool_size)
        .def_readonly("actual_pool_size", &IterationInfo::actual_pool_size)
        .def_readonly("final_threshold", &IterationInfo::final_threshold)
        .def_readonly("pool_building_time", &IterationInfo::pool_building_time)
        .def_readonly("trim_m", &IterationInfo::trim_m)
        .def_readonly("trim_k", &IterationInfo::trim_k)
        .def_readonly("raw_dets_count", &IterationInfo::raw_dets_count)
        .def_readonly("raw_energy", &IterationInfo::raw_energy)
        .def_readonly("energy_change", &IterationInfo::energy_change)
        .def_readonly("converged", &IterationInfo::converged)
        .def_readonly("core_set_size_after", &IterationInfo::core_set_size_after)
        .def_readonly("iteration_time", &IterationInfo::iteration_time)
        .def_readonly("cumulative_time", &IterationInfo::cumulative_time)
        .def_readonly("stop_reason", &IterationInfo::stop_reason);

    py::class_<IterativeWorkflowResult64>(m, "IterativeWorkflowResult")
        .def(py::init<>())
        .def_readonly("final_energy", &IterativeWorkflowResult64::final_energy)
        .def_readonly("final_dets", &IterativeWorkflowResult64::final_dets)
        .def_readonly("final_coeffs", &IterativeWorkflowResult64::final_coeffs)
        .def_readonly("iteration_history", &IterativeWorkflowResult64::iteration_history)
        .def_readonly("total_time", &IterativeWorkflowResult64::total_time)
        .def_readonly("total_iterations", &IterativeWorkflowResult64::total_iterations)
        .def_readonly("success", &IterativeWorkflowResult64::success)
        .def_readonly("error_message", &IterativeWorkflowResult64::error_message);

    m.def("iterative_workflow_cpp", &iterative_workflow,
          py::arg("h1"), py::arg("eri"),
          py::arg("n_alpha"), py::arg("n_beta"), py::arg("n_orb"),
          py::arg("system_name"),
          py::arg("initial_dets"), py::arg("initial_coeffs"),
          py::arg("nuclear_repulsion"),
          py::arg("params"),
          py::call_guard<py::gil_scoped_release>(),
          "C++ implementation of iterative_workflow.");

    m.def("iterative_workflow_cpp_np",
          [](py::array_t<double, py::array::c_style | py::array::forcecast> h1_np,
             py::array_t<double, py::array::c_style | py::array::forcecast> eri_np,
             int n_alpha, int n_beta, int n_orb,
             const std::string& system_name,
             const std::vector<Determinant64>& initial_dets,
             const std::vector<double>& initial_coeffs,
             double nuclear_repulsion,
             const IterativeWorkflowParams& params) {
              auto h1_buf = h1_np.request();
              if (h1_buf.ndim != 2) throw std::runtime_error("h1 must be 2D array");
              size_t h1_rows = h1_buf.shape[0], h1_cols = h1_buf.shape[1];
              double* h1_ptr = static_cast<double*>(h1_buf.ptr);
              std::vector<std::vector<double>> h1(h1_rows, std::vector<double>(h1_cols));
              for (size_t i = 0; i < h1_rows; ++i)
                  for (size_t j = 0; j < h1_cols; ++j)
                      h1[i][j] = h1_ptr[i * h1_cols + j];
              auto eri_buf = eri_np.request();
              std::vector<double> eri(static_cast<double*>(eri_buf.ptr),
                                     static_cast<double*>(eri_buf.ptr) + eri_buf.size);
              return iterative_workflow(h1, eri, n_alpha, n_beta, n_orb,
                                        system_name, initial_dets, initial_coeffs,
                                        nuclear_repulsion, params);
          },
          py::arg("h1"), py::arg("eri"),
          py::arg("n_alpha"), py::arg("n_beta"), py::arg("n_orb"),
          py::arg("system_name"),
          py::arg("initial_dets"), py::arg("initial_coeffs"),
          py::arg("nuclear_repulsion"),
          py::arg("params"),
          "Numpy-optimized C++ iterative_workflow.");

    bind_scalable_workflow<2>(m, "128");
    bind_scalable_workflow<3>(m, "192");
    bind_scalable_workflow<4>(m, "256");
    bind_scalable_workflow<5>(m, "320");
    bind_scalable_workflow<6>(m, "384");
    bind_scalable_workflow<7>(m, "448");
    bind_scalable_workflow<8>(m, "512");

#ifdef TRIMCI_HAS_GPU
    // GPU-accelerated iterative workflow bindings
    m.def("iterative_workflow_gpu_128",
          &trimci_gpu::iterative_workflow_gpu_128,
          py::arg("h1"), py::arg("eri"),
          py::arg("n_alpha"), py::arg("n_beta"), py::arg("n_orb"),
          py::arg("system_name"),
          py::arg("initial_dets"), py::arg("initial_coeffs"),
          py::arg("nuclear_repulsion"),
          py::arg("params"),
          py::call_guard<py::gil_scoped_release>(),
          "GPU-accelerated iterative workflow (128-bit determinants, up to 128 orbitals).");

    m.def("iterative_workflow_gpu_64",
          &trimci_gpu::iterative_workflow_gpu_64,
          py::arg("h1"), py::arg("eri"),
          py::arg("n_alpha"), py::arg("n_beta"), py::arg("n_orb"),
          py::arg("system_name"),
          py::arg("initial_dets"), py::arg("initial_coeffs"),
          py::arg("nuclear_repulsion"),
          py::arg("params"),
          py::call_guard<py::gil_scoped_release>(),
          "GPU-accelerated iterative workflow (64-bit determinants, up to 64 orbitals).");

    // Multi-run variants: share a single GpuContext + integrals upload across N runs.
    // Each run can have its own initial determinants (Phase 0 ensemble with
    // random seed states). initial_dets_list / initial_coeffs_list must have
    // size == num_runs (per-run) or size == 1 (broadcast shared state).
    m.def("iterative_workflow_gpu_multi_128",
          &trimci_gpu::iterative_workflow_gpu_multi_128,
          py::arg("h1"), py::arg("eri"),
          py::arg("n_alpha"), py::arg("n_beta"), py::arg("n_orb"),
          py::arg("system_name"),
          py::arg("initial_dets_list"), py::arg("initial_coeffs_list"),
          py::arg("nuclear_repulsion"),
          py::arg("params"),
          py::arg("num_runs"),
          py::arg("seeds") = std::vector<uint64_t>{},
          py::call_guard<py::gil_scoped_release>(),
          "GPU multi-run workflow (128-bit dets); uploads integrals once and runs num_runs "
          "independent TRIM workflows sharing the GpuContext. initial_dets_list / "
          "initial_coeffs_list have size == num_runs for per-run state or size == 1 to "
          "broadcast. seeds[r] > 0 overrides; otherwise derives from params.random_seed "
          "(0 = wall-clock).");

    m.def("iterative_workflow_gpu_multi_64",
          &trimci_gpu::iterative_workflow_gpu_multi_64,
          py::arg("h1"), py::arg("eri"),
          py::arg("n_alpha"), py::arg("n_beta"), py::arg("n_orb"),
          py::arg("system_name"),
          py::arg("initial_dets_list"), py::arg("initial_coeffs_list"),
          py::arg("nuclear_repulsion"),
          py::arg("params"),
          py::arg("num_runs"),
          py::arg("seeds") = std::vector<uint64_t>{},
          py::call_guard<py::gil_scoped_release>(),
          "GPU multi-run workflow (64-bit dets); see multi_128 for details.");

    // GPU 4-index integral transform (Phase 1 of GPU BFGS orbital opt).
    // Returns (h1_new_flat, eri_new_flat), both row-major flattened.
    m.def("transform_integrals_gpu",
          &trimci_gpu::transform_integrals_gpu_alloc,
          py::arg("h1_flat"), py::arg("eri_flat"), py::arg("U_flat"),
          py::arg("n_orb"),
          py::call_guard<py::gil_scoped_release>(),
          "GPU 4-index integral transform via cuBLAS DGEMM chain. "
          "All inputs row-major flattened. Returns (h1_new_flat, eri_new_flat).");

    m.def("compute_rdm_gpu_128",
          &trimci_gpu::compute_rdm_gpu_alloc_128,
          py::arg("dets_alpha"), py::arg("dets_beta"), py::arg("coeffs"),
          py::arg("n_orb"), py::arg("n_alpha"), py::arg("n_beta"),
          py::call_guard<py::gil_scoped_release>(),
          "GPU 2-RDM (and 1-RDM from contraction) for 128-bit determinants. "
          "dets_alpha/beta: list of length-2 uint64 arrays. coeffs: float list. "
          "Returns (rdm1_flat [n_orb²], rdm2_tri [n_orb²(n_orb²+1)/2]).");

    m.def("compute_orbital_gradient_gpu_128",
          &trimci_gpu::compute_orbital_gradient_gpu_alloc_128,
          py::arg("dets_alpha"), py::arg("dets_beta"), py::arg("coeffs"),
          py::arg("h1_flat"), py::arg("eri_flat"),
          py::arg("n_orb"), py::arg("n_alpha"), py::arg("n_beta"),
          py::call_guard<py::gil_scoped_release>(),
          "GPU orbital gradient: 2*(F - F^T) from generalized Fock (RDMs × integrals). "
          "Returns grad_flat [n_orb²] row-major.");

    m.def("has_gpu", []() { return true; }, "Check if GPU support is available.");
#else
    m.def("has_gpu", []() { return false; }, "Check if GPU support is available.");
#endif
}
