// Bindings for orbital optimization
#include "bind_common.hpp"

#include "determinant.hpp"
#include "orbital_opt.hpp"

using namespace trimci_core;

template<size_t N>
void bind_scalable_orbital_opt(py::module& m, const std::string& suffix) {
    using DetType = DeterminantT<std::array<uint64_t, N>>;
    m.def(("orbital_gradient_" + suffix).c_str(),
          &orbital_gradient_t<std::array<uint64_t, N>>,
          py::arg("dets"), py::arg("coeffs"), py::arg("h1"), py::arg("eri"),
          py::arg("n_orb"), py::arg("n_alpha") = -1, py::arg("n_beta") = -1);
    m.def(("orbital_approx_newton_step_" + suffix).c_str(),
          &orbital_approx_newton_step_t<std::array<uint64_t, N>>,
          py::arg("dets"), py::arg("coeffs"), py::arg("h1"), py::arg("eri"),
          py::arg("n_orb"), py::arg("n_alpha") = -1, py::arg("n_beta") = -1,
          py::arg("rotate_integrals") = true, py::arg("step_scale") = 1.0, py::arg("min_hess") = 1e-5,
          py::arg("max_param") = 0.2);
    m.def(("orbital_approx_newton_optimize_" + suffix).c_str(),
          &orbital_approx_newton_optimize_t<std::array<uint64_t, N>>,
          py::arg("dets"), py::arg("coeffs"), py::arg("h1"), py::arg("eri"),
          py::arg("n_orb"), py::arg("n_alpha") = -1, py::arg("n_beta") = -1,
          py::arg("max_iter") = 50, py::arg("grad_tol") = 1e-6, py::arg("energy_tol") = 1e-8,
          py::arg("step_scale") = 1.0, py::arg("min_hess") = 1e-5, py::arg("max_param") = 0.2,
          py::arg("line_search") = true, py::arg("ls_max_iter") = 6,
          py::arg("ls_shrink") = 0.5, py::arg("ls_energy_tol") = 0.0,
          py::arg("davidson_tol") = 1e-3, py::arg("davidson_max_iter") = 200,
          py::arg("rotate_integrals") = true, py::arg("record_trace") = false);
    using ST = std::array<uint64_t, N>;
    m.def(("orbital_bfgs_optimize_" + suffix).c_str(),
          [](const std::vector<DeterminantT<ST>>& dets, const std::vector<double>& coeffs,
             const std::vector<std::vector<double>>& h1, const std::vector<double>& eri,
             int n_orb, int na, int nb, int mi, double gt, double et, double ss, double mh,
             double mp, double bd, double but, double bsf, double bsd, double bsm, double bid,
             bool ls, int lmi, double lsh, double let, double dt, int dmi, bool ri, bool rt) {
              return orbital_bfgs_optimize_t<ST>(
                  dets, coeffs, h1, eri, n_orb, na, nb, mi, gt, et, ss, mh, mp,
                  bd, but, bsf, bsd, bsm, bid, ls, lmi, lsh, let, dt, dmi, ri, rt, nullptr);
          },
          py::arg("dets"), py::arg("coeffs"), py::arg("h1"), py::arg("eri"),
          py::arg("n_orb"), py::arg("n_alpha") = -1, py::arg("n_beta") = -1,
          py::arg("max_iter") = 50, py::arg("grad_tol") = 1e-6, py::arg("energy_tol") = 1e-8,
          py::arg("step_scale") = 1.0, py::arg("min_hess") = 1e-5, py::arg("max_param") = 0.2,
          py::arg("bfgs_damp") = 1e-3, py::arg("bfgs_update_tol") = 1e-6,
          py::arg("bfgs_step_factor") = 5.0, py::arg("bfgs_step_decay") = 0.97,
          py::arg("bfgs_step_min") = 0.25, py::arg("bfgs_init_div") = 4.0,
          py::arg("line_search") = true, py::arg("ls_max_iter") = 6,
          py::arg("ls_shrink") = 0.5, py::arg("ls_energy_tol") = 0.0,
          py::arg("davidson_tol") = 1e-3, py::arg("davidson_max_iter") = 200,
          py::arg("rotate_integrals") = true, py::arg("record_trace") = false);
}

void bind_orbital_opt(py::module& m) {
    m.def("orbital_gradient", &orbital_gradient_t<uint64_t>,
          py::arg("dets"), py::arg("coeffs"), py::arg("h1"), py::arg("eri"),
          py::arg("n_orb"), py::arg("n_alpha") = -1, py::arg("n_beta") = -1);
    m.def("orbital_gradient_64", &orbital_gradient_t<uint64_t>,
          py::arg("dets"), py::arg("coeffs"), py::arg("h1"), py::arg("eri"),
          py::arg("n_orb"), py::arg("n_alpha") = -1, py::arg("n_beta") = -1);
    m.def("orbital_approx_newton_step", &orbital_approx_newton_step_t<uint64_t>,
          py::arg("dets"), py::arg("coeffs"), py::arg("h1"), py::arg("eri"),
          py::arg("n_orb"), py::arg("n_alpha") = -1, py::arg("n_beta") = -1,
          py::arg("rotate_integrals") = true, py::arg("step_scale") = 1.0, py::arg("min_hess") = 1e-5,
          py::arg("max_param") = 0.2);
    m.def("orbital_approx_newton_optimize", &orbital_approx_newton_optimize_t<uint64_t>,
          py::arg("dets"), py::arg("coeffs"), py::arg("h1"), py::arg("eri"),
          py::arg("n_orb"), py::arg("n_alpha") = -1, py::arg("n_beta") = -1,
          py::arg("max_iter") = 50, py::arg("grad_tol") = 1e-6, py::arg("energy_tol") = 1e-8,
          py::arg("step_scale") = 1.0, py::arg("min_hess") = 1e-5, py::arg("max_param") = 0.2,
          py::arg("line_search") = true, py::arg("ls_max_iter") = 6,
          py::arg("ls_shrink") = 0.5, py::arg("ls_energy_tol") = 0.0,
          py::arg("davidson_tol") = 1e-3, py::arg("davidson_max_iter") = 200,
          py::arg("rotate_integrals") = true, py::arg("record_trace") = false);
    m.def("orbital_bfgs_optimize",
          [](const std::vector<DeterminantT<uint64_t>>& dets, const std::vector<double>& coeffs,
             const std::vector<std::vector<double>>& h1, const std::vector<double>& eri,
             int n_orb, int na, int nb, int mi, double gt, double et, double ss, double mh,
             double mp, double bd, double but, double bsf, double bsd, double bsm, double bid,
             bool ls, int lmi, double lsh, double let, double dt, int dmi, bool ri, bool rt) {
              return orbital_bfgs_optimize_t<uint64_t>(
                  dets, coeffs, h1, eri, n_orb, na, nb, mi, gt, et, ss, mh, mp,
                  bd, but, bsf, bsd, bsm, bid, ls, lmi, lsh, let, dt, dmi, ri, rt, nullptr);
          },
          py::arg("dets"), py::arg("coeffs"), py::arg("h1"), py::arg("eri"),
          py::arg("n_orb"), py::arg("n_alpha") = -1, py::arg("n_beta") = -1,
          py::arg("max_iter") = 50, py::arg("grad_tol") = 1e-6, py::arg("energy_tol") = 1e-8,
          py::arg("step_scale") = 1.0, py::arg("min_hess") = 1e-5, py::arg("max_param") = 0.2,
          py::arg("bfgs_damp") = 1e-3, py::arg("bfgs_update_tol") = 1e-6,
          py::arg("bfgs_step_factor") = 5.0, py::arg("bfgs_step_decay") = 0.97,
          py::arg("bfgs_step_min") = 0.25, py::arg("bfgs_init_div") = 4.0,
          py::arg("line_search") = true, py::arg("ls_max_iter") = 6,
          py::arg("ls_shrink") = 0.5, py::arg("ls_energy_tol") = 0.0,
          py::arg("davidson_tol") = 1e-3, py::arg("davidson_max_iter") = 200,
          py::arg("rotate_integrals") = true, py::arg("record_trace") = false);
    m.def("orbital_approx_newton_step_64", &orbital_approx_newton_step_t<uint64_t>,
          py::arg("dets"), py::arg("coeffs"), py::arg("h1"), py::arg("eri"),
          py::arg("n_orb"), py::arg("n_alpha") = -1, py::arg("n_beta") = -1,
          py::arg("rotate_integrals") = true, py::arg("step_scale") = 1.0, py::arg("min_hess") = 1e-5,
          py::arg("max_param") = 0.2);
    m.def("orbital_approx_newton_optimize_64", &orbital_approx_newton_optimize_t<uint64_t>,
          py::arg("dets"), py::arg("coeffs"), py::arg("h1"), py::arg("eri"),
          py::arg("n_orb"), py::arg("n_alpha") = -1, py::arg("n_beta") = -1,
          py::arg("max_iter") = 50, py::arg("grad_tol") = 1e-6, py::arg("energy_tol") = 1e-8,
          py::arg("step_scale") = 1.0, py::arg("min_hess") = 1e-5, py::arg("max_param") = 0.2,
          py::arg("line_search") = true, py::arg("ls_max_iter") = 6,
          py::arg("ls_shrink") = 0.5, py::arg("ls_energy_tol") = 0.0,
          py::arg("davidson_tol") = 1e-3, py::arg("davidson_max_iter") = 200,
          py::arg("rotate_integrals") = true, py::arg("record_trace") = false);
    m.def("orbital_bfgs_optimize_64",
          [](const std::vector<DeterminantT<uint64_t>>& dets, const std::vector<double>& coeffs,
             const std::vector<std::vector<double>>& h1, const std::vector<double>& eri,
             int n_orb, int na, int nb, int mi, double gt, double et, double ss, double mh,
             double mp, double bd, double but, double bsf, double bsd, double bsm, double bid,
             bool ls, int lmi, double lsh, double let, double dt, int dmi, bool ri, bool rt) {
              return orbital_bfgs_optimize_t<uint64_t>(
                  dets, coeffs, h1, eri, n_orb, na, nb, mi, gt, et, ss, mh, mp,
                  bd, but, bsf, bsd, bsm, bid, ls, lmi, lsh, let, dt, dmi, ri, rt, nullptr);
          },
          py::arg("dets"), py::arg("coeffs"), py::arg("h1"), py::arg("eri"),
          py::arg("n_orb"), py::arg("n_alpha") = -1, py::arg("n_beta") = -1,
          py::arg("max_iter") = 50, py::arg("grad_tol") = 1e-6, py::arg("energy_tol") = 1e-8,
          py::arg("step_scale") = 1.0, py::arg("min_hess") = 1e-5, py::arg("max_param") = 0.2,
          py::arg("bfgs_damp") = 1e-3, py::arg("bfgs_update_tol") = 1e-6,
          py::arg("bfgs_step_factor") = 5.0, py::arg("bfgs_step_decay") = 0.97,
          py::arg("bfgs_step_min") = 0.25, py::arg("bfgs_init_div") = 4.0,
          py::arg("line_search") = true, py::arg("ls_max_iter") = 6,
          py::arg("ls_shrink") = 0.5, py::arg("ls_energy_tol") = 0.0,
          py::arg("davidson_tol") = 1e-3, py::arg("davidson_max_iter") = 200,
          py::arg("rotate_integrals") = true, py::arg("record_trace") = false);

    bind_scalable_orbital_opt<2>(m, "128");
    bind_scalable_orbital_opt<3>(m, "192");
    bind_scalable_orbital_opt<4>(m, "256");
    bind_scalable_orbital_opt<5>(m, "320");
    bind_scalable_orbital_opt<6>(m, "384");
    bind_scalable_orbital_opt<7>(m, "448");
    bind_scalable_orbital_opt<8>(m, "512");
}
