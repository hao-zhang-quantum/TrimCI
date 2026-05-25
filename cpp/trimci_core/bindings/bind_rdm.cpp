// Bindings for RDM and RDM-full
#include "bind_common.hpp"

#include "determinant.hpp"
#include "rdm.hpp"
#include "rdm_full.hpp"

using namespace trimci_core;

template<size_t N>
void bind_scalable_rdm(py::module& m, const std::string& suffix) {
    using DetType = DeterminantT<std::array<uint64_t, N>>;
    m.def(("compute_orbital_gradient_" + suffix).c_str(), &compute_orbital_gradient_t<std::array<uint64_t, N>>,
          py::arg("dets"), py::arg("coeffs"), py::arg("h1"), py::arg("eri"), py::arg("n_orb"), py::arg("n_elec"),
          py::arg("attentive_orbitals") = std::vector<int>{},
          py::call_guard<py::gil_scoped_release>());
}

void bind_rdm_full(py::module& m) {
    m.def("compute_1rdm", &compute_1rdm,
          py::arg("dets"), py::arg("coeffs"), py::arg("n_orb"),
          py::call_guard<py::gil_scoped_release>(),
          "Compute exact 1-RDM: gamma[p,q] = <Psi|a+_p a_q|Psi>");

    m.def("compute_2rdm", &compute_2rdm,
          py::arg("dets"), py::arg("coeffs"), py::arg("n_orb"),
          py::call_guard<py::gil_scoped_release>(),
          "Compute exact 2-RDM: Gamma[p,q,r,s] = <Psi|a+_p a+_q a_s a_r|Psi>");

    m.def("contract_4rdm_cumulant", &contract_4rdm_cumulant,
          py::arg("Gamma2"), py::arg("c_ijab"),
          py::arg("p"), py::arg("q"), py::arg("r"), py::arg("s"),
          py::arg("n_orb"),
          "Contract 4-RDM (via cumulant) with amplitude tensor on-the-fly.");

    m.def("energy_from_rdm", &energy_from_rdm,
          py::arg("gamma"), py::arg("Gamma2"),
          py::arg("h1"), py::arg("eri"),
          py::arg("E_nuc"), py::arg("n_orb"),
          py::call_guard<py::gil_scoped_release>(),
          "Compute energy from RDMs: E = tr(h1 @ gamma) + 0.5 * tr(eri @ Gamma2) + E_nuc");

    m.def("compute_1rdm_spin_resolved", &compute_1rdm_spin_resolved,
          py::arg("dets"), py::arg("coeffs"), py::arg("n_orb"),
          py::call_guard<py::gil_scoped_release>(),
          "Compute spin-resolved 1-RDM: (gamma_aa, gamma_bb)");

    m.def("compute_2rdm_spin_resolved", &compute_2rdm_spin_resolved,
          py::arg("dets"), py::arg("coeffs"), py::arg("n_orb"),
          py::call_guard<py::gil_scoped_release>(),
          "Compute spin-resolved 2-RDM: (Gamma_aa, Gamma_ab, Gamma_bb)");

    m.def("compute_transition_1rdm_spin_resolved", &compute_transition_1rdm_spin_resolved,
          py::arg("dets_I"), py::arg("coeffs_I"),
          py::arg("dets_J"), py::arg("coeffs_J"),
          py::arg("n_orb"),
          py::call_guard<py::gil_scoped_release>(),
          "Compute transition 1-RDM: gamma^{IJ}[p,q] = <Psi_I|a+_p a_q|Psi_J>");

    m.def("compute_transition_2rdm_spin_resolved", &compute_transition_2rdm_spin_resolved,
          py::arg("dets_I"), py::arg("coeffs_I"),
          py::arg("dets_J"), py::arg("coeffs_J"),
          py::arg("n_orb"),
          py::call_guard<py::gil_scoped_release>(),
          "Compute transition 2-RDM: Gamma^{IJ}[p,q,r,s]");

    m.def("compute_3rdm_spin_resolved", &compute_3rdm_spin_resolved,
          py::arg("dets"), py::arg("coeffs"), py::arg("n_orb"),
          py::call_guard<py::gil_scoped_release>(),
          "Compute spin-resolved 3-RDM (exact). Returns (aaa, aab, abb, bbb).");

    m.def("compute_3rdm_cumulant_spin_resolved", &compute_3rdm_cumulant_spin_resolved,
          py::arg("gamma_aa"), py::arg("gamma_bb"),
          py::arg("Gamma_aa"), py::arg("Gamma_ab"), py::arg("Gamma_bb"),
          py::arg("n_orb"),
          py::call_guard<py::gil_scoped_release>(),
          "Compute spin-resolved 3-RDM via cumulant approximation.");

    m.def("compute_4rdm_spin_resolved", &compute_4rdm_spin_resolved,
          py::arg("dets"), py::arg("coeffs"), py::arg("n_orb"),
          py::call_guard<py::gil_scoped_release>(),
          "Compute spin-resolved 4-RDM (exact). Returns 5 spin components.");

    m.def("compute_4rdm_cumulant_from_2rdm_spin_resolved", &compute_4rdm_cumulant_from_2rdm_spin_resolved,
          py::arg("gamma_aa"), py::arg("gamma_bb"),
          py::arg("Gamma_aa"), py::arg("Gamma_ab"), py::arg("Gamma_bb"),
          py::arg("n_orb"),
          py::call_guard<py::gil_scoped_release>(),
          "Compute spin-resolved 4-RDM via cumulant (from 2-RDM).");

    m.def("compute_4rdm_cumulant_from_3rdm_spin_resolved", &compute_4rdm_cumulant_from_3rdm_spin_resolved,
          py::arg("gamma_aa"), py::arg("gamma_bb"),
          py::arg("Gamma_aa"), py::arg("Gamma_ab"), py::arg("Gamma_bb"),
          py::arg("Gamma3_aaa"), py::arg("Gamma3_aab"), py::arg("Gamma3_abb"), py::arg("Gamma3_bbb"),
          py::arg("n_orb"),
          py::call_guard<py::gil_scoped_release>(),
          "Compute spin-resolved 4-RDM cumulant using exact 3-RDM.");

    m.def("compute_rdm_element_cumulant2", &compute_rdm_element_cumulant2,
          py::arg("creators"), py::arg("annihilators"),
          py::arg("gamma_aa"), py::arg("gamma_bb"),
          py::arg("Gamma_aa"), py::arg("Gamma_ab"), py::arg("Gamma_bb"),
          py::arg("n_orb"),
          "Compute a single n-RDM element using cumulant-2 approximation.");

    m.def("compute_5rdm_spin_resolved", &compute_5rdm_spin_resolved,
          py::arg("dets"), py::arg("coeffs"), py::arg("n_orb"),
          py::call_guard<py::gil_scoped_release>(),
          "Compute spin-resolved 5-RDM (exact). Returns 6 spin components.");

    m.def("compute_6rdm_spin_resolved", &compute_6rdm_spin_resolved,
          py::arg("dets"), py::arg("coeffs"), py::arg("n_orb"),
          py::call_guard<py::gil_scoped_release>(),
          "Compute spin-resolved 6-RDM (exact). Returns 7 spin components.");
}

void bind_rdm(py::module& m) {
    m.def("compute_orbital_gradient", &compute_orbital_gradient,
          py::arg("dets"), py::arg("coeffs"), py::arg("h1"), py::arg("eri"), py::arg("n_orb"), py::arg("n_elec"),
          py::arg("attentive_orbitals") = std::vector<int>{},
          py::call_guard<py::gil_scoped_release>());

    m.def("compute_orbital_gradient_64", &compute_orbital_gradient,
          py::arg("dets"), py::arg("coeffs"), py::arg("h1"), py::arg("eri"), py::arg("n_orb"), py::arg("n_elec"),
          py::arg("attentive_orbitals") = std::vector<int>{},
          py::call_guard<py::gil_scoped_release>());

    bind_scalable_rdm<2>(m, "128");
    bind_scalable_rdm<3>(m, "192");
    bind_scalable_rdm<4>(m, "256");
    bind_scalable_rdm<5>(m, "320");
    bind_scalable_rdm<6>(m, "384");
    bind_scalable_rdm<7>(m, "448");
    bind_scalable_rdm<8>(m, "512");
}
