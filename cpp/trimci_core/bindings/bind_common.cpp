// Bindings for determinants and Hamiltonian
#include "bind_common.hpp"
#include <sstream>
#include <bitset>
#include <array>

#include "determinant.hpp"
#include "hamiltonian.hpp"
#include "spin_operator.hpp"

using namespace trimci_core;

// Helper function to create string representation for array-based determinants
template<size_t N>
std::string array_to_bitstring(const std::array<uint64_t, N>& arr) {
    std::ostringstream oss;
    for (size_t i = 0; i < N; ++i) {
        if (i > 0) oss << "|";
        oss << std::bitset<64>(arr[i]);
    }
    return oss.str();
}

// Template helpers for scalable bindings
template<size_t N>
void bind_scalable_determinant(py::module& m, const std::string& suffix) {
    using DetType = DeterminantT<std::array<uint64_t, N>>;
    std::string className = "Determinant" + suffix;
    py::class_<DetType>(m, className.c_str())
        .def(py::init<std::array<uint64_t, N>, std::array<uint64_t, N>>())
        .def_readwrite("alpha", &DetType::alpha)
        .def_readwrite("beta",  &DetType::beta)
        .def(py::self == py::self)
        .def("__repr__", [className](const DetType& d) {
            std::ostringstream oss;
            oss << className << "(alpha=" << array_to_bitstring(d.alpha)
                << ", beta=" << array_to_bitstring(d.beta) << ")";
            return oss.str();
        })
        .def(py::pickle(
            [](const DetType& d) { return py::make_tuple(d.alpha, d.beta); },
            [](py::tuple t) {
                if (t.size() != 2) throw std::runtime_error("Invalid state");
                return new DetType(t[0].cast<std::array<uint64_t, N>>(), t[1].cast<std::array<uint64_t, N>>());
            }
        ));

    m.def(("generate_reference_det_" + suffix).c_str(),
          [](int n_alpha, int n_beta) { return generate_reference_det_t<std::array<uint64_t, N>>(n_alpha, n_beta); },
          py::arg("n_alpha"), py::arg("n_beta"));

    m.def(("generate_excitations_" + suffix).c_str(),
          [](const DetType& det, int n_orb) { return generate_excitations_t<std::array<uint64_t, N>>(det, n_orb); },
          py::arg("det"), py::arg("n_orb"));
}

template<size_t N>
void bind_scalable_hamiltonian(py::module& m, const std::string& suffix) {
    m.def(("load_or_create_Hij_cache_" + suffix).c_str(), &load_or_create_Hij_cache_t<std::array<uint64_t, N>>,
          py::arg("mol_name"), py::arg("n_elec"), py::arg("n_orb"), py::arg("cache_dir") = std::string("cache"));
    using HijFunc = double(*)(const DeterminantT<std::array<uint64_t, N>>&,
                              const DeterminantT<std::array<uint64_t, N>>&,
                              const std::vector<std::vector<double>>&,
                              const std::vector<double>&);
    m.def(("compute_H_ij_" + suffix).c_str(), static_cast<HijFunc>(&compute_H_ij_t<std::array<uint64_t, N>>),
          py::arg("det_i"), py::arg("det_j"), py::arg("h1"), py::arg("eri"));
}

void bind_determinants(py::module& m) {
    // Determinant64 (uint64_t storage) - Exposed as standard Determinant for backward compatibility
    py::class_<Determinant64>(m, "Determinant")
        .def(py::init<uint64_t, uint64_t>())
        .def_readwrite("alpha", &Determinant64::alpha)
        .def_readwrite("beta",  &Determinant64::beta)
        .def(py::self == py::self)
        .def("__repr__", [](const Determinant64& d) {
            std::ostringstream oss;
            oss << "Determinant(alpha=" << std::bitset<64>(d.alpha)
                << ", beta=" << std::bitset<64>(d.beta) << ")";
            return oss.str();
        })
        .def(py::pickle(
            [](const Determinant64& d) {
                return py::make_tuple(d.alpha, d.beta);
            },
            [](py::tuple t) {
                if (t.size() != 2)
                    throw std::runtime_error("Invalid state for Determinant");
                return new Determinant64(
                    t[0].cast<uint64_t>(),
                    t[1].cast<uint64_t>()
                );
            }
        ));

    // Helper functions for Determinant64 (Default)
    m.def("generate_reference_det", &generate_reference_det,
          py::arg("n_alpha"), py::arg("n_beta"));
    m.def("generate_excitations",   &generate_excitations,
          py::arg("det"), py::arg("n_orb"));

    // Aliases for 64-bit to match scalable naming convention
    m.attr("Determinant64") = m.attr("Determinant");
    m.def("generate_reference_det_64", &generate_reference_det, py::arg("n_alpha"), py::arg("n_beta"));
    m.def("generate_excitations_64", &generate_excitations, py::arg("det"), py::arg("n_orb"));

    // Scalable determinants bindings
    bind_scalable_determinant<2>(m, "128");
    bind_scalable_determinant<3>(m, "192");
    bind_scalable_determinant<4>(m, "256");
    bind_scalable_determinant<5>(m, "320");
    bind_scalable_determinant<6>(m, "384");
    bind_scalable_determinant<7>(m, "448");
    bind_scalable_determinant<8>(m, "512");
}

void bind_hamiltonian(py::module& m) {
    m.def("extract_mol_name", &extract_mol_name, py::arg("atom_str"));

    // Hamiltonian functions for Determinant64 (Default)
    m.def("load_or_create_Hij_cache", &load_or_create_Hij_cache,
          py::arg("mol_name"), py::arg("n_elec"), py::arg("n_orb"),
          py::arg("cache_dir") = std::string("cache"));
    m.def("compute_H_ij", &compute_H_ij,
          py::arg("det_i"), py::arg("det_j"),
          py::arg("h1"), py::arg("eri"));

    m.def("pair_key", &pair_key, py::arg("d1"), py::arg("d2"));

    // Aliases for 64-bit
    m.def("load_or_create_Hij_cache_64", &load_or_create_Hij_cache,
          py::arg("mol_name"), py::arg("n_elec"), py::arg("n_orb"),
          py::arg("cache_dir") = std::string("cache"));
    m.def("compute_H_ij_64", &compute_H_ij,
          py::arg("det_i"), py::arg("det_j"),
          py::arg("h1"), py::arg("eri"));

    // Scalable hamiltonian bindings
    bind_scalable_hamiltonian<2>(m, "128");
    bind_scalable_hamiltonian<3>(m, "192");
    bind_scalable_hamiltonian<4>(m, "256");
    bind_scalable_hamiltonian<5>(m, "320");
    bind_scalable_hamiltonian<6>(m, "384");
    bind_scalable_hamiltonian<7>(m, "448");
    bind_scalable_hamiltonian<8>(m, "512");

    // CI Energy Evaluation
    m.def("evaluate_ci_energy", &evaluate_ci_energy,
          py::arg("dets_alpha"), py::arg("dets_beta"), py::arg("coeffs"),
          py::arg("h1"), py::arg("eri"), py::arg("n_orb"),
          py::call_guard<py::gil_scoped_release>(),
          R"doc(
Evaluate CI energy given determinants and coefficients.

Computes E = Σ_ij c_i c_j ⟨D_i|H|D_j⟩ using Slater-Condon rules.
OpenMP parallelized for efficiency.

Args:
    dets_alpha: Alpha bitstrings for each determinant
    dets_beta: Beta bitstrings for each determinant
    coeffs: CI coefficients
    h1: One-body integrals (n_orb x n_orb, as vector<vector>)
    eri: Two-body integrals (flattened n_orb^4)
    n_orb: Number of orbitals

Returns:
    CI energy (float)
)doc");

    // ── S² operator (diagnostic) ────────────────────────────────────
    m.def("compute_S2_ij", &compute_S2_ij,
          py::arg("det_i"), py::arg("det_j"),
          "⟨det_i|S²|det_j⟩ for 64-bit spinful fermion determinants.");
    m.def("compute_S2_ij_64", &compute_S2_ij,
          py::arg("det_i"), py::arg("det_j"));
    m.def("compute_S2_ij_128",
          static_cast<double(*)(
              const DeterminantT<std::array<uint64_t,2>>&,
              const DeterminantT<std::array<uint64_t,2>>&)>(
              &compute_S2_ij_t<std::array<uint64_t,2>>),
          py::arg("det_i"), py::arg("det_j"));

    m.def("evaluate_S2", &evaluate_S2,
          py::arg("dets_alpha"), py::arg("dets_beta"), py::arg("coeffs"),
          py::call_guard<py::gil_scoped_release>(),
          R"doc(
Evaluate ⟨Ψ|S²|Ψ⟩ for a CI wavefunction (64-bit determinants).

  ⟨S²⟩ = Σ_ij c_i c_j ⟨D_i|S²|D_j⟩

S² = S_z(S_z+1) + S_-S_+. Off-diagonal partners enumerated by
spin-exchange (α-singly ↔ β-singly swap at two orbitals), so the
cost is O(N · N_α_singly · N_β_singly · hash) rather than O(N²).

For a pure spin eigenstate: ⟨S²⟩ = S(S+1), so S=0 ⇒ 0, triplet ⇒ 2, etc.

Args:
    dets_alpha: uint64 array of α bitstrings
    dets_beta:  uint64 array of β bitstrings
    coeffs:     float64 CI coefficients

Returns:
    ⟨S²⟩ (float)
)doc");

    m.def("evaluate_S2_128", &evaluate_S2_128,
          py::arg("dets_alpha"), py::arg("dets_beta"), py::arg("coeffs"),
          py::call_guard<py::gil_scoped_release>(),
          "128-bit version of evaluate_S2 (alpha/beta are lists of "
          "array<uint64_t,2>; used when n_orb > 64).");
}
