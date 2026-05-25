#include "bind_common.hpp"

PYBIND11_MODULE(trimci_core, m) {
    m.doc() = "TrimCI core: Determinant, Hamiltonian, Screening, Trim, Orbital Optimization, Fast Expansion, Iterative Workflow";

    bind_determinants(m);
    bind_hamiltonian(m);
    bind_screening(m);
    bind_trim(m);
    bind_rdm(m);
    bind_rdm_full(m);
    bind_orbital_opt(m);
    bind_iterative_workflow(m);
    bind_direct_builders(m);
    bind_davidson_gep(m);
    bind_matfree_davidson(m);
    bind_fast_expansion(m);
    bind_davidson_utils(m);
}
