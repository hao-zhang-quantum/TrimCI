
#include "determinant.hpp"

namespace trimci_core {

// Wrapper implementation for generate_excitations
std::vector<Determinant> generate_excitations(const Determinant& det, int n_orb) {
    return generate_excitations_t<uint64_t>(det, n_orb);
}

// Explicit instantiations for generate_excitations_t to ensure symbols exist if not fully inlined elsewhere
template std::vector<DeterminantT<uint64_t>> generate_excitations_t<uint64_t>(const DeterminantT<uint64_t>&, int);
template std::vector<DeterminantT<std::array<uint64_t, 2>>> generate_excitations_t<std::array<uint64_t, 2>>(const DeterminantT<std::array<uint64_t, 2>>&, int);
template std::vector<DeterminantT<std::array<uint64_t, 3>>> generate_excitations_t<std::array<uint64_t, 3>>(const DeterminantT<std::array<uint64_t, 3>>&, int);
template std::vector<DeterminantT<std::array<uint64_t, 4>>> generate_excitations_t<std::array<uint64_t, 4>>(const DeterminantT<std::array<uint64_t, 4>>&, int);
template std::vector<DeterminantT<std::array<uint64_t, 5>>> generate_excitations_t<std::array<uint64_t, 5>>(const DeterminantT<std::array<uint64_t, 5>>&, int);
template std::vector<DeterminantT<std::array<uint64_t, 6>>> generate_excitations_t<std::array<uint64_t, 6>>(const DeterminantT<std::array<uint64_t, 6>>&, int);
template std::vector<DeterminantT<std::array<uint64_t, 7>>> generate_excitations_t<std::array<uint64_t, 7>>(const DeterminantT<std::array<uint64_t, 7>>&, int);
template std::vector<DeterminantT<std::array<uint64_t, 8>>> generate_excitations_t<std::array<uint64_t, 8>>(const DeterminantT<std::array<uint64_t, 8>>&, int);

} // namespace trimci_core
