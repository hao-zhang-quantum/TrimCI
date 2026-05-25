#include "orbital_opt.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <unordered_map>

#include <omp.h>
#include <Eigen/Eigenvalues>

#include "hamiltonian.hpp"
#include "trim.hpp"

namespace trimci_core {
namespace detail {

template<typename StorageType>
int popcount_storage(const StorageType& s) {
    if constexpr (std::is_same_v<StorageType, uint64_t>) {
        return __builtin_popcountll(s);
    } else {
        int count = 0;
        for (size_t i = 0; i < s.size(); ++i) {
            count += __builtin_popcountll(s[i]);
        }
        return count;
    }
}

template<typename StorageType>
int count_bits_below(const StorageType& s, int pos) {
    if (pos <= 0) return 0;
    if constexpr (std::is_same_v<StorageType, uint64_t>) {
        uint64_t mask = (pos >= 64) ? ~0ULL : ((uint64_t(1) << pos) - 1ULL);
        return __builtin_popcountll(s & mask);
    } else {
        int count = 0;
        int unit = pos / 64;
        int offset = pos % 64;
        for (int i = 0; i < unit; ++i) {
            count += __builtin_popcountll(s[static_cast<size_t>(i)]);
        }
        if (unit >= 0 && unit < static_cast<int>(s.size()) && offset > 0) {
            uint64_t mask = (uint64_t(1) << offset) - 1ULL;
            count += __builtin_popcountll(s[static_cast<size_t>(unit)] & mask);
        }
        return count;
    }
}

template<typename StorageType>
struct DiffResult {
    int n_diffs = 0;
    std::array<int, 2> left_only{{-1, -1}};
    std::array<int, 2> right_only{{-1, -1}};
    int permutation_factor = 1;
};

template<typename StorageType>
DiffResult<StorageType> diff_bits(const StorageType& self, const StorageType& other) {
    DiffResult<StorageType> result;
    using BitOpsType = HamiltonianBitOps<StorageType>;
    StorageType left = BitOpsType::and_not(self, other);
    StorageType right = BitOpsType::and_not(other, self);
    int left_cnt = BitOpsType::storage_to_indices_inline(left, result.left_only.data(), 2);
    int right_cnt = BitOpsType::storage_to_indices_inline(right, result.right_only.data(), 2);
    result.n_diffs = left_cnt;
    if (left_cnt != right_cnt) {
        result.permutation_factor = 0;
        return result;
    }
    if (result.n_diffs == 0) {
        result.permutation_factor = 1;
    } else if (result.n_diffs == 1) {
        int from = result.right_only[0];
        int to = result.left_only[0];
        result.permutation_factor = cre_des_sign_t(from, to, self);
    } else if (result.n_diffs == 2) {
        int m = std::min(result.right_only[0], result.right_only[1]);
        int n = std::max(result.right_only[0], result.right_only[1]);
        int p = std::min(result.left_only[0], result.left_only[1]);
        int q = std::max(result.left_only[0], result.left_only[1]);
        int phase1 = cre_des_sign_t(m, p, self);
        StorageType tmp = self;
        ::trimci_core::BitOps<StorageType>::set_bit(tmp, m);
        ::trimci_core::BitOps<StorageType>::clear_bit(tmp, p);
        int phase2 = cre_des_sign_t(n, q, tmp);
        result.permutation_factor = phase1 * phase2;
    } else {
        result.permutation_factor = 0;
    }
    return result;
}

template<typename StorageType>
int permfac_ccaa(StorageType halfket, unsigned p, unsigned q, unsigned r, unsigned s) {
    int counter = 0;
    unsigned orbs_a[2] = {s, r};
    for (unsigned i = 0; i < 2; ++i) {
        unsigned orb = orbs_a[i];
        counter += count_bits_below(halfket, static_cast<int>(orb));
        ::trimci_core::BitOps<StorageType>::clear_bit(halfket, static_cast<int>(orb));
    }
    unsigned orbs_c[2] = {q, p};
    for (unsigned i = 0; i < 2; ++i) {
        unsigned orb = orbs_c[i];
        counter += count_bits_below(halfket, static_cast<int>(orb));
        ::trimci_core::BitOps<StorageType>::set_bit(halfket, static_cast<int>(orb));
    }
    return (counter % 2 == 0) ? 1 : -1;
}

inline size_t combine4_2rdm(int n_orb, unsigned p, unsigned q, unsigned r, unsigned s) {
    size_t a = static_cast<size_t>(p) * n_orb + s;
    size_t b = static_cast<size_t>(q) * n_orb + r;
    if (a >= b) {
        return (a * (a + 1)) / 2 + b;
    }
    return (b * (b + 1)) / 2 + a;
}

template<typename StorageType>
std::vector<StorageType> get_singles(const StorageType& config, int n_orb) {
    std::vector<StorageType> singles;
    if constexpr (std::is_same_v<StorageType, uint64_t>) {
        uint64_t c = config;
        uint64_t holes = (~c) & (n_orb == 64 ? ~0ULL : ((uint64_t(1) << n_orb) - 1ULL));
        while (c) {
            int p = __builtin_ctzll(c);
            c &= ~(uint64_t(1) << p);
            uint64_t h = holes;
            while (h) {
                int q = __builtin_ctzll(h);
                h &= ~(uint64_t(1) << q);
                uint64_t new_conf = (config ^ (uint64_t(1) << p)) | (uint64_t(1) << q);
                singles.push_back(new_conf);
            }
        }
    } else {
        for (int p = 0; p < n_orb; ++p) {
            if (!::trimci_core::BitOps<StorageType>::get_bit(config, p)) continue;
            for (int q = 0; q < n_orb; ++q) {
                if (::trimci_core::BitOps<StorageType>::get_bit(config, q)) continue;
                StorageType new_conf = config;
                ::trimci_core::BitOps<StorageType>::flip_bit(new_conf, p);
                ::trimci_core::BitOps<StorageType>::flip_bit(new_conf, q);
                singles.push_back(new_conf);
            }
        }
    }
    return singles;
}

template<typename StorageType>
std::vector<std::vector<int>>
build_connections_upper_t(const std::vector<DeterminantT<StorageType>>& dets, int n_orb) {
    const int dim = static_cast<int>(dets.size());
    std::vector<std::vector<int>> connections(static_cast<size_t>(dim));

    struct StorageHash {
        size_t operator()(const StorageType& s) const noexcept {
            if constexpr (std::is_same_v<StorageType, uint64_t>) {
                return std::hash<uint64_t>{}(s);
            } else {
                size_t seed = 0;
                for (const auto& v : s) {
                    seed ^= std::hash<uint64_t>{}(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
                }
                return seed;
            }
        }
    };

    std::unordered_map<StorageType, std::vector<int>, StorageHash> alpha_map;
    std::unordered_map<StorageType, std::vector<int>, StorageHash> beta_map;
    alpha_map.reserve(static_cast<size_t>(dim));
    beta_map.reserve(static_cast<size_t>(dim));
    for (int i = 0; i < dim; ++i) {
        alpha_map[dets[i].alpha].push_back(i);
        beta_map[dets[i].beta].push_back(i);
    }

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < dim; ++i) {
        std::vector<int> local;
        local.reserve(64);

        {
            auto it = beta_map.find(dets[i].beta);
            if (it != beta_map.end()) {
                for (int j : it->second) {
                    if (j > i) continue;
                    int diff = HamiltonianBitOps<StorageType>::count_differences(
                        dets[i].alpha, dets[j].alpha);
                    if (diff <= 4) {
                        local.push_back(j);
                    }
                }
            }
        }

        {
            auto it = alpha_map.find(dets[i].alpha);
            if (it != alpha_map.end()) {
                for (int j : it->second) {
                    if (j >= i) continue;
                    int diff = HamiltonianBitOps<StorageType>::count_differences(
                        dets[i].beta, dets[j].beta);
                    if (diff <= 4) {
                        local.push_back(j);
                    }
                }
            }
        }

        {
            auto a_singles = get_singles(dets[i].alpha, n_orb);
            for (const auto& a_new : a_singles) {
                auto it = alpha_map.find(a_new);
                if (it == alpha_map.end()) continue;
                for (int j : it->second) {
                    if (j >= i) continue;
                    int b_diff = HamiltonianBitOps<StorageType>::count_differences(
                        dets[i].beta, dets[j].beta);
                    if (b_diff == 2) {
                        local.push_back(j);
                    }
                }
            }
        }

        std::sort(local.begin(), local.end());
        local.erase(std::unique(local.begin(), local.end()), local.end());
        connections[static_cast<size_t>(i)] = std::move(local);
    }

    return connections;
}

struct DenseIntegrals {
    int n_orb = 0;
    size_t n_orb_2 = 0;
    size_t n_orb_3 = 0;
    std::vector<double> two_body;
    std::vector<double> one_body;

    void allocate(int n) {
        n_orb = n;
        n_orb_2 = static_cast<size_t>(n) * n;
        n_orb_3 = n_orb_2 * n;
        two_body.assign(n_orb_3 * n, 0.0);
        one_body.assign(n_orb_2, 0.0);
    }

    inline double& get_2b(int p, int q, int r, int s) {
        size_t ind = static_cast<size_t>(p) * n_orb_3 +
                     static_cast<size_t>(q) * n_orb_2 +
                     static_cast<size_t>(r) * n_orb +
                     static_cast<size_t>(s);
        return two_body[ind];
    }

    inline double get_2b(int p, int q, int r, int s) const {
        size_t ind = static_cast<size_t>(p) * n_orb_3 +
                     static_cast<size_t>(q) * n_orb_2 +
                     static_cast<size_t>(r) * n_orb +
                     static_cast<size_t>(s);
        return two_body[ind];
    }

    inline double& get_1b(int p, int q) {
        size_t ind = static_cast<size_t>(p) * n_orb + static_cast<size_t>(q);
        return one_body[ind];
    }

    inline double get_1b(int p, int q) const {
        size_t ind = static_cast<size_t>(p) * n_orb + static_cast<size_t>(q);
        return one_body[ind];
    }
};

} // namespace detail

template<typename StorageType>
class RDMBuilderT {
public:
    RDMBuilderT(int n_orb,
                int n_alpha,
                int n_beta,
                const std::vector<DeterminantT<StorageType>>& dets,
                const std::vector<double>& coeffs)
        : dets_(dets),
          coeffs_(coeffs),
          n_orb_(n_orb),
          n_up_(n_alpha),
          n_dn_(n_beta) {}

    void compute_2rdm(const std::vector<std::vector<int>>& connections) {
        const size_t n_orb_ll = static_cast<size_t>(n_orb_);
        const size_t size_two_rdm = n_orb_ll * n_orb_ll * (n_orb_ll * n_orb_ll + 1) / 2;
        two_rdm_.assign(size_two_rdm, 0.0);

        // Thread-local buffers to avoid atomic writes
        const int n_threads = omp_get_max_threads();
        std::vector<std::vector<double>> local_rdms(n_threads,
            std::vector<double>(size_two_rdm, 0.0));

#pragma omp parallel
        {
            double* my_buf = local_rdms[omp_get_thread_num()].data();
#pragma omp for schedule(dynamic, 10)
            for (int i_det = 0; i_det < static_cast<int>(connections.size()); ++i_det) {
                const auto& this_det = dets_[i_det];
                for (size_t j_pos = 0; j_pos < connections[i_det].size(); ++j_pos) {
                    size_t j_det = static_cast<size_t>(connections[i_det][j_pos]);
                    const auto& connected_det = dets_[j_det];
                    get_2rdm_elements(connected_det, j_det, this_det, i_det, my_buf);
                }
            }
        }

        // Reduce thread-local buffers
        for (int t = 0; t < n_threads; ++t) {
            const double* src = local_rdms[t].data();
            for (size_t k = 0; k < size_two_rdm; ++k) {
                two_rdm_[k] += src[k];
            }
        }
    }

    void compute_1rdm_from_2rdm() {
        one_rdm_ = Eigen::MatrixXd::Zero(n_orb_, n_orb_);
        const double denom = static_cast<double>(n_up_ + n_dn_ - 1);
        if (denom <= 0.0) return;
        for (int p = 0; p < n_orb_; ++p) {
            for (int s = 0; s < n_orb_; ++s) {
                double val = 0.0;
                for (int k = 0; k < n_orb_; ++k) {
                    val += two_rdm_elem(p, k, k, s) / denom;
                }
                one_rdm_(p, s) = val;
            }
        }
    }

    const Eigen::MatrixXd& one_rdm_matrix() const { return one_rdm_; }

    double two_rdm_elem(unsigned p, unsigned q, unsigned r, unsigned s) const {
        return two_rdm_[detail::combine4_2rdm(n_orb_, p, q, r, s)];
    }

    double energy_from_integrals(const std::vector<std::vector<double>>& h1,
                                 const std::vector<double>& eri) const {
        double onebody = 0.0;
        double twobody = 0.0;
        for (int p = 0; p < n_orb_; ++p) {
            for (int q = 0; q < n_orb_; ++q) {
                onebody += one_rdm_(p, q) * h1[static_cast<size_t>(p)][static_cast<size_t>(q)];
            }
        }
        for (int p = 0; p < n_orb_; ++p) {
            for (int q = 0; q < n_orb_; ++q) {
                for (int r = 0; r < n_orb_; ++r) {
                    for (int s = 0; s < n_orb_; ++s) {
                        twobody += 0.5 * two_rdm_elem(p, q, r, s) *
                                   get_eri(eri, n_orb_, p, s, q, r);
                    }
                }
            }
        }
        return onebody + twobody;
    }

private:
    void get_2rdm_elements(const DeterminantT<StorageType>& connected_det,
                           size_t j_det,
                           const DeterminantT<StorageType>& this_det,
                           size_t i_det,
                           double* buf) {
        auto occ_up = detail::HamiltonianBitOps<StorageType>::storage_to_indices(this_det.alpha);
        auto occ_dn = detail::HamiltonianBitOps<StorageType>::storage_to_indices(this_det.beta);

        auto diff_up = detail::diff_bits(connected_det.alpha, this_det.alpha);
        auto diff_dn = detail::diff_bits(connected_det.beta, this_det.beta);
        auto diff_up_rev = detail::diff_bits(this_det.alpha, connected_det.alpha);
        auto diff_dn_rev = detail::diff_bits(this_det.beta, connected_det.beta);

        if (diff_up.n_diffs == 0) {
            if (diff_dn.n_diffs == 0) {
                for (unsigned i_elec = 0; i_elec < static_cast<unsigned>(n_up_); ++i_elec) {
                    for (unsigned j_elec = i_elec + 1; j_elec < static_cast<unsigned>(n_up_); ++j_elec) {
                        unsigned s = static_cast<unsigned>(occ_up[i_elec]);
                        unsigned r = static_cast<unsigned>(occ_up[j_elec]);
                        write_in_2rdm(s, r, r, s, 1.0, i_det, j_det, buf);
                        write_in_2rdm(s, r, s, r, -1.0, i_det, j_det, buf);
                        write_in_2rdm(r, s, s, r, 1.0, i_det, j_det, buf);
                        write_in_2rdm(r, s, r, s, -1.0, i_det, j_det, buf);
                    }
                }
                for (unsigned i_elec = 0; i_elec < static_cast<unsigned>(n_dn_); ++i_elec) {
                    for (unsigned j_elec = i_elec + 1; j_elec < static_cast<unsigned>(n_dn_); ++j_elec) {
                        unsigned s = static_cast<unsigned>(occ_dn[i_elec]);
                        unsigned r = static_cast<unsigned>(occ_dn[j_elec]);
                        write_in_2rdm(s, r, r, s, 1.0, i_det, j_det, buf);
                        write_in_2rdm(s, r, s, r, -1.0, i_det, j_det, buf);
                        write_in_2rdm(r, s, s, r, 1.0, i_det, j_det, buf);
                        write_in_2rdm(r, s, r, s, -1.0, i_det, j_det, buf);
                    }
                }
                for (unsigned i_elec = 0; i_elec < static_cast<unsigned>(n_up_); ++i_elec) {
                    for (unsigned j_elec = 0; j_elec < static_cast<unsigned>(n_dn_); ++j_elec) {
                        unsigned s = static_cast<unsigned>(occ_up[i_elec]);
                        unsigned r = static_cast<unsigned>(occ_dn[j_elec]);
                        write_in_2rdm(s, r, r, s, 1.0, i_det, j_det, buf);
                        write_in_2rdm(r, s, s, r, 1.0, i_det, j_det, buf);
                    }
                }
            } else if (diff_dn.n_diffs == 1) {
                unsigned b1 = static_cast<unsigned>(diff_dn.right_only[0]);
                unsigned b2 = static_cast<unsigned>(diff_dn.left_only[0]);
                for (unsigned i_elec = 0; i_elec < static_cast<unsigned>(n_dn_); ++i_elec) {
                    unsigned p = static_cast<unsigned>(occ_dn[i_elec]);
                    if ((p != b1) && (p != b2)) {
                        double signed_factor = detail::permfac_ccaa(this_det.beta, p, b2, b1, p);
                        write_in_2rdm(p, b2, b1, p, signed_factor, i_det, j_det, buf);
                        write_in_2rdm(b2, p, b1, p, -signed_factor, i_det, j_det, buf);
                        write_in_2rdm(p, b2, p, b1, -signed_factor, i_det, j_det, buf);
                        write_in_2rdm(b2, p, p, b1, signed_factor, i_det, j_det, buf);

                        signed_factor = detail::permfac_ccaa(connected_det.beta, p, b1, b2, p);
                        write_in_2rdm(p, b1, b2, p, signed_factor, i_det, j_det, buf);
                        write_in_2rdm(b1, p, b2, p, -signed_factor, i_det, j_det, buf);
                        write_in_2rdm(p, b1, p, b2, -signed_factor, i_det, j_det, buf);
                        write_in_2rdm(b1, p, p, b2, signed_factor, i_det, j_det, buf);
                    }
                }
                for (unsigned i_elec = 0; i_elec < static_cast<unsigned>(n_up_); ++i_elec) {
                    unsigned p = static_cast<unsigned>(occ_up[i_elec]);
                    double signed_factor = static_cast<double>(diff_dn.permutation_factor);
                    write_in_2rdm(p, b2, b1, p, signed_factor, i_det, j_det, buf);
                    write_in_2rdm(b2, p, p, b1, signed_factor, i_det, j_det, buf);

                    signed_factor = static_cast<double>(diff_dn_rev.permutation_factor);
                    write_in_2rdm(p, b1, b2, p, signed_factor, i_det, j_det, buf);
                    write_in_2rdm(b1, p, p, b2, signed_factor, i_det, j_det, buf);
                }
            } else if (diff_dn.n_diffs == 2) {
                unsigned b1 = static_cast<unsigned>(diff_dn.right_only[0]);
                unsigned b2 = static_cast<unsigned>(diff_dn.right_only[1]);
                unsigned b3 = static_cast<unsigned>(diff_dn.left_only[0]);
                unsigned b4 = static_cast<unsigned>(diff_dn.left_only[1]);
                double signed_factor = static_cast<double>(diff_dn.permutation_factor);
                write_in_2rdm(b3, b4, b2, b1, signed_factor, i_det, j_det, buf);
                write_in_2rdm(b3, b4, b1, b2, -signed_factor, i_det, j_det, buf);
                write_in_2rdm(b4, b3, b2, b1, -signed_factor, i_det, j_det, buf);
                write_in_2rdm(b4, b3, b1, b2, signed_factor, i_det, j_det, buf);

                signed_factor = static_cast<double>(diff_dn_rev.permutation_factor);
                write_in_2rdm(b1, b2, b4, b3, signed_factor, i_det, j_det, buf);
                write_in_2rdm(b1, b2, b3, b4, -signed_factor, i_det, j_det, buf);
                write_in_2rdm(b2, b1, b4, b3, -signed_factor, i_det, j_det, buf);
                write_in_2rdm(b2, b1, b3, b4, signed_factor, i_det, j_det, buf);
            }
        } else if (diff_up.n_diffs == 1) {
            if (diff_dn.n_diffs == 0) {
                unsigned a1 = static_cast<unsigned>(diff_up.right_only[0]);
                unsigned a2 = static_cast<unsigned>(diff_up.left_only[0]);
                for (unsigned i_elec = 0; i_elec < static_cast<unsigned>(n_up_); ++i_elec) {
                    unsigned p = static_cast<unsigned>(occ_up[i_elec]);
                    if ((p != a1) && (p != a2)) {
                        double signed_factor = detail::permfac_ccaa(this_det.alpha, p, a2, a1, p);
                        write_in_2rdm(p, a2, a1, p, signed_factor, i_det, j_det, buf);
                        write_in_2rdm(p, a2, p, a1, -signed_factor, i_det, j_det, buf);
                        write_in_2rdm(a2, p, a1, p, -signed_factor, i_det, j_det, buf);
                        write_in_2rdm(a2, p, p, a1, signed_factor, i_det, j_det, buf);

                        signed_factor = detail::permfac_ccaa(connected_det.alpha, p, a1, a2, p);
                        write_in_2rdm(p, a1, a2, p, signed_factor, i_det, j_det, buf);
                        write_in_2rdm(p, a1, p, a2, -signed_factor, i_det, j_det, buf);
                        write_in_2rdm(a1, p, a2, p, -signed_factor, i_det, j_det, buf);
                        write_in_2rdm(a1, p, p, a2, signed_factor, i_det, j_det, buf);
                    }
                }
                for (unsigned i_elec = 0; i_elec < static_cast<unsigned>(n_dn_); ++i_elec) {
                    unsigned p = static_cast<unsigned>(occ_dn[i_elec]);
                    double signed_factor = static_cast<double>(diff_up.permutation_factor);
                    write_in_2rdm(a2, p, p, a1, signed_factor, i_det, j_det, buf);
                    write_in_2rdm(p, a2, a1, p, signed_factor, i_det, j_det, buf);

                    signed_factor = static_cast<double>(diff_up_rev.permutation_factor);
                    write_in_2rdm(a1, p, p, a2, signed_factor, i_det, j_det, buf);
                    write_in_2rdm(p, a1, a2, p, signed_factor, i_det, j_det, buf);
                }
            } else if (diff_dn.n_diffs == 1) {
                unsigned a1 = static_cast<unsigned>(diff_up.right_only[0]);
                unsigned a2 = static_cast<unsigned>(diff_up.left_only[0]);
                unsigned b1 = static_cast<unsigned>(diff_dn.right_only[0]);
                unsigned b2 = static_cast<unsigned>(diff_dn.left_only[0]);
                double signed_factor = static_cast<double>(diff_up.permutation_factor * diff_dn.permutation_factor);
                write_in_2rdm(a2, b2, b1, a1, signed_factor, i_det, j_det, buf);
                write_in_2rdm(b2, a2, a1, b1, signed_factor, i_det, j_det, buf);

                signed_factor = static_cast<double>(diff_up_rev.permutation_factor * diff_dn_rev.permutation_factor);
                write_in_2rdm(a1, b1, b2, a2, signed_factor, i_det, j_det, buf);
                write_in_2rdm(b1, a1, a2, b2, signed_factor, i_det, j_det, buf);
            }
        } else if (diff_up.n_diffs == 2) {
            if (diff_dn.n_diffs == 0) {
                unsigned a1 = static_cast<unsigned>(diff_up.right_only[0]);
                unsigned a2 = static_cast<unsigned>(diff_up.right_only[1]);
                unsigned a3 = static_cast<unsigned>(diff_up.left_only[0]);
                unsigned a4 = static_cast<unsigned>(diff_up.left_only[1]);
                double signed_factor = static_cast<double>(diff_up.permutation_factor);
                write_in_2rdm(a3, a4, a2, a1, signed_factor, i_det, j_det, buf);
                write_in_2rdm(a3, a4, a1, a2, -signed_factor, i_det, j_det, buf);
                write_in_2rdm(a4, a3, a2, a1, -signed_factor, i_det, j_det, buf);
                write_in_2rdm(a4, a3, a1, a2, signed_factor, i_det, j_det, buf);

                signed_factor = static_cast<double>(diff_up_rev.permutation_factor);
                write_in_2rdm(a1, a2, a4, a3, signed_factor, i_det, j_det, buf);
                write_in_2rdm(a1, a2, a3, a4, -signed_factor, i_det, j_det, buf);
                write_in_2rdm(a2, a1, a4, a3, -signed_factor, i_det, j_det, buf);
                write_in_2rdm(a2, a1, a3, a4, signed_factor, i_det, j_det, buf);
            }
        }
    }

    void write_in_2rdm(unsigned p, unsigned q, unsigned r, unsigned s, double factor,
                       size_t i_det, size_t j_det, double* buf) {
        size_t a = static_cast<size_t>(p) * n_orb_ + s;
        size_t b = static_cast<size_t>(q) * n_orb_ + r;
        if (a >= b) {
            size_t idx = (a * (a + 1)) / 2 + b;
            double value = coeffs_[i_det] * coeffs_[j_det] * factor;
            buf[idx] += value;
        }
    }

    const std::vector<DeterminantT<StorageType>>& dets_;
    const std::vector<double>& coeffs_;
    int n_orb_;
    int n_up_;
    int n_dn_;
    Eigen::MatrixXd one_rdm_;
    std::vector<double> two_rdm_;
};

template<typename StorageType>
Eigen::MatrixXd generalized_fock_from_rdm(
    const Eigen::MatrixXd& one_rdm,
    const RDMBuilderT<StorageType>& rdm,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    int n_orb) {
    Eigen::MatrixXd fock(n_orb, n_orb);
#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int m = 0; m < n_orb; ++m) {
        for (int n = 0; n < n_orb; ++n) {
            double elem = 0.0;
            for (int q = 0; q < n_orb; ++q) {
                elem += one_rdm(m, q) * h1[static_cast<size_t>(n)][static_cast<size_t>(q)];
            }
            for (int q = 0; q < n_orb; ++q) {
                for (int r = 0; r < n_orb; ++r) {
                    for (int s = 0; s < n_orb; ++s) {
                        elem += rdm.two_rdm_elem(m, r, s, q) *
                                get_eri(eri, n_orb, n, q, r, s);
                    }
                }
            }
            fock(m, n) = elem;
        }
    }
    return fock;
}

template<typename StorageType>
Eigen::MatrixXd compute_rotation_from_parameters(
    const Eigen::VectorXd& params,
    const std::vector<std::pair<int, int>>& param_indices,
    int n_orb) {
    Eigen::MatrixXd X = Eigen::MatrixXd::Zero(n_orb, n_orb);
    for (size_t i = 0; i < param_indices.size(); ++i) {
        int p = param_indices[i].first;
        int q = param_indices[i].second;
        double val = params(static_cast<Eigen::Index>(i));
        X(p, q) = -val;
        X(q, p) = val;
    }

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(X * X);
    Eigen::VectorXd tau2 = es.eigenvalues();
    Eigen::MatrixXd W = es.eigenvectors();
    Eigen::MatrixXd rot = Eigen::MatrixXd::Zero(n_orb, n_orb);

#pragma omp parallel for
    for (int i = 0; i < n_orb; ++i) {
        for (int j = 0; j < n_orb; ++j) {
            double val = 0.0;
            for (int k = 0; k < n_orb; ++k) {
                double cos_tau = 1.0;
                double sinc_tau = 1.0;
                if (std::abs(tau2(k)) >= 1e-10) {
                    double tau = std::sqrt(-tau2(k));
                    cos_tau = std::cos(tau);
                    sinc_tau = std::sin(tau) / tau;
                }
                val += cos_tau * W(i, k) * W(j, k);
                double sum_l = 0.0;
                for (int l = 0; l < n_orb; ++l) {
                    sum_l += W(l, k) * X(l, j);
                }
                val += sinc_tau * W(i, k) * sum_l;
            }
            rot(i, j) = val;
        }
    }

    return rot;
}

static void rotate_integrals_with_matrix(
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    const Eigen::MatrixXd& rot,
    std::vector<std::vector<double>>& h1_out,
    std::vector<double>& eri_out) {
    const int N = static_cast<int>(h1.size());
    const size_t N2 = static_cast<size_t>(N) * N;
    const size_t N3 = N2 * N;
    const size_t N4 = N3 * N;

    // Two buffers for ping-pong (row-major: data[p*N3 + q*N2 + r*N + s])
    std::vector<double> cur(N4), next(N4);
    std::copy(eri.begin(), eri.begin() + static_cast<ptrdiff_t>(N4), cur.begin());

    // Eigen column-major Map on row-major data:
    //   Map(data, rows, cols): element(i,j) = data[i + j*rows]
    //   Row-major data[outer*stride + inner] maps to element(inner, outer).

    // Pass 1: Transform 1st index (p, stride N^3)
    // Map(data, N^3, N): element(qrs, p) = data[p*N^3 + qrs]
    // next = cur * rot  (each column p of result is Σ_i cur(:,i)*rot(i,p))
    {
        Eigen::Map<Eigen::MatrixXd> cm(cur.data(), N3, N);
        Eigen::Map<Eigen::MatrixXd> nm(next.data(), N3, N);
        nm.noalias() = cm * rot;
    }
    std::swap(cur, next);

    // Pass 2: Transform 2nd index (q, stride N^2)
    // For each p-slice: Map(slice, N^2, N)
    #pragma omp parallel for schedule(static)
    for (int p = 0; p < N; ++p) {
        Eigen::Map<Eigen::MatrixXd> cm(cur.data() + p * static_cast<ptrdiff_t>(N3), static_cast<int>(N2), N);
        Eigen::Map<Eigen::MatrixXd> nm(next.data() + p * static_cast<ptrdiff_t>(N3), static_cast<int>(N2), N);
        nm.noalias() = cm * rot;
    }
    std::swap(cur, next);

    // Pass 3: Transform 3rd index (r, stride N)
    // For each (p,q)-slice: Map(slice, N, N)
    #pragma omp parallel for schedule(static)
    for (int pq = 0; pq < static_cast<int>(N2); ++pq) {
        Eigen::Map<Eigen::MatrixXd> cm(cur.data() + pq * static_cast<ptrdiff_t>(N2), N, N);
        Eigen::Map<Eigen::MatrixXd> nm(next.data() + pq * static_cast<ptrdiff_t>(N2), N, N);
        nm.noalias() = cm * rot;
    }
    std::swap(cur, next);

    // Pass 4: Transform 4th index (s, stride 1)
    // Map(data, N, N^3): element(s, pqr) = data[pqr*N + s]
    // next = rot^T * cur
    {
        Eigen::Map<Eigen::MatrixXd> cm(cur.data(), N, N3);
        Eigen::Map<Eigen::MatrixXd> nm(next.data(), N, N3);
        nm.noalias() = rot.transpose() * cm;
    }

    eri_out = std::move(next);

    // h1 rotation: h1_out = rot^T * h1 * rot
    Eigen::MatrixXd h1_mat(N, N);
    for (int p = 0; p < N; ++p)
        for (int q = 0; q < N; ++q)
            h1_mat(p, q) = h1[static_cast<size_t>(p)][static_cast<size_t>(q)];
    Eigen::MatrixXd h1_rot = rot.transpose() * h1_mat * rot;
    h1_out.assign(static_cast<size_t>(N), std::vector<double>(static_cast<size_t>(N), 0.0));
    for (int p = 0; p < N; ++p)
        for (int q = 0; q < N; ++q)
            h1_out[static_cast<size_t>(p)][static_cast<size_t>(q)] = h1_rot(p, q);
}

template<typename StorageType>
Eigen::MatrixXd orbital_gradient_t(
    const std::vector<DeterminantT<StorageType>>& dets,
    const std::vector<double>& coeffs,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    int n_orb,
    int n_alpha,
    int n_beta) {
    if (dets.empty() || coeffs.empty()) {
        return Eigen::MatrixXd();
    }
    int n_orb_use = n_orb > 0 ? n_orb : static_cast<int>(h1.size());
    int n_alpha_use = n_alpha;
    int n_beta_use = n_beta;
    if (n_alpha_use <= 0 || n_beta_use <= 0) {
        n_alpha_use = detail::popcount_storage(dets[0].alpha);
        n_beta_use = detail::popcount_storage(dets[0].beta);
    }

    auto connections = detail::build_connections_upper_t(dets, n_orb_use);
    RDMBuilderT<StorageType> rdm(n_orb_use, n_alpha_use, n_beta_use, dets, coeffs);
    rdm.compute_2rdm(connections);
    rdm.compute_1rdm_from_2rdm();

    Eigen::MatrixXd fock = generalized_fock_from_rdm(rdm.one_rdm_matrix(), rdm, h1, eri, n_orb_use);
    Eigen::MatrixXd grad = 2.0 * (fock - fock.transpose());
    return grad;
}

template<typename StorageType>
struct ApproxNewtonParamsT {
    std::vector<std::pair<int, int>> param_indices;
    Eigen::VectorXd params_base;
    double grad_norm = 0.0;
};

template<typename StorageType>
static ApproxNewtonParamsT<StorageType> compute_approx_newton_params(
    const std::vector<DeterminantT<StorageType>>& dets,
    const std::vector<double>& coeffs,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    int n_orb,
    int n_alpha,
    int n_beta,
    double min_hess) {
    ApproxNewtonParamsT<StorageType> out;
    if (dets.empty() || coeffs.empty()) {
        return out;
    }

    auto connections = detail::build_connections_upper_t(dets, n_orb);
    RDMBuilderT<StorageType> rdm(n_orb, n_alpha, n_beta, dets, coeffs);
    rdm.compute_2rdm(connections);
    rdm.compute_1rdm_from_2rdm();

    Eigen::MatrixXd fock = generalized_fock_from_rdm(rdm.one_rdm_matrix(), rdm, h1, eri, n_orb);

    out.param_indices.reserve(static_cast<size_t>(n_orb * (n_orb - 1) / 2));
    for (int i = 0; i < n_orb; ++i) {
        for (int j = i + 1; j < n_orb; ++j) {
            out.param_indices.emplace_back(i, j);
        }
    }

    const Eigen::Index n_params = static_cast<Eigen::Index>(out.param_indices.size());
    if (n_params == 0) {
        return out;
    }

    Eigen::VectorXd grad(n_params);
    Eigen::VectorXd hess_diag(n_params);
#pragma omp parallel for schedule(dynamic)
    for (Eigen::Index idx = 0; idx < n_params; ++idx) {
        int p = out.param_indices[static_cast<size_t>(idx)].first;
        int q = out.param_indices[static_cast<size_t>(idx)].second;
        grad(idx) = 2.0 * (fock(p, q) - fock(q, p));

        auto hessian_part = [&](int p0, int q0, int r0, int s0) {
            double elem = 0.0;
            elem += 2.0 * rdm.one_rdm_matrix()(p0, r0) * h1[static_cast<size_t>(q0)][static_cast<size_t>(s0)];
            if (q0 == s0) {
                elem -= (fock(p0, r0) + fock(r0, p0));
            }
            double y_elem = 0.0;
            for (int m = 0; m < n_orb; ++m) {
                for (int n = 0; n < n_orb; ++n) {
                    y_elem += (rdm.two_rdm_elem(p0, r0, n, m) + rdm.two_rdm_elem(p0, n, r0, m)) *
                              get_eri(eri, n_orb, q0, m, n, s0);
                    y_elem += rdm.two_rdm_elem(p0, m, n, r0) *
                              get_eri(eri, n_orb, q0, s0, m, n);
                }
            }
            elem += 2.0 * y_elem;
            return elem;
        };

        double hdiag = hessian_part(p, q, p, q) - 2.0 * hessian_part(p, q, q, p) +
                       hessian_part(q, p, q, p);
        hess_diag(idx) = hdiag;
    }

    out.params_base.resize(n_params);
    for (Eigen::Index i = 0; i < n_params; ++i) {
        double denom = std::max(hess_diag(i), min_hess);
        out.params_base(i) = -grad(i) / denom;
    }
    out.grad_norm = grad.norm();
    return out;
}

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
    int n_alpha,
    int n_beta,
    bool rotate_integrals,
    double step_scale,
    double min_hess,
    double max_param) {
    Eigen::MatrixXd rot;
    std::vector<std::vector<double>> h1_rot;
    std::vector<double> eri_rot;
    if (dets.empty() || coeffs.empty()) {
        return {rot, h1_rot, eri_rot, 0.0, 0.0};
    }
    int n_orb_use = n_orb > 0 ? n_orb : static_cast<int>(h1.size());
    int n_alpha_use = n_alpha;
    int n_beta_use = n_beta;
    if (n_alpha_use <= 0 || n_beta_use <= 0) {
        n_alpha_use = detail::popcount_storage(dets[0].alpha);
        n_beta_use = detail::popcount_storage(dets[0].beta);
    }
    auto params_data = compute_approx_newton_params(dets, coeffs, h1, eri, n_orb_use,
                                                    n_alpha_use, n_beta_use, min_hess);
    Eigen::VectorXd params = params_data.params_base * step_scale;
    if (!params.allFinite()) {
        params.setZero();
    }
    if (max_param > 0.0 && params.size() > 0) {
        double max_abs = params.cwiseAbs().maxCoeff();
        if (max_abs > max_param) {
            params *= (max_param / max_abs);
        }
    }

    rot = compute_rotation_from_parameters<StorageType>(params, params_data.param_indices, n_orb_use);
    if (rotate_integrals) {
        rotate_integrals_with_matrix(h1, eri, rot, h1_rot, eri_rot);
    }

    double grad_norm = params_data.grad_norm;
    double energy = 0.0;
    if (rotate_integrals) {
        auto connections = detail::build_connections_upper_t(dets, n_orb_use);
        RDMBuilderT<StorageType> rdm(n_orb_use, n_alpha_use, n_beta_use, dets, coeffs);
        rdm.compute_2rdm(connections);
        rdm.compute_1rdm_from_2rdm();
        energy = rdm.energy_from_integrals(h1_rot, eri_rot);
    }

    return {rot, h1_rot, eri_rot, grad_norm, energy};
}

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
    int n_alpha,
    int n_beta,
    int max_iter,
    double grad_tol,
    double energy_tol,
    double step_scale,
    double min_hess,
    double max_param,
    bool line_search,
    int ls_max_iter,
    double ls_shrink,
    double ls_energy_tol,
    double davidson_tol,
    int davidson_max_iter,
    bool rotate_integrals,
    bool record_trace) {
    Eigen::MatrixXd rot_total;
    std::vector<std::vector<double>> h1_out;
    std::vector<double> eri_out;
    std::vector<std::array<double, 2>> trace;

    if (dets.empty() || h1.empty()) {
        return {rot_total, h1_out, eri_out, 0.0, 0.0, false, 0, trace};
    }

    int n_orb_use = n_orb > 0 ? n_orb : static_cast<int>(h1.size());
    int n_alpha_use = n_alpha;
    int n_beta_use = n_beta;
    if (n_alpha_use <= 0 || n_beta_use <= 0) {
        n_alpha_use = detail::popcount_storage(dets[0].alpha);
        n_beta_use = detail::popcount_storage(dets[0].beta);
    }

    rot_total = Eigen::MatrixXd::Identity(n_orb_use, n_orb_use);
    std::vector<std::vector<double>> h1_cur = h1;
    std::vector<double> eri_cur = eri;

    HijCacheT<StorageType> cache;
    double energy = 0.0;
    std::vector<double> coeffs_cur = coeffs;
    std::tie(energy, coeffs_cur) =
        diagonalize_subspace_davidson_t(dets, h1_cur, eri_cur, cache, false,
                                        davidson_max_iter, davidson_tol, false, n_orb_use);

    if (max_iter <= 0) {
        if (rotate_integrals) {
            h1_out = std::move(h1_cur);
            eri_out = std::move(eri_cur);
        }
        return {rot_total, h1_out, eri_out, 0.0, energy, false, 0, trace};
    }

    bool converged = false;
    int iter_done = 0;
    double grad_norm = 0.0;
    double prev_energy = energy;
    double best_energy = energy;
    double best_grad_norm = grad_norm;
    Eigen::MatrixXd best_rot = rot_total;
    std::vector<std::vector<double>> best_h1 = h1_cur;
    std::vector<double> best_eri = eri_cur;
    bool fell_back = false;

    for (int iter = 0; iter < max_iter; ++iter) {
        auto params_data = compute_approx_newton_params(dets, coeffs_cur, h1_cur, eri_cur,
                                                        n_orb_use, n_alpha_use, n_beta_use, min_hess);
        grad_norm = params_data.grad_norm;
        iter_done = iter + 1;

        if (!std::isfinite(grad_norm)) {
            fell_back = true;
            break;
        }

        if (iter == 0 || energy < best_energy) {
            best_energy = energy;
            best_grad_norm = grad_norm;
            best_rot = rot_total;
            best_h1 = h1_cur;
            best_eri = eri_cur;
        }

        if (grad_norm < grad_tol) {
            if (record_trace) {
                trace.push_back({energy, grad_norm});
            }
            converged = true;
            break;
        }

        bool accepted = false;
        double alpha = 1.0;
        int ls_iters = line_search ? std::max(1, ls_max_iter) : 1;
        double energy_slack = std::max(ls_energy_tol,
                                       1e-12 * std::max(1.0, std::abs(prev_energy)));
        for (int ls = 0; ls < ls_iters; ++ls) {
            Eigen::VectorXd params = params_data.params_base * (step_scale * alpha);
            if (!params.allFinite()) {
                alpha *= ls_shrink;
                continue;
            }
            if (max_param > 0.0 && params.size() > 0) {
                double max_abs = params.cwiseAbs().maxCoeff();
                if (max_abs > max_param) {
                    params *= (max_param / max_abs);
                }
            }

            Eigen::MatrixXd rot_step =
                compute_rotation_from_parameters<StorageType>(params, params_data.param_indices, n_orb_use);
            if (rot_step.size() == 0) {
                break;
            }

            std::vector<std::vector<double>> h1_rot;
            std::vector<double> eri_rot;
            rotate_integrals_with_matrix(h1_cur, eri_cur, rot_step, h1_rot, eri_rot);

            double energy_try = 0.0;
            std::vector<double> coeffs_try;
            std::tie(energy_try, coeffs_try) =
                diagonalize_subspace_davidson_t(dets, h1_rot, eri_rot, cache, false,
                                                davidson_max_iter, davidson_tol, false, n_orb_use);

            if (!std::isfinite(energy_try)) {
                alpha *= ls_shrink;
                continue;
            }

            if (!line_search || energy_try <= prev_energy + energy_slack) {
                accepted = true;
                rot_total = rot_total * rot_step;
                h1_cur = std::move(h1_rot);
                eri_cur = std::move(eri_rot);
                energy = energy_try;
                coeffs_cur = std::move(coeffs_try);
                break;
            }

            alpha *= ls_shrink;
        }

        if (!accepted) {
            fell_back = true;
            break;
        }

        if (record_trace) {
            trace.push_back({energy, grad_norm});
        }

        double delta_energy = std::abs(energy - prev_energy);
        if (energy_tol > 0.0 && std::isfinite(delta_energy) && delta_energy < energy_tol) {
            converged = true;
            break;
        }
        prev_energy = energy;
    }

    if (fell_back) {
        rot_total = best_rot;
        h1_cur = std::move(best_h1);
        eri_cur = std::move(best_eri);
        energy = best_energy;
        grad_norm = best_grad_norm;
        converged = false;
    }

    if (rotate_integrals) {
        h1_out = std::move(h1_cur);
        eri_out = std::move(eri_cur);
    }

    return {rot_total, h1_out, eri_out, grad_norm, energy, converged, iter_done, trace};
}

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
    int n_alpha,
    int n_beta,
    int max_iter,
    double grad_tol,
    double energy_tol,
    double step_scale,
    double min_hess,
    double max_param,
    double bfgs_damp,
    double bfgs_update_tol,
    double bfgs_step_factor,
    double bfgs_step_decay,
    double bfgs_step_min,
    double bfgs_init_div,
    bool line_search,
    int ls_max_iter,
    double ls_shrink,
    double ls_energy_tol,
    double davidson_tol,
    int davidson_max_iter,
    bool rotate_integrals,
    bool record_trace,
    DiagFunc<StorageType> diag_func) {
    Eigen::MatrixXd rot_total;
    std::vector<std::vector<double>> h1_out;
    std::vector<double> eri_out;
    std::vector<std::array<double, 2>> trace;

    if (dets.empty() || h1.empty()) {
        return {rot_total, h1_out, eri_out, 0.0, 0.0, false, 0, trace};
    }

    int n_orb_use = n_orb > 0 ? n_orb : static_cast<int>(h1.size());
    int n_alpha_use = n_alpha;
    int n_beta_use = n_beta;
    if (n_alpha_use <= 0 || n_beta_use <= 0) {
        n_alpha_use = detail::popcount_storage(dets[0].alpha);
        n_beta_use = detail::popcount_storage(dets[0].beta);
    }

    rot_total = Eigen::MatrixXd::Identity(n_orb_use, n_orb_use);
    std::vector<std::vector<double>> h1_cur = h1;
    std::vector<double> eri_cur = eri;

    HijCacheT<StorageType> cache;
    double energy = 0.0;
    std::vector<double> coeffs_cur = coeffs;

    // Davidson dispatch: use callback if provided, otherwise standard solver
    auto do_diagonalize = [&](const std::vector<std::vector<double>>& h1_in,
                              const std::vector<double>& eri_in,
                              const std::vector<double>& guess_in)
        -> std::tuple<double, std::vector<double>>
    {
        if (diag_func) {
            return diag_func(h1_in, eri_in, guess_in);
        }
        return diagonalize_subspace_davidson_t(
            dets, h1_in, eri_in, cache, false,
            davidson_max_iter, davidson_tol, false, n_orb_use, guess_in);
    };

    std::tie(energy, coeffs_cur) = do_diagonalize(h1_cur, eri_cur, coeffs_cur);

    if (max_iter <= 0) {
        if (rotate_integrals) {
            h1_out = std::move(h1_cur);
            eri_out = std::move(eri_cur);
        }
        return {rot_total, h1_out, eri_out, 0.0, energy, false, 0, trace};
    }

    std::vector<std::pair<int, int>> param_indices;
    param_indices.reserve(static_cast<size_t>(n_orb_use * (n_orb_use - 1) / 2));
    for (int i = 0; i < n_orb_use; ++i) {
        for (int j = i + 1; j < n_orb_use; ++j) {
            param_indices.emplace_back(i, j);
        }
    }

    auto connections = detail::build_connections_upper_t(dets, n_orb_use);
    const Eigen::Index dim = static_cast<Eigen::Index>(param_indices.size());
    Eigen::VectorXd grad(dim);
    Eigen::VectorXd grad_prev = Eigen::VectorXd::Zero(dim);
    Eigen::VectorXd update_prev = Eigen::VectorXd::Zero(dim);
    Eigen::MatrixXd hess = Eigen::MatrixXd::Zero(dim, dim);

    bool converged = false;
    bool first_iter = true;
    int iter_done = 0;
    double grad_norm = 0.0;
    double prev_energy = energy;
    double best_energy = energy;
    double best_grad_norm = grad_norm;
    Eigen::MatrixXd best_rot = rot_total;
    std::vector<std::vector<double>> best_h1 = h1_cur;
    std::vector<double> best_eri = eri_cur;
    bool fell_back = false;
    double initial_update_norm = 0.0;
    double step_size_factor = bfgs_step_factor;

    // Profiling accumulators
    using ProfClock = std::chrono::high_resolution_clock;
    double prof_rdm_s = 0.0, prof_fock_s = 0.0, prof_rotate_s = 0.0, prof_diag_s = 0.0;

    for (int iter = 0; iter < max_iter; ++iter) {
        auto t_rdm0 = ProfClock::now();
        RDMBuilderT<StorageType> rdm(n_orb_use, n_alpha_use, n_beta_use, dets, coeffs_cur);
        rdm.compute_2rdm(connections);
        rdm.compute_1rdm_from_2rdm();
        prof_rdm_s += std::chrono::duration<double>(ProfClock::now() - t_rdm0).count();

        auto t_fock0 = ProfClock::now();
        Eigen::MatrixXd fock = generalized_fock_from_rdm(rdm.one_rdm_matrix(), rdm, h1_cur, eri_cur, n_orb_use);

#pragma omp parallel for schedule(dynamic)
        for (Eigen::Index idx = 0; idx < dim; ++idx) {
            int p = param_indices[static_cast<size_t>(idx)].first;
            int q = param_indices[static_cast<size_t>(idx)].second;
            grad(idx) = 2.0 * (fock(p, q) - fock(q, p));
        }
        prof_fock_s += std::chrono::duration<double>(ProfClock::now() - t_fock0).count();

        grad_norm = grad.norm();
        iter_done = iter + 1;

        if (!std::isfinite(grad_norm)) {
            fell_back = true;
            break;
        }

        if (iter == 0 || energy < best_energy) {
            best_energy = energy;
            best_grad_norm = grad_norm;
            best_rot = rot_total;
            best_h1 = h1_cur;
            best_eri = eri_cur;
        }

        if (first_iter) {
            Eigen::VectorXd hess_diag(dim);

            auto hessian_part = [&](int p0, int q0, int r0, int s0) {
                double elem = 0.0;
                elem += 2.0 * rdm.one_rdm_matrix()(p0, r0) *
                        h1_cur[static_cast<size_t>(q0)][static_cast<size_t>(s0)];
                if (q0 == s0) {
                    elem -= (fock(p0, r0) + fock(r0, p0));
                }
                double y_elem = 0.0;
                for (int m = 0; m < n_orb_use; ++m) {
                    for (int n = 0; n < n_orb_use; ++n) {
                        y_elem += (rdm.two_rdm_elem(p0, r0, n, m) + rdm.two_rdm_elem(p0, n, r0, m)) *
                                  get_eri(eri_cur, n_orb_use, q0, m, n, s0);
                        y_elem += rdm.two_rdm_elem(p0, m, n, r0) *
                                  get_eri(eri_cur, n_orb_use, q0, s0, m, n);
                    }
                }
                elem += 2.0 * y_elem;
                return elem;
            };

#pragma omp parallel for schedule(dynamic)
            for (Eigen::Index idx = 0; idx < dim; ++idx) {
                int p = param_indices[static_cast<size_t>(idx)].first;
                int q = param_indices[static_cast<size_t>(idx)].second;
                double hdiag = hessian_part(p, q, p, q) - 2.0 * hessian_part(p, q, q, p) +
                               hessian_part(q, p, q, p);
                hess_diag(idx) = std::max(hdiag / bfgs_init_div, min_hess);
            }

            hess = hess_diag.asDiagonal();
            grad_prev.setZero();
            update_prev.setZero();
        } else {
            const Eigen::VectorXd y = grad - grad_prev;
            const Eigen::VectorXd hs = hess * update_prev;
            const double ys = y.dot(update_prev);
            const double shs = hs.dot(update_prev);
            if (ys > bfgs_update_tol && std::abs(shs) > bfgs_update_tol) {
                hess += (y * y.transpose()) / ys - (hs * hs.transpose()) / shs;
            }
        }

        Eigen::MatrixXd hess_shift = hess;
        hess_shift.diagonal().array() += bfgs_damp;
        Eigen::VectorXd new_param = hess_shift.householderQr().solve(-grad);
        if (!new_param.allFinite()) {
            new_param.setZero();
        }
        if (step_scale != 1.0) {
            new_param *= step_scale;
        }

        double update_norm = new_param.norm();
        if (first_iter) {
            initial_update_norm = update_norm;
            first_iter = false;
        } else if (initial_update_norm > 0.0) {
            double max_norm = step_size_factor * initial_update_norm;
            if (update_norm > max_norm) {
                new_param *= max_norm / update_norm;
                update_norm = max_norm;
            }
            step_size_factor = std::max(bfgs_step_min, step_size_factor * bfgs_step_decay);
        }
        if (max_param > 0.0 && new_param.size() > 0) {
            double max_abs = new_param.cwiseAbs().maxCoeff();
            if (max_abs > max_param) {
                new_param *= (max_param / max_abs);
            }
        }

        if (grad_norm < grad_tol) {
            if (record_trace) {
                trace.push_back({energy, grad_norm});
            }
            converged = true;
            break;
        }

        bool accepted = false;
        double alpha = 1.0;
        int ls_iters = line_search ? std::max(1, ls_max_iter) : 1;
        Eigen::VectorXd accepted_param;
        double energy_slack = std::max(ls_energy_tol,
                                       1e-12 * std::max(1.0, std::abs(prev_energy)));
        for (int ls = 0; ls < ls_iters; ++ls) {
            Eigen::VectorXd params = new_param * alpha;
            if (!params.allFinite()) {
                alpha *= ls_shrink;
                continue;
            }
            Eigen::MatrixXd rot_step =
                compute_rotation_from_parameters<StorageType>(params, param_indices, n_orb_use);
            if (rot_step.size() == 0) {
                break;
            }

            auto t_rot0 = ProfClock::now();
            std::vector<std::vector<double>> h1_rot;
            std::vector<double> eri_rot;
            rotate_integrals_with_matrix(h1_cur, eri_cur, rot_step, h1_rot, eri_rot);
            prof_rotate_s += std::chrono::duration<double>(ProfClock::now() - t_rot0).count();

            auto t_diag0 = ProfClock::now();
            double energy_try = 0.0;
            std::vector<double> coeffs_try;
            std::tie(energy_try, coeffs_try) =
                do_diagonalize(h1_rot, eri_rot, coeffs_cur);
            prof_diag_s += std::chrono::duration<double>(ProfClock::now() - t_diag0).count();

            if (!std::isfinite(energy_try)) {
                alpha *= ls_shrink;
                continue;
            }

            if (!line_search || energy_try <= prev_energy + energy_slack) {
                accepted = true;
                accepted_param = std::move(params);
                rot_total = rot_total * rot_step;
                h1_cur = std::move(h1_rot);
                eri_cur = std::move(eri_rot);
                energy = energy_try;
                coeffs_cur = std::move(coeffs_try);
                break;
            }

            alpha *= ls_shrink;
        }

        if (!accepted) {
            fell_back = true;
            break;
        }

        if (record_trace) {
            trace.push_back({energy, grad_norm});
        }

        grad_prev = grad;
        update_prev = std::move(accepted_param);

        double delta_energy = std::abs(energy - prev_energy);
        if (energy_tol > 0.0 && delta_energy < energy_tol) {
            converged = true;
            break;
        }
        prev_energy = energy;
    }

    // Print profiling summary
    {
        double prof_total = prof_rdm_s + prof_fock_s + prof_rotate_s + prof_diag_s;
        std::cout << "  [OrbOpt Profile] " << iter_done << " iters: "
                  << "2RDM=" << std::fixed << std::setprecision(2) << prof_rdm_s << "s ("
                  << (prof_total > 0 ? prof_rdm_s/prof_total*100 : 0) << "%), "
                  << "Fock+grad=" << prof_fock_s << "s ("
                  << (prof_total > 0 ? prof_fock_s/prof_total*100 : 0) << "%), "
                  << "rotate=" << prof_rotate_s << "s ("
                  << (prof_total > 0 ? prof_rotate_s/prof_total*100 : 0) << "%), "
                  << "diag=" << prof_diag_s << "s ("
                  << (prof_total > 0 ? prof_diag_s/prof_total*100 : 0) << "%)"
                  << std::endl;
    }

    if (fell_back) {
        rot_total = best_rot;
        h1_cur = std::move(best_h1);
        eri_cur = std::move(best_eri);
        energy = best_energy;
        grad_norm = best_grad_norm;
        converged = false;
    }

    if (rotate_integrals) {
        h1_out = std::move(h1_cur);
        eri_out = std::move(eri_cur);
    }

    return {rot_total, h1_out, eri_out, grad_norm, energy, converged, iter_done, trace};
}

template Eigen::MatrixXd orbital_gradient_t<uint64_t>(
    const std::vector<DeterminantT<uint64_t>>&,
    const std::vector<double>&,
    const std::vector<std::vector<double>>&,
    const std::vector<double>&,
    int, int, int);
template Eigen::MatrixXd orbital_gradient_t<std::array<uint64_t, 2>>(
    const std::vector<DeterminantT<std::array<uint64_t, 2>>>&,
    const std::vector<double>&,
    const std::vector<std::vector<double>>&,
    const std::vector<double>&,
    int, int, int);
template Eigen::MatrixXd orbital_gradient_t<std::array<uint64_t, 3>>(
    const std::vector<DeterminantT<std::array<uint64_t, 3>>>&,
    const std::vector<double>&,
    const std::vector<std::vector<double>>&,
    const std::vector<double>&,
    int, int, int);
template Eigen::MatrixXd orbital_gradient_t<std::array<uint64_t, 4>>(
    const std::vector<DeterminantT<std::array<uint64_t, 4>>>&,
    const std::vector<double>&,
    const std::vector<std::vector<double>>&,
    const std::vector<double>&,
    int, int, int);
template Eigen::MatrixXd orbital_gradient_t<std::array<uint64_t, 5>>(
    const std::vector<DeterminantT<std::array<uint64_t, 5>>>&,
    const std::vector<double>&,
    const std::vector<std::vector<double>>&,
    const std::vector<double>&,
    int, int, int);
template Eigen::MatrixXd orbital_gradient_t<std::array<uint64_t, 6>>(
    const std::vector<DeterminantT<std::array<uint64_t, 6>>>&,
    const std::vector<double>&,
    const std::vector<std::vector<double>>&,
    const std::vector<double>&,
    int, int, int);
template Eigen::MatrixXd orbital_gradient_t<std::array<uint64_t, 7>>(
    const std::vector<DeterminantT<std::array<uint64_t, 7>>>&,
    const std::vector<double>&,
    const std::vector<std::vector<double>>&,
    const std::vector<double>&,
    int, int, int);
template Eigen::MatrixXd orbital_gradient_t<std::array<uint64_t, 8>>(
    const std::vector<DeterminantT<std::array<uint64_t, 8>>>&,
    const std::vector<double>&,
    const std::vector<std::vector<double>>&,
    const std::vector<double>&,
    int, int, int);

template std::tuple<Eigen::MatrixXd,
                    std::vector<std::vector<double>>,
                    std::vector<double>,
                    double,
                    double>
orbital_approx_newton_step_t<uint64_t>(
    const std::vector<DeterminantT<uint64_t>>&,
    const std::vector<double>&,
    const std::vector<std::vector<double>>&,
    const std::vector<double>&,
    int, int, int, bool, double, double, double);
template std::tuple<Eigen::MatrixXd,
                    std::vector<std::vector<double>>,
                    std::vector<double>,
                    double,
                    double>
orbital_approx_newton_step_t<std::array<uint64_t, 2>>(
    const std::vector<DeterminantT<std::array<uint64_t, 2>>>&,
    const std::vector<double>&,
    const std::vector<std::vector<double>>&,
    const std::vector<double>&,
    int, int, int, bool, double, double, double);
template std::tuple<Eigen::MatrixXd,
                    std::vector<std::vector<double>>,
                    std::vector<double>,
                    double,
                    double>
orbital_approx_newton_step_t<std::array<uint64_t, 3>>(
    const std::vector<DeterminantT<std::array<uint64_t, 3>>>&,
    const std::vector<double>&,
    const std::vector<std::vector<double>>&,
    const std::vector<double>&,
    int, int, int, bool, double, double, double);
template std::tuple<Eigen::MatrixXd,
                    std::vector<std::vector<double>>,
                    std::vector<double>,
                    double,
                    double>
orbital_approx_newton_step_t<std::array<uint64_t, 4>>(
    const std::vector<DeterminantT<std::array<uint64_t, 4>>>&,
    const std::vector<double>&,
    const std::vector<std::vector<double>>&,
    const std::vector<double>&,
    int, int, int, bool, double, double, double);
template std::tuple<Eigen::MatrixXd,
                    std::vector<std::vector<double>>,
                    std::vector<double>,
                    double,
                    double>
orbital_approx_newton_step_t<std::array<uint64_t, 5>>(
    const std::vector<DeterminantT<std::array<uint64_t, 5>>>&,
    const std::vector<double>&,
    const std::vector<std::vector<double>>&,
    const std::vector<double>&,
    int, int, int, bool, double, double, double);
template std::tuple<Eigen::MatrixXd,
                    std::vector<std::vector<double>>,
                    std::vector<double>,
                    double,
                    double>
orbital_approx_newton_step_t<std::array<uint64_t, 6>>(
    const std::vector<DeterminantT<std::array<uint64_t, 6>>>&,
    const std::vector<double>&,
    const std::vector<std::vector<double>>&,
    const std::vector<double>&,
    int, int, int, bool, double, double, double);
template std::tuple<Eigen::MatrixXd,
                    std::vector<std::vector<double>>,
                    std::vector<double>,
                    double,
                    double>
orbital_approx_newton_step_t<std::array<uint64_t, 7>>(
    const std::vector<DeterminantT<std::array<uint64_t, 7>>>&,
    const std::vector<double>&,
    const std::vector<std::vector<double>>&,
    const std::vector<double>&,
    int, int, int, bool, double, double, double);
template std::tuple<Eigen::MatrixXd,
                    std::vector<std::vector<double>>,
                    std::vector<double>,
                    double,
                    double>
orbital_approx_newton_step_t<std::array<uint64_t, 8>>(
    const std::vector<DeterminantT<std::array<uint64_t, 8>>>&,
    const std::vector<double>&,
    const std::vector<std::vector<double>>&,
    const std::vector<double>&,
    int, int, int, bool, double, double, double);

template std::tuple<Eigen::MatrixXd,
                    std::vector<std::vector<double>>,
                    std::vector<double>,
                    double,
                    double,
                    bool,
                    int,
                    std::vector<std::array<double, 2>>>
orbital_approx_newton_optimize_t<uint64_t>(
    const std::vector<DeterminantT<uint64_t>>&,
    const std::vector<double>&,
    const std::vector<std::vector<double>>&,
    const std::vector<double>&,
    int, int, int, int, double, double, double, double, double, bool, int, double, double, double, int, bool, bool);
template std::tuple<Eigen::MatrixXd,
                    std::vector<std::vector<double>>,
                    std::vector<double>,
                    double,
                    double,
                    bool,
                    int,
                    std::vector<std::array<double, 2>>>
orbital_approx_newton_optimize_t<std::array<uint64_t, 2>>(
    const std::vector<DeterminantT<std::array<uint64_t, 2>>>&,
    const std::vector<double>&,
    const std::vector<std::vector<double>>&,
    const std::vector<double>&,
    int, int, int, int, double, double, double, double, double, bool, int, double, double, double, int, bool, bool);
template std::tuple<Eigen::MatrixXd,
                    std::vector<std::vector<double>>,
                    std::vector<double>,
                    double,
                    double,
                    bool,
                    int,
                    std::vector<std::array<double, 2>>>
orbital_approx_newton_optimize_t<std::array<uint64_t, 3>>(
    const std::vector<DeterminantT<std::array<uint64_t, 3>>>&,
    const std::vector<double>&,
    const std::vector<std::vector<double>>&,
    const std::vector<double>&,
    int, int, int, int, double, double, double, double, double, bool, int, double, double, double, int, bool, bool);
template std::tuple<Eigen::MatrixXd,
                    std::vector<std::vector<double>>,
                    std::vector<double>,
                    double,
                    double,
                    bool,
                    int,
                    std::vector<std::array<double, 2>>>
orbital_approx_newton_optimize_t<std::array<uint64_t, 4>>(
    const std::vector<DeterminantT<std::array<uint64_t, 4>>>&,
    const std::vector<double>&,
    const std::vector<std::vector<double>>&,
    const std::vector<double>&,
    int, int, int, int, double, double, double, double, double, bool, int, double, double, double, int, bool, bool);
template std::tuple<Eigen::MatrixXd,
                    std::vector<std::vector<double>>,
                    std::vector<double>,
                    double,
                    double,
                    bool,
                    int,
                    std::vector<std::array<double, 2>>>
orbital_approx_newton_optimize_t<std::array<uint64_t, 5>>(
    const std::vector<DeterminantT<std::array<uint64_t, 5>>>&,
    const std::vector<double>&,
    const std::vector<std::vector<double>>&,
    const std::vector<double>&,
    int, int, int, int, double, double, double, double, double, bool, int, double, double, double, int, bool, bool);
template std::tuple<Eigen::MatrixXd,
                    std::vector<std::vector<double>>,
                    std::vector<double>,
                    double,
                    double,
                    bool,
                    int,
                    std::vector<std::array<double, 2>>>
orbital_approx_newton_optimize_t<std::array<uint64_t, 6>>(
    const std::vector<DeterminantT<std::array<uint64_t, 6>>>&,
    const std::vector<double>&,
    const std::vector<std::vector<double>>&,
    const std::vector<double>&,
    int, int, int, int, double, double, double, double, double, bool, int, double, double, double, int, bool, bool);
template std::tuple<Eigen::MatrixXd,
                    std::vector<std::vector<double>>,
                    std::vector<double>,
                    double,
                    double,
                    bool,
                    int,
                    std::vector<std::array<double, 2>>>
orbital_approx_newton_optimize_t<std::array<uint64_t, 7>>(
    const std::vector<DeterminantT<std::array<uint64_t, 7>>>&,
    const std::vector<double>&,
    const std::vector<std::vector<double>>&,
    const std::vector<double>&,
    int, int, int, int, double, double, double, double, double, bool, int, double, double, double, int, bool, bool);
template std::tuple<Eigen::MatrixXd,
                    std::vector<std::vector<double>>,
                    std::vector<double>,
                    double,
                    double,
                    bool,
                    int,
                    std::vector<std::array<double, 2>>>
orbital_approx_newton_optimize_t<std::array<uint64_t, 8>>(
    const std::vector<DeterminantT<std::array<uint64_t, 8>>>&,
    const std::vector<double>&,
    const std::vector<std::vector<double>>&,
    const std::vector<double>&,
    int, int, int, int, double, double, double, double, double, bool, int, double, double, double, int, bool, bool);

template std::tuple<Eigen::MatrixXd,
                    std::vector<std::vector<double>>,
                    std::vector<double>,
                    double,
                    double,
                    bool,
                    int,
                    std::vector<std::array<double, 2>>>
orbital_bfgs_optimize_t<uint64_t>(
    const std::vector<DeterminantT<uint64_t>>&,
    const std::vector<double>&,
    const std::vector<std::vector<double>>&,
    const std::vector<double>&,
    int, int, int, int, double, double, double, double, double, double, double, double, double, double, double, bool, int, double, double, double, int, bool, bool, DiagFunc<uint64_t>);
template std::tuple<Eigen::MatrixXd,
                    std::vector<std::vector<double>>,
                    std::vector<double>,
                    double,
                    double,
                    bool,
                    int,
                    std::vector<std::array<double, 2>>>
orbital_bfgs_optimize_t<std::array<uint64_t, 2>>(
    const std::vector<DeterminantT<std::array<uint64_t, 2>>>&,
    const std::vector<double>&,
    const std::vector<std::vector<double>>&,
    const std::vector<double>&,
    int, int, int, int, double, double, double, double, double, double, double, double, double, double, double, bool, int, double, double, double, int, bool, bool, DiagFunc<std::array<uint64_t, 2>>);
template std::tuple<Eigen::MatrixXd,
                    std::vector<std::vector<double>>,
                    std::vector<double>,
                    double,
                    double,
                    bool,
                    int,
                    std::vector<std::array<double, 2>>>
orbital_bfgs_optimize_t<std::array<uint64_t, 3>>(
    const std::vector<DeterminantT<std::array<uint64_t, 3>>>&,
    const std::vector<double>&,
    const std::vector<std::vector<double>>&,
    const std::vector<double>&,
    int, int, int, int, double, double, double, double, double, double, double, double, double, double, double, bool, int, double, double, double, int, bool, bool, DiagFunc<std::array<uint64_t, 3>>);
template std::tuple<Eigen::MatrixXd,
                    std::vector<std::vector<double>>,
                    std::vector<double>,
                    double,
                    double,
                    bool,
                    int,
                    std::vector<std::array<double, 2>>>
orbital_bfgs_optimize_t<std::array<uint64_t, 4>>(
    const std::vector<DeterminantT<std::array<uint64_t, 4>>>&,
    const std::vector<double>&,
    const std::vector<std::vector<double>>&,
    const std::vector<double>&,
    int, int, int, int, double, double, double, double, double, double, double, double, double, double, double, bool, int, double, double, double, int, bool, bool, DiagFunc<std::array<uint64_t, 4>>);
template std::tuple<Eigen::MatrixXd,
                    std::vector<std::vector<double>>,
                    std::vector<double>,
                    double,
                    double,
                    bool,
                    int,
                    std::vector<std::array<double, 2>>>
orbital_bfgs_optimize_t<std::array<uint64_t, 5>>(
    const std::vector<DeterminantT<std::array<uint64_t, 5>>>&,
    const std::vector<double>&,
    const std::vector<std::vector<double>>&,
    const std::vector<double>&,
    int, int, int, int, double, double, double, double, double, double, double, double, double, double, double, bool, int, double, double, double, int, bool, bool, DiagFunc<std::array<uint64_t, 5>>);
template std::tuple<Eigen::MatrixXd,
                    std::vector<std::vector<double>>,
                    std::vector<double>,
                    double,
                    double,
                    bool,
                    int,
                    std::vector<std::array<double, 2>>>
orbital_bfgs_optimize_t<std::array<uint64_t, 6>>(
    const std::vector<DeterminantT<std::array<uint64_t, 6>>>&,
    const std::vector<double>&,
    const std::vector<std::vector<double>>&,
    const std::vector<double>&,
    int, int, int, int, double, double, double, double, double, double, double, double, double, double, double, bool, int, double, double, double, int, bool, bool, DiagFunc<std::array<uint64_t, 6>>);
template std::tuple<Eigen::MatrixXd,
                    std::vector<std::vector<double>>,
                    std::vector<double>,
                    double,
                    double,
                    bool,
                    int,
                    std::vector<std::array<double, 2>>>
orbital_bfgs_optimize_t<std::array<uint64_t, 7>>(
    const std::vector<DeterminantT<std::array<uint64_t, 7>>>&,
    const std::vector<double>&,
    const std::vector<std::vector<double>>&,
    const std::vector<double>&,
    int, int, int, int, double, double, double, double, double, double, double, double, double, double, double, bool, int, double, double, double, int, bool, bool, DiagFunc<std::array<uint64_t, 7>>);
template std::tuple<Eigen::MatrixXd,
                    std::vector<std::vector<double>>,
                    std::vector<double>,
                    double,
                    double,
                    bool,
                    int,
                    std::vector<std::array<double, 2>>>
orbital_bfgs_optimize_t<std::array<uint64_t, 8>>(
    const std::vector<DeterminantT<std::array<uint64_t, 8>>>&,
    const std::vector<double>&,
    const std::vector<std::vector<double>>&,
    const std::vector<double>&,
    int, int, int, int, double, double, double, double, double, double, double, double, double, double, double, bool, int, double, double, double, int, bool, bool, DiagFunc<std::array<uint64_t, 8>>);

} // namespace trimci_core
