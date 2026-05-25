#include "ab_index.hpp"
#include "omp_compat.hpp"
#include <algorithm>
#include <numeric>
#include <fstream>
#include <stdexcept>
#include <cstring>

namespace trimci_core {
namespace fe {

// ============================================================================
// Build unique strings + per-det alpha/beta ids
// ============================================================================
template<typename StorageType>
void ABIndex<StorageType>::build_unique_strings(
    const std::vector<DeterminantT<StorageType>>& dets)
{
    alpha_to_id_.clear();
    beta_to_id_.clear();
    unique_alphas_.clear();
    unique_betas_.clear();

    det_alpha_id_.resize(dets.size());
    det_beta_id_.resize(dets.size());

    for (size_t i = 0; i < dets.size(); ++i) {
        // Alpha
        auto [it_a, inserted_a] = alpha_to_id_.emplace(
            dets[i].alpha, static_cast<uint32_t>(unique_alphas_.size()));
        if (inserted_a) {
            unique_alphas_.push_back(dets[i].alpha);
        }
        det_alpha_id_[i] = it_a->second;

        // Beta
        auto [it_b, inserted_b] = beta_to_id_.emplace(
            dets[i].beta, static_cast<uint32_t>(unique_betas_.size()));
        if (inserted_b) {
            unique_betas_.push_back(dets[i].beta);
        }
        det_beta_id_[i] = it_b->second;
    }
}

// ============================================================================
// Build CSR: alpha_id -> det indices, beta_id -> det indices
// ============================================================================
template<typename StorageType>
void ABIndex<StorageType>::build_csr(
    const std::vector<DeterminantT<StorageType>>& dets)
{
    const size_t N = dets.size();
    const size_t n_alpha = unique_alphas_.size();
    const size_t n_beta = unique_betas_.size();

    // --- Alpha CSR (two-pass) ---
    // Pass 1: count
    alpha_det_ptr_.assign(n_alpha + 1, 0);
    for (size_t i = 0; i < N; ++i) {
        alpha_det_ptr_[det_alpha_id_[i] + 1]++;
    }
    // Prefix sum
    for (size_t a = 1; a <= n_alpha; ++a) {
        alpha_det_ptr_[a] += alpha_det_ptr_[a - 1];
    }
    // Pass 2: fill
    alpha_det_flat_.resize(N);
    alpha_beta_ids_.resize(N);
    alpha_beta_strings_.resize(N);
    std::vector<size_t> alpha_pos(n_alpha);
    for (size_t a = 0; a < n_alpha; ++a) alpha_pos[a] = alpha_det_ptr_[a];
    for (size_t i = 0; i < N; ++i) {
        uint32_t a = det_alpha_id_[i];
        size_t pos = alpha_pos[a]++;
        alpha_det_flat_[pos] = static_cast<uint32_t>(i);
        alpha_beta_ids_[pos] = det_beta_id_[i];
        alpha_beta_strings_[pos] = dets[i].beta;
    }

    // --- Beta CSR (two-pass) ---
    beta_det_ptr_.assign(n_beta + 1, 0);
    for (size_t i = 0; i < N; ++i) {
        beta_det_ptr_[det_beta_id_[i] + 1]++;
    }
    for (size_t b = 1; b <= n_beta; ++b) {
        beta_det_ptr_[b] += beta_det_ptr_[b - 1];
    }
    beta_det_flat_.resize(N);
    beta_alpha_ids_.resize(N);
    beta_alpha_strings_.resize(N);
    std::vector<size_t> beta_pos(n_beta);
    for (size_t b = 0; b < n_beta; ++b) beta_pos[b] = beta_det_ptr_[b];
    for (size_t i = 0; i < N; ++i) {
        uint32_t b = det_beta_id_[i];
        size_t pos = beta_pos[b]++;
        beta_det_flat_[pos] = static_cast<uint32_t>(i);
        beta_alpha_ids_[pos] = det_alpha_id_[i];
        beta_alpha_strings_[pos] = dets[i].alpha;
    }
}

// ============================================================================
// Build singles neighbor table via abm1
// ============================================================================
template<typename StorageType>
void ABIndex<StorageType>::build_singles_table()
{
    using BitOpsType = detail::HamiltonianBitOps<StorageType>;

    // --- Alpha singles ---
    {
        // Build abm1: (alpha minus 1 orbital) -> list of alpha_ids
        fe_map<StorageType, std::vector<uint32_t>, SplitMix64Hash> alpha_m1;

        for (uint32_t a = 0; a < unique_alphas_.size(); ++a) {
            int occ[512];
            int n_occ = BitOpsType::storage_to_indices_inline(unique_alphas_[a], occ, 512);
            for (int k = 0; k < n_occ; ++k) {
                StorageType m1 = unique_alphas_[a];
                BitOps<StorageType>::clear_bit(m1, occ[k]);
                alpha_m1[m1].push_back(a);
            }
        }

        // Derive singles: for each alpha_id a, collect all a' sharing an abm1 key
        size_t n_alpha = unique_alphas_.size();
        // First pass: count neighbors
        std::vector<size_t> counts(n_alpha, 0);

        for (uint32_t a = 0; a < n_alpha; ++a) {
            fe_set<uint32_t> neighbors;
            int occ[512];
            int n_occ = BitOpsType::storage_to_indices_inline(unique_alphas_[a], occ, 512);
            for (int k = 0; k < n_occ; ++k) {
                StorageType m1 = unique_alphas_[a];
                BitOps<StorageType>::clear_bit(m1, occ[k]);
                auto it = alpha_m1.find(m1);
                if (it != alpha_m1.end()) {
                    for (uint32_t a2 : it->second) {
                        if (a2 != a) neighbors.insert(a2);
                    }
                }
            }
            counts[a] = neighbors.size();
        }

        // Build CSR
        alpha_singles_ptr_.resize(n_alpha + 1);
        alpha_singles_ptr_[0] = 0;
        for (size_t a = 0; a < n_alpha; ++a) {
            alpha_singles_ptr_[a + 1] = alpha_singles_ptr_[a] + counts[a];
        }
        alpha_singles_flat_.resize(alpha_singles_ptr_[n_alpha]);

        // Second pass: fill
        for (uint32_t a = 0; a < n_alpha; ++a) {
            fe_set<uint32_t> neighbors;
            int occ[512];
            int n_occ = BitOpsType::storage_to_indices_inline(unique_alphas_[a], occ, 512);
            for (int k = 0; k < n_occ; ++k) {
                StorageType m1 = unique_alphas_[a];
                BitOps<StorageType>::clear_bit(m1, occ[k]);
                auto it = alpha_m1.find(m1);
                if (it != alpha_m1.end()) {
                    for (uint32_t a2 : it->second) {
                        if (a2 != a) neighbors.insert(a2);
                    }
                }
            }
            size_t pos = alpha_singles_ptr_[a];
            for (uint32_t nb : neighbors) {
                alpha_singles_flat_[pos++] = nb;
            }
            // Sort for reproducibility
            std::sort(alpha_singles_flat_.begin() + alpha_singles_ptr_[a],
                      alpha_singles_flat_.begin() + alpha_singles_ptr_[a + 1]);
        }
    }

    // --- Beta singles (same algorithm) ---
    {
        fe_map<StorageType, std::vector<uint32_t>, SplitMix64Hash> beta_m1;

        for (uint32_t b = 0; b < unique_betas_.size(); ++b) {
            int occ[512];
            int n_occ = BitOpsType::storage_to_indices_inline(unique_betas_[b], occ, 512);
            for (int k = 0; k < n_occ; ++k) {
                StorageType m1 = unique_betas_[b];
                BitOps<StorageType>::clear_bit(m1, occ[k]);
                beta_m1[m1].push_back(b);
            }
        }

        size_t n_beta = unique_betas_.size();
        std::vector<size_t> counts(n_beta, 0);

        for (uint32_t b = 0; b < n_beta; ++b) {
            fe_set<uint32_t> neighbors;
            int occ[512];
            int n_occ = BitOpsType::storage_to_indices_inline(unique_betas_[b], occ, 512);
            for (int k = 0; k < n_occ; ++k) {
                StorageType m1 = unique_betas_[b];
                BitOps<StorageType>::clear_bit(m1, occ[k]);
                auto it = beta_m1.find(m1);
                if (it != beta_m1.end()) {
                    for (uint32_t b2 : it->second) {
                        if (b2 != b) neighbors.insert(b2);
                    }
                }
            }
            counts[b] = neighbors.size();
        }

        beta_singles_ptr_.resize(n_beta + 1);
        beta_singles_ptr_[0] = 0;
        for (size_t b = 0; b < n_beta; ++b) {
            beta_singles_ptr_[b + 1] = beta_singles_ptr_[b] + counts[b];
        }
        beta_singles_flat_.resize(beta_singles_ptr_[n_beta]);

        for (uint32_t b = 0; b < n_beta; ++b) {
            fe_set<uint32_t> neighbors;
            int occ[512];
            int n_occ = BitOpsType::storage_to_indices_inline(unique_betas_[b], occ, 512);
            for (int k = 0; k < n_occ; ++k) {
                StorageType m1 = unique_betas_[b];
                BitOps<StorageType>::clear_bit(m1, occ[k]);
                auto it = beta_m1.find(m1);
                if (it != beta_m1.end()) {
                    for (uint32_t b2 : it->second) {
                        if (b2 != b) neighbors.insert(b2);
                    }
                }
            }
            size_t pos = beta_singles_ptr_[b];
            for (uint32_t nb : neighbors) {
                beta_singles_flat_[pos++] = nb;
            }
            std::sort(beta_singles_flat_.begin() + beta_singles_ptr_[b],
                      beta_singles_flat_.begin() + beta_singles_ptr_[b + 1]);
        }
    }
}

// ============================================================================
// Public: build
// ============================================================================
template<typename StorageType>
void ABIndex<StorageType>::build(
    const std::vector<DeterminantT<StorageType>>& dets, int n_orb)
{
    n_dets_ = dets.size();
    n_orb_ = n_orb;
    build_unique_strings(dets);
    build_csr(dets);
    build_singles_table();
}

// ============================================================================
// Public: update (incremental)
// ============================================================================
template<typename StorageType>
void ABIndex<StorageType>::update(
    const std::vector<DeterminantT<StorageType>>& dets, size_t old_size)
{
    // For now, rebuild from scratch. Incremental optimization is Phase 1 enhancement.
    (void)old_size;
    build(dets, n_orb_);
}

// for_each_connection and for_each_alpha_group are defined inline in ab_index.hpp

// ============================================================================
// Public: memory_bytes
// ============================================================================
template<typename StorageType>
size_t ABIndex<StorageType>::memory_bytes() const
{
    size_t bytes = 0;
    bytes += unique_alphas_.capacity() * sizeof(StorageType);
    bytes += unique_betas_.capacity() * sizeof(StorageType);
    bytes += alpha_det_flat_.capacity() * sizeof(uint32_t);
    bytes += alpha_det_ptr_.capacity() * sizeof(size_t);
    bytes += alpha_beta_ids_.capacity() * sizeof(uint32_t);
    bytes += beta_det_flat_.capacity() * sizeof(uint32_t);
    bytes += beta_det_ptr_.capacity() * sizeof(size_t);
    bytes += beta_alpha_ids_.capacity() * sizeof(uint32_t);
    bytes += alpha_singles_flat_.capacity() * sizeof(uint32_t);
    bytes += alpha_singles_ptr_.capacity() * sizeof(size_t);
    bytes += beta_singles_flat_.capacity() * sizeof(uint32_t);
    bytes += beta_singles_ptr_.capacity() * sizeof(size_t);
    bytes += alpha_beta_strings_.capacity() * sizeof(StorageType);
    bytes += beta_alpha_strings_.capacity() * sizeof(StorageType);
    bytes += det_alpha_id_.capacity() * sizeof(uint32_t);
    bytes += det_beta_id_.capacity() * sizeof(uint32_t);
    // Hash maps are harder to estimate precisely; rough approximation
    bytes += alpha_to_id_.size() * (sizeof(StorageType) + sizeof(uint32_t) + 16);
    bytes += beta_to_id_.size() * (sizeof(StorageType) + sizeof(uint32_t) + 16);
    return bytes;
}

// ============================================================================
// Public: clear
// ============================================================================
template<typename StorageType>
void ABIndex<StorageType>::clear()
{
    n_dets_ = 0;
    unique_alphas_.clear(); unique_betas_.clear();
    alpha_to_id_.clear(); beta_to_id_.clear();
    alpha_det_flat_.clear(); alpha_det_ptr_.clear(); alpha_beta_ids_.clear();
    alpha_beta_strings_.clear();
    beta_det_flat_.clear(); beta_det_ptr_.clear(); beta_alpha_ids_.clear();
    beta_alpha_strings_.clear();
    alpha_singles_flat_.clear(); alpha_singles_ptr_.clear();
    beta_singles_flat_.clear(); beta_singles_ptr_.clear();
    det_alpha_id_.clear(); det_beta_id_.clear();
}

// ============================================================================
// Public: save (binary serialization)
// ============================================================================
template<typename StorageType>
void ABIndex<StorageType>::save(const std::string& path) const
{
    std::ofstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("ABIndex::save: cannot open " + path);

    // --- Header (256 bytes) ---
    f.write("ABIX", 4);
    const uint32_t version = 1;
    f.write(reinterpret_cast<const char*>(&version), 4);

    const uint64_t n_dets_u64 = n_dets_;
    f.write(reinterpret_cast<const char*>(&n_dets_u64), 8);

    const int32_t n_orb_i32 = n_orb_;
    const int32_t pad0 = 0;
    f.write(reinterpret_cast<const char*>(&n_orb_i32), 4);
    f.write(reinterpret_cast<const char*>(&pad0), 4);

    const uint32_t storage_bits = static_cast<uint32_t>(sizeof(StorageType) * 8);
    const uint32_t pad1 = 0;
    f.write(reinterpret_cast<const char*>(&storage_bits), 4);
    f.write(reinterpret_cast<const char*>(&pad1), 4);

    const uint64_t sizes[16] = {
        unique_alphas_.size(),       unique_betas_.size(),
        alpha_det_flat_.size(),      alpha_det_ptr_.size(),
        alpha_beta_ids_.size(),      beta_det_flat_.size(),
        beta_det_ptr_.size(),        beta_alpha_ids_.size(),
        alpha_singles_flat_.size(),  alpha_singles_ptr_.size(),
        beta_singles_flat_.size(),   beta_singles_ptr_.size(),
        alpha_beta_strings_.size(),  beta_alpha_strings_.size(),
        det_alpha_id_.size(),        det_beta_id_.size(),
    };
    f.write(reinterpret_cast<const char*>(sizes), 16 * 8);

    const uint64_t reserved[8] = {};
    f.write(reinterpret_cast<const char*>(reserved), 64);

    // --- Data (16 arrays, sequential) ---
    auto write_vec = [&](const auto& vec) {
        if (!vec.empty())
            f.write(reinterpret_cast<const char*>(vec.data()),
                    static_cast<std::streamsize>(
                        vec.size() * sizeof(typename std::remove_reference_t<decltype(vec)>::value_type)));
    };

    write_vec(unique_alphas_);       write_vec(unique_betas_);
    write_vec(alpha_det_flat_);      write_vec(alpha_det_ptr_);
    write_vec(alpha_beta_ids_);      write_vec(beta_det_flat_);
    write_vec(beta_det_ptr_);        write_vec(beta_alpha_ids_);
    write_vec(alpha_singles_flat_);  write_vec(alpha_singles_ptr_);
    write_vec(beta_singles_flat_);   write_vec(beta_singles_ptr_);
    write_vec(alpha_beta_strings_);  write_vec(beta_alpha_strings_);
    write_vec(det_alpha_id_);        write_vec(det_beta_id_);

    if (!f.good())
        throw std::runtime_error("ABIndex::save: write error on " + path);
}

// ============================================================================
// Public: load (binary deserialization)
// ============================================================================
template<typename StorageType>
ABIndex<StorageType> ABIndex<StorageType>::load(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("ABIndex::load: cannot open " + path);

    ABIndex<StorageType> idx;

    // --- Header ---
    char magic[4];
    f.read(magic, 4);
    if (std::memcmp(magic, "ABIX", 4) != 0)
        throw std::runtime_error("ABIndex::load: invalid magic in " + path);

    uint32_t version;
    f.read(reinterpret_cast<char*>(&version), 4);
    if (version != 1)
        throw std::runtime_error("ABIndex::load: unsupported version " +
                                 std::to_string(version) + " in " + path);

    uint64_t n_dets_u64;
    f.read(reinterpret_cast<char*>(&n_dets_u64), 8);
    idx.n_dets_ = static_cast<size_t>(n_dets_u64);

    int32_t n_orb_i32, pad0;
    f.read(reinterpret_cast<char*>(&n_orb_i32), 4);
    f.read(reinterpret_cast<char*>(&pad0), 4);
    idx.n_orb_ = n_orb_i32;

    uint32_t storage_bits, pad1;
    f.read(reinterpret_cast<char*>(&storage_bits), 4);
    f.read(reinterpret_cast<char*>(&pad1), 4);

    const uint32_t expected_bits = static_cast<uint32_t>(sizeof(StorageType) * 8);
    if (storage_bits != expected_bits)
        throw std::runtime_error("ABIndex::load: StorageType mismatch: file has " +
                                 std::to_string(storage_bits) + " bits, expected " +
                                 std::to_string(expected_bits));

    uint64_t sizes[16];
    f.read(reinterpret_cast<char*>(sizes), 16 * 8);

    uint64_t reserved[8];
    f.read(reinterpret_cast<char*>(reserved), 64);

    // --- Data ---
    auto read_vec = [&](auto& vec, uint64_t count) {
        vec.resize(static_cast<size_t>(count));
        if (count > 0)
            f.read(reinterpret_cast<char*>(vec.data()),
                   static_cast<std::streamsize>(
                       count * sizeof(typename std::remove_reference_t<decltype(vec)>::value_type)));
    };

    read_vec(idx.unique_alphas_,       sizes[0]);
    read_vec(idx.unique_betas_,        sizes[1]);
    read_vec(idx.alpha_det_flat_,      sizes[2]);
    read_vec(idx.alpha_det_ptr_,       sizes[3]);
    read_vec(idx.alpha_beta_ids_,      sizes[4]);
    read_vec(idx.beta_det_flat_,       sizes[5]);
    read_vec(idx.beta_det_ptr_,        sizes[6]);
    read_vec(idx.beta_alpha_ids_,      sizes[7]);
    read_vec(idx.alpha_singles_flat_,  sizes[8]);
    read_vec(idx.alpha_singles_ptr_,   sizes[9]);
    read_vec(idx.beta_singles_flat_,   sizes[10]);
    read_vec(idx.beta_singles_ptr_,    sizes[11]);
    read_vec(idx.alpha_beta_strings_,  sizes[12]);
    read_vec(idx.beta_alpha_strings_,  sizes[13]);
    read_vec(idx.det_alpha_id_,        sizes[14]);
    read_vec(idx.det_beta_id_,         sizes[15]);

    if (!f.good())
        throw std::runtime_error("ABIndex::load: read error on " + path);

    // Rebuild hash maps from unique string vectors
    idx.alpha_to_id_.reserve(idx.unique_alphas_.size());
    for (uint32_t i = 0; i < static_cast<uint32_t>(idx.unique_alphas_.size()); ++i)
        idx.alpha_to_id_[idx.unique_alphas_[i]] = i;

    idx.beta_to_id_.reserve(idx.unique_betas_.size());
    for (uint32_t i = 0; i < static_cast<uint32_t>(idx.unique_betas_.size()); ++i)
        idx.beta_to_id_[idx.unique_betas_[i]] = i;

    return idx;
}

// ============================================================================
// Explicit template instantiations
// ============================================================================
template class ABIndex<uint64_t>;
template class ABIndex<std::array<uint64_t, 2>>;
template class ABIndex<std::array<uint64_t, 4>>;

// Explicit instantiations of template member functions
// (for_each_connection and for_each_alpha_group are header-instantiated via templates)

}  // namespace fe
}  // namespace trimci_core
