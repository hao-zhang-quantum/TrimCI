#include "expansion_loop.hpp"
#include "streaming_pt2.hpp"
#include "ab_index.hpp"
#include "detspace_matvec.hpp"
#include "detspace_davidson.hpp"
#include "sparse_update_davidson.hpp"
#include "screening.hpp"
#include "hamiltonian.hpp"
#include "orbital_opt.hpp"
#include "morton_sort.hpp"
#ifdef TRIMCI_HAS_GPU
#include "detspace_matvec_gpu.hpp"
#endif
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <unordered_map>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <sys/stat.h>

#include "parallel_sort.hpp"

#ifdef __APPLE__
#include <mach/mach.h>
#elif defined(__linux__)
#include <fstream>
#endif

namespace trimci_core {
namespace {
// Returns current RSS in MB (cross-platform)
double get_rss_mb() {
#ifdef __APPLE__
    struct mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  (task_info_t)&info, &count) == KERN_SUCCESS) {
        return info.resident_size / (1024.0 * 1024.0);
    }
    return 0.0;
#elif defined(__linux__)
    std::ifstream ifs("/proc/self/status");
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.compare(0, 6, "VmRSS:") == 0) {
            // VmRSS: <value> kB
            size_t pos = line.find_first_of("0123456789");
            if (pos != std::string::npos)
                return std::stod(line.substr(pos)) / 1024.0;
        }
    }
    return 0.0;
#else
    return 0.0;
#endif
}
// Save checkpoint: binary file with dets + coeffs + metadata.
// Format: "TCPT" magic (4B) | version uint32 (4B) | n_dets uint64 (8B) |
//         round int32 (4B) | pad (4B) | energy float64 (8B) |
//         alpha[n_dets] uint64 | beta[n_dets] uint64 | coeffs[n_dets] float64
// Python can read with numpy.frombuffer.
template<typename StorageType>
static void save_checkpoint(
    const std::string& dir, int round,
    const std::vector<DeterminantT<StorageType>>& dets,
    const std::vector<double>& coeffs,
    double energy)
{
    using Clock = std::chrono::high_resolution_clock;
    auto t0 = Clock::now();

    // Ensure directory exists (POSIX mkdir, ok if already exists)
    mkdir(dir.c_str(), 0755);

    std::string path = dir + "/checkpoint_round_" + std::to_string(round) + ".bin";
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "[Checkpoint] ERROR: cannot open " << path << std::endl;
        return;
    }

    const uint64_t n = dets.size();
    const uint32_t version = 1;
    const int32_t rnd = round;
    const int32_t pad = 0;

    f.write("TCPT", 4);
    f.write(reinterpret_cast<const char*>(&version), 4);
    f.write(reinterpret_cast<const char*>(&n), 8);
    f.write(reinterpret_cast<const char*>(&rnd), 4);
    f.write(reinterpret_cast<const char*>(&pad), 4);
    f.write(reinterpret_cast<const char*>(&energy), 8);

    // Write alpha and beta using a staging buffer for bulk writes.
    // Element-by-element writes through ofstream are unreliable at >10 GB scale.
    // For multi-segment types (e.g. std::array<uint64_t, 2>), write all segments contiguously.
    {
        // Determine bytes per spin component
        constexpr size_t BYTES_PER_SPIN = sizeof(StorageType);
        constexpr size_t BUF_BYTES = 32 * 1024 * 1024;  // 32 MB staging buffer
        constexpr size_t BUF_ENTRIES = BUF_BYTES / BYTES_PER_SPIN;
        std::vector<StorageType> buf(BUF_ENTRIES);

        for (uint64_t off = 0; off < n; off += BUF_ENTRIES) {
            size_t chunk = std::min<size_t>(BUF_ENTRIES, n - off);
            for (size_t j = 0; j < chunk; ++j)
                buf[j] = dets[off + j].alpha;
            f.write(reinterpret_cast<const char*>(buf.data()), chunk * BYTES_PER_SPIN);
        }
        for (uint64_t off = 0; off < n; off += BUF_ENTRIES) {
            size_t chunk = std::min<size_t>(BUF_ENTRIES, n - off);
            for (size_t j = 0; j < chunk; ++j)
                buf[j] = dets[off + j].beta;
            f.write(reinterpret_cast<const char*>(buf.data()), chunk * BYTES_PER_SPIN);
        }
    }

    // Coeffs are contiguous — write in 256 MB chunks to avoid single huge write()
    constexpr size_t COEFF_CHUNK = 32 * 1024 * 1024;  // 256 MB per write
    for (uint64_t off = 0; off < n; off += COEFF_CHUNK) {
        size_t chunk = std::min<size_t>(COEFF_CHUNK, n - off);
        f.write(reinterpret_cast<const char*>(coeffs.data() + off), chunk * 8);
    }

    f.flush();
    if (!f.good()) {
        std::cerr << "[Checkpoint] ERROR: write failed on " << path
                  << " (expected " << (32 + n * 24) << " bytes)" << std::endl;
        f.close();
        return;
    }
    f.close();

    double size_mb = static_cast<double>(32 + n * 24) / (1024.0 * 1024.0);
    double t_s = std::chrono::duration<double>(Clock::now() - t0).count();
    std::cout << "[Checkpoint] Saved round " << round << ": " << n << " dets, "
        << std::fixed << std::setprecision(1) << size_mb << " MB -> " << path
        << " (" << std::setprecision(1) << t_s << "s)" << std::endl;
}

// Save integrals checkpoint: h1 + eri + U_total.
// When per_round=true (orbopt), saves as checkpoint_integrals_round_N.bin
// in addition to the latest checkpoint_integrals.bin.
// Format: "TCPI" (4B) | version uint32 (4B) | n_orb int32 (4B) | round int32 (4B)
//         | h1[n_orb²] float64 | eri[n_orb⁴] float64 | U_total[n_orb²] float64
static void save_integrals_checkpoint(
    const std::string& dir, int round,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    const std::vector<double>& U_total,
    int n_orb,
    bool per_round = false)
{
    mkdir(dir.c_str(), 0755);
    std::string path = dir + "/checkpoint_integrals.bin";
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "[Checkpoint] ERROR: cannot open " << path << std::endl;
        return;
    }

    const uint32_t version = 1;
    const int32_t norb32 = n_orb;
    const int32_t rnd = round;

    f.write("TCPI", 4);
    f.write(reinterpret_cast<const char*>(&version), 4);
    f.write(reinterpret_cast<const char*>(&norb32), 4);
    f.write(reinterpret_cast<const char*>(&rnd), 4);

    // h1: n_orb × n_orb, row-major
    for (int i = 0; i < n_orb; ++i)
        f.write(reinterpret_cast<const char*>(h1[i].data()), n_orb * sizeof(double));

    // eri: n_orb^4 flat
    f.write(reinterpret_cast<const char*>(eri.data()), eri.size() * sizeof(double));

    // U_total: n_orb × n_orb, row-major
    f.write(reinterpret_cast<const char*>(U_total.data()), U_total.size() * sizeof(double));

    f.close();

    double size_mb = static_cast<double>(16 + (2 * n_orb * n_orb + (size_t)n_orb * n_orb * n_orb * n_orb) * 8) / (1024.0 * 1024.0);
    std::cout << "[Checkpoint] Saved integrals (round " << round << "): "
        << std::fixed << std::setprecision(1) << size_mb << " MB -> " << path
        << std::endl;

    // When orbopt is active, also save a per-round copy so each round's
    // integrals can be loaded independently for post-hoc analysis.
    if (per_round) {
        std::string rpath = dir + "/checkpoint_integrals_round_"
                          + std::to_string(round) + ".bin";
        std::ifstream src(path, std::ios::binary);
        std::ofstream dst(rpath, std::ios::binary);
        dst << src.rdbuf();
        std::cout << "[Checkpoint] Copied integrals -> " << rpath << std::endl;
    }
}

}  // anonymous namespace
namespace fe {

// ============================================================================
// TeeBuf: intercepts std::cout at streambuf level so ALL cout output
// (including from pool_build_t) is also written to the log file.
// ============================================================================
class TeeBuf : public std::streambuf {
    std::streambuf* orig_;
    std::ofstream file_;
public:
    TeeBuf(std::streambuf* orig, const std::string& filename)
        : orig_(orig)
    {
        if (!filename.empty()) file_.open(filename, std::ios::app);
    }

    ~TeeBuf() override = default;

protected:
    int overflow(int c) override {
        if (c == EOF) return c;
        // Write to original (stdout)
        if (orig_->sputc(c) == EOF) return EOF;
        // Write to log file
        if (file_.is_open()) {
            file_.put(static_cast<char>(c));
            if (c == '\n') file_.flush();
        }
        return c;
    }

    std::streamsize xsputn(const char* s, std::streamsize n) override {
        orig_->sputn(s, n);
        if (file_.is_open()) {
            file_.write(s, n);
            // Flush if there's a newline in the chunk
            for (std::streamsize i = 0; i < n; ++i) {
                if (s[i] == '\n') { file_.flush(); break; }
            }
        }
        return n;
    }

    int sync() override {
        orig_->pubsync();
        if (file_.is_open()) file_.flush();
        return 0;
    }
};

// RAII guard: install TeeBuf on cout, restore on destruction
class CoutTeeGuard {
    std::streambuf* saved_;
    TeeBuf tee_buf_;
public:
    CoutTeeGuard(const std::string& filename)
        : saved_(std::cout.rdbuf())
        , tee_buf_(saved_, filename)
    {
        if (!filename.empty()) std::cout.rdbuf(&tee_buf_);
    }
    ~CoutTeeGuard() { std::cout.rdbuf(saved_); }
};

// ============================================================================
// Cache statistics: |H_ij| distribution analysis for memory optimization
// ============================================================================
template<typename StorageType>
static void print_cache_stats(const ConnectionCache<StorageType>& cache) {
    const size_t nnz = cache.val.size();
    if (nnz == 0) return;

    // Log-scale histogram: bins for [10^k, 10^{k+1})
    // bins[0] = exactly zero, bins[1..21] = [1e-20, 1e-19), ..., [1e0, 1e1)
    constexpr int N_BINS = 22;  // 0: zero, 1-21: decades 1e-20 to 1e1
    std::vector<size_t> hist(N_BINS, 0);

    double max_abs = 0.0;
    double sum_abs = 0.0;
    double sum_sq = 0.0;
    size_t n_zero = 0;

    // float precision analysis
    size_t float_exact = 0;   // round-trips perfectly
    double max_float_err = 0.0;
    double sum_float_err_sq = 0.0;

    for (size_t k = 0; k < nnz; ++k) {
        const double v = cache.val[k];
        const double a = std::abs(v);

        if (a == 0.0) {
            n_zero++;
            hist[0]++;
            float_exact++;
            continue;
        }

        if (a > max_abs) max_abs = a;
        sum_abs += a;
        sum_sq += a * a;

        // Decade bin: log10(a) in [-20, 1)
        int decade = static_cast<int>(std::floor(std::log10(a)));
        int bin = decade + 21;  // shift so 1e-20 -> bin 1
        if (bin < 1) bin = 1;
        if (bin >= N_BINS) bin = N_BINS - 1;
        hist[bin]++;

        // Float round-trip error
        float vf = static_cast<float>(v);
        double v_back = static_cast<double>(vf);
        double err = std::abs(v - v_back);
        double rel_err = (a > 0) ? err / a : 0.0;
        if (rel_err > max_float_err) max_float_err = rel_err;
        sum_float_err_sq += rel_err * rel_err;
        if (err == 0.0) float_exact++;
    }

    double mean_abs = sum_abs / nnz;
    double rms = std::sqrt(sum_sq / nnz);
    double float_rms_rel = std::sqrt(sum_float_err_sq / nnz);

    std::cout << "\n[ConnCache Stats] nnz=" << nnz
              << ", max|H|=" << std::scientific << std::setprecision(3) << max_abs
              << ", mean|H|=" << mean_abs
              << ", rms|H|=" << rms << "\n";

    // Histogram
    std::cout << "[ConnCache Stats] |H_ij| distribution (log10 decades):\n";
    std::cout << "  exactly 0:   " << std::setw(12) << hist[0]
              << "  (" << std::fixed << std::setprecision(2)
              << 100.0 * hist[0] / nnz << "%)\n";

    size_t cumul = hist[0];
    // Print thresholds at key decades
    int decades_to_print[] = {-15, -12, -10, -8, -7, -6, -5, -4, -3, -2, -1, 0};
    for (int d : decades_to_print) {
        int bin = d + 21;
        if (bin < 1 || bin >= N_BINS) continue;
        // Cumulative: entries with |H| < 10^d
        size_t below = hist[0];
        for (int b = 1; b < bin; ++b) below += hist[b];
        std::cout << "  |H| < 1e" << std::setw(3) << d << ": "
                  << std::setw(12) << below
                  << "  (" << std::fixed << std::setprecision(2)
                  << 100.0 * below / nnz << "%)"
                  << "  truncate saves "
                  << below * 12 / (1024*1024) << " MB\n";
    }

    // Float analysis
    std::cout << "[ConnCache Stats] float precision analysis:\n"
              << "  float exact round-trip: " << float_exact << " / " << nnz
              << " (" << std::fixed << std::setprecision(2)
              << 100.0 * float_exact / nnz << "%)\n"
              << "  max float relative error: " << std::scientific << std::setprecision(3)
              << max_float_err << "\n"
              << "  RMS float relative error: " << float_rms_rel << "\n";

    // Memory projections
    size_t mem_double = nnz * 12;
    size_t mem_float = nnz * 8;
    std::cout << "[ConnCache Stats] memory:\n"
              << "  current (uint32+double): " << mem_double / (1024*1024) << " MB\n"
              << "  float   (uint32+float):  " << mem_float / (1024*1024) << " MB"
              << " (saves " << (mem_double - mem_float) / (1024*1024) << " MB)\n"
              << std::endl;
}

// ============================================================================
// Build warm-start vector: map old coefficients to new det positions
// ============================================================================
template<typename StorageType>
static std::vector<double> build_warm_start(
    const std::vector<DeterminantT<StorageType>>& old_dets,
    const std::vector<double>& old_coeffs,
    const std::vector<DeterminantT<StorageType>>& new_dets)
{
    std::unordered_map<DeterminantT<StorageType>, double> coeff_map;
    coeff_map.reserve(old_dets.size());
    for (size_t i = 0; i < old_dets.size(); ++i) {
        coeff_map[old_dets[i]] = old_coeffs[i];
    }

    std::vector<double> warm(new_dets.size(), 0.0);
    for (size_t i = 0; i < new_dets.size(); ++i) {
        auto it = coeff_map.find(new_dets[i]);
        if (it != coeff_map.end()) {
            warm[i] = it->second;
        }
    }

    double norm = 0.0;
    for (double x : warm) norm += x * x;
    norm = std::sqrt(norm);
    if (norm > 1e-15) {
        for (double& x : warm) x /= norm;
    } else {
        warm[0] = 1.0;
    }
    return warm;
}

// ============================================================================
// Main expansion loop
// ============================================================================
template<typename StorageType>
ExpansionResult<StorageType> run_expansion(
    const std::vector<DeterminantT<StorageType>>& initial_dets,
    const std::vector<double>& initial_coeffs,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    int n_orb,
    const ExpansionConfig<StorageType>& config)
{
    using Clock = std::chrono::high_resolution_clock;
    auto t_total = Clock::now();

    // Install tee on std::cout: all cout output (ours + pool_build_t) -> log file
    CoutTeeGuard tee_guard(config.log_file);

    ExpansionResult<StorageType> result;
    result.dets = initial_dets;
    result.coefficients = initial_coeffs;

    if (config.verbose >= 1) {
        std::cout << "\n"
            << "============================================================\n"
            << " Fast Expansion: iterative TrimCI expansion loop\n"
            << "============================================================\n"
            << " initial_dets = " << initial_dets.size()
            << ", n_orb = " << n_orb
            << ", max_n_dets = " << config.max_n_dets << "\n"
            << " growth_factor = " << config.growth_factor
            << ", energy_tol = " << config.expansion_energy_tol << "\n"
            << " threshold = " << config.threshold
            << ", threshold_decay = " << config.threshold_decay << "\n"
            << "============================================================\n"
            << std::endl;
    }

    // Mutable integral copies for orbital optimization
    std::vector<std::vector<double>> h1_work = h1;
    std::vector<double> eri_work = eri;

    // Cumulative orbital rotation matrix (row-major flat, initialized to identity)
    std::vector<double> U_total(n_orb * n_orb, 0.0);
    for (int i = 0; i < n_orb; ++i) U_total[i * n_orb + i] = 1.0;

    // State
    std::vector<DeterminantT<StorageType>> dets = initial_dets;
    std::vector<double> coeffs = initial_coeffs;
    double prev_energy = 0.0;
    double threshold = config.threshold;
    ABIndex<StorageType> ab_index;
    HijCacheT<StorageType> cache;
    std::string cache_file;

    // Adaptive PT2: inherit converged eps across rounds.
    // Initialized to 0 (= use config default); updated after each adaptive cycle.
    double last_converged_eps = 0.0;
    double last_ffm_wall_time = 0.0;

    // Hard guard against infinite FE loop on symmetry-saturated systems
    // (e.g. H4 singlet hits 20 dets, the spin-complete max, then HB returns 0
    // candidates round after round). Triggers even when dets_conv_ratio is
    // disabled (Python wrapper default = -1).
    int consecutive_zero_growth = 0;

    for (int round = 0; round < config.max_expansion_rounds; ++round) {
        auto t_round = Clock::now();

        // ==================================================================
        // Step 1: Expand
        // ==================================================================
        size_t old_size = dets.size();
        std::vector<DeterminantT<StorageType>> new_dets;
        double final_threshold = threshold;
        bool space_converged = false;

        if (config.evaluate_only) {
            // Skip expansion: run Davidson+PT2+dressed on existing dets
            new_dets = dets;
            if (config.verbose >= 1) {
                std::cout << "[FE Round " << round << "] Evaluate-only: "
                    << old_size << " dets (no expansion)" << std::endl;
            }
        } else {
            size_t target_size = std::min(
                static_cast<size_t>(std::ceil(old_size * config.growth_factor)),
                config.max_n_dets);
            if (target_size <= old_size) target_size = config.max_n_dets;

            if (config.verbose >= 1) {
                std::cout << "[FE Round " << round << "] Expanding: "
                    << old_size << " -> target " << target_size
                    << ", threshold=" << std::scientific << std::setprecision(4)
                    << threshold << std::fixed << std::endl;
            }

            auto t_expand = Clock::now();

            PoolBuildParams pb_params;
            pb_params.screening_mode = config.screening_mode;
            pb_params.max_rounds = 1;
            pb_params.threshold_decay = config.threshold_decay;
            pb_params.strict_target_size = config.strict_target_size;
            pb_params.e0 = prev_energy;

            // Compute MI weights for "mi" screening mode
            if (config.screening_mode == "mi") {
                size_t N = dets.size();
                std::vector<double> mi_w(n_orb * n_orb, 0.0);
                // <n_i> and <n_i n_j> from current wavefunction
                std::vector<double> n_occ(n_orb, 0.0);
                std::vector<double> nn(n_orb * n_orb, 0.0);
                for (size_t k = 0; k < N; ++k) {
                    double c2 = coeffs[k] * coeffs[k];
                    // Build occupation vector from getOccupied lists
                    std::vector<int> occ(n_orb, 0);
                    for (int o : dets[k].getOccupiedAlpha()) occ[o]++;
                    for (int o : dets[k].getOccupiedBeta()) occ[o]++;
                    for (int p = 0; p < n_orb; ++p) {
                        n_occ[p] += c2 * occ[p];
                        for (int q = p+1; q < n_orb; ++q) {
                            nn[p * n_orb + q] += c2 * occ[p] * occ[q];
                        }
                    }
                }
                // C(p,q) = <n_p n_q> - <n_p><n_q>
                for (int p = 0; p < n_orb; ++p)
                    for (int q = p+1; q < n_orb; ++q) {
                        double c = std::abs(nn[p * n_orb + q] - n_occ[p] * n_occ[q]);
                        mi_w[p * n_orb + q] = c;
                        mi_w[q * n_orb + p] = c;
                    }
                pb_params.mi_weights = std::move(mi_w);
                pb_params.mi_n_orb = n_orb;
                if (config.verbose >= 2 && round == 0) {
                    // Print top-5 MI pairs on first round
                    std::vector<std::tuple<double, int, int>> mi_pairs;
                    for (int p = 0; p < n_orb; ++p)
                        for (int q = p+1; q < n_orb; ++q)
                            mi_pairs.push_back({pb_params.mi_weights[p*n_orb+q], p, q});
                    std::sort(mi_pairs.rbegin(), mi_pairs.rend());
                    std::cout << "[MI] Top-5 correlated pairs:" << std::endl;
                    for (int i = 0; i < std::min(5, (int)mi_pairs.size()); ++i) {
                        auto [w, p, q] = mi_pairs[i];
                        std::cout << "  (" << p << "," << q << "): " << w << std::endl;
                    }
                }
            }

            auto [expanded_dets, ft] = pool_build_t<StorageType>(
                dets, coeffs, n_orb, h1_work, eri_work,
                threshold, target_size,
                cache, cache_file,
                /*attentive_orbitals=*/{},
                /*verbosity=*/config.verbose,
                pb_params);
            new_dets = std::move(expanded_dets);
            final_threshold = ft;

            double t_expand_s = std::chrono::duration<double>(Clock::now() - t_expand).count();

            if (config.verbose >= 1) {
                std::cout << "[FE Round " << round << "] Expanded: "
                    << old_size << " -> " << new_dets.size()
                    << " (" << std::fixed << std::setprecision(1)
                    << (double)new_dets.size() / old_size << "x)"
                    << ", time=" << std::setprecision(2)
                    << t_expand_s << "s" << std::endl;
            }

            double growth_ratio = static_cast<double>(new_dets.size()) / old_size;
            space_converged = (growth_ratio < config.dets_conv_ratio) && (round > 0);

            // Hard guard: 2 consecutive strict zero-growth rounds = saturated.
            // Independent of dets_conv_ratio so symmetry-complete systems (e.g.
            // H4 singlet saturating at 20 dets) cannot loop indefinitely when
            // the ratio check is disabled.
            if (new_dets.size() == old_size && round > 0) {
                consecutive_zero_growth++;
                if (consecutive_zero_growth >= 2 && !space_converged) {
                    space_converged = true;
                    if (config.verbose >= 1) {
                        std::cout << "[FE Round " << round
                            << "] Space converged (hard guard): "
                            << consecutive_zero_growth
                            << " consecutive zero-growth rounds" << std::endl;
                    }
                }
            } else {
                consecutive_zero_growth = 0;
            }
        }
        size_t new_size = new_dets.size();

        if (space_converged && config.verbose >= 1) {
            double gr = static_cast<double>(new_size) / old_size;
            std::cout << "[FE Round " << round << "] Space converged: "
                << "growth=" << std::setprecision(4) << gr
                << " < " << config.dets_conv_ratio << std::endl;
        }

        // ==================================================================
        // Step 1.5: Sort dets for cache locality
        // ==================================================================
        auto t_sort = Clock::now();
        if (config.use_morton_sort) {
            fe::morton_sort_dets(new_dets, n_orb);  // Z-order: balanced 2D locality
        } else {
            trimci::parallel_sort(new_dets.begin(), new_dets.end());  // alpha-first lexicographic
        }
        double t_sort_s = std::chrono::duration<double>(Clock::now() - t_sort).count();
        if (config.verbose >= 1) {
            std::cout << "[FE Round " << round << "] Sort"
                << (config.use_morton_sort ? " (Morton)" : " (alpha)") << ": "
                << std::setprecision(2) << t_sort_s << "s" << std::endl;
        }

        // ==================================================================
        // Step 1.6: Pool-build-only early exit (skip ABIndex, Diagonals, Davidson)
        // ==================================================================
        if (config.pool_build_only) {
            coeffs.assign(new_dets.size(), 0.0);
            coeffs[0] = 1.0;
            dets = std::move(new_dets);

            // Save checkpoint (energy=0, not computed)
            int ckpt_round = round + config.checkpoint_round_offset;
            if (!config.checkpoint_dir.empty()) {
                save_checkpoint<StorageType>(config.checkpoint_dir, ckpt_round, dets, coeffs, 0.0);
            }

            result.ndets_history.push_back(dets.size());
            result.energy_history.push_back(0.0);

            double t_round_s = std::chrono::duration<double>(Clock::now() - t_round).count();
            if (config.verbose >= 1) {
                std::cout << "[FE Round " << round << "] Pool-build-only complete: "
                    << dets.size() << " dets, "
                    << std::setprecision(1) << t_round_s << "s" << std::endl;
            }
            break;
        }

        // ==================================================================
        // Step 2: Build ABIndex
        // ==================================================================
        auto t_index = Clock::now();
        ab_index.build(new_dets, n_orb);
        double t_index_s = std::chrono::duration<double>(Clock::now() - t_index).count();

        if (config.verbose >= 1) {
            std::cout << "[FE Round " << round << "] ABIndex: "
                << ab_index.n_unique_alpha() << "α × "
                << ab_index.n_unique_beta() << "β, "
                << ab_index.memory_bytes() / 1024 << " KB, "
                << std::setprecision(2) << t_index_s << "s" << std::endl;
        }

        // ==================================================================
        // Step 3: Compute diagonals
        // ==================================================================
        auto t_diag = Clock::now();
        auto diag = compute_diagonals<StorageType>(new_dets, h1_work, eri_work, n_orb);
        double t_diag_s = std::chrono::duration<double>(Clock::now() - t_diag).count();

        if (config.verbose >= 1) {
            std::cout << "[FE Round " << round << "] Diagonals: "
                << std::setprecision(2) << t_diag_s << "s" << std::endl;
        }

        // ==================================================================
        // Step 3.5: Build connection cache (if enabled)
        // ==================================================================
        ConnectionCache<StorageType> conn_cache;

        // Skip the cache build when GPU is going to handle matvec — the
        // matvec_fn lambda dispatches to GPU first and never reads the
        // cache, so building it is pure waste (O(N²) time + memory). This
        // matters at 1M+ dets where cache build alone can be minutes.
        const bool gpu_will_handle_matvec = (config.backend == "gpu");
        if (gpu_will_handle_matvec && config.use_connection_cache && config.verbose >= 1) {
            std::cout << "[FE Round " << round
                << "] use_connection_cache=true ignored under backend=\"gpu\""
                << " (cache would not be read; saving build time + memory)."
                << std::endl;
        }

        if (config.use_connection_cache && !gpu_will_handle_matvec) {
            auto t_cache = Clock::now();
            conn_cache = build_connection_cache<StorageType>(
                ab_index, new_dets, h1_work, eri_work, n_orb);
            double t_cache_s = std::chrono::duration<double>(Clock::now() - t_cache).count();

            if (config.verbose >= 1) {
                std::cout << "[FE Round " << round << "] ConnCache: "
                    << conn_cache.col.size() << " entries, "
                    << conn_cache.memory_bytes() / (1024*1024) << " MB, "
                    << std::setprecision(2) << t_cache_s << "s" << std::endl;
            }

            if (config.verbose >= 2) {
                print_cache_stats(conn_cache);
            }
        } else if (config.verbose >= 1) {
            std::string mode;
            if (gpu_will_handle_matvec) mode = "GPU";
            else if (config.use_dual_order) mode = "dual-order";
            else if (config.symmetric_sigma) mode = "symmetric";
            else mode = "full-matrix";
            std::cout << "[FE Round " << round << "] Matvec: on-the-fly"
                << " (" << mode << ")" << std::endl;
        }

        // ==================================================================
        // Step 4: Davidson with warm start
        // ==================================================================
        auto t_dav = Clock::now();

        std::vector<std::vector<double>> guess;
        if (!config.no_warm_start && !coeffs.empty()) {
            auto warm = build_warm_start<StorageType>(dets, coeffs, new_dets);
            guess.push_back(std::move(warm));
        }

        MatvecContext<StorageType> ctx{new_dets, ab_index, h1_work, eri_work, n_orb, diag};

        // Optional GPU matvec session (one per round). Built only if the
        // user requested backend="gpu" AND the build has GPU support.
        // Python wrapper is expected to resolve "auto" → "cpu"/"gpu"
        // before this point; unknown values fall back to CPU.
        const bool want_gpu = (config.backend == "gpu");
#ifdef TRIMCI_HAS_GPU
        std::unique_ptr<fe::GpuMatvecSession<StorageType>> gpu_session;
        if (want_gpu) {
            auto t_gpu = Clock::now();
            gpu_session = std::make_unique<fe::GpuMatvecSession<StorageType>>(
                ab_index, new_dets, h1_work, eri_work, n_orb, config.verbose);
            double t_gpu_s = std::chrono::duration<double>(Clock::now() - t_gpu).count();
            if (config.verbose >= 1) {
                std::cout << "[FE Round " << round << "] GPU matvec session built ("
                          << std::fixed << std::setprecision(2) << t_gpu_s << "s)"
                          << std::endl;
            }
        }
#else
        if (want_gpu && config.verbose >= 1) {
            std::cout << "[FE Round " << round << "] WARNING: backend=\"gpu\" but"
                      << " this build has no GPU support; falling back to CPU matvec."
                      << std::endl;
        }
#endif

        // Check if cache was actually built — if user requested backend="gpu"
        // we skipped the cache build above, so the empty conn_cache must not
        // be passed to matvec_cached (would segfault). On CPU-only build the
        // GPU session won't exist, and we fall back gracefully to other CPU
        // matvec variants.
        const bool cache_built = config.use_connection_cache && !gpu_will_handle_matvec;
        auto matvec_fn = [&](const double* v, double* sigma, size_t n) {
            std::fill(sigma, sigma + n, 0.0);
#ifdef TRIMCI_HAS_GPU
            if (gpu_session) {
                gpu_session->apply(v, sigma, n);
                return;
            }
#endif
            if (cache_built) {
                matvec_cached<StorageType>(conn_cache, diag, v, sigma, n);
            } else if (config.use_dual_order) {
                matvec_dual_order<StorageType>(ctx, v, sigma, n);
            } else if (config.symmetric_sigma) {
                matvec_dressed<StorageType>(ctx, v, sigma, n);
            } else {
                matvec_dressed_full<StorageType>(ctx, v, sigma, n);
            }
        };

        // Block matvec: share H_ij across multiple trial vectors (only for full-matrix mode)
        BlockMatvecFunc block_mv = nullptr;
        if (config.davidson_block_size > 1 && !config.use_connection_cache && !config.symmetric_sigma) {
            block_mv = [&](const double* const* vb, double** sb, size_t n, int nv) {
                matvec_dressed_full_block<StorageType>(ctx, vb, sb, n, nv);
            };
        }

        // ==================================================================
        // Step 4a: Perturbative warm-start enhancement (solver-agnostic)
        // ==================================================================
        // Applied BEFORE either solver so both standard and sparse Davidson
        // start from the same enhanced guess. For new dets j (warm[j] ≈ 0):
        //   sigma = H * warm  →  warm[j] = sigma[j] / (θ₀ - diag[j])
        // Cost: ONE extra full matvec. Saves multiple solver iterations.
        if (config.perturbative_warmstart && !guess.empty()) {
            auto t_pt = Clock::now();
            auto& warm = guess[0];

            // Compute sigma = H * warm
            std::vector<double> sigma(new_size, 0.0);
            matvec_fn(warm.data(), sigma.data(), new_size);

            // θ₀ = warm · sigma (Rayleigh quotient of current guess)
            double theta0 = 0.0;
            for (size_t i = 0; i < new_size; ++i) theta0 += warm[i] * sigma[i];

            // Enhance zero-coefficient dets
            size_t n_new_dets = 0, n_enhanced = 0;
            for (size_t j = 0; j < new_size; ++j) {
                if (std::abs(warm[j]) < 1e-15) {
                    ++n_new_dets;
                    if (std::abs(sigma[j]) > 1e-15) {
                        double denom = theta0 - diag[j];
                        if (std::abs(denom) < 1e-6)
                            denom = (denom >= 0) ? 1e-6 : -1e-6;
                        warm[j] = sigma[j] / denom;
                        ++n_enhanced;
                    }
                }
            }

            if (n_enhanced > 0) {
                double norm = 0.0;
                for (size_t i = 0; i < new_size; ++i) norm += warm[i] * warm[i];
                norm = std::sqrt(norm);
                if (norm > 1e-15)
                    for (size_t i = 0; i < new_size; ++i) warm[i] /= norm;
            }

            double t_pt_s = std::chrono::duration<double>(Clock::now() - t_pt).count();
            if (config.verbose >= 1) {
                std::cout << "[FE Round " << round << "] Perturbative warm-start: "
                          << n_enhanced << "/" << n_new_dets << " new dets enhanced, "
                          << "time=" << std::fixed << std::setprecision(2) << t_pt_s << "s"
                          << std::endl;
            }
        }

        DavidsonParams dav_params = config.davidson_params;
        dav_params.verbose = config.verbose;
        dav_params.block_size = config.davidson_block_size;

        DavidsonResult dav_result;
        if (config.use_sparse_update) {
            SparseUpdateParams sp;
            sp.max_subspace = dav_params.max_subspace;
            sp.max_iter = dav_params.max_iter;
            sp.residual_tol = dav_params.residual_tol;
            sp.energy_tol = dav_params.energy_tol;
            sp.verbose = config.verbose;
            sp.truncation_ratio = config.sparse_truncation_ratio;
            sp.min_active_dets = config.sparse_min_active;
            sp.n_warm_iters = config.sparse_n_warm_iters;
            sp.capture_threshold = config.sparse_capture_threshold;
            sp.adaptive_K = config.sparse_adaptive_K;
            sp.truncation_mode = config.sparse_truncation_mode;
            sp.warm_ratio = config.sparse_warm_ratio;
            sp.random_sample_ratio = config.sparse_random_sample_ratio;
            sp.ucb_kappa = config.sparse_ucb_kappa;
            sp.olsen_correction = config.sparse_olsen_correction;
            sp.thick_restart_size = config.sparse_thick_restart_size;
            sp.momentum_beta = config.sparse_momentum_beta;
            sp.block_size = dav_params.block_size;  // Block sparse Davidson (1 = single-vector)
            sp.K_schedule = config.sparse_K_schedule;
            sp.final_full_matvec = config.sparse_final_full_matvec;

            dav_result = sparse_update_solve<StorageType>(
                ctx, matvec_fn, new_size, sp, guess);
        } else {
            dav_result = davidson_solve(matvec_fn, diag, new_size, dav_params, guess, block_mv);
        }
        double t_dav_s = std::chrono::duration<double>(Clock::now() - t_dav).count();
        last_ffm_wall_time = dav_result.ffm_wall_time;

        double energy = dav_result.eigenvalues.empty() ? 0.0 : dav_result.eigenvalues[0];

        if (config.verbose >= 1) {
            std::cout << "[FE Round " << round << "] Davidson: E = "
                << std::fixed << std::setprecision(10) << energy
                << ", iters=" << dav_result.n_iters
                << ", resid=" << std::scientific << std::setprecision(2)
                << dav_result.residual_norm
                << ", time=" << std::fixed << std::setprecision(2)
                << t_dav_s << "s" << std::endl;
        }

        // Update state
        dets = std::move(new_dets);
        coeffs = dav_result.eigenvectors.empty()
            ? std::vector<double>(dets.size(), 0.0)
            : dav_result.eigenvectors[0];

        // ==================================================================
        // Step 4.5: Orbital optimization (BFGS with FE matvec callback)
        // ==================================================================
        if (config.orbital_optimization) {
            auto t_orbopt = Clock::now();

            // FE Davidson callback: reuses ABIndex, rebuilds cache/context per call
            double cb_diag_s = 0.0, cb_cache_s = 0.0, cb_davidson_s = 0.0;
            int cb_calls = 0;

            DiagFunc<StorageType> fe_diag =
                [&](const std::vector<std::vector<double>>& h1_in,
                    const std::vector<double>& eri_in,
                    const std::vector<double>& guess_in)
                -> std::tuple<double, std::vector<double>>
            {
                ++cb_calls;
                auto t0 = Clock::now();
                auto diag_orb = compute_diagonals<StorageType>(dets, h1_in, eri_in, n_orb);
                cb_diag_s += std::chrono::duration<double>(Clock::now() - t0).count();

                // Build cache or context once per callback; Davidson's ~20 matvec iters reuse it
                ConnectionCache<StorageType> cc_orb;
                MatvecContext<StorageType> ctx_orb{dets, ab_index, h1_in, eri_in, n_orb, diag_orb};
                if (config.use_connection_cache) {
                    auto t1 = Clock::now();
                    cc_orb = build_connection_cache<StorageType>(ab_index, dets, h1_in, eri_in, n_orb);
                    cb_cache_s += std::chrono::duration<double>(Clock::now() - t1).count();
                }

                auto mv = [&](const double* v, double* sigma, size_t nn) {
                    std::fill(sigma, sigma + nn, 0.0);
                    if (config.use_connection_cache) {
                        matvec_cached<StorageType>(cc_orb, diag_orb, v, sigma, nn);
                    } else if (config.symmetric_sigma) {
                        matvec_dressed<StorageType>(ctx_orb, v, sigma, nn);
                    } else {
                        matvec_dressed_full<StorageType>(ctx_orb, v, sigma, nn);
                    }
                };

                DavidsonParams dp;
                dp.residual_tol = 1e-3;
                dp.max_iter = 200;
                std::vector<std::vector<double>> guesses;
                if (!guess_in.empty()) guesses.push_back(guess_in);
                auto t2 = Clock::now();
                auto dr = davidson_solve(mv, diag_orb, dets.size(), dp, guesses);
                cb_davidson_s += std::chrono::duration<double>(Clock::now() - t2).count();
                double e = dr.eigenvalues.empty() ? 0.0 : dr.eigenvalues[0];
                auto c = dr.eigenvectors.empty()
                    ? std::vector<double>{}
                    : dr.eigenvectors[0];
                return {e, c};
            };

            auto [U_opt, h1_opt, eri_opt, grad_norm_opt, energy_opt,
                  orbopt_converged, orbopt_iters, orbopt_trace] =
                orbital_bfgs_optimize_t<StorageType>(
                    dets, coeffs, h1_work, eri_work, n_orb,
                    -1, -1,
                    config.orbital_opt_max_iter,
                    config.orbital_opt_grad_tol,
                    config.orbital_opt_energy_tol, 1.0, 1e-5, 0.2,
                    1e-3, 1e-6, 5.0, 0.97, 0.25, 4.0,
                    true, 6, 0.5, 0.0,
                    1e-3, 200,
                    true, false,
                    fe_diag);

            double t_orbopt_s = std::chrono::duration<double>(Clock::now() - t_orbopt).count();

            if (!h1_opt.empty()) {
                double dE_orbopt = energy_opt - energy;
                h1_work = std::move(h1_opt);
                eri_work = std::move(eri_opt);
                energy = energy_opt;

                // Accumulate U_total = U_total_old × U_opt
                {
                    std::vector<double> U_new(n_orb * n_orb, 0.0);
                    for (int ii = 0; ii < n_orb; ++ii)
                        for (int jj = 0; jj < n_orb; ++jj)
                            for (int kk = 0; kk < n_orb; ++kk)
                                U_new[ii * n_orb + jj] += U_total[ii * n_orb + kk] * U_opt(kk, jj);
                    U_total = std::move(U_new);
                }

                if (config.verbose >= 1) {
                    std::cout << "[FE Round " << round << "] OrbOpt: "
                        << orbopt_iters << " iters, "
                        << (orbopt_converged ? "converged" : "not converged")
                        << ", |grad|=" << std::scientific << std::setprecision(2)
                        << grad_norm_opt
                        << ", dE=" << dE_orbopt
                        << std::fixed << ", time=" << std::setprecision(2)
                        << t_orbopt_s << "s" << std::endl;
                    std::cout << "  [Callback Profile] " << cb_calls << " calls: "
                        << "diag=" << cb_diag_s << "s, "
                        << "cache=" << cb_cache_s << "s, "
                        << "davidson=" << cb_davidson_s << "s"
                        << std::endl;
                }
            }
        }

        result.energy_history.push_back(energy);
        result.ndets_history.push_back(dets.size());

        // ==================================================================
        // Step 5: Per-round Sketch PT2 (if enabled)
        // ==================================================================
        if (!config.pt2_config.has_value()) {
            // Push zeros to keep pt2/variance/screening arrays aligned with energy/ndets
            result.pt2_history.push_back(0.0);
            result.variance_ext_history.push_back(0.0);
            result.screening_error_history.push_back(0.0);
        }
        if (config.pt2_config.has_value()) {
            auto t_pt2 = Clock::now();
            auto pt2_cfg = config.pt2_config.value();
            const double screening_target = pt2_cfg.screening_error_target;

            PT2Result<StorageType> pt2_result;

            if (screening_target > 0 && pt2_cfg.eps_hc_filter > 0) {
                // ---- Adaptive screening: progressively refine eps ----
                //
                // Motivation: fixed eps_hc_filter becomes unreliable at large N_dets.
                // Benchmark (Fe4S4 CAS(54,36)): eps=1e-6 error grows 31%(10M) → 64%(40M).
                //
                // Algorithm: eps *= 0.3 each step, compare PT2(eps_prev) vs PT2(eps_curr).
                // Converge when |diff| / |PT2_curr| < target.
                //
                // Error calibration (power-law model: err(eps) = A·eps^α, α ≈ 1.26):
                //   diff = A·eps^α · (1 - 0.3^α)
                //   true_remaining = A·(0.3·eps)^α = A·eps^α · 0.3^α
                //   diff / true_remaining = (1 - 0.3^α) / 0.3^α = 3^α - 1 ≈ 2.9
                //   → true_error ≈ target / 2.9  (e.g. target=3% → ~1% actual)
                //
                // Choice of 0.3 (÷3): 0.1 (÷10) overestimates 17×, wastes compute;
                //   0.5 (÷2) only 1.4× margin, insufficient. 0.3 is well-calibrated.
                //
                // Cross-round inheritance: last_converged_eps avoids redundant coarse steps.
                //
                double eps = (last_converged_eps > 0) ? last_converged_eps
                                                      : pt2_cfg.eps_hc_filter;
                pt2_cfg.screening_error_target = 0;  // prevent recursion
                pt2_cfg.eps_hc_filter = eps;
                pt2_cfg.verbose = (config.verbose >= 2) ? 1 : 0;

                auto prev_result = compute_pt2<StorageType>(
                    dets, coeffs, energy, h1_work, eri_work, n_orb, pt2_cfg);

                if (config.verbose >= 1) {
                    std::cout << "[PT2 adaptive] eps=" << std::scientific
                              << std::setprecision(1) << eps
                              << ", ΔE_PT2=" << std::fixed << std::setprecision(10)
                              << prev_result.delta_pt2
                              << ", time=" << std::setprecision(1)
                              << prev_result.time_seconds << "s (initial)" << std::endl;
                }

                constexpr int max_adaptive = 8;       // covers 1e-5 → ~1.5e-9
                constexpr double eps_scale = 0.3;      // eps *= 0.3 each step (÷3.3)
                bool converged = false;
                for (int attempt = 0; attempt < max_adaptive; ++attempt) {
                    eps *= eps_scale;
                    pt2_cfg.eps_hc_filter = eps;
                    auto curr_result = compute_pt2<StorageType>(
                        dets, coeffs, energy, h1_work, eri_work, n_orb, pt2_cfg);

                    double diff = std::abs(curr_result.delta_pt2 - prev_result.delta_pt2);
                    double ratio = (std::abs(curr_result.delta_pt2) > 1e-15)
                                   ? diff / std::abs(curr_result.delta_pt2) : 0.0;
                    curr_result.screening_error_estimate = diff;

                    if (config.verbose >= 1) {
                        std::cout << "[PT2 adaptive] eps=" << std::scientific
                                  << std::setprecision(1) << eps
                                  << ", ΔE_PT2=" << std::fixed << std::setprecision(10)
                                  << curr_result.delta_pt2
                                  << ", screen_err=" << std::scientific << std::setprecision(3)
                                  << diff << " (" << std::fixed << std::setprecision(2)
                                  << 100.0 * ratio << "%)"
                                  << ", time=" << std::setprecision(1)
                                  << curr_result.time_seconds << "s" << std::endl;
                    }

                    if (ratio < screening_target) {
                        pt2_result = curr_result;
                        last_converged_eps = eps / eps_scale;  // coarser eps as start for next round
                        converged = true;
                        break;
                    }
                    prev_result = curr_result;
                    pt2_result = curr_result;  // use best even if not converged
                }
                if (!converged) {
                    last_converged_eps = eps;  // tightest eps tried
                    if (config.verbose >= 1) {
                        std::cout << "[PT2 adaptive] WARNING: did not converge within "
                                  << max_adaptive << " steps, using eps="
                                  << std::scientific << std::setprecision(1) << eps
                                  << std::endl;
                    }
                }
            } else {
                // ---- Single-shot mode (screening_error_target=0 or eps=0) ----
                pt2_result = compute_pt2<StorageType>(
                    dets, coeffs, energy, h1_work, eri_work, n_orb, pt2_cfg);
            }

            double t_pt2_s = std::chrono::duration<double>(Clock::now() - t_pt2).count();

            result.pt2_history.push_back(pt2_result.delta_pt2);
            result.variance_ext_history.push_back(pt2_result.variance_ext);
            result.screening_error_history.push_back(pt2_result.screening_error_estimate);

            if (config.verbose >= 1) {
                std::cout << "[FE Round " << round << "] PT2: ΔE = "
                    << std::fixed << std::setprecision(10) << pt2_result.delta_pt2
                    << ", σ²_ext = " << pt2_result.variance_ext
                    << ", E_var+PT2 = " << (energy + pt2_result.delta_pt2)
                    << ", time=" << std::setprecision(1) << t_pt2_s << "s"
                    << std::endl;
                if (pt2_result.screening_error_estimate > 0) {
                    std::cout << "[FE Round " << round << "] PT2 screen err: "
                        << std::scientific << std::setprecision(3)
                        << pt2_result.screening_error_estimate << " Ha ("
                        << std::fixed << std::setprecision(2)
                        << 100.0 * pt2_result.screening_error_estimate
                           / std::abs(pt2_result.delta_pt2) << "%)"
                        << std::endl;
                }
            }
        }

        // ==================================================================
        // Step 5b: Per-round Dressed CI
        //   Mode 1: dressed_energy=true → hybrid on-the-fly (uses PT2 config)
        //   Mode 2: dressed_ci_config set → original per-det sketch
        // ==================================================================
        if (config.dressed_energy && config.pt2_config.has_value()) {
            // --- Hybrid mode: on-the-fly dressed matvec ---
            // Self-consistent Brillouin-Wigner: iterate with updated E_ref
            auto t_dressed = Clock::now();
            double E_ref = energy;  // start from variational energy
            DressedCIResult<StorageType> dressed_result;
            const int sc_max = config.dressed_sc_max_iter;

            for (int sc_iter = 0; sc_iter < sc_max; ++sc_iter) {
                dressed_result = compute_dressed_ci_hybrid<StorageType>(
                    dets, coeffs, E_ref, h1_work, eri_work, n_orb,
                    config.pt2_config.value(), config.davidson_params);

                double dE_sc = std::abs(dressed_result.energy_dressed - E_ref);

                if (config.verbose >= 1 && sc_max > 1) {
                    std::cout << "[FE Round " << round << "] DressedCI SC iter "
                        << sc_iter << ": E_ref=" << std::fixed << std::setprecision(10)
                        << E_ref << " -> E_dressed="
                        << dressed_result.energy_dressed
                        << " (dE=" << std::scientific << std::setprecision(2)
                        << dE_sc << ")" << std::fixed << std::endl;
                }

                if (sc_iter > 0 && dE_sc < config.dressed_sc_energy_tol) {
                    if (config.verbose >= 1) {
                        std::cout << "[FE Round " << round
                            << "] DressedCI SC converged in " << (sc_iter + 1)
                            << " iters" << std::endl;
                    }
                    break;
                }
                E_ref = dressed_result.energy_dressed;
            }

            double t_dressed_s = std::chrono::duration<double>(
                Clock::now() - t_dressed).count();

            result.dressed_energy_history.push_back(
                dressed_result.energy_dressed);
            result.dressed_pt2_energy_history.push_back(
                dressed_result.energy_dressed);  // no separate PT2 in hybrid mode

            if (config.verbose >= 1) {
                std::cout << "[FE Round " << round << "] DressedCI(hybrid): E = "
                    << std::fixed << std::setprecision(10)
                    << dressed_result.energy_dressed
                    << ", iters=" << dressed_result.davidson_iters
                    << (dressed_result.davidson_converged
                        ? " (conv)" : " (NOT conv)")
                    << ", time=" << std::setprecision(1)
                    << t_dressed_s << "s" << std::endl;
            }

            if (!dressed_result.coefficients_dressed.empty()
                && dressed_result.davidson_converged) {
                coeffs = dressed_result.coefficients_dressed;
            }
        } else if (config.dressed_ci_config.has_value()) {
            // --- Original per-det sketch mode ---
            auto t_dressed = Clock::now();
            DressedCIConfig dc_cfg = config.dressed_ci_config.value();

            auto dressed_result = compute_dressed_ci<StorageType>(
                dets, coeffs, energy, h1_work, eri_work, n_orb, dc_cfg);

            double t_dressed_s = std::chrono::duration<double>(
                Clock::now() - t_dressed).count();

            result.dressed_energy_history.push_back(
                dressed_result.energy_dressed);

            // When n_external == 0 (e.g. FCI space), PT2-on-dressed is not
            // computed and the fields remain at 0.0.  Fall back to energy_dressed.
            double E_dressed_pt2;
            if (dressed_result.n_external > 0) {
                E_dressed_pt2 = dressed_result.energy_var_dressed
                              + dressed_result.delta_pt2_dressed;
            } else {
                E_dressed_pt2 = dressed_result.energy_dressed;
            }
            result.dressed_pt2_energy_history.push_back(E_dressed_pt2);

            if (config.verbose >= 1) {
                std::cout << "[FE Round " << round << "] DressedCI: E_dressed = "
                    << std::fixed << std::setprecision(10)
                    << dressed_result.energy_dressed
                    << ", E_d(ref)+PT2 = " << E_dressed_pt2
                    << ", iters=" << dressed_result.davidson_iters
                    << (dressed_result.davidson_converged
                        ? " (conv)" : " (NOT conv)")
                    << ", time=" << std::setprecision(1)
                    << t_dressed_s << "s" << std::endl;
            }

            // Warm-start: use dressed coefficients for next round's Davidson
            // (only if Davidson converged, to avoid propagating noise)
            if (!dressed_result.coefficients_dressed.empty()
                && dressed_result.davidson_converged) {
                coeffs = dressed_result.coefficients_dressed;
            }
        }

        // ==================================================================
        // Checkpoint: save wavefunction after each round
        // ==================================================================
        if (!config.checkpoint_dir.empty()) {
            int ckpt_round = round + config.checkpoint_round_offset;
            save_checkpoint<StorageType>(config.checkpoint_dir, ckpt_round, dets, coeffs, energy);
            save_integrals_checkpoint(config.checkpoint_dir, ckpt_round,
                                      h1_work, eri_work, U_total, n_orb,
                                      config.orbital_optimization);
        }

        double t_round_s = std::chrono::duration<double>(Clock::now() - t_round).count();
        double t_total_s = std::chrono::duration<double>(Clock::now() - t_total).count();

        if (config.verbose >= 1) {
            std::cout << "[FE Round " << round << "] Round: "
                << t_round_s << "s, Total: " << t_total_s
                << "s, RSS: " << std::fixed << std::setprecision(0)
                << get_rss_mb() << " MB" << std::endl;
            if (round > 0) {
                std::cout << "[FE Round " << round << "] dE = "
                    << std::scientific << std::setprecision(2)
                    << (energy - prev_energy) << std::fixed << std::endl;
            }
            std::cout << "------------------------------------------------------------\n";
        }

        // ==================================================================
        // Step 5: Check convergence
        // ==================================================================
        bool energy_converged = (round > 0) &&
            (std::abs(energy - prev_energy) < config.expansion_energy_tol);
        bool size_limit = (dets.size() >= config.max_n_dets);

        if (config.evaluate_only) {
            if (config.verbose >= 1) {
                std::cout << "[FE] Evaluate-only: done after 1 round." << std::endl;
            }
            prev_energy = energy;
            break;
        }
        if (energy_converged) {
            if (config.verbose >= 1) {
                std::cout << "[FE] Energy converged: |dE| = "
                    << std::scientific << std::setprecision(2)
                    << std::abs(energy - prev_energy)
                    << " < " << config.expansion_energy_tol << std::endl;
            }
            prev_energy = energy;
            break;
        }
        if (space_converged) {
            if (config.verbose >= 1) {
                std::cout << "[FE] Space converged, no more dets to add." << std::endl;
            }
            prev_energy = energy;
            break;
        }
        if (size_limit) {
            if (config.verbose >= 1) {
                std::cout << "[FE] Size limit reached: " << dets.size()
                    << " >= " << config.max_n_dets << std::endl;
            }
            prev_energy = energy;
            break;
        }

        prev_energy = energy;
        threshold = final_threshold;
    }

    // ==================================================================
    // Final result
    // ==================================================================
    result.dets = std::move(dets);
    result.coefficients = std::move(coeffs);
    result.energy_var = prev_energy;
    result.ffm_wall_time = last_ffm_wall_time;
    result.n_rounds = static_cast<int>(result.energy_history.size());
    result.U_total = std::move(U_total);
    result.h1_optimized = std::move(h1_work);
    result.eri_optimized = std::move(eri_work);

    double t_wall = std::chrono::duration<double>(Clock::now() - t_total).count();

    if (config.verbose >= 1) {
        std::cout << "\n[FE] Done: " << result.n_rounds << " rounds, "
            << result.dets.size() << " dets, E_var = "
            << std::fixed << std::setprecision(10)
            << result.energy_var
            << ", wall=" << std::setprecision(1) << t_wall << "s"
            << std::endl;
    }

    return result;
}

// ============================================================================
// run_expansion_phased: multi-phase wrapper
// ============================================================================
template<typename StorageType>
ExpansionResult<StorageType> run_expansion_phased(
    const std::vector<DeterminantT<StorageType>>& initial_dets,
    const std::vector<double>& initial_coeffs,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    int n_orb,
    const std::vector<ExpansionConfig<StorageType>>& phase_configs)
{
    auto dets = initial_dets;
    auto coeffs = initial_coeffs;
    auto h1_work = h1;
    auto eri_work = eri;

    // Cumulative U across all phases
    std::vector<double> U_total(n_orb * n_orb, 0.0);
    for (int i = 0; i < n_orb; ++i) U_total[i * n_orb + i] = 1.0;

    ExpansionResult<StorageType> combined;

    for (size_t p = 0; p < phase_configs.size(); ++p) {
        std::cout << "\n[FE] ========== Phase " << p + 1
                  << " / " << phase_configs.size() << " ==========\n" << std::endl;

        auto result = run_expansion(dets, coeffs, h1_work, eri_work,
                                    n_orb, phase_configs[p]);

        // Transfer state for next phase
        dets = std::move(result.dets);
        coeffs = std::move(result.coefficients);
        h1_work = std::move(result.h1_optimized);
        eri_work = std::move(result.eri_optimized);

        // Accumulate U_total = U_total_old × U_phase
        if (!result.U_total.empty()) {
            std::vector<double> U_new(n_orb * n_orb, 0.0);
            for (int i = 0; i < n_orb; ++i)
                for (int j = 0; j < n_orb; ++j)
                    for (int k = 0; k < n_orb; ++k)
                        U_new[i * n_orb + j] += U_total[i * n_orb + k]
                                               * result.U_total[k * n_orb + j];
            U_total = std::move(U_new);
        }

        // Merge histories
        for (auto& e : result.energy_history) combined.energy_history.push_back(e);
        for (auto& n : result.ndets_history) combined.ndets_history.push_back(n);
        for (auto& v : result.pt2_history) combined.pt2_history.push_back(v);
        for (auto& v : result.variance_ext_history) combined.variance_ext_history.push_back(v);
        for (auto& v : result.screening_error_history) combined.screening_error_history.push_back(v);
        for (auto& v : result.dressed_energy_history) combined.dressed_energy_history.push_back(v);
        for (auto& v : result.dressed_pt2_energy_history) combined.dressed_pt2_energy_history.push_back(v);
        combined.n_rounds += result.n_rounds;
    }

    combined.dets = std::move(dets);
    combined.coefficients = std::move(coeffs);
    combined.energy_var = combined.energy_history.empty()
        ? 0.0 : combined.energy_history.back();
    combined.U_total = std::move(U_total);
    combined.h1_optimized = std::move(h1_work);
    combined.eri_optimized = std::move(eri_work);
    return combined;
}

// Explicit instantiations
template ExpansionResult<uint64_t> run_expansion<uint64_t>(
    const std::vector<DeterminantT<uint64_t>>&,
    const std::vector<double>&,
    const std::vector<std::vector<double>>&,
    const std::vector<double>&, int,
    const ExpansionConfig<uint64_t>&);

template ExpansionResult<uint64_t> run_expansion_phased<uint64_t>(
    const std::vector<DeterminantT<uint64_t>>&,
    const std::vector<double>&,
    const std::vector<std::vector<double>>&,
    const std::vector<double>&, int,
    const std::vector<ExpansionConfig<uint64_t>>&);


// 128-bit explicit instantiations
using Bit128 = std::array<uint64_t, 2>;

template ExpansionResult<Bit128> run_expansion<Bit128>(
    const std::vector<DeterminantT<Bit128>>&,
    const std::vector<double>&,
    const std::vector<std::vector<double>>&,
    const std::vector<double>&, int,
    const ExpansionConfig<Bit128>&);

template ExpansionResult<Bit128> run_expansion_phased<Bit128>(
    const std::vector<DeterminantT<Bit128>>&,
    const std::vector<double>&,
    const std::vector<std::vector<double>>&,
    const std::vector<double>&, int,
    const std::vector<ExpansionConfig<Bit128>>&);

}  // namespace fe
}  // namespace trimci_core
