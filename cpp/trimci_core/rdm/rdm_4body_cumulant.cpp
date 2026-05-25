/**
 * 4-RDM cumulant approximation using exact 3-RDM.
 *
 * Approximation: set only λ⁴ = 0 (4-body cumulant).
 * Uses exact Γ¹, Γ², Γ³ as input for better accuracy than pure 2-body cumulant.
 *
 * Closed-form reconstruction:
 *   Γ⁴ = A_S[Γ³ ⊗ γ] + A_S[λ² ⊗ λ²] − 3·det₄(γ)
 * where λ² = Γ² − det₂(γ) is the 2-body cumulant.
 *
 * Derivation:
 *   Cumulant decomposition with λ⁴ = 0:
 *     Γ⁴ = A[λ³⊗γ] + A[λ²⊗λ²] + A[λ²⊗γ⊗γ] + det₄(γ)
 *   Substitute λ³ = Γ³ − A[λ²⊗γ] − det₃(γ):
 *     A[λ³⊗γ] = A[Γ³⊗γ] − A[λ²⊗γ⊗γ] − A[det₃(γ)⊗γ]
 *   The A[λ²⊗γ⊗γ] terms cancel, giving:
 *     Γ⁴ = A[Γ³⊗γ] + A[λ²⊗λ²] − A[det₃(γ)⊗γ] + det₄(γ)
 *   For all spin sectors: A[det₃(γ)⊗γ] = 4·det₄(γ), hence:
 *     Γ⁴ = A[Γ³⊗γ] + A[λ²⊗λ²] − 3·det₄(γ)
 *
 * Term counts per element:
 *   aaaa/bbbb: 16 (Γ³⊗γ) + 9 (λ²⊗λ²) + det₄
 *   aaab/abbb: 10 (Γ³⊗γ) + 9 (λ²⊗λ²) + det₄
 *   aabb:       8 (Γ³⊗γ) + 5 (λ²⊗λ²) + det₄
 */

#include "rdm_full.hpp"
#include <cmath>
#include <stdexcept>
#include <string>
#include <omp.h>

namespace trimci_core {

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>,
           std::vector<double>, std::vector<double>>
compute_4rdm_cumulant_from_3rdm_spin_resolved(
    const std::vector<double>& gamma_aa,
    const std::vector<double>& gamma_bb,
    const std::vector<double>& Gamma_aa,
    const std::vector<double>& Gamma_ab,
    const std::vector<double>& Gamma_bb,
    const std::vector<double>& Gamma3_aaa,
    const std::vector<double>& Gamma3_aab,
    const std::vector<double>& Gamma3_abb,
    const std::vector<double>& Gamma3_bbb,
    int n_orb
) {
    const int N = n_orb;
    const size_t sN = static_cast<size_t>(N);
    const size_t N2 = sN * sN;
    const size_t N3 = N2 * sN;
    const size_t N4 = N3 * sN;
    const size_t N5 = N4 * sN;
    const size_t N6 = N5 * sN;
    const size_t N7 = N6 * sN;
    const size_t N8 = N7 * sN;

    if (N > 14) {
        size_t storage_mb = (N8 * 8 * 5) / (1024 * 1024);
        throw std::runtime_error(
            "compute_4rdm_cumulant_from_3rdm: n_orb=" + std::to_string(N) +
            " requires ~" + std::to_string(storage_mb) + "MB, aborting");
    }

    // Raw data pointers for fast indexing
    const double* ga = gamma_aa.data();
    const double* gb = gamma_bb.data();
    const double* G3a   = Gamma3_aaa.data();
    const double* G3aab = Gamma3_aab.data();
    const double* G3abb = Gamma3_abb.data();
    const double* G3b   = Gamma3_bbb.data();

    // Index helpers
    auto I2 = [sN](int p, int q) -> size_t {
        return static_cast<size_t>(p) * sN + static_cast<size_t>(q);
    };
    auto I4 = [sN, N2, N3](int p, int q, int r, int s) -> size_t {
        return static_cast<size_t>(p)*N3 + static_cast<size_t>(q)*N2
             + static_cast<size_t>(r)*sN + static_cast<size_t>(s);
    };
    auto I6 = [sN, N2, N3, N4, N5](int p, int q, int r,
                                     int s, int t, int u) -> size_t {
        return static_cast<size_t>(p)*N5 + static_cast<size_t>(q)*N4
             + static_cast<size_t>(r)*N3 + static_cast<size_t>(s)*N2
             + static_cast<size_t>(t)*sN + static_cast<size_t>(u);
    };
    auto I8 = [sN, N2, N3, N4, N5, N6, N7](int p, int q, int r, int s,
                                              int t, int u, int v, int w) -> size_t {
        return static_cast<size_t>(p)*N7 + static_cast<size_t>(q)*N6
             + static_cast<size_t>(r)*N5 + static_cast<size_t>(s)*N4
             + static_cast<size_t>(t)*N3 + static_cast<size_t>(u)*N2
             + static_cast<size_t>(v)*sN + static_cast<size_t>(w);
    };

    // det₃ helpers: 3×3 determinant of same-spin 1-RDM
    auto det3_a = [ga, sN](int a, int b, int c, int d, int e, int f) -> double {
        double ad = ga[a*sN+d], ae = ga[a*sN+e], af = ga[a*sN+f];
        double bd = ga[b*sN+d], be = ga[b*sN+e], bf = ga[b*sN+f];
        double cd = ga[c*sN+d], ce = ga[c*sN+e], cf = ga[c*sN+f];
        return ad*(be*cf - bf*ce) - ae*(bd*cf - bf*cd) + af*(bd*ce - be*cd);
    };
    auto det3_b = [gb, sN](int a, int b, int c, int d, int e, int f) -> double {
        double ad = gb[a*sN+d], ae = gb[a*sN+e], af = gb[a*sN+f];
        double bd = gb[b*sN+d], be = gb[b*sN+e], bf = gb[b*sN+f];
        double cd = gb[c*sN+d], ce = gb[c*sN+e], cf = gb[c*sN+f];
        return ad*(be*cf - bf*ce) - ae*(bd*cf - bf*cd) + af*(bd*ce - be*cd);
    };

    // ========= Pre-compute λ² = Γ² − det₂(γ) =========
    std::vector<double> L_aa(N4, 0.0);
    std::vector<double> L_ab(N4, 0.0);
    std::vector<double> L_bb(N4, 0.0);

    #pragma omp parallel for collapse(2)
    for (int p = 0; p < N; ++p) {
        for (int q = 0; q < N; ++q) {
            for (int r = 0; r < N; ++r) {
                for (int s = 0; s < N; ++s) {
                    size_t i = I4(p,q,r,s);
                    // λ²_aa = Γ²_aa - (γ_a(p,r)γ_a(q,s) - γ_a(p,s)γ_a(q,r))
                    L_aa[i] = Gamma_aa[i]
                        - ga[I2(p,r)]*ga[I2(q,s)] + ga[I2(p,s)]*ga[I2(q,r)];
                    // λ²_ab = Γ²_ab - γ_a(p,r)γ_b(q,s)
                    L_ab[i] = Gamma_ab[i] - ga[I2(p,r)]*gb[I2(q,s)];
                    // λ²_bb = Γ²_bb - (γ_b(p,r)γ_b(q,s) - γ_b(p,s)γ_b(q,r))
                    L_bb[i] = Gamma_bb[i]
                        - gb[I2(p,r)]*gb[I2(q,s)] + gb[I2(p,s)]*gb[I2(q,r)];
                }
            }
        }
    }

    const double* La  = L_aa.data();
    const double* Lab = L_ab.data();
    const double* Lb  = L_bb.data();

    // Allocate output
    std::vector<double> G4_aaaa(N8, 0.0);
    std::vector<double> G4_aaab(N8, 0.0);
    std::vector<double> G4_aabb(N8, 0.0);
    std::vector<double> G4_abbb(N8, 0.0);
    std::vector<double> G4_bbbb(N8, 0.0);

    // ===================== aaaa sector =====================
    // Γ⁴_aaaa = A_S[Γ³_aaa ⊗ γ_a] + A_S[λ²_aa ⊗ λ²_aa] − 3·det₄(γ_a)
    #pragma omp parallel for collapse(4)
    for (int p = 0; p < N; ++p) {
      for (int q = 0; q < N; ++q) {
        for (int r = 0; r < N; ++r) {
          for (int s = 0; s < N; ++s) {
            for (int t = 0; t < N; ++t) {
              for (int u = 0; u < N; ++u) {
                for (int v = 0; v < N; ++v) {
                  for (int w = 0; w < N; ++w) {
                    double val = 0.0;

                    // --- A_S[Γ³_aaa ⊗ γ_a]: 16 terms ---
                    // sign = (-1)^{k+l}, k=removed creator, l=removed annihilator (1-indexed)
                    // k=1 (spectator=p)
                    val += G3a[I6(q,r,s, u,v,w)] * ga[I2(p,t)];
                    val -= G3a[I6(q,r,s, t,v,w)] * ga[I2(p,u)];
                    val += G3a[I6(q,r,s, t,u,w)] * ga[I2(p,v)];
                    val -= G3a[I6(q,r,s, t,u,v)] * ga[I2(p,w)];
                    // k=2 (spectator=q)
                    val -= G3a[I6(p,r,s, u,v,w)] * ga[I2(q,t)];
                    val += G3a[I6(p,r,s, t,v,w)] * ga[I2(q,u)];
                    val -= G3a[I6(p,r,s, t,u,w)] * ga[I2(q,v)];
                    val += G3a[I6(p,r,s, t,u,v)] * ga[I2(q,w)];
                    // k=3 (spectator=r)
                    val += G3a[I6(p,q,s, u,v,w)] * ga[I2(r,t)];
                    val -= G3a[I6(p,q,s, t,v,w)] * ga[I2(r,u)];
                    val += G3a[I6(p,q,s, t,u,w)] * ga[I2(r,v)];
                    val -= G3a[I6(p,q,s, t,u,v)] * ga[I2(r,w)];
                    // k=4 (spectator=s)
                    val -= G3a[I6(p,q,r, u,v,w)] * ga[I2(s,t)];
                    val += G3a[I6(p,q,r, t,v,w)] * ga[I2(s,u)];
                    val -= G3a[I6(p,q,r, t,u,w)] * ga[I2(s,v)];
                    val += G3a[I6(p,q,r, t,u,v)] * ga[I2(s,w)];

                    // --- A_S[λ²_aa ⊗ λ²_aa]: 9 terms ---
                    // Creation: {pq,rs}(+), {pr,qs}(-), {ps,qr}(+)
                    // Annihilation: {tu,vw}(+), {tv,uw}(-), {tw,uv}(+)
                    val += La[I4(p,q,t,u)] * La[I4(r,s,v,w)];
                    val -= La[I4(p,q,t,v)] * La[I4(r,s,u,w)];
                    val += La[I4(p,q,t,w)] * La[I4(r,s,u,v)];
                    val -= La[I4(p,r,t,u)] * La[I4(q,s,v,w)];
                    val += La[I4(p,r,t,v)] * La[I4(q,s,u,w)];
                    val -= La[I4(p,r,t,w)] * La[I4(q,s,u,v)];
                    val += La[I4(p,s,t,u)] * La[I4(q,r,v,w)];
                    val -= La[I4(p,s,t,v)] * La[I4(q,r,u,w)];
                    val += La[I4(p,s,t,w)] * La[I4(q,r,u,v)];

                    // --- −3·det₄(γ_a) via Laplace along first row ---
                    double d4 = ga[I2(p,t)] * det3_a(q,r,s, u,v,w)
                              - ga[I2(p,u)] * det3_a(q,r,s, t,v,w)
                              + ga[I2(p,v)] * det3_a(q,r,s, t,u,w)
                              - ga[I2(p,w)] * det3_a(q,r,s, t,u,v);
                    val -= 3.0 * d4;

                    G4_aaaa[I8(p,q,r,s,t,u,v,w)] = val;
                  }
                }
              }
            }
          }
        }
      }
    }

    // ===================== bbbb sector =====================
    // Mirror of aaaa with β
    #pragma omp parallel for collapse(4)
    for (int p = 0; p < N; ++p) {
      for (int q = 0; q < N; ++q) {
        for (int r = 0; r < N; ++r) {
          for (int s = 0; s < N; ++s) {
            for (int t = 0; t < N; ++t) {
              for (int u = 0; u < N; ++u) {
                for (int v = 0; v < N; ++v) {
                  for (int w = 0; w < N; ++w) {
                    double val = 0.0;

                    // A_S[Γ³_bbb ⊗ γ_b]: 16 terms
                    val += G3b[I6(q,r,s, u,v,w)] * gb[I2(p,t)];
                    val -= G3b[I6(q,r,s, t,v,w)] * gb[I2(p,u)];
                    val += G3b[I6(q,r,s, t,u,w)] * gb[I2(p,v)];
                    val -= G3b[I6(q,r,s, t,u,v)] * gb[I2(p,w)];
                    val -= G3b[I6(p,r,s, u,v,w)] * gb[I2(q,t)];
                    val += G3b[I6(p,r,s, t,v,w)] * gb[I2(q,u)];
                    val -= G3b[I6(p,r,s, t,u,w)] * gb[I2(q,v)];
                    val += G3b[I6(p,r,s, t,u,v)] * gb[I2(q,w)];
                    val += G3b[I6(p,q,s, u,v,w)] * gb[I2(r,t)];
                    val -= G3b[I6(p,q,s, t,v,w)] * gb[I2(r,u)];
                    val += G3b[I6(p,q,s, t,u,w)] * gb[I2(r,v)];
                    val -= G3b[I6(p,q,s, t,u,v)] * gb[I2(r,w)];
                    val -= G3b[I6(p,q,r, u,v,w)] * gb[I2(s,t)];
                    val += G3b[I6(p,q,r, t,v,w)] * gb[I2(s,u)];
                    val -= G3b[I6(p,q,r, t,u,w)] * gb[I2(s,v)];
                    val += G3b[I6(p,q,r, t,u,v)] * gb[I2(s,w)];

                    // A_S[λ²_bb ⊗ λ²_bb]: 9 terms
                    val += Lb[I4(p,q,t,u)] * Lb[I4(r,s,v,w)];
                    val -= Lb[I4(p,q,t,v)] * Lb[I4(r,s,u,w)];
                    val += Lb[I4(p,q,t,w)] * Lb[I4(r,s,u,v)];
                    val -= Lb[I4(p,r,t,u)] * Lb[I4(q,s,v,w)];
                    val += Lb[I4(p,r,t,v)] * Lb[I4(q,s,u,w)];
                    val -= Lb[I4(p,r,t,w)] * Lb[I4(q,s,u,v)];
                    val += Lb[I4(p,s,t,u)] * Lb[I4(q,r,v,w)];
                    val -= Lb[I4(p,s,t,v)] * Lb[I4(q,r,u,w)];
                    val += Lb[I4(p,s,t,w)] * Lb[I4(q,r,u,v)];

                    // −3·det₄(γ_b)
                    double d4 = gb[I2(p,t)] * det3_b(q,r,s, u,v,w)
                              - gb[I2(p,u)] * det3_b(q,r,s, t,v,w)
                              + gb[I2(p,v)] * det3_b(q,r,s, t,u,w)
                              - gb[I2(p,w)] * det3_b(q,r,s, t,u,v);
                    val -= 3.0 * d4;

                    G4_bbbb[I8(p,q,r,s,t,u,v,w)] = val;
                  }
                }
              }
            }
          }
        }
      }
    }

    // ===================== aaab sector =====================
    // [p_α, q_α, r_α, s_β, t_α, u_α, v_α, w_β]
    // Antisymmetric in α-creators (p,q,r) and α-annihilators (t,u,v)
    #pragma omp parallel for collapse(4)
    for (int p = 0; p < N; ++p) {
      for (int q = 0; q < N; ++q) {
        for (int r = 0; r < N; ++r) {
          for (int s = 0; s < N; ++s) {
            for (int t = 0; t < N; ++t) {
              for (int u = 0; u < N; ++u) {
                for (int v = 0; v < N; ++v) {
                  for (int w = 0; w < N; ++w) {
                    double val = 0.0;

                    // --- A_S[Γ³⊗γ]: 10 terms ---
                    // Type 1: k=4(s_β), l=4(w_β) → Γ³_aaa · γ_b
                    val += G3a[I6(p,q,r, t,u,v)] * gb[I2(s,w)];
                    // Type 2: k∈{1,2,3}(α), l∈{1,2,3}(α) → Γ³_aab · γ_a
                    // k=1(p), l=1(t): +
                    val += G3aab[I6(q,r,s, u,v,w)] * ga[I2(p,t)];
                    // k=1, l=2(u): -
                    val -= G3aab[I6(q,r,s, t,v,w)] * ga[I2(p,u)];
                    // k=1, l=3(v): +
                    val += G3aab[I6(q,r,s, t,u,w)] * ga[I2(p,v)];
                    // k=2(q), l=1: -
                    val -= G3aab[I6(p,r,s, u,v,w)] * ga[I2(q,t)];
                    // k=2, l=2: +
                    val += G3aab[I6(p,r,s, t,v,w)] * ga[I2(q,u)];
                    // k=2, l=3: -
                    val -= G3aab[I6(p,r,s, t,u,w)] * ga[I2(q,v)];
                    // k=3(r), l=1: +
                    val += G3aab[I6(p,q,s, u,v,w)] * ga[I2(r,t)];
                    // k=3, l=2: -
                    val -= G3aab[I6(p,q,s, t,v,w)] * ga[I2(r,u)];
                    // k=3, l=3: +
                    val += G3aab[I6(p,q,s, t,u,w)] * ga[I2(r,v)];

                    // --- A_S[λ²⊗λ²]: 9 terms (λ²_aa · λ²_ab) ---
                    // Creation: {pq,rs}_αα,αβ(+), {pr,qs}(-), {qr,ps}(+)
                    // Annihilation: {tu,vw}_αα,αβ(+), {tv,uw}(-), {uv,tw}(+)
                    val += La[I4(p,q,t,u)] * Lab[I4(r,s,v,w)];
                    val -= La[I4(p,q,t,v)] * Lab[I4(r,s,u,w)];
                    val += La[I4(p,q,u,v)] * Lab[I4(r,s,t,w)];
                    val -= La[I4(p,r,t,u)] * Lab[I4(q,s,v,w)];
                    val += La[I4(p,r,t,v)] * Lab[I4(q,s,u,w)];
                    val -= La[I4(p,r,u,v)] * Lab[I4(q,s,t,w)];
                    val += La[I4(q,r,t,u)] * Lab[I4(p,s,v,w)];
                    val -= La[I4(q,r,t,v)] * Lab[I4(p,s,u,w)];
                    val += La[I4(q,r,u,v)] * Lab[I4(p,s,t,w)];

                    // --- −3·det₄(γ)_aaab = −3·det₃(γ_a)(p,q,r;t,u,v)·γ_b(s,w) ---
                    val -= 3.0 * det3_a(p,q,r, t,u,v) * gb[I2(s,w)];

                    G4_aaab[I8(p,q,r,s,t,u,v,w)] = val;
                  }
                }
              }
            }
          }
        }
      }
    }

    // ===================== abbb sector =====================
    // [p_α, q_β, r_β, s_β, t_α, u_β, v_β, w_β]
    // Antisymmetric in β-creators (q,r,s) and β-annihilators (u,v,w)
    #pragma omp parallel for collapse(4)
    for (int p = 0; p < N; ++p) {
      for (int q = 0; q < N; ++q) {
        for (int r = 0; r < N; ++r) {
          for (int s = 0; s < N; ++s) {
            for (int t = 0; t < N; ++t) {
              for (int u = 0; u < N; ++u) {
                for (int v = 0; v < N; ++v) {
                  for (int w = 0; w < N; ++w) {
                    double val = 0.0;

                    // --- A_S[Γ³⊗γ]: 10 terms ---
                    // Type 1: k=1(p_α), l=1(t_α) → Γ³_bbb · γ_a
                    val += G3b[I6(q,r,s, u,v,w)] * ga[I2(p,t)];
                    // Type 2: k∈{2,3,4}(β), l∈{2,3,4}(β) → Γ³_abb · γ_b
                    // k=2(q), l=2(u): +
                    val += G3abb[I6(p,r,s, t,v,w)] * gb[I2(q,u)];
                    // k=2, l=3(v): -
                    val -= G3abb[I6(p,r,s, t,u,w)] * gb[I2(q,v)];
                    // k=2, l=4(w): +
                    val += G3abb[I6(p,r,s, t,u,v)] * gb[I2(q,w)];
                    // k=3(r), l=2: -
                    val -= G3abb[I6(p,q,s, t,v,w)] * gb[I2(r,u)];
                    // k=3, l=3: +
                    val += G3abb[I6(p,q,s, t,u,w)] * gb[I2(r,v)];
                    // k=3, l=4: -
                    val -= G3abb[I6(p,q,s, t,u,v)] * gb[I2(r,w)];
                    // k=4(s), l=2: +
                    val += G3abb[I6(p,q,r, t,v,w)] * gb[I2(s,u)];
                    // k=4, l=3: -
                    val -= G3abb[I6(p,q,r, t,u,w)] * gb[I2(s,v)];
                    // k=4, l=4: +
                    val += G3abb[I6(p,q,r, t,u,v)] * gb[I2(s,w)];

                    // --- A_S[λ²⊗λ²]: 9 terms (λ²_ab · λ²_bb) ---
                    // Creation: {pq,rs}_αβ,ββ(+), {pr,qs}(-), {ps,qr}(+)
                    // Annihilation: {tu,vw}_αβ,ββ(+), {tv,uw}(-), {tw,uv}(+)
                    val += Lab[I4(p,q,t,u)] * Lb[I4(r,s,v,w)];
                    val -= Lab[I4(p,q,t,v)] * Lb[I4(r,s,u,w)];
                    val += Lab[I4(p,q,t,w)] * Lb[I4(r,s,u,v)];
                    val -= Lab[I4(p,r,t,u)] * Lb[I4(q,s,v,w)];
                    val += Lab[I4(p,r,t,v)] * Lb[I4(q,s,u,w)];
                    val -= Lab[I4(p,r,t,w)] * Lb[I4(q,s,u,v)];
                    val += Lab[I4(p,s,t,u)] * Lb[I4(q,r,v,w)];
                    val -= Lab[I4(p,s,t,v)] * Lb[I4(q,r,u,w)];
                    val += Lab[I4(p,s,t,w)] * Lb[I4(q,r,u,v)];

                    // --- −3·det₄(γ)_abbb = −3·γ_a(p,t)·det₃(γ_b)(q,r,s;u,v,w) ---
                    val -= 3.0 * ga[I2(p,t)] * det3_b(q,r,s, u,v,w);

                    G4_abbb[I8(p,q,r,s,t,u,v,w)] = val;
                  }
                }
              }
            }
          }
        }
      }
    }

    // ===================== aabb sector =====================
    // [p_α, q_α, r_β, s_β, t_α, u_α, v_β, w_β]
    // Antisymmetric in α-creators (p,q), α-annihilators (t,u),
    //                   β-creators (r,s), β-annihilators (v,w)
    #pragma omp parallel for collapse(4)
    for (int p = 0; p < N; ++p) {
      for (int q = 0; q < N; ++q) {
        for (int r = 0; r < N; ++r) {
          for (int s = 0; s < N; ++s) {
            for (int t = 0; t < N; ++t) {
              for (int u = 0; u < N; ++u) {
                for (int v = 0; v < N; ++v) {
                  for (int w = 0; w < N; ++w) {
                    double val = 0.0;

                    // --- A_S[Γ³⊗γ]: 8 terms ---
                    // α spectator: Γ³_abb · γ_a (4 terms)
                    // Γ³_abb[α,β,β, α,β,β]
                    // k=1(p), l=1(t): +
                    val += G3abb[I6(q,r,s, u,v,w)] * ga[I2(p,t)];
                    // k=1(p), l=2(u): -
                    val -= G3abb[I6(q,r,s, t,v,w)] * ga[I2(p,u)];
                    // k=2(q), l=1(t): -
                    val -= G3abb[I6(p,r,s, u,v,w)] * ga[I2(q,t)];
                    // k=2(q), l=2(u): +
                    val += G3abb[I6(p,r,s, t,v,w)] * ga[I2(q,u)];

                    // β spectator: Γ³_aab · γ_b (4 terms)
                    // Γ³_aab[α,α,β, α,α,β]
                    // k=3(r), l=3(v): +
                    val += G3aab[I6(p,q,s, t,u,w)] * gb[I2(r,v)];
                    // k=3(r), l=4(w): -
                    val -= G3aab[I6(p,q,s, t,u,v)] * gb[I2(r,w)];
                    // k=4(s), l=3(v): -
                    val -= G3aab[I6(p,q,r, t,u,w)] * gb[I2(s,v)];
                    // k=4(s), l=4(w): +
                    val += G3aab[I6(p,q,r, t,u,v)] * gb[I2(s,w)];

                    // --- A_S[λ²⊗λ²]: 5 terms ---
                    // P1×Q1: λ²_aa(p,q;t,u) · λ²_bb(r,s;v,w)
                    val += La[I4(p,q,t,u)] * Lb[I4(r,s,v,w)];
                    // P2×Q2: λ²_ab(p,r;t,v) · λ²_ab(q,s;u,w)
                    val += Lab[I4(p,r,t,v)] * Lab[I4(q,s,u,w)];
                    // P2×Q3: −λ²_ab(p,r;t,w) · λ²_ab(q,s;u,v)
                    val -= Lab[I4(p,r,t,w)] * Lab[I4(q,s,u,v)];
                    // P3×Q2: −λ²_ab(p,s;t,v) · λ²_ab(q,r;u,w)
                    val -= Lab[I4(p,s,t,v)] * Lab[I4(q,r,u,w)];
                    // P3×Q3: +λ²_ab(p,s;t,w) · λ²_ab(q,r;u,v)
                    val += Lab[I4(p,s,t,w)] * Lab[I4(q,r,u,v)];

                    // --- −3·det₄(γ)_aabb = −3·det₂(γ_a)(p,q;t,u)·det₂(γ_b)(r,s;v,w) ---
                    double det2a = ga[I2(p,t)]*ga[I2(q,u)] - ga[I2(p,u)]*ga[I2(q,t)];
                    double det2b = gb[I2(r,v)]*gb[I2(s,w)] - gb[I2(r,w)]*gb[I2(s,v)];
                    val -= 3.0 * det2a * det2b;

                    G4_aabb[I8(p,q,r,s,t,u,v,w)] = val;
                  }
                }
              }
            }
          }
        }
      }
    }

    return std::make_tuple(std::move(G4_aaaa), std::move(G4_aaab),
                           std::move(G4_aabb), std::move(G4_abbb),
                           std::move(G4_bbbb));
}

} // namespace trimci_core
