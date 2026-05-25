"""
TrimCI three-phase workflow skill.

A production-ready, self-contained script for running TrimCI calculations.
Copy to your experiment directory, edit the configuration, and run.
After copy, rename the script to avoid confusion of this TrimCI_skill template with the actual experiment script.

=== Three Phases ===

Phase 0 — Core + OrbOpt Discovery
    Multi-run stochastic sampling + multi-cycle orbital optimization with
    a small det space (e.g. 100–10k dets).
    Goal: find high-quality initial orbitals and determinant space.
    Key params: num_runs, cycles, max_final_dets
    Cost: hours (dominated by num_runs × cycles × max_final_dets)
    Full param reference:
      - top-level TrimCI knobs → py/trimci/TrimCI_runner/config.py (DEFAULT_CONFIG)
      - optimizer_options_dict → py/trimci/TrimCI_runner/orbital_optimization.py
      - dispatcher entry       → py/trimci/TrimCI_runner/trimci_driver.py (run_full)

Phase 1 — Core + OrbOpt Refinement (Expansion)
    Starting from Phase 0 best result, gradually expand the det space
    while continuing orbital optimization at every step.
    Goal: let orbitals co-evolve with the growing det space,
    avoiding the convergence slowdown from frozen orbitals.
    Key params: max_n_dets, growth_factor, orbital_opt_max_iter
    Cost: hours–days (depends on max_n_dets and integral sparsity)

Phase 2 — Core Expansion Only
    Freeze orbitals, rapidly expand to final target det count,
    optionally with PT2 energy correction.
    Goal: squeeze out remaining variational energy with a large det space.
    Key params: max_n_dets, growth_factor
    Cost: hours (fast without orbopt overhead)

Phase 1 + Phase 2 share one implementation; full param reference:
  - py/trimci/TrimCI_runner/run_expansion.py
      EXPANSION_DEFAULTS    — growth, pool-build, screening, warm-start
      ORBOPT_DEFAULTS       — BFGS convergence (Phase 1 only)
      DAVIDSON_DEFAULTS     — subspace, iters, tols
      PT2_DEFAULTS          — sketch, semistochastic partition, eps adaptation

=== Phase Switches ===

    ENABLE_PHASE1 = True/False   — skip Phase 1 (go directly Phase 0 → Phase 2)
    ENABLE_PHASE2 = True/False   — skip Phase 2 (stop after Phase 1)

    Setting both to False runs Phase 0 only (orbopt discovery).

=== Usage ===

    1. Copy this file to your experiment directory
    2. Place FCIDUMP in the same directory (or edit FCIDUMP path below)
    3. Edit configuration parameters
    4. python TrimCI_skill.py
"""
import json
import os
import shutil
import sys
import time

import numpy as np

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
FCIDUMP = os.path.join(WORK_DIR, "FCIDUMP")

# ============================================================
#  Phase Switches
# ============================================================
ENABLE_PHASE1 = True   # Set False to skip orbopt refinement (Phase 0 → Phase 2)
ENABLE_PHASE2 = True   # Set False to stop after Phase 1 (no frozen expansion)

# ============================================================
#  Phase 0: Core + OrbOpt Discovery
# ============================================================
PHASE0_CONFIG = {
    # "random": [n_per_run, pool_size] — each run starts from 1 reference det
    # + n_per_run random dets sampled from a pool of pool_size candidates.
    # Larger pool → more diverse starting points → better chance of finding
    # good orbital rotation directions. Cost: ~seconds per run for pool generation.
    # Recommended: 10000 (quick), 100000 (thorough for degenerate systems)
    "initial_dets_dict": {"reference": 1, "random": [1, 10000]},
    # backend — CI workflow (pool build + Davidson) execution backend.
    # "auto": pick CPU if max_final_dets < 1000 else GPU (recommended default)
    # "cpu":  force CPU, regardless of GPU availability
    # "gpu":  force GPU, fallback to CPU if build lacks CUDA support
    # Note: GPU overhead dominates below ~1000 dets (measured 3.7× slower
    # on Fe4S4 N=36/100 dets, 2026-04-23). GPU wins at ≥10k dets.
    "backend": "auto",
    # orbital_optimization — set False to skip orbopt entirely in Phase 0
    # (useful for quick baseline runs or systems where HF orbitals are already good)
    "orbital_optimization": True,
    "optimizer_options_dict": {
        # optimizer — orbital optimizer algorithm.
        # "auto_bfgs": pick cpp_bfgs for small n_orb, gpu_bfgs for n_orb ≥ 16
        # "cpp_bfgs" / "bfgs": explicit CPU BFGS with analytic gradient
        # "gpu_bfgs": explicit GPU BFGS (uses compute_orbital_gradient_gpu_128 +
        #             transform_integrals_gpu + compute_rdm_gpu_128)
        # The auto heuristic is separate from backend: you can have
        # backend='cpu' + gpu_bfgs (CI on CPU, orbopt on GPU) and vice versa.
        "optimizer": "auto_bfgs",
        # cycles — number of orbopt-then-CI iterations.
        # Each cycle: BFGS rotates orbitals → re-run all num_runs TrimCI.
        # More cycles = better orbitals but much longer wall time.
        # 5 cycles may get stuck in local minima; 10 cycles recommended.
        # Empirical: run09-run0 needed 5 cycles to escape a bad basin.
        "cycles": 10,
        # maxiter — BFGS steps per cycle. 100 usually enough; BFGS converges
        # or hits maxiter, then CI re-runs with the new orbitals.
        "maxiter": 100,
        "tracking_dets": False,
        "davidson_tol": 1e-7,
        "ftol": 1e-8,
    },
    "threshold": 1e-2,
    "pool_core_ratio": 40,
    "num_groups": 20,
    "max_rounds": 4,
    # core_set_ratio — [min_ratio, max_ratio] for det space growth per iteration.
    # [1, 2] means det count can at most double each TrimCI iteration.
    # Controls how fast dets grow from first_cycle_keep_size to max_final_dets.
    # Recommended: [1, 1.1] (conservative), [1, 2] (aggressive, faster to max_final_dets)
    "core_set_ratio": [1, 1.1],
    # max_final_dets — target det count at the END of Phase 0.
    # Larger → better orbopt quality (2-RDM more accurate) but slower.
    # Recommended: 100 (quick test), 1000 (balanced), 10000 (thorough orbopt)
    "max_final_dets": 100,
    "first_cycle_keep_size": 10,
    "pool_build_strategy": "heat_bath",
    "local_trim_keep_ratio": 4,
    # num_runs — independent TrimCI runs per cycle, each with different
    # random initialization. The BEST run's energy determines the cycle result.
    # Run-to-run variance is large (>3 Ha observed on 6x6 Hubbard).
    # More runs = better sampling of the stochastic landscape.
    # Recommended: 64 (quick), 256 (production)
    "num_runs": 64,
    # n_cpu_workers — concurrent CPU subprocesses (each its own OMP pool).
    # Total CPU use = n_cpu_workers × (OMP_NUM_THREADS / n_cpu_workers).
    # Rule of thumb: each worker gets ≥4 OMP threads.
    # Recommended: 1 (laptop), 4 (workstation), 16 (cluster with 128+ cores)
    # Legacy alias `n_parallel` still accepted for backwards compatibility.
    "n_cpu_workers": 1,
    # n_gpu_workers — concurrent GPU-pinned subprocesses, 1:1 to GPUs.
    # Each worker claims one GPU via CUDA_VISIBLE_DEVICES at spawn time.
    # Mutually exclusive with n_cpu_workers > 1 (a pool is either CPU or GPU).
    # Requires `request_gpus = n_gpu_workers` in the HTCondor submit file.
    # Recommended: 0 (CPU path), 4 (4-GPU node), 8 (H100 x8 or similar)
    "n_gpu_workers": 0,
    "verbosity": 1,
}

# ============================================================
#  Phase 1: Core + OrbOpt Refinement (Expansion with orbopt)
# ============================================================
PHASE1_CONFIG = {
    # backend — Davidson matvec backend (mirrors PHASE0_CONFIG/run_full API).
    #   "cpu"  : CPU matvec (default; same as Phase 1 always was)
    #   "gpu"  : Offload Davidson H*v to CUDA (uses Phase 0 GPU infrastructure
    #            kernel_ch1/ch2/ch3). Pool build, orbopt, PT2 stay on CPU.
    #            Mutually exclusive with use_connection_cache and sparse_update.
    #   "auto" : Pick "gpu" if max_n_dets ≥ 100k AND GPU available, else "cpu".
    # Only meaningful when Davidson dominates round time (≥1M dets on 1 GPU).
    "backend": "cpu",
    # max_n_dets — det count where orbopt stops and Phase 2 takes over.
    # Larger → orbitals adapt to bigger det spaces → better final energy,
    # but orbopt cost grows (Davidson matvec + 2-RDM at each step).
    # Recommended: 1M (molecules, dense integrals), 64M (lattice models, sparse integrals)
    "max_n_dets": 1_000_000,
    # growth_factor — det count multiplier per expansion round.
    # 1.1 = 10% growth, ~90 rounds from 10k to 64M.
    # Slow growth lets orbitals track smoothly; too fast may miss
    # orbital adaptation opportunities.
    "growth_factor": 1.1,
    # orbital_optimization — set False to freeze orbitals from Phase 0
    # and run expansion only (effectively merging Phase 1 into Phase 2)
    "orbital_optimization": True,
    "orbital_opt_max_iter": 50,
    # use_connection_cache — cache all H_ij elements in Davidson matvec.
    # Dramatically speeds up (which calls Davidson many times per round)
    # at the cost of O(min(N², N*n_connected)) memory. Safe for Phase 1.
    # Set False if running out of memory on machines with small RAM.
    "use_connection_cache": True,
    # davidson.energy_tol — Davidson stops when |ΔE_iter| < tol.
    # Phase 1 dets grow 10%/round, so round-to-round dE is already ~1e-3
    # to 1e-4. Converging Davidson below 1e-4 wastes iterations.
    # Orbopt uses its own grad_tol=1e-3 as primary convergence criterion.
    "davidson": {"energy_tol": 1e-4},
}

# ============================================================
#  Phase 2: Core Expansion Only
# ============================================================
PHASE2_CONFIG = {
    # backend — see PHASE1_CONFIG.backend notes.
    #   Phase 2 hits the largest det counts (10M-100M+); GPU expected to
    #   help most here. "auto" recommended for production runs.
    "backend": "cpu",
    # max_n_dets — final target. Orbitals are frozen from Phase 1;
    # only det space grows. Diminishing returns beyond ~100M for most systems.
    # Recommended: 10M (quick), 100M (standard), 300M+ (if orbopt plateau is high)
    "max_n_dets": 100_000_000,
    # growth_factor — 2.0 = doubling each round. Faster than Phase 1 because
    # no orbopt overhead. Each round: pool build + Davidson + optional PT2.
    "growth_factor": 2.0,
    "pt2_correction": True,
    # davidson.energy_tol — Phase 2 is the final stage, so target tighter
    # than Phase 1. 1e-5 Ha = 10 μHa, still 160× below chemical accuracy
    # (1.6 mHa). Saves ~2× Davidson time vs 1e-7 default.
    "davidson": {"energy_tol": 1e-5},
}


# ============================================================
#  Implementation
# ============================================================

def _save_summary(out_dir, summary):
    """Atomically write summary.json (write tmp then rename)."""
    path = os.path.join(out_dir, "summary.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(summary, f, indent=2)
    os.replace(tmp, path)


def _find_orbopt_fcidump(details):
    """Locate fcidump_orbopt from run_full details (or by filesystem scan).

    Three defensive layers:
      1. details['fcidump_orbopt_path'] — authoritative, set by
         run_orbital_optimization since trimci 0.2.0.
      2. details['orbopt_dir']         — derive `fcidump_orbopt` from the
         orbopt directory; also covers resumed jobs.
      3. filesystem scan of WORK_DIR/trimci_orbopt_results/* — last-resort
         safety net for older result_dirs or unusual calling patterns.
    """
    if details and isinstance(details, dict):
        # Layer 1: explicit path
        p = details.get("fcidump_orbopt_path")
        if p and os.path.exists(p):
            return p
        # Layer 2: orbopt_dir + 'fcidump_orbopt' (or latest fcidump_cycle_*)
        orbopt_dir = details.get("orbopt_dir")
        if orbopt_dir and os.path.isdir(orbopt_dir):
            p = os.path.join(orbopt_dir, "fcidump_orbopt")
            if os.path.exists(p):
                return p
            cycles = sorted([f for f in os.listdir(orbopt_dir)
                             if f.startswith("fcidump_cycle_")])
            if cycles:
                return os.path.join(orbopt_dir, cycles[-1])

    # Layer 3: filesystem scan (legacy / sub-run-only details)
    if not os.path.isdir(WORK_DIR):
        return None
    for d in sorted(os.listdir(WORK_DIR), reverse=True):
        if not d.startswith("trimci_orbopt_results"):
            continue
        base = os.path.join(WORK_DIR, d)
        for sub in sorted(os.listdir(base), reverse=True):
            sub_path = os.path.join(base, sub)
            if not os.path.isdir(sub_path):
                continue
            p = os.path.join(sub_path, "fcidump_orbopt")
            if os.path.exists(p):
                return p
            cycles = sorted([f for f in os.listdir(sub_path)
                             if f.startswith("fcidump_cycle_")])
            if cycles:
                return os.path.join(sub_path, cycles[-1])
    return None


def run_phase0(fcidump_path, out_dir):
    """Phase 0: multi-run stochastic sampling + multi-cycle orbital optimization."""
    from trimci import run_full

    cfg = PHASE0_CONFIG
    opt = cfg["optimizer_options_dict"]

    print("=" * 70)
    print("  Phase 0: Core + OrbOpt Discovery")
    print(f"  num_runs={cfg['num_runs']}, n_parallel={cfg['n_parallel']}, "
          f"cycles={opt['cycles']}, max_final_dets={cfg['max_final_dets']}")
    print("=" * 70)
    sys.stdout.flush()

    t0 = time.time()
    energy, dets, coeffs, details, _ = run_full(
        fcidump_path=fcidump_path,
        config_dict=cfg,
    )
    elapsed = time.time() - t0

    if not dets:
        raise RuntimeError("Phase 0 returned no determinants — check run_full output")

    # .alpha/.beta for Determinant objects, indexing for plain tuples
    # For n_orb > 64, .alpha/.beta are arrays of uint64 segments (e.g. [low, high] for 128-bit).
    # The npz format expected by run_expansion is [alpha_segs..., beta_segs...] per row.
    if hasattr(dets[0], "alpha"):
        a0 = dets[0].alpha
        if isinstance(a0, int) or not hasattr(a0, '__len__'):
            # Determinant64: scalar uint64
            alpha = np.array([d.alpha for d in dets], dtype=np.uint64).reshape(-1, 1)
            beta = np.array([d.beta for d in dets], dtype=np.uint64).reshape(-1, 1)
        else:
            # Determinant128+: array of uint64 segments
            alpha = np.array([list(d.alpha) for d in dets], dtype=np.uint64)
            beta = np.array([list(d.beta) for d in dets], dtype=np.uint64)
    else:
        alpha = np.array([d[0] for d in dets], dtype=np.uint64).reshape(-1, 1)
        beta = np.array([d[1] for d in dets], dtype=np.uint64).reshape(-1, 1)
    coeffs_arr = np.array(coeffs, dtype=np.float64)
    dets_path = os.path.join(out_dir, "dets_phase0.npz")
    np.savez_compressed(dets_path, dets=np.column_stack([alpha, beta]), dets_coeffs=coeffs_arr)

    # Locate orbopt FCIDUMP
    fcidump_out = os.path.join(out_dir, "fcidump_phase0")
    orbopt_fcidump = _find_orbopt_fcidump(details)
    if orbopt_fcidump:
        shutil.copy2(orbopt_fcidump, fcidump_out)
        print(f"  OrbOpt FCIDUMP: {orbopt_fcidump}")
    else:
        print("  WARNING: No orbopt FCIDUMP found, using original integrals")
        shutil.copy2(fcidump_path, fcidump_out)

    result = {
        "energy": float(energy),
        "n_dets": len(dets),
        "time_s": round(elapsed, 1),
        "dets_path": dets_path,
        "fcidump_path": fcidump_out,
    }
    print(f"\n  Phase 0 complete: E = {energy:.8f}, N_dets = {len(dets)}, time = {elapsed:.1f}s")
    sys.stdout.flush()
    return result


def run_expansion_phases(fcidump_path, dets_path, out_dir):
    """Run enabled expansion phases (Phase 1 and/or Phase 2)."""
    from trimci.TrimCI_runner.run_expansion import run_expansion

    phases = []
    phase_labels = []

    if ENABLE_PHASE1:
        phases.append(PHASE1_CONFIG)
        phase_labels.append(
            f"Phase 1: orbopt refine -> {PHASE1_CONFIG['max_n_dets']:,} dets "
            f"(growth={PHASE1_CONFIG['growth_factor']})")

    if ENABLE_PHASE2:
        phases.append(PHASE2_CONFIG)
        phase_labels.append(
            f"Phase 2: expansion   -> {PHASE2_CONFIG['max_n_dets']:,} dets "
            f"(growth={PHASE2_CONFIG['growth_factor']})")

    if not phases:
        return None

    label = "\n  ".join(phase_labels)

    result = run_expansion(
        fcidump_path=fcidump_path,
        dets_path=dets_path,
        phases=phases,
        checkpoint_dir=os.path.join(out_dir, "checkpoints"),
        output_prefix=os.path.join(out_dir, "result"),
        label=label,
    )
    return result


def main():
    os.chdir(WORK_DIR)
    out_dir = os.path.join(WORK_DIR, "output")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isfile(FCIDUMP):
        print(f"  ERROR: FCIDUMP not found at {FCIDUMP}")
        print(f"  Place FCIDUMP in {WORK_DIR}/ or edit the FCIDUMP path in this script.")
        sys.exit(1)

    t_start = time.time()
    summary = {"enable_phase1": ENABLE_PHASE1, "enable_phase2": ENABLE_PHASE2}

    # --- Phase 0: always runs ---
    p0 = run_phase0(FCIDUMP, out_dir)
    summary["phase0"] = {
        "energy": p0["energy"],
        "n_dets": p0["n_dets"],
        "time_s": p0["time_s"],
    }
    _save_summary(out_dir, summary)

    # --- Phase 1 and/or Phase 2 ---
    result = run_expansion_phases(p0["fcidump_path"], p0["dets_path"], out_dir)

    t_total = time.time() - t_start

    if result is not None:
        summary.update({
            "final_energy_var": float(result["energy_var"]),
            "final_n_dets": result["n_final"],
            "total_time_s": round(t_total, 1),
            "ndets_history": result["ndets_history"],
            "energy_history": result["energy_history"],
            "pt2_history": result.get("pt2_history", []),
        })
    else:
        summary.update({
            "final_energy_var": p0["energy"],
            "final_n_dets": p0["n_dets"],
            "total_time_s": round(t_total, 1),
        })

    _save_summary(out_dir, summary)

    # --- Final report ---
    print(f"\n{'='*70}")
    print(f"  Workflow complete  ({t_total/3600:.1f} h)")
    print(f"  Phase 0 (orbopt discovery):  E = {p0['energy']:.8f}  @ {p0['n_dets']:,} dets  ({p0['time_s']:.0f}s)")
    if result is not None:
        if ENABLE_PHASE1:
            print(f"  Phase 1 (orbopt refine):     -> {PHASE1_CONFIG['max_n_dets']:,} dets")
        if ENABLE_PHASE2:
            print(f"  Phase 2 (expansion):         -> {result['n_final']:,} dets")
        print(f"  Final:  E_var = {result['energy_var']:.8f}")
        pt2_history = result.get("pt2_history", [])
        if pt2_history and any(abs(x) > 1e-15 for x in pt2_history):
            last_pt2 = pt2_history[-1]
            print(f"  Final:  E_var + PT2 = {result['energy_var'] + last_pt2:.8f}")
    else:
        print(f"  (Phase 1 & 2 disabled — Phase 0 only)")
        print(f"  Final:  E_var = {p0['energy']:.8f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
