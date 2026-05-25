"""High-level entry point for fast expansion.

Usage from experiment scripts::

    # Option 1: from files
    from trimci.TrimCI_runner.run_expansion import run_expansion

    result = run_expansion(
        fcidump_path="path/to/fcidump",
        dets_path="path/to/dets.npz",
        phases=[{
            "max_n_dets": 1_000_000,
            "growth_factor": 1.1,
            "orbital_optimization": True,
        }],
        checkpoint_dir="checkpoints/",
        output_prefix="my_run",
    )

    # Option 2: from raw arrays (e.g. loaded from checkpoints)
    from trimci.checkpoint import load_checkpoint, load_integrals_checkpoint

    ckpt = load_checkpoint("checkpoint_round_478.bin")
    intg = load_integrals_checkpoint("checkpoint_integrals.bin")

    result = run_expansion(
        h1=intg["h1"], eri=intg["eri"], e_nuc=-330.123,
        alpha_init=ckpt["alpha"], beta_init=ckpt["beta"],
        coeffs_init=ckpt["coeffs"],
        phases=[...],
        checkpoint_dir="checkpoints/",
    )
"""
import json
import os
import sys
import time

import numpy as np

from trimci import trimci_core
from .io_utils import read_fcidump

fe = trimci_core.fast_expansion


# ============================================================
#  Defaults (maps 1:1 to C++ ExpansionConfig fields)
# ============================================================

EXPANSION_DEFAULTS = dict(
    # ==== growth target ====
    # ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    # ⭐⭐ Most commonly tuned: target size and growth rate
    max_n_dets           = 1_000_000,   # target determinant count
    growth_factor        = 1.1,         # N_dets *= growth_factor each round
    # ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    max_expansion_rounds = 10_000,      # safety cap on total rounds

    # ==== GPU matvec ====
    # backend — Davidson matvec backend, mirrors Phase 0's args.backend API.
    #   "cpu"  : CPU matvec (default; same as Phase 1+2 always was)
    #   "gpu"  : Offload Davidson H*v to CUDA kernel_ch1/ch2/ch3 (Phase 0 infra)
    #            Uploads integrals + dets + AB-factored groups once per round,
    #            each Davidson iter does H2D v → kernel → D2H sigma.
    #            Pool build, PT2, dressed CI, orbopt stay on CPU regardless.
    #            Mutually exclusive with use_connection_cache, sparse_update,
    #            davidson_block_size > 1.
    #   "auto" : Pick "gpu" if max_n_dets ≥ 100000 AND GPU available, else "cpu".
    #            (Threshold tuned for Phase 1/2 Davidson-dominant rounds; below
    #            ~100k dets CPU matvec is competitive with GPU launch overhead.)
    # Requires trimci build with TRIMCI_HAS_GPU. Expected GPU speedup at
    # ≥10^6 dets on A100/L40S: 3-10× (Davidson-bound rounds).
    backend              = "cpu",

    # ==== pool build ====
    screening_mode       = "hb",        # "hb" | "hb-pt2" | "mi" (mutual-info weighted) | "uniform" (random)
    threshold            = 1e-4,        # initial |H_ij·c_j| > θ to include
    threshold_decay      = 0.5,         # if pool < target: θ *= decay, retry
    strict_target_size   = True,        # truncate pool to exact target N

    # ==== Davidson warm start ====
    no_warm_start        = False,        # True: skip warm-start (use random guess each round)

    # ==== convergence (early stop, -1 = disabled) ====
    expansion_energy_tol = -1,          # |ΔE| < tol → stop
    dets_conv_ratio      = -1,          # N_new/N_old < ratio → space saturated

    # ==== per-round corrections ====
    # ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    # ⭐⭐ Main feature toggles
    orbital_optimization = False,        # enable per-round orbital rotation
    pt2_correction       = False,        # enable per-round PT2 energy correction
    dressed_energy       = False,        # enable dressed CI (auto-enables PT2)
    dressed_sc_max_iter  = 1,           # self-consistent BW iterations (1=single-shot, >1=self-consistent)
    dressed_sc_energy_tol = 1e-4,       # SC convergence: |ΔE_dressed| < tol
    evaluate_only        = False,        # skip expansion, evaluate on existing dets (1 round)
    pool_build_only      = False,        # expand + sort + ABIndex only, skip Davidson/PT2 (for distributed Davidson)
    checkpoint_round_offset = 0,          # added to round counter for checkpoint filenames (e.g. 4 to start at round_4)
    # ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓

    # ==== performance ====
    # ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    # ⭐⭐ Speed vs memory: cache H_ij in Davidson
    use_connection_cache = False,        # Davidson matvec: cache all H_ij (faster, more memory)
    symmetric_sigma      = True,         # True: upper-triangle + per-thread buffers (O(N*T) mem)
                                         # False: full-matrix row-only (2x H_ij, O(N) mem)
    davidson_block_size  = 1,            # >1: block Davidson with shared H_ij (requires symmetric_sigma=False)
                                         # ⚠️  OVERRIDE PITFALL: set via cfg.davidson_block_size (top-level field).
                                         #    Do NOT use cfg.davidson_params.block_size — expansion_loop.cpp
                                         #    overwrites dav_params.block_size from config.davidson_block_size,
                                         #    silently discarding any value set on davidson_params. (exp09e bug)
    # ==== Sparse-update Davidson (exp09d) — 2.6-3.7x speedup @ 1M+ dets ====
    use_sparse_update        = False,    # True: sparse matvec (top-K trial vector), scales O(K*C) vs O(N*C)
    sparse_truncation_ratio  = 0.05,    # K/N fraction to keep (0.02=faster, 0.05=balanced accuracy)
    sparse_min_active        = 1000,    # K lower bound (safety floor for small N)
    sparse_n_warm_iters      = 0,       # full matvec warm-up iters before switching to sparse (0=always sparse)
    sparse_capture_threshold = 0.0,    # auto-switch: sparse when top-K norm capture > threshold (0=disabled)
    # ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓

    # ==== output ====
    verbose              = 2,           # 0=silent, 1=per-phase, 2=per-round
    log_file             = "",          # "" = stdout only

)

ORBOPT_DEFAULTS = dict(
    orbital_opt_max_iter   = 100,       # max BFGS steps per round
    # ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    # ⭐ Orbital optimization convergence threshold
    orbital_opt_grad_tol   = 1e-3,      # convergence: ||grad|| < tol
    orbital_opt_energy_tol = 1e-6,      # convergence: |ΔE| < tol
    # ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
)

# Davidson solver used in each round's diagonalization (and orbopt inner loop)
DAVIDSON_DEFAULTS = dict(
    max_subspace = 60,                  # max Krylov subspace vectors
    max_iter     = 500,                 # max solver iterations
    # ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    # ⭐ Davidson convergence thresholds
    residual_tol = 1e-7,                # residual norm: ||Hx - Ex|| < tol
    energy_tol   = 1e-6,               # energy change: |ΔE| < tol
    # ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    n_states     = 1,                   # number of lowest eigenstates
    verbose      = 0,                   # solver output level
)

# Semistochastic PT2 energy correction (dual-sketch algorithm)
PT2_DEFAULTS = dict(
    # ==== sketch data structure ====
    sketch_width          = 200_000,    # w: buckets per row (memory ∝ w × d)
    sketch_depth          = 5,          # d: rows for median-of-means debiasing
    sketch_seed           = -1,         # hash function seed (-1 = random per run)

    # ==== auxiliary data structures ====
    bloom_fp_rate         = 0.001,      # Bloom filter false positive rate
    hll_precision         = 14,         # HyperLogLog: m=2^p registers (14 → ~0.8% error)

    # ==== numerical ====
    intruder_threshold    = 1e-6,       # skip external dets with |d_a| < this

    # ==== semistochastic partitioning ====
    # Core dets split into two phases by |c_i| ranking:
    #   Phase 1 (exact sum):  top dets, all their excitations enumerated exactly
    #   Phase 2 (sampling):   remaining dets, importance-sampled
    # ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    # ⭐ Accuracy vs speed tradeoff
    norm_target           = 0.99,       # adaptive k: smallest k s.t. ||c_{1:k}||²/||c||² >= target; overrides deterministic_fraction when >0
    # ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    deterministic_fraction = 0.2,       # Phase 1 as fraction of N_core (0.2 captures >99.5% of ||c||²); overridden by norm_target
    n_deterministic       = 0,          # Phase 1 absolute count; only used when fraction == 0
    stochastic_fraction   = 0.02,       # Phase 2 sample size = fraction × N_core
    n_stochastic_samples  = 1000,       # Phase 2 fallback count when stochastic_fraction == 0
    stochastic_seed       = -1,         # Phase 2 RNG seed (-1 = random per run)

    # ==== SHCI-style integral screening (adaptive) ====
    # ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    # ⭐ Adaptive PT2 screening: auto-refines eps until error < target
    screening_error_target = 0.03,      # target: |ΔE(eps_prev)-ΔE(eps)|/|ΔE(eps)| < 3% (eps×0.3 each step). 0 = single-shot, no error estimation
    # ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    eps_hc_filter         = 1e-5,       # starting eps for adaptive refinement (coarsest). 0 = exact (no screening)


    # ==== dressed CI overrides ====
    dressed_sketch_width  = -1,         # -1 = sketch_width (default), >0 = explicit override
    dressed_build_norm_target = 1.0,    # build-phase truncation: 1.0 = off; try 0.999 for large systems to trade ~1mHa for speed
    dressed_query_norm_target = 1.0,    # query-phase truncation: 1.0 = off; try 0.999 for large systems (keeps 45% of 2M dets, ~1.8mHa loss)
    dressed_davidson_energy_tol = 1e-4, # dressed Davidson energy_tol override: 0 = use davidson energy_tol as-is

    # ==== output ====
    verbose               = 1,          # 0=silent, 1=summary, 2=progress
)

# Keys routed to ExpansionConfig (flat attributes, not nested)
_EXPANSION_KEYS = set(EXPANSION_DEFAULTS) | set(ORBOPT_DEFAULTS)


# ============================================================
#  Config builder
# ============================================================

def _build_config(phase_dict, checkpoint_dir, use_128=False):
    """Build an ExpansionConfig from a flat parameter dict.

    Special keys:
        "davidson": dict  → applied to config.davidson_params
        "pt2":      dict  → PT2Config overrides (used when pt2_correction=True)
        use_128: bool     → use ExpansionConfig128 for n_orb > 64
    """
    merged = {**EXPANSION_DEFAULTS, **ORBOPT_DEFAULTS, **phase_dict}
    davidson_kw = {**DAVIDSON_DEFAULTS, **merged.pop("davidson", {})}
    pt2_overrides = merged.pop("pt2", {})
    enable_pt2 = merged.pop("pt2_correction", False)
    enable_dressed = merged.pop("dressed_energy", False)

    # Deprecation guard: legacy `use_gpu` was renamed to `backend` in 2026-04.
    # Check the *user's* phase_dict (not the post-merge), since merged is
    # already populated with EXPANSION_DEFAULTS["backend"]="cpu" — testing
    # `merged.setdefault("backend", ...)` would never override that.
    if "use_gpu" in phase_dict:
        legacy = phase_dict["use_gpu"]
        target = "gpu" if legacy else "cpu"
        # Only override if the user did NOT also pass an explicit `backend`.
        user_backend_explicit = "backend" in phase_dict
        action = (f"keeping user's backend={phase_dict['backend']!r}"
                  if user_backend_explicit else f"converting → backend={target!r}")
        print(f"[run_expansion] WARNING: 'use_gpu' is deprecated. "
              f"Use backend='gpu'/'cpu'/'auto' instead. ({action})")
        if not user_backend_explicit:
            merged["backend"] = target
        merged.pop("use_gpu", None)

    # Resolve backend = "auto" → concrete "cpu"/"gpu" before passing to C++.
    # Threshold 100k tuned for Phase 1/2 Davidson-bound rounds (below that
    # CPU matvec is competitive with GPU launch + setup overhead).
    # Normalize string before dispatching: 'GPU' / ' gpu ' / 'Auto' all map
    # to canonical lowercase, otherwise the C++ string-equality check would
    # silently fall through to CPU.
    requested_backend = str(merged.get("backend", "cpu")).strip().lower()
    merged["backend"] = requested_backend
    if requested_backend == "auto":
        try:
            import trimci.trimci_core as _tc
            gpu_available = _tc.has_gpu()
        except (ImportError, AttributeError):
            gpu_available = False
        max_n_dets = int(merged.get("max_n_dets", 0))
        if gpu_available and max_n_dets >= 100_000:
            merged["backend"] = "gpu"
            print(f"[run_expansion] backend='auto' → 'gpu' "
                  f"(max_n_dets={max_n_dets:,}, GPU available)")
        else:
            merged["backend"] = "cpu"
            print(f"[run_expansion] backend='auto' → 'cpu' "
                  f"(max_n_dets={max_n_dets:,}, gpu_available={gpu_available})")
    elif requested_backend == "gpu":
        # User forced GPU. C++ side will warn + fallback if no CUDA build.
        try:
            import trimci.trimci_core as _tc
            if not _tc.has_gpu():
                print("[run_expansion] WARNING: backend='gpu' requested but "
                      "trimci has no GPU support; will fall back to CPU at "
                      "the C++ dispatcher.")
        except (ImportError, AttributeError):
            pass
    elif requested_backend != "cpu":
        # Defensive: unknown value → fall back to cpu rather than silently
        # going through the wrong path in C++.
        print(f"[run_expansion] WARNING: unknown backend='{requested_backend}', "
              f"treating as 'cpu'.")
        merged["backend"] = "cpu"

    # Route top-level PT2 keys (e.g. eps_hc_filter) into pt2_overrides
    _pt2_keys = set(PT2_DEFAULTS)
    for k in list(merged.keys()):
        if k in _pt2_keys and k not in _EXPANSION_KEYS:
            pt2_overrides.setdefault(k, merged.pop(k))

    # Dressed CI requires PT2 (uses PT2's sketch/screening infrastructure)
    if enable_dressed and not enable_pt2:
        enable_pt2 = True

    ConfigClass = fe.ExpansionConfig128 if use_128 else fe.ExpansionConfig
    config = ConfigClass()
    for k, v in merged.items():
        if k in _EXPANSION_KEYS:
            if hasattr(config, k):
                setattr(config, k, v)

    config.dressed_energy = enable_dressed
    config.checkpoint_dir = checkpoint_dir
    for k, v in davidson_kw.items():
        setattr(config.davidson_params, k, v)

    if enable_pt2:
        pt2_merged = {**PT2_DEFAULTS, **pt2_overrides}
        for seed_key in ("sketch_seed", "stochastic_seed"):
            if pt2_merged[seed_key] < 0:
                pt2_merged[seed_key] = int.from_bytes(os.urandom(8), "little") >> 1
        PT2Class = fe.PT2Config128 if use_128 else fe.PT2Config
        pt2 = PT2Class()
        for k, v in pt2_merged.items():
            setattr(pt2, k, v)
        config.pt2_config = pt2
    else:
        config.pt2_config = None

    return config


# ============================================================
#  Main entry point
# ============================================================

def run_expansion(
    fcidump_path=None,
    dets_path=None,
    phases=None,
    checkpoint_dir="checkpoints",
    output_prefix=None,
    label="",
    *,
    h1=None,
    eri=None,
    e_nuc=0.0,
    alpha_init=None,
    beta_init=None,
    coeffs_init=None,
    measure_s2=False,
):
    """Load data, run multi-phase fast expansion, print summary, save results.

    Integrals can be provided as a FCIDUMP file path OR as raw arrays:
        - fcidump_path: Path to FCIDUMP file.
        - h1, eri, e_nuc: Raw arrays (h1: n_orb×n_orb, eri: n_orb⁴ flat or 4D).

    Initial dets can be provided as a .npz file OR as raw arrays:
        - dets_path: Path to dets.npz (keys: "dets", "dets_coeffs").
        - alpha_init, beta_init, coeffs_init: Raw uint64/float64 arrays.

    Args:
        phases: List of dicts, one per phase. Each dict is flat with
            expansion/orbopt keys at top level, plus optional nested:
            - "davidson": {max_subspace, max_iter, residual_tol, ...}
            - "pt2": {sketch_width, ...} overrides (used when pt2_correction=True)
        checkpoint_dir: Base directory for checkpoints.
        output_prefix: If set, saves {prefix}_results.json and {prefix}_dets.npz.
        label: Banner text (e.g. "Phase B: BS-2").

    Returns:
        dict with keys: energy_var, e_nuc, n_final, total_time_s,
        ndets_history, energy_history, pt2_history, variance_ext_history,
        alphas, betas, coeffs.
    """
    if phases is None:
        raise ValueError("phases must be provided")
    _flush = sys.stdout.flush

    # --- banner ---
    print("=" * 70)
    if label:
        print(f"  {label}")
    for i, ph in enumerate(phases):
        n = ph.get("max_n_dets", EXPANSION_DEFAULTS["max_n_dets"])
        g = ph.get("growth_factor", EXPANSION_DEFAULTS["growth_factor"])
        orbopt = ph.get("orbital_optimization", EXPANSION_DEFAULTS["orbital_optimization"])
        pt2 = ph.get("pt2_correction", EXPANSION_DEFAULTS["pt2_correction"])
        dressed = ph.get("dressed_energy", EXPANSION_DEFAULTS["dressed_energy"])
        eval_only = ph.get("evaluate_only", EXPANSION_DEFAULTS["evaluate_only"])
        flags = f"orbopt={'ON' if orbopt else 'OFF'}, PT2={'ON' if pt2 or dressed else 'OFF'}"
        if dressed:
            flags += ", Dressed=ON"
        pbo = ph.get("pool_build_only", EXPANSION_DEFAULTS["pool_build_only"])
        if pbo:
            flags += ", PoolBuildOnly=ON"
            print(f"  Phase {i+1}: expand → {n:,} dets (no Davidson), {flags}")
        elif eval_only:
            flags += ", EvalOnly=ON"
            print(f"  Phase {i+1}: evaluate {n:,} dets (no expansion), {flags}")
        else:
            print(f"  Phase {i+1}: → {n:,} dets, growth={g}, {flags}")
    if fcidump_path:
        print(f"  FCIDUMP: {fcidump_path}")
    else:
        print(f"  Integrals: raw arrays (n_orb={h1.shape[0]})")
    if dets_path:
        print(f"  Initial dets: {dets_path}")
    elif alpha_init is not None:
        print(f"  Initial dets: raw arrays ({len(alpha_init)} dets)")
    else:
        print(f"  Initial dets: (not provided yet)")
    print(f"  Checkpoint: {checkpoint_dir}")
    print("=" * 70)
    _flush()

    # --- load integrals ---
    if h1 is not None:
        n_orb = h1.shape[0]
        eri_flat = np.asarray(eri, dtype=np.float64)
        if eri_flat.ndim > 1:
            eri_flat = eri_flat.ravel()
        h1 = np.asarray(h1, dtype=np.float64)
        print(f"  Using raw integrals: n_orb={n_orb}, e_nuc={e_nuc:.6f}")
        _flush()
    elif fcidump_path is not None:
        print(f"\nLoading FCIDUMP...")
        _flush()
        h1, eri_4d, _, n_orb, e_nuc, n_alpha, n_beta, _ = read_fcidump(fcidump_path)
        eri_flat = eri_4d.ravel().astype(np.float64)
        print(f"  n_orb={n_orb}, n_alpha={n_alpha}, n_beta={n_beta}, e_nuc={e_nuc:.6f}")
        _flush()
    else:
        raise ValueError("Provide either fcidump_path or h1/eri/e_nuc")

    # --- load initial dets ---
    if alpha_init is not None:
        alpha_init = np.asarray(alpha_init, dtype=np.uint64)
        beta_init = np.asarray(beta_init, dtype=np.uint64)
        coeffs_init = np.asarray(coeffs_init, dtype=np.float64)
        print(f"  Using raw dets: {len(alpha_init)} dets")
        _flush()
    elif dets_path is not None:
        print(f"Loading initial dets...")
        data = np.load(dets_path)
        dets_arr = data["dets"]
        coeffs_init = data["dets_coeffs"]
        n_cols = dets_arr.shape[1]
        if n_cols == 2:
            # 64-bit determinants: [alpha, beta]
            alpha_init = dets_arr[:, 0].astype(np.uint64)
            beta_init = dets_arr[:, 1].astype(np.uint64)
        elif n_cols == 4:
            # 128-bit determinants: [alpha_seg0, alpha_seg1, beta_seg0, beta_seg1]
            alpha_init = dets_arr[:, :2].astype(np.uint64)
            beta_init = dets_arr[:, 2:].astype(np.uint64)
        elif n_cols == 6:
            # 192-bit determinants
            alpha_init = dets_arr[:, :3].astype(np.uint64)
            beta_init = dets_arr[:, 3:].astype(np.uint64)
        else:
            # General: assume first half is alpha, second half is beta
            half = n_cols // 2
            alpha_init = dets_arr[:, :half].astype(np.uint64)
            beta_init = dets_arr[:, half:].astype(np.uint64)
        print(f"  Initial dets: {len(alpha_init)} ({n_cols // 2} segments per spin)")
        _flush()
    else:
        raise ValueError("Provide either dets_path or alpha_init/beta_init/coeffs_init")

    # --- detect 128-bit mode ---
    use_128 = (alpha_init.ndim == 2 and alpha_init.shape[1] == 2)
    if use_128:
        print(f"  128-bit determinant mode (n_orb > 64)")
        _flush()

    # --- build configs ---
    os.makedirs(checkpoint_dir, exist_ok=True)
    configs = []
    for i, ph in enumerate(phases):
        ckpt = os.path.join(checkpoint_dir, f"phase{i+1}") if len(phases) > 1 else checkpoint_dir
        configs.append(_build_config(ph, ckpt, use_128=use_128))

    print(f"\nStarting expansion ({len(phases)} phase{'s' if len(phases)>1 else ''})...")
    _flush()

    # --- run ---
    t0 = time.time()
    if use_128:
        result = fe.run_expansion_phased_128(
            alpha_init, beta_init, coeffs_init,
            h1, eri_flat, n_orb, configs)
    else:
        result = fe.run_expansion_phased(
            alpha_init, beta_init, coeffs_init,
            h1, eri_flat, n_orb, configs)
    t_total = time.time() - t0

    # --- extract results ---
    alphas, betas = result.get_alpha_beta()
    alphas = np.array(alphas, dtype=np.uint64)
    betas = np.array(betas, dtype=np.uint64)
    coeffs = np.array(result.get_coefficients())
    n_final = len(alphas)
    E_var = result.energy_var
    ndets_history = list(result.ndets_history)
    energy_history = list(result.energy_history)
    pt2_history = list(result.pt2_history)
    var_history = list(result.variance_ext_history)
    screen_err_history = list(result.screening_error_history)
    dressed_history = list(result.dressed_energy_history)
    dressed_pt2_history = list(result.dressed_pt2_energy_history)

    # --- optional ⟨S²⟩ measurement ---
    s2_value = None
    s_value = None
    if measure_s2:
        from .s2_probe import measure_s2 as _measure_s2
        t_s2 = time.time()
        s2_value, s_value = _measure_s2(
            alphas=alphas, betas=betas, coeffs=coeffs, use_128=use_128)
        print(f"  ⟨S²⟩ = {s2_value:.6f}   "
              f"S ≈ {s_value:.4f}   "
              f"(S²-eval: {time.time()-t_s2:.2f}s)")
        _flush()

    # --- summary ---
    print(f"\n{'='*70}")
    print(f"  Expansion complete: {n_final:,} dets")
    print(f"  E_var = {E_var:.10f} (electronic)")
    print(f"  E_var = {E_var + e_nuc:.10f} (total)")
    if s2_value is not None:
        print(f"  ⟨S²⟩  = {s2_value:.6f}   (S ≈ {s_value:.4f})")
    print(f"  Total time: {t_total:.1f}s ({t_total/3600:.2f}h)")
    print(f"{'='*70}")

    # --- per-round table ---
    has_pt2 = any(abs(x) > 1e-15 for x in pt2_history)
    has_screen_err = any(abs(x) > 1e-15 for x in screen_err_history)
    has_dressed = any(abs(x) > 1e-15 for x in dressed_history)
    if has_pt2:
        hdr = f"  {'Round':>5} {'N_dets':>12} {'E_var':>18} {'ΔE_PT2':>14} {'E_var+PT2':>18}"
        if has_dressed:
            hdr += f" {'E_dressed':>18}"
        if has_screen_err:
            hdr += f" {'screen_err':>12}"
        print(f"\n{hdr}")
        sep = f"  {'-'*5} {'-'*12} {'-'*18} {'-'*14} {'-'*18}"
        if has_dressed:
            sep += f" {'-'*18}"
        if has_screen_err:
            sep += f" {'-'*12}"
        print(sep)
        for i in range(len(ndets_history)):
            pt2 = pt2_history[i] if i < len(pt2_history) else 0.0
            line = (f"  {i:>5} {ndets_history[i]:>12,} {energy_history[i]:>18.10f}"
                    f" {pt2:>14.10f} {energy_history[i]+pt2:>18.10f}")
            if has_dressed and i < len(dressed_history):
                line += f" {dressed_history[i]:>18.10f}"
            if has_screen_err and i < len(screen_err_history):
                line += f" {screen_err_history[i]:>12.2e}"
            print(line)
    else:
        print(f"\n  {'Round':>5} {'N_dets':>12} {'E_var':>18} {'dE (mHa)':>12}")
        print(f"  {'-'*5} {'-'*12} {'-'*18} {'-'*12}")
        for i in range(len(ndets_history)):
            de = (energy_history[i] - energy_history[i-1]) * 1000 if i > 0 else 0
            print(f"  {i:>5} {ndets_history[i]:>12,} {energy_history[i]:>18.10f} {de:>+12.4f}")
    _flush()

    # --- save ---
    if output_prefix:
        results_dict = {
            "n_final": n_final,
            "e_var_electronic": float(E_var),
            "e_var_total": float(E_var + e_nuc),
            "e_nuc": float(e_nuc),
            "total_time_s": t_total,
            "ndets_history": [int(x) for x in ndets_history],
            "energy_history": [float(x) for x in energy_history],
            "pt2_history": [float(x) for x in pt2_history],
            "variance_ext_history": [float(x) for x in var_history],
            "screening_error_history": [float(x) for x in screen_err_history],
            "dressed_energy_history": [float(x) for x in dressed_history],
            "dressed_pt2_energy_history": [float(x) for x in dressed_pt2_history],
            "s_squared": None if s2_value is None else float(s2_value),
            "spin":      None if s_value  is None else float(s_value),
        }
        json_path = f"{output_prefix}_results.json"
        with open(json_path, "w") as f:
            json.dump(results_dict, f, indent=2)

        npz_path = f"{output_prefix}_dets.npz"
        np.savez(npz_path,
                 dets=np.column_stack([alphas, betas]),
                 dets_coeffs=coeffs)

        print(f"\n  Saved: {json_path}")
        print(f"  Saved: {npz_path}")
        _flush()

    return {
        "energy_var": E_var,
        "e_nuc": e_nuc,
        "n_final": n_final,
        "total_time_s": t_total,
        "ndets_history": ndets_history,
        "energy_history": energy_history,
        "pt2_history": pt2_history,
        "variance_ext_history": var_history,
        "screening_error_history": screen_err_history,
        "dressed_energy_history": dressed_history,
        "dressed_pt2_energy_history": dressed_pt2_history,
        "alphas": alphas,
        "betas": betas,
        "coeffs": coeffs,
        "s_squared": s2_value,
        "spin": s_value,
    }
