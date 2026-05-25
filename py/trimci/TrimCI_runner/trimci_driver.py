"""
TrimCI driver: main entry points and multi-run orchestration.
"""
import logging
import os
import time
from datetime import datetime
from math import comb
from pathlib import Path

import numpy as np

from trimci import (
    extract_mol_name,
    setup_molecule,
    get_functions_for_system,
)
from trimci.trimci_logging import setup_logging, log_important, log_verbose

from .io_utils import read_fcidump, generate_unique_timestamp
from .config import load_configurations
from .det_utils import (
    dets_to_array, get_top_determinants,
    generate_initial_states, load_initial_dets_from_file,
)
from .iterative_workflow import iterative_workflow
from .orbital_optimization import run_orbital_optimization
from .report_generator import generate_multi_run_report

RED_BOLD = "\033[1;31m"
RESET = "\033[0m"


def run_full(fcidump_path: str = None,
             molecule: str = None, basis: str = "sto-3g", spin: int = 0,
             h1=None, eri=None, n_elec: int = None, ms2: int = 0, e_nuc: float = 0.0,
             psym: int = 1, mol_name: str = None,
             trimci_config_path: str = None, config_dict: dict = None, **overrides):
    """
    MAIN ENTRY POINT for all TrimCI calculations.
    See run_full_calculation() for full documentation.
    """
    return run_full_calculation(fcidump_path=fcidump_path,
                                molecule=molecule, basis=basis, spin=spin,
                                h1=h1, eri=eri, n_elec=n_elec, ms2=ms2, e_nuc=e_nuc,
                                psym=psym, mol_name=mol_name,
                                trimci_config_path=trimci_config_path,
                                config_dict=config_dict, **overrides)


def run_full_calculation(fcidump_path: str = None,
                         molecule: str = None, basis: str = "sto-3g", spin: int = 0,
                         h1=None, eri=None, n_elec: int = None, ms2: int = 0,
                         e_nuc: float = 0.0, psym: int = 1, mol_name: str = None,
                         trimci_config_path: str = None, config_dict: dict = None, **overrides):
    """
    Unified entry point that dispatches to different execution modes.

    Input modes (mutually exclusive):
      - ``fcidump_path``: read integrals from FCIDUMP file
      - ``molecule`` + ``basis`` + ``spin``: build via PySCF
      - ``h1`` + ``eri`` + ``n_elec`` + ``ms2`` + ``e_nuc``: raw integrals
        (no file roundtrip)

    Execution Branches:
      1. ORBITAL OPTIMIZATION MODE (orbital_optimization=True) -> run_orbital_optimization()
      2. PARALLEL MODE (n_parallel > 1) -> run_trimci_main_calculation_parallel()
      3. SEQUENTIAL MULTI-RUN (num_runs > 1) -> run_trimci_main_calculation_single()
      4. SINGLE RUN (default) -> run_trimci_main_calculation_single()

    Returns:
        Tuple: (final_energy, dets, coeffs, details, run_args)
    """
    setup_logging(1)  # Initialize logging format early to avoid default WARNING:root: prefix

    n_modes = sum(x is not None for x in
                  [fcidump_path, molecule, (h1 if h1 is not None else None)])
    if n_modes == 0:
        raise ValueError(
            "Must provide one of: fcidump_path, molecule, or (h1, eri, n_elec)"
        )
    if n_modes > 1:
        raise ValueError(
            "fcidump_path, molecule, and (h1, eri) inputs are mutually exclusive"
        )

    if fcidump_path:
        h1, eri, n_elec, n_orb, nuclear_repulsion, n_alpha, n_beta, psym = read_fcidump(fcidump_path)

        force_spinless = overrides.get('force_spinless', False) or (config_dict and config_dict.get('force_spinless', False))
        if force_spinless:
            log_important(f"Force spinless mode enabled: Setting n_beta=0, n_alpha={n_elec}")
            n_alpha = n_elec
            n_beta = 0

        args = load_configurations(str(Path(fcidump_path).parent), trimci_config_path)
        if config_dict:
            for key, value in config_dict.items():
                setattr(args, key, value)
        for key, value in overrides.items():
            setattr(args, key, value)
        args.psym = psym
        args.fcidump_path = fcidump_path
        args.nuclear_repulsion = nuclear_repulsion
        mol_name = f"FCIDUMP_{n_elec}e_{n_orb}o"
    elif molecule:
        mol, mf, h1, eri = setup_molecule(molecule, basis, spin=spin)
        n_elec = mol.nelectron
        spin = mol.spin
        n_orb = len(h1)
        n_alpha = (n_elec + spin) // 2
        n_beta = (n_elec - spin) // 2

        nuclear_repulsion = mol.energy_nuc()
        mol_name = extract_mol_name(molecule)
        if config_dict:
            args = load_configurations(".", trimci_config_path, save_if_not_exist=False)
            for key, value in config_dict.items():
                setattr(args, key, value)
        else:
            args = load_configurations(".", trimci_config_path)

        for key, value in overrides.items():
            setattr(args, key, value)
        args.molecule_spec = molecule
        args.nuclear_repulsion = nuclear_repulsion
    else:
        # Direct-integrals path — no FCIDUMP roundtrip, no PySCF dependency.
        if eri is None or n_elec is None:
            raise ValueError(
                "Direct-integrals mode requires h1, eri, and n_elec."
            )
        import numpy as _np
        h1 = _np.asarray(h1, dtype=_np.float64)
        eri = _np.asarray(eri, dtype=_np.float64)
        n_elec = int(n_elec)
        n_orb = int(h1.shape[0])
        n_alpha = (n_elec + int(ms2)) // 2
        n_beta = (n_elec - int(ms2)) // 2
        if n_alpha + n_beta != n_elec:
            raise ValueError(
                f"n_elec={n_elec} and ms2={ms2} inconsistent: "
                f"n_alpha={n_alpha}, n_beta={n_beta}"
            )
        nuclear_repulsion = float(e_nuc)
        if mol_name is None:
            mol_name = f"integrals_{n_elec}e_{n_orb}o"
        args = load_configurations(".", trimci_config_path, save_if_not_exist=False)
        if config_dict:
            for key, value in config_dict.items():
                setattr(args, key, value)
        for key, value in overrides.items():
            setattr(args, key, value)
        args.psym = int(psym)
        args.nuclear_repulsion = nuclear_repulsion

    setup_logging(args.verbose)

    try:
        _ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _num_runs = getattr(args, 'num_runs', 1)
        _max_dets = getattr(args, 'max_final_dets', 'N/A')
        with open("realtime_progress.out", "a", encoding="utf-8") as f:
            f.write(f"\n==== [ {_ts} ] RUN_FULL START ====\n")
            f.write(f"System: {mol_name} | electrons: {n_elec} | orbitals: {n_orb} | "
                    f"num_runs: {_num_runs} | max_final_dets: {_max_dets}\n")
    except Exception:
        pass

    verbosity = getattr(args, 'verbosity', 1 if getattr(args, 'verbose', False) else 0)
    if verbosity >= 1:
        print(args)
    n_total = int(comb(n_orb, n_alpha) * comb(n_orb, n_beta))

    # Auto-determine max_final_dets
    max_final_dets = getattr(args, 'max_final_dets', None)
    if max_final_dets == "auto" or max_final_dets == -1:
        auto_dets = int(3000 / (n_orb ** 1.5))
        auto_dets = max(50, min(500, auto_dets))
        auto_dets = min(auto_dets, n_total)
        log_important(f"max_final_dets='auto': n_orb={n_orb} -> max_final_dets={auto_dets}")
        args.max_final_dets = auto_dets
    elif max_final_dets is not None and max_final_dets > n_total:
        log_important(f"max_final_dets ({max_final_dets}) exceeds total ({n_total}). Adjusting to {n_total}.")
        args.max_final_dets = n_total

    measure_s2 = getattr(args, 'measure_s2', False)

    if getattr(args, 'orbital_optimization', False):
        result = run_orbital_optimization(h1, eri, n_alpha, n_beta, n_orb, mol_name, args, nuclear_repulsion)
    else:
        result = run_trimci_main_calculation(h1, eri, n_alpha, n_beta, n_orb, mol_name, args, nuclear_repulsion)

    # Optional ⟨S²⟩ diagnostic on the final wavefunction.
    # Uses the C++ kernel in trimci_core (cpp/trimci_core/common/spin_operator.cpp).
    # The probe is O(N · N_α_singly · N_β_singly), usually negligible vs Davidson.
    if measure_s2:
        import json
        import traceback
        try:
            from .s2_probe import measure_s2 as _measure_s2
            final_dets = result[1]
            final_coeffs = result[2]
            iteration_details = result[3] if isinstance(result[3], dict) else None

            s2, s = _measure_s2(dets=final_dets, coeffs=final_coeffs)
            log_important(f"⟨S²⟩ = {s2:.6f}   (S ≈ {s:.4f})")

            # Stash into iteration_details so it's visible to downstream code
            # and the trimci_results.json on disk.
            if iteration_details is not None:
                iteration_details['s_squared'] = float(s2)
                iteration_details['spin']      = float(s)

                # Patch the JSON that iterative_workflow already wrote. We only
                # add the two top-level fields + the iteration_summary copy;
                # dets.npz is untouched.
                results_dir = iteration_details.get('results_dir', '')
                if results_dir:
                    json_path = os.path.join(results_dir, 'trimci_results.json')
                    if os.path.exists(json_path):
                        with open(json_path, 'r') as f:
                            data = json.load(f)
                        data['s_squared'] = float(s2)
                        data['spin']      = float(s)
                        # iteration_summary reference may already be updated
                        # in-place (same dict object), but be explicit.
                        if isinstance(data.get('iteration_summary'), dict):
                            data['iteration_summary']['s_squared'] = float(s2)
                            data['iteration_summary']['spin']      = float(s)
                        with open(json_path, 'w') as f:
                            json.dump(data, f, indent=2)
                        log_important(f"⟨S²⟩ written to {json_path}")
        except (AttributeError, IndexError, TypeError, RuntimeError, ValueError,
                OSError, json.JSONDecodeError) as e:
            # Narrow the catch so real bugs (import errors, C++ crashes) propagate.
            log_important(f"⟨S²⟩ measurement failed: {e}\n{traceback.format_exc()}")
    return result


def run_trimci_main_calculation(h1, eri, n_alpha, n_beta, n_orb, mol_name, args, nuclear_repulsion, folder=None):
    """Dispatch to single, CPU-parallel, or GPU-parallel execution.

    Worker pool shape:
      args.n_gpu_workers > 0 — GPU-parallel, one worker per GPU
      args.n_cpu_workers > 1 OR args.n_parallel > 1 — CPU-parallel
      otherwise — single-process (may still do sequential multi-run via num_runs)
    """
    n_gpu_workers = int(getattr(args, 'n_gpu_workers', 0))
    n_cpu_workers = int(getattr(args, 'n_cpu_workers',
                                getattr(args, 'n_parallel', 1)))

    if n_gpu_workers > 0 or n_cpu_workers > 1:
        from .parallel_runner import run_trimci_main_calculation_parallel
        return run_trimci_main_calculation_parallel(
            h1, eri, n_alpha, n_beta, n_orb, mol_name, args, nuclear_repulsion,
            folder=folder,
            num_runs=getattr(args, 'num_runs', 50),
            n_parallel=(n_gpu_workers if n_gpu_workers > 0 else n_cpu_workers),
            omp_per_run=getattr(args, 'omp_per_run', None)
        )
    return _run_single(h1, eri, n_alpha, n_beta, n_orb, mol_name, args, nuclear_repulsion, folder)


def _run_single(h1, eri, n_alpha, n_beta, n_orb, mol_name, args, nuclear_repulsion, folder=None):
    """Single or sequential multi-run execution."""
    num_runs = getattr(args, 'num_runs', 1)

    if num_runs > 1:
        return _run_multi(h1, eri, n_alpha, n_beta, n_orb, mol_name, args, nuclear_repulsion, folder, num_runs)

    # Single run
    start_total = time.perf_counter()
    if folder is None:
        single_run_folder = str(Path("trimci_single_run_results"))
    else:
        single_run_folder = str(Path(folder))

    Path(single_run_folder).mkdir(parents=True, exist_ok=True)
    log_file = os.path.join(single_run_folder, "realtime_progress.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.WARNING)
    file_handler.setFormatter(logging.Formatter("%(asctime)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logging.getLogger().addHandler(file_handler)

    from .summary_logger import TrimCISummaryLogger
    args_dict = vars(args).copy() if hasattr(args, '__dict__') else dict(args)
    summary_logger = TrimCISummaryLogger(os.path.join(single_run_folder, "single_summary.log"), mode="single")
    summary_logger.write_header(mol_name, n_orb, n_alpha, n_beta, args_dict)

    _fe, _cd, _cc, _id, _ra = iterative_workflow(h1, eri, n_alpha, n_beta, n_orb, mol_name,
                                  args, nuclear_repulsion, start_total, single_run_folder)

    _elapsed = time.perf_counter() - start_total
    n_dets = len(_cd) if _cd else 0
    n_iters = _id.get('total_iterations', 0)

    summary_logger.write_single_summary(_fe, n_dets, n_iters, _elapsed, _id.get('results_dir', ''))
    summary_logger.write_iteration_table(_id)
    summary_logger.close()

    try:
        _ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("realtime_progress.out", "a", encoding="utf-8") as f:
            f.write(f"==== [ {_ts} ] RUN_FULL END ====\n")
            f.write(f"Summary: single | energy: {_fe:.8f} | kept_dir: {_id.get('results_dir', '')}\n\n")
    except Exception:
        pass

    logging.getLogger().removeHandler(file_handler)
    file_handler.close()
    return (_fe, _cd, _cc, _id, _ra)


def _resolve_backend(args, n_orb):
    """Resolve args.backend ('auto'|'cpu'|'gpu') to a concrete routing choice.

    Heuristic for 'auto':
      - If GPU not available → 'cpu'.
      - If max_final_dets < 1000 → 'cpu' (GPU kernel overhead > work;
        measured ~3.7× slower on Fe4S4 N=36/100 dets, 2026-04).
      - Otherwise → 'gpu'.

    Returns 'cpu' or 'gpu'. Announces the pick so users can see why a
    particular path was taken.
    """
    requested = getattr(args, 'backend', 'auto')
    if requested not in ('auto', 'cpu', 'gpu'):
        log_important(f"Unknown backend '{requested}', treating as 'auto'.")
        requested = 'auto'

    if requested == 'cpu':
        return 'cpu'

    try:
        import trimci.trimci_core as _tc
        gpu_available = _tc.has_gpu()
    except (ImportError, AttributeError):
        gpu_available = False

    if requested == 'gpu':
        if not gpu_available:
            log_important("backend='gpu' requested but this build has no GPU support; falling back to CPU.")
            return 'cpu'
        return 'gpu'

    # 'auto'
    if not gpu_available:
        return 'cpu'
    max_dets = getattr(args, 'max_final_dets', None)
    # "auto" / -1 / None are all small-enough sentinels — driver will pick a
    # small auto_dets via its 3000/n_orb^1.5 heuristic before calling us.
    if not isinstance(max_dets, int) or max_dets < 1000:
        log_important(f"[backend] auto → cpu (max_final_dets={max_dets}, "
                      f"GPU overhead dominates below ~1k dets)")
        return 'cpu'
    log_important(f"[backend] auto → gpu (max_final_dets={max_dets})")
    return 'gpu'


def _run_multi(h1, eri, n_alpha, n_beta, n_orb, mol_name, args, nuclear_repulsion, folder, num_runs):
    """Sequential multi-run execution.

    Routing via ``args.backend`` ∈ {'auto', 'cpu', 'gpu'}. Default 'auto':
    small det spaces (< 1000) go CPU, larger go GPU. GPU path shares a single
    GpuContext across num_runs. Falls back to CPU if GPU unavailable.
    """
    choice = _resolve_backend(args, n_orb)
    if choice == 'gpu':
        try:
            return _run_multi_gpu(h1, eri, n_alpha, n_beta, n_orb,
                                  mol_name, args, nuclear_repulsion,
                                  folder, num_runs)
        except (ImportError, AttributeError) as e:
            log_important(f"GPU multi-run import failed ({e}); falling back to CPU.")

    return _run_multi_cpu(h1, eri, n_alpha, n_beta, n_orb, mol_name,
                          args, nuclear_repulsion, folder, num_runs)


def _run_multi_gpu(h1, eri, n_alpha, n_beta, n_orb, mol_name, args, nuclear_repulsion, folder, num_runs):
    """GPU multi-run: share a single GpuContext across N independent runs.

    Generates num_runs distinct initial determinant sets via generate_initial_states
    (uses args.initial_dets_dict, same mechanism as CPU path), packs them into a
    list-of-lists, and calls the C++ iterative_workflow_gpu_multi_{64,128}. The
    integrals + excitation table are uploaded to the GPU once rather than once
    per run.

    Per-run TRIM partition RNG seed is derived from args.random_seed (or wall-clock
    fallback) by mixing with run index.
    """
    import trimci.trimci_core as trimci_core
    from trimci.trimci_core import IterativeWorkflowParams

    log_important(f"Running {num_runs} independent GPU calculations (shared GpuContext)...")

    h1_np = np.asarray(h1, dtype=np.float64)
    if h1_np.ndim == 1:
        h1_np = h1_np.reshape(n_orb, n_orb)
    if isinstance(eri, np.ndarray) and eri.ndim == 4:
        eri_flat = eri.reshape(-1)
    else:
        eri_flat = np.asarray(eri, dtype=np.float64).ravel()

    # Per-run initial states. Each generate_initial_states call uses a fresh
    # np.random.default_rng(), so N calls produce N independent draws.
    initial_dets_dict = getattr(args, 'initial_dets_dict', None) or {"hf": 1.0}
    initial_dets_list = []
    initial_coeffs_list = []
    for _r in range(num_runs):
        _dets, _coeffs = generate_initial_states(n_alpha, n_beta, n_orb, initial_dets_dict)
        initial_dets_list.append(list(_dets))
        initial_coeffs_list.append([float(c) for c in _coeffs])

    # Populate IterativeWorkflowParams from args, mirroring iterative_workflow.py
    params = IterativeWorkflowParams()
    params.max_iterations = getattr(args, 'max_iterations', 200000)
    params.max_final_dets = getattr(args, 'max_final_dets', -1)
    params.core_set_ratio = list(getattr(args, 'core_set_ratio', [2.0]))
    params.initial_pool_size = getattr(args, 'initial_pool_size', 100)
    params.core_set_schedule = list(getattr(args, 'core_set_schedule', []))
    params.first_cycle_keep_size = getattr(args, 'first_cycle_keep_size', 10)
    params.pool_core_ratio = getattr(args, 'pool_core_ratio', 10)
    params.threshold = getattr(args, 'threshold', 0.01)
    params.threshold_decay = getattr(args, 'threshold_decay', 0.9)
    params.max_rounds = getattr(args, 'max_rounds', 1)
    params.num_groups = getattr(args, 'num_groups', 10)
    params.num_groups_ratio = getattr(args, 'num_groups_ratio', 0.0)
    params.local_trim_keep_ratio = getattr(args, 'local_trim_keep_ratio', 0.0)
    params.keep_ratio = getattr(args, 'keep_ratio', 0.1)
    params.verbosity = getattr(args, 'verbosity', 1)

    master_seed = getattr(args, 'random_seed', 0) or int(time.time())
    params.random_seed = int(master_seed) & 0xFFFFFFFFFFFFFFFF
    seeds = [(int(master_seed) + r * 0x9e3779b1) & 0xFFFFFFFFFFFFFFFF
             for r in range(num_runs)]

    gpu_multi = (trimci_core.iterative_workflow_gpu_multi_64
                 if n_orb <= 64 else
                 trimci_core.iterative_workflow_gpu_multi_128)

    # pybind11 bindings take std::vector<std::vector<double>> for h1 and
    # std::vector<double> for eri; convert explicitly (numpy arrays don't
    # auto-convert to these STL containers in the current bindings).
    h1_input = [list(row) for row in h1_np]
    eri_input = eri_flat.tolist() if hasattr(eri_flat, 'tolist') else list(eri_flat)

    log_important(f"Launching GPU multi-run (n_orb={n_orb}, dets/run={len(initial_dets_list[0])})...")
    t0 = time.perf_counter()
    results = gpu_multi(
        h1_input, eri_input,
        n_alpha, n_beta, n_orb,
        mol_name,
        initial_dets_list, initial_coeffs_list,
        nuclear_repulsion, params,
        num_runs, seeds
    )
    gpu_wall = time.perf_counter() - t0
    log_important(f"GPU multi-run wall time: {gpu_wall:.2f}s "
                  f"({gpu_wall/num_runs:.2f}s/run avg)")

    energies = [float(r.final_energy) for r in results]
    best_idx = int(np.argmin(energies))
    best = results[best_idx]

    log_important(f"Summary of all {num_runs} GPU runs:")
    for i, e in enumerate(energies):
        marker = " [BEST]" if i == best_idx else ""
        log_important(f"  Run {i+1}: Energy = {e:.8f}{marker}")
    log_important(f"Best result from run {best_idx+1} with energy: {best.final_energy:.8f}")

    best_details = {
        'source': 'gpu_multi_run',
        'run_idx': best_idx + 1,
        'total_iterations': int(getattr(best, 'total_iterations', 0)),
        'gpu_wall_time': gpu_wall,
        'num_runs': num_runs,
        'all_energies': energies,
    }
    return (float(best.final_energy), list(best.final_dets),
            list(best.final_coeffs), best_details, args)


def _run_multi_cpu(h1, eri, n_alpha, n_beta, n_orb, mol_name, args, nuclear_repulsion, folder, num_runs):
    """Sequential multi-run execution (CPU backend)."""
    import shutil

    log_important(f"Running {num_runs} independent calculations to find the best result...")

    best_energy = float('inf')
    best_result = None
    all_results = []

    unique_ts = generate_unique_timestamp()
    if folder is None:
        multi_run_folder = str(Path("trimci_multi_run_results") / f"{mol_name}_{unique_ts}")
    else:
        multi_run_folder = str(Path(folder) / f"multi_{mol_name}_{unique_ts}")

    Path(multi_run_folder).mkdir(parents=True, exist_ok=True)
    log_file = os.path.join(multi_run_folder, "realtime_progress.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.WARNING)
    file_handler.setFormatter(logging.Formatter("%(asctime)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logging.getLogger().addHandler(file_handler)

    from .summary_logger import TrimCISummaryLogger
    args_dict = vars(args).copy() if hasattr(args, '__dict__') else dict(args)
    summary_logger = TrimCISummaryLogger(os.path.join(multi_run_folder, "multi_summary.log"), mode="multi")
    summary_logger.write_header(mol_name, n_orb, n_alpha, n_beta, args_dict, num_runs=num_runs)

    multi_start_time = time.perf_counter()

    for run_idx in range(num_runs):
        start_total = time.perf_counter()
        log_important(f"Starting run {run_idx + 1}/{num_runs}")

        final_energy, current_dets, current_coeffs, iteration_details, run_args = iterative_workflow(
            h1, eri, n_alpha, n_beta, n_orb, mol_name, args, nuclear_repulsion, start_total, multi_run_folder
        )

        results_dir = iteration_details.get('results_dir', '')
        run_result = {
            'run_idx': run_idx + 1,
            'final_energy': final_energy,
            'current_dets': current_dets,
            'current_coeffs': current_coeffs,
            'iteration_details': iteration_details,
            'run_args': run_args,
            'results_dir': results_dir
        }
        all_results.append(run_result)

        log_important(f"Run {run_idx + 1} completed with energy: {final_energy:.8f}")

        run_elapsed = time.perf_counter() - start_total
        n_iters = iteration_details.get('total_iterations')
        summary_logger.log_run_complete(run_idx + 1, final_energy, run_elapsed, n_iters)

        if final_energy < best_energy:
            best_energy = final_energy
            best_result = run_result
            log_important(f"New best energy found: {final_energy:.8f}")

        log_important(f"Current best energy: {best_energy:.8f}")

    # Summary of all runs
    log_important(f"Summary of all {num_runs} runs:")
    for result in all_results:
        marker = " [BEST]" if result['run_idx'] == best_result['run_idx'] else ""
        log_important(f"  Run {result['run_idx']}: Energy = {result['final_energy']:.8f}{marker}")

    log_important(f"Best result from run {best_result['run_idx']} with energy: {best_result['final_energy']:.8f}")

    # Clean up non-best results directories
    all_results.sort(key=lambda x: x['final_energy'])

    num_keep = getattr(args, 'num_runs_keep_top_k', 1)
    top_k_results = all_results[:num_keep]
    top_k_indices = set(r['run_idx'] for r in top_k_results)

    log_important(f"Keeping top {num_keep} results (Runs: {sorted(list(top_k_indices))})")

    for result in all_results:
        if result['run_idx'] not in top_k_indices and result['results_dir']:
            try:
                if os.path.exists(result['results_dir']):
                    shutil.rmtree(result['results_dir'])
                    log_verbose(f"Removed non-top-{num_keep} results directory: {result['results_dir']}")
            except Exception as e:
                log_verbose(f"Failed to remove directory {result['results_dir']}: {e}")

    # Combine determinants from top k results
    unique_dets_list, unique_coeffs_list = _combine_top_k_dets(top_k_results, args, h1, eri, n_alpha, n_beta, n_orb, mol_name)

    # Save combined determinants
    _save_combined_results(unique_dets_list, unique_coeffs_list, top_k_results, multi_run_folder, num_keep)

    # Prepare combined results info
    combined_results_info = {
        'num_combined_dets': len(unique_dets_list),
        'renormalized': getattr(args, 'num_runs_keep_top_renormalize', False),
        'file_path': os.path.join(multi_run_folder, "dets_combine.npz"),
        'top_10_determinants': get_top_determinants(unique_dets_list, unique_coeffs_list, top_n=10),
    }

    generate_multi_run_report(all_results, best_result, mol_name, args,
                              folder=multi_run_folder,
                              combined_results_info=combined_results_info)

    try:
        _ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        best_results_dir = best_result.get('results_dir', '')
        with open("realtime_progress.out", "a", encoding="utf-8") as f:
            f.write(f"==== [ {_ts} ] RUN_FULL END ====\n")
            f.write(f"Summary: best=run_{best_result['run_idx']} | energy: {best_result['final_energy']:.8f} | kept_dir: {best_results_dir}\n\n")
    except Exception:
        pass

    total_elapsed = time.perf_counter() - multi_start_time
    for r in all_results:
        r['elapsed'] = r.get('elapsed', 0)
    summary_logger.write_multi_summary(all_results, best_result, total_elapsed)
    summary_logger.close()

    logging.getLogger().removeHandler(file_handler)
    file_handler.close()

    return (best_result['final_energy'], best_result['current_dets'],
            best_result['current_coeffs'], best_result['iteration_details'],
            best_result['run_args'])


def _combine_top_k_dets(top_k_results, args, h1, eri, n_alpha, n_beta, n_orb, mol_name):
    """Combine determinants from top-k runs, optionally renormalizing."""
    num_keep = len(top_k_results)
    log_important(f"Combining determinants from top {num_keep} runs...")

    unique_dets_map = {}
    for result in top_k_results:
        current_dets = result['current_dets']
        current_coeffs = result['current_coeffs']
        for det, coeff in zip(current_dets, current_coeffs):
            if isinstance(det.alpha, list):
                key = (tuple(det.alpha), tuple(det.beta))
            else:
                key = (det.alpha, det.beta)
            if key not in unique_dets_map:
                unique_dets_map[key] = (det, coeff)

    unique_dets_list = [v[0] for v in unique_dets_map.values()]
    unique_coeffs_list = [v[1] for v in unique_dets_map.values()]

    log_important(f"Combined {len(unique_dets_list)} unique determinants.")

    if getattr(args, 'num_runs_keep_top_renormalize', False):
        log_important("Renormalizing combined determinants using trim()...")
        eri_flat = eri.reshape(-1) if hasattr(eri, 'reshape') and eri.ndim == 4 else eri
        try:
            funcs = get_functions_for_system(n_orb)
            run_trim_func = funcs['run_trim']
            try:
                t_energy, unique_dets_list, unique_coeffs_list = run_trim_func(
                    unique_dets_list, h1, eri_flat,
                    mol_name, n_alpha+n_beta, n_orb,
                    [1], [len(unique_dets_list)],
                    False, False, [], 1e-3
                )
                log_important(f"Renormalization complete. Energy: {t_energy:.8f}")
            except Exception as e:
                log_important(f"Renormalization failed: {e}")
                norm = np.linalg.norm(unique_coeffs_list)
                if norm > 1e-12:
                    unique_coeffs_list = [c/norm for c in unique_coeffs_list]
        except Exception as e:
            log_important(f"Renormalization failed: {e}")
            norm = np.linalg.norm(unique_coeffs_list)
            if norm > 1e-12:
                unique_coeffs_list = [c/norm for c in unique_coeffs_list]
    else:
        norm = np.linalg.norm(unique_coeffs_list)
        if norm > 1e-12:
            unique_coeffs_list = [c/norm for c in unique_coeffs_list]

    return unique_dets_list, unique_coeffs_list


def _save_combined_results(unique_dets_list, unique_coeffs_list, top_k_results, multi_run_folder, num_keep):
    """Save combined determinants and per-run data."""
    dets_combine_path = os.path.join(multi_run_folder, "dets_combine.npz")
    try:
        np.savez_compressed(dets_combine_path,
                            dets=dets_to_array(unique_dets_list),
                            dets_coeffs=np.array(unique_coeffs_list))
        log_important(f"Saved combined determinants to {dets_combine_path}")
    except Exception as e:
        log_important(f"Failed to save combined determinants: {e}")

    dets_multi_run_path = os.path.join(multi_run_folder, "dets_multi_run.npz")
    try:
        all_wfs = []
        for result in top_k_results:
            wf_data = {
                'dets': dets_to_array(result['current_dets']),
                'coeffs': np.array(result['current_coeffs']),
                'energy': result['final_energy'],
                'n_dets': len(result['current_dets'])
            }
            all_wfs.append(wf_data)
        np.savez_compressed(dets_multi_run_path,
                            all_wfs=np.array(all_wfs, dtype=object),
                            n_wfs=num_keep,
                            energies=np.array([r['final_energy'] for r in top_k_results]))
        log_important(f"Saved {num_keep} wavefunctions to {dets_multi_run_path}")
    except Exception as e:
        log_important(f"Failed to save multi-run dets: {e}")
