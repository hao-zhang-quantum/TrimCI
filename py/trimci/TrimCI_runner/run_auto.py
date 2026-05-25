"""
Auto-run interface: automatically selects reasonable parameters
and optionally explores nearby configurations.
"""
import math
import os
import time
from datetime import datetime
from math import comb
from pathlib import Path
from types import SimpleNamespace as Namespace

import numpy as np

from trimci import (
    extract_mol_name,
    screening,
    setup_molecule,
    get_functions_for_system,
)
from trimci.trimci_logging import setup_logging, log_important, log_verbose

from .io_utils import read_fcidump
from .config import load_configurations
from .det_utils import dets_to_array
from .iterative_workflow import iterative_workflow
from .report_generator import generate_multi_run_report


def run_auto(fcidump_path: str = None,
             molecule: str = None, basis: str = "sto-3g", spin: int = 0,
             goal: str = "balanced",
             ndets: int = None,
             ndets_explore: int = None,
             nexploration: int = None,
             trimci_config_path: str = None, config_dict: dict = None, **overrides):
    """
    High-level auto-run interface.

    Automatically selects reasonable parameters based on the problem size
    (e.g., number of orbitals/electrons), and optionally explores nearby
    configurations to pick the best result.

    Args:
        fcidump_path: Path to FCIDUMP file (mutually exclusive with molecule)
        molecule: Molecule specification string (mutually exclusive with fcidump_path)
        basis: Basis for molecule setup when using molecule mode
        spin: Spin multiplicity offset (2S) for molecule mode
        goal: Tuning goal: one of {"balanced", "speed", "accuracy"}
        ndets: Max final determinants budget (final target budget)
        ndets_explore: Determinant budget for exploration phase
        nexploration: Number of exploration variants
        trimci_config_path: Optional config file path to seed defaults
        config_dict: Optional explicit config dict to seed/override defaults
        overrides: Keyword overrides that take final precedence

    Returns:
        Tuple (final_energy, current_dets, current_coeffs, iteration_details, run_args)
    """
    if fcidump_path is None and molecule is None:
        raise ValueError("Either fcidump_path or molecule must be provided")
    if fcidump_path and molecule:
        raise ValueError("fcidump_path and molecule are mutually exclusive")

    # --- Prepare system data ---
    if fcidump_path:
        h1, eri, n_elec, n_orb, nuclear_repulsion, n_alpha, n_beta, _ = read_fcidump(fcidump_path)

        force_spinless = overrides.get('force_spinless', False) or (config_dict and config_dict.get('force_spinless', False))
        if force_spinless:
            log_important(f"Force spinless mode enabled: Setting n_beta=0, n_alpha={n_elec}")
            n_alpha = n_elec
            n_beta = 0

        config_dir = str(Path(fcidump_path).parent)
        args = load_configurations(config_dir, trimci_config_path, save_if_not_exist=False)
        mol_name = f"FCIDUMP_{n_elec}e_{n_orb}o"
    else:
        mol, mf, h1, eri = setup_molecule(molecule, basis, spin=spin)
        n_elec = mol.nelectron
        spin = mol.spin
        n_orb = len(h1)
        n_alpha = (n_elec + spin) // 2
        n_beta = (n_elec - spin) // 2
        nuclear_repulsion = mol.energy_nuc()
        mol_name = extract_mol_name(molecule)
        args = load_configurations(".", trimci_config_path, save_if_not_exist=False)

    if config_dict:
        for k, v in config_dict.items():
            setattr(args, k, v)

    setup_logging(getattr(args, 'verbose', False))

    # Write run_auto start block
    try:
        _ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _ndets_explore_hint = ndets_explore if ndets_explore is not None else "auto"
        with open("realtime_progress.out", "a", encoding="utf-8") as f:
            f.write(f"\n==== [ {_ts} ] RUN_AUTO START ====\n")
            f.write(f"System: {mol_name} | electrons: {n_elec} | orbitals: {n_orb} | "
                    f"goal: {goal} | ndets: {ndets} | ndets_explore: {_ndets_explore_hint}\n")
    except Exception:
        pass

    def clone(ns):
        return Namespace(**vars(ns))

    total_configs = comb(n_orb, n_alpha) * comb(n_orb, n_beta)
    log10_configs = math.log10(total_configs) if total_configs > 0 else 0.0

    if log10_configs < 8:
        if ndets is None:
            ndets = 50
        cat = "small"
    elif log10_configs < 16:
        if ndets is None:
            ndets = 100
        cat = "medium"
    elif log10_configs < 24:
        if ndets is None:
            ndets = 200
        cat = "large"
    else:
        if ndets is None:
            ndets = 500
        cat = "xlarge"

    def auto_tune(base, n_orb_val, n_alpha_val, n_beta_val, goal_val, ndets_val=None):
        tuned = clone(base)

        if cat == "small":
            if log10_configs < 4:
                tuned.initial_pool_size = 50
                tuned.pool_core_ratio = 4
                tuned.local_trim_keep_ratio = 0.20
                tuned.threshold = 0.06
                tuned.max_final_dets = 20
                tuned.max_rounds = 2
                tuned.num_groups = 4
            else:
                tuned.initial_pool_size = 100
                tuned.pool_core_ratio = 10
                tuned.local_trim_keep_ratio = 0.15
                tuned.threshold = 0.06
                tuned.max_final_dets = 50
                tuned.max_rounds = 2
                tuned.num_groups = 10
        elif cat == "medium":
            tuned.initial_pool_size = 200
            tuned.pool_core_ratio = 20
            tuned.local_trim_keep_ratio = 0.10
            tuned.threshold = 0.06
            tuned.max_final_dets = 100
            tuned.max_rounds = 2
            tuned.num_groups = 12
        elif cat == "large":
            tuned.initial_pool_size = 500
            tuned.pool_core_ratio = 25
            tuned.local_trim_keep_ratio = 0.08
            tuned.threshold = 0.06
            tuned.max_final_dets = 200
            tuned.max_rounds = 2
            tuned.num_groups = 14
        else:
            tuned.initial_pool_size = 1000
            tuned.pool_core_ratio = 30
            tuned.local_trim_keep_ratio = 0.05
            tuned.threshold = 0.06
            tuned.max_final_dets = 400
            tuned.max_rounds = 4
            tuned.num_groups = 16

        tuned.core_set_ratio = [1, 1.2]
        tuned.max_final_dets = int(ndets_val)
        tuned.pool_build_strategy = "hb"
        tuned.verbose = False
        tuned.load_initial_dets = False
        tuned.first_cycle_keep_size = 10

        if goal_val == "speed":
            tuned.local_trim_keep_ratio = tuned.local_trim_keep_ratio * 1.3
            tuned.core_set_ratio = [1.5]
        elif goal_val == "accuracy":
            tuned.local_trim_keep_ratio = tuned.local_trim_keep_ratio * 0.7
            tuned.threshold = max(0.02, tuned.threshold * 0.8)
            tuned.core_set_ratio = [1, 1.05]

        return tuned

    baseline = auto_tune(args, n_orb, n_alpha, n_beta, goal, ndets)

    if ndets_explore is not None:
        exploration_ndets = int(ndets_explore)
    else:
        if cat == "small":
            exploration_ndets = max(10, int(ndets ** 0.5))
        elif cat == "medium":
            exploration_ndets = max(100, int(ndets ** 0.5))
        elif cat == "large":
            exploration_ndets = max(200, int(ndets ** 0.5))
        else:
            exploration_ndets = max(1000, int(ndets ** 0.5))

    for k, v in overrides.items():
        _verbosity = getattr(baseline, 'verbosity', 1 if getattr(baseline, 'verbose', False) else 0)
        if _verbosity >= 1:
            print(k)
        if k == "max_final_dets":
            continue
        elif k == "explore_final_dets":
            exploration_ndets = v
            log_verbose(f"Overriding max_final_dets to {v} for exploration")
            setattr(baseline, "max_final_dets", v)
        else:
            setattr(baseline, k, v)

    baseline_explore = clone(baseline)
    baseline_explore.max_final_dets = exploration_ndets
    baseline_explore.max_final_dets = min(baseline_explore.max_final_dets, total_configs)
    baseline_explore.initial_pool_size = min(baseline_explore.initial_pool_size, total_configs)

    variants = [baseline_explore]

    if nexploration is None:
        if goal == "accuracy":
            nexploration = 50
        elif goal == "speed":
            nexploration = 10
        else:
            nexploration = 20
    if total_configs < 1000:
        nexploration = max(1, nexploration // 10)

    if cat == "small":
        n_random = 50
    elif cat == "medium":
        n_random = 100
    elif cat == "large":
        n_random = 500
    else:
        n_random = 1000

    for _ in range(nexploration):
        var = clone(baseline_explore)
        tweak = {"initial_dets_dict": {
                    "random": [1, n_random],
                    "hf": [1, 1]
                    }}
        for key, value in tweak.items():
            setattr(var, key, value)
        variants.append(var)

    # Run each variant and pick the best
    explore_start_time = time.perf_counter()
    try:
        _ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("realtime_progress.out", "a", encoding="utf-8") as f:
            f.write(f"---- [ {_ts} ] EXPLORE START ----\n")
            f.write(f"exploration_ndets: {exploration_ndets} | variants: {len(variants)}\n")
            f.write(f"total_configs: {int(total_configs):,} | log10_configs: {log10_configs:.2f} | scale: {cat}\n")
    except Exception:
        pass
    log_important(f"Explore: explore_ndets={exploration_ndets}, exploring {len(variants)-1} nexploration")

    best_energy = float('inf')
    best_result = None
    all_results = []

    for idx, run_args in enumerate(variants):
        start_total = time.perf_counter()
        label = f"explore_{idx}"
        log_important(f"Starting {label} ({idx+1}/{len(variants)})")

        final_energy, current_dets, current_coeffs, iteration_details, run_ns = iterative_workflow(
            h1, eri, n_alpha, n_beta, n_orb, mol_name, run_args, nuclear_repulsion, start_total
        )

        results_dir = iteration_details.get('results_dir', '')

        try:
            total_runs = len(variants)
            progress_ratio = (idx + 1) / total_runs
            elapsed_seconds = time.perf_counter() - start_total
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            progress_msg = (
                f"[{timestamp}] Progress: {idx + 1}/{total_runs} ({progress_ratio:.2%}) "
                f"- {label} - energy: {final_energy:.8f} "
                f"- elapsed_s: {elapsed_seconds:.2f} - exploration_ndets: {exploration_ndets}"
            )
            with open("realtime_progress.out", "a", encoding="utf-8") as f:
                f.write(progress_msg + "\n")
        except Exception:
            pass

        run_result = {
            'run_idx': idx + 1,
            'final_energy': final_energy,
            'current_dets': current_dets,
            'current_coeffs': current_coeffs,
            'iteration_details': iteration_details,
            'run_args': run_ns,
            'results_dir': results_dir,
            'label': label,
        }

        all_results.append(run_result)

        log_important(f"{label} completed with energy: {final_energy:.8f}")
        if final_energy < best_energy:
            best_energy = final_energy
            best_result = run_result
            log_important(f"New best energy: {final_energy:.8f} ({label})")

    # Mark explore end
    try:
        _ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _explore_elapsed = time.perf_counter() - explore_start_time
        _best_label = best_result['label'] if best_result else "None"
        _best_energy = f"{best_result['final_energy']:.8f}" if best_result else "N/A"
        with open("realtime_progress.out", "a", encoding="utf-8") as f:
            f.write(f"---- [ {_ts} ] EXPLORE END ----\n")
            f.write(f"best: {_best_label} | energy: {_best_energy} | elapsed_s: {_explore_elapsed:.2f} | completed_runs: {len(variants)}\n")
    except Exception:
        pass

    # Final run at target ndets
    final_result = None
    if ndets is not None and best_result is not None:
        final_args = clone(best_result['run_args'])
        final_args.load_initial_dets = True
        final_args.initial_dets_path = os.path.join(best_result['results_dir'], "dets.npz")
        final_args.max_final_dets = int(ndets)
        log_important(f"Starting final_ndets run with max_final_dets={final_args.max_final_dets}")
        final_start_time = time.perf_counter()
        final_energy, current_dets, current_coeffs, iteration_details, run_ns = iterative_workflow(
            h1, eri, n_alpha, n_beta, n_orb, mol_name, final_args, nuclear_repulsion, final_start_time
        )
        final_result = {
            'run_idx': len(all_results) + 1,
            'final_energy': final_energy,
            'current_dets': current_dets,
            'current_coeffs': current_coeffs,
            'iteration_details': iteration_details,
            'run_args': run_ns,
            'results_dir': iteration_details.get('results_dir', ''),
            'label': 'final_ndets',
        }
        all_results.append(final_result)
        log_important(f"final_ndets completed with energy: {final_energy:.8f}")

        try:
            _ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            _final_elapsed = time.perf_counter() - final_start_time
            with open("realtime_progress.out", "a", encoding="utf-8") as f:
                f.write(f"---- [ {_ts} ] FINAL RUN END ----\n")
                f.write(f"label: {final_result['label']} | energy: {final_result['final_energy']:.8f} | ndets: {final_args.max_final_dets} | elapsed_s: {_final_elapsed:.2f}\n")
        except Exception:
            pass

    # Clean up non-final results directories
    best_dir = best_result['results_dir'] if best_result else ''
    final_dir = final_result['results_dir'] if final_result else ''
    for result in all_results:
        if result['results_dir'] and (result['results_dir'] != final_dir and result['results_dir'] != best_dir):
            try:
                if os.path.exists(result['results_dir']):
                    import shutil
                    shutil.rmtree(result['results_dir'])
                    log_verbose(f"Removed non-final results directory: {result['results_dir']}")
            except Exception as e:
                log_verbose(f"Failed to remove directory {result['results_dir']}: {e}")

    # Return best result
    def _write_auto_end(label, energy, kept_dir):
        try:
            _ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open("realtime_progress.out", "a", encoding="utf-8") as f:
                f.write(f"==== [ {_ts} ] RUN_AUTO END ====\n")
                f.write(f"Summary: {label} | energy: {energy:.8f} | kept_dir: {kept_dir}\n\n")
        except Exception:
            pass

    if final_result:
        generate_multi_run_report(all_results, final_result, mol_name, baseline, final_run=True)
        log_important(f"Final: {final_result['label']} with energy {final_result['final_energy']:.8f}")
        _write_auto_end(f"final={final_result['label']}", final_result['final_energy'], final_result['results_dir'])
        return (final_result['final_energy'], final_result['current_dets'],
                final_result['current_coeffs'], final_result['iteration_details'],
                final_result['run_args'])
    elif best_result:
        generate_multi_run_report(all_results, best_result, mol_name, baseline)
        log_important(f"Best: {best_result['label']} with energy {best_result['final_energy']:.8f}")
        _write_auto_end(f"best={best_result['label']}", best_result['final_energy'], best_result['results_dir'])
        return (best_result['final_energy'], best_result['current_dets'],
                best_result['current_coeffs'], best_result['iteration_details'],
                best_result['run_args'])
    else:
        _baseline_start_time = time.perf_counter()
        _fe, _cd, _cc, _id, _ra = iterative_workflow(
            h1, eri, n_alpha, n_beta, n_orb, mol_name,
            baseline, nuclear_repulsion, _baseline_start_time
        )
        _write_auto_end("fallback=baseline", _fe, _id.get('results_dir', ''))
        return (_fe, _cd, _cc, _id, _ra)
