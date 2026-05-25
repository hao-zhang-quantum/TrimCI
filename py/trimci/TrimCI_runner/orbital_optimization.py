"""
Orbital optimization mode: iteratively runs energy calculation and orbital optimization.
"""
import copy
import os
from datetime import datetime
from pathlib import Path

import numpy as np

from trimci import get_functions_for_system
from trimci.trimci_logging import log_important, log_verbose

from .io_utils import generate_unique_timestamp
from .det_utils import dets_to_array, load_initial_dets_from_file


def _expand_schedule(schedule):
    """Expand schedule with repeat syntax: [64, "32*5", 16] → [64, 32, 32, 32, 32, 32, 16]"""
    if schedule is None:
        return None
    result = []
    for item in schedule:
        if isinstance(item, str) and '*' in item:
            val, count = item.split('*')
            result.extend([int(val)] * int(count))
        else:
            result.append(int(item))
    return result


def _save_fcidump(orblab, tools, path, h1, eri, psym, n_orb, n_elec, nuclear_repulsion, ms2):
    """Symmetrize integrals and write FCIDUMP. Honors psym across the round-trip
    (pyscf's writer drops PSYM info, breaking k-space / complex-orbital SCI)."""
    sym_h1 = orblab.symmetrize_h1(h1)
    sym_eri = orblab.symmetrize_eri(eri, psym)
    orblab.write_fcidump_with_psym(path, sym_h1, sym_eri, n_orb, n_elec, nuclear_repulsion, ms2, psym=psym)


def run_orbital_optimization(h1, eri, n_alpha, n_beta, n_orb, mol_name, args, nuclear_repulsion):
    """
    Run orbital optimization mode — Phase 0 of the TrimCI workflow.

    Iteratively alternates between CI (run_trimci_main_calculation) and
    BFGS orbital rotation. At each cycle the best-energy run's final
    orbitals are rotated and re-fed to the next cycle's CI.

    ``args.optimizer_options_dict`` keys (defaults in parentheses):

        optimizer           ("L-BFGS-B")   "L-BFGS-B" | "bfgs"
        cycles              (5)            number of alternating BFGS↔CI passes
        maxiter             (100)          BFGS steps per pass
        ftol                (1e-8)         BFGS function-value tolerance
        davidson_tol        (1e-7)         inner Davidson residual_tol
        tracking_dets       (False)        reuse same det list across cycles
        select_best_cycle   (True)         return the lowest-energy cycle's orbitals
        record_optimization (False)        save per-step BFGS log (large)
        dets_schedule       (None)         override max_final_dets per cycle
                                           e.g. [64, "32*5", 16]
        num_runs_schedule   (None)         override num_runs per cycle
        load_initial_dets_as_initial_wf (False)  seed BFGS from loaded dets
        initial_wf_dets_num (None)         truncate initial WF to first N dets

    Top-level ``args`` fields are inherited from
    TrimCI_runner/config.py DEFAULT_CONFIG (threshold, max_final_dets,
    num_runs, num_groups, pool_core_ratio, etc.).
    """
    from .. import orblab
    from pyscf import tools

    # Lazy import to avoid circular dependency
    from .trimci_driver import run_trimci_main_calculation

    args = copy.deepcopy(args)

    unique_ts = generate_unique_timestamp()
    orbopt_dir = str(Path("trimci_orbopt_results") / f"orbopt_{mol_name}_{unique_ts}")
    os.makedirs(orbopt_dir, exist_ok=True)

    orbopt_log_path = os.path.join(orbopt_dir, "orbopt_summary.log")

    def log_orbopt(msg, newline=False):
        log_important(msg)
        with open(orbopt_log_path, "a") as f:
            if newline:
                f.write(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {msg}\n")
            else:
                f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {msg}\n")

    psym = getattr(args, 'psym', 8)
    n_elec = n_alpha + n_beta
    ms2 = abs(n_alpha - n_beta)

    h1_original = h1.copy()
    eri_original = eri.copy()

    # Construct optimizer_options_dict
    optimizer_options_dict = getattr(args, 'optimizer_options_dict', {}).copy()
    if optimizer_options_dict.get('optimizer', None) is None:
        optimizer_options_dict['optimizer'] = 'L-BFGS-B'
    if optimizer_options_dict.get('cycles', None) is None:
        optimizer_options_dict['cycles'] = 5
    if optimizer_options_dict.get('record_optimization', None) is None:
        optimizer_options_dict['record_optimization'] = False

    num_cycles = optimizer_options_dict['cycles']
    dets_schedule = _expand_schedule(optimizer_options_dict.get('dets_schedule', None))
    num_runs_schedule = _expand_schedule(optimizer_options_dict.get('num_runs_schedule', None))

    optimizer = orblab.OrbitalOptimizer(
        n_orb, n_elec=n_elec, mol_name=mol_name,
        verbose=getattr(args, 'verbose', False)
    )
    optimizer.nuclear_repulsion = nuclear_repulsion

    current_h1 = h1.copy()
    current_eri = eri.copy()

    log_orbopt(f"Starting Orbital Optimization Run")
    for k, v in optimizer_options_dict.items():
        log_orbopt(f"{k}: {v}")
    log_orbopt(f"results will be saved in {orbopt_dir}")

    U_total = np.eye(n_orb)

    # Initial Wavefunction Generation
    log_orbopt("Initial Wavefunction Generation")

    # Schedules override num_cycles: use shortest schedule length - 1
    schedule_num_cycles = num_cycles
    if dets_schedule is not None:
        args.max_final_dets = dets_schedule[0]
        log_orbopt(f"Initial max_final_dets: {args.max_final_dets}")
        schedule_num_cycles = min(schedule_num_cycles, len(dets_schedule) - 1)
    if num_runs_schedule is not None:
        args.num_runs = num_runs_schedule[0]
        log_orbopt(f"Initial num_runs: {args.num_runs}")
        schedule_num_cycles = min(schedule_num_cycles, len(num_runs_schedule) - 1)
    if schedule_num_cycles != num_cycles:
        num_cycles = schedule_num_cycles
        log_orbopt(f"Schedule overrides num_cycles to {num_cycles}")
    args_initial = copy.deepcopy(args)

    load_initial_dets_as_initial_wf = optimizer_options_dict.get('load_initial_dets_as_initial_wf', False)

    if load_initial_dets_as_initial_wf:
        initial_dets_path = getattr(args, 'initial_dets_path', None)
        if initial_dets_path is None:
            raise ValueError("load_initial_dets_as_initial_wf=True but initial_dets_path is not set")

        log_orbopt(f"Loading initial wavefunction directly from: {initial_dets_path}")

        dets, loaded_coeffs = load_initial_dets_from_file(initial_dets_path, core_set=False)

        if dets is None:
            raise ValueError(f"Failed to load determinants from {initial_dets_path}")

        loaded_coeffs = np.array(loaded_coeffs) if not isinstance(loaded_coeffs, np.ndarray) else loaded_coeffs

        initial_wf_dets_num = optimizer_options_dict.get('initial_wf_dets_num', None)
        if initial_wf_dets_num is not None and initial_wf_dets_num < len(dets):
            original_count = len(dets)
            sorted_idx = np.argsort(np.abs(loaded_coeffs))[::-1][:initial_wf_dets_num]
            dets = [dets[i] for i in sorted_idx]
            loaded_coeffs = loaded_coeffs[sorted_idx]
            log_orbopt(f"Truncated initial WF to top-{initial_wf_dets_num} determinants (from {original_count})")

        coeffs = loaded_coeffs.tolist() if hasattr(loaded_coeffs, 'tolist') else list(loaded_coeffs)

        norm = np.linalg.norm(coeffs)
        if norm > 1e-12:
            coeffs = [c / norm for c in coeffs]

        log_orbopt(f"Computing initial energy with {len(dets)} determinants...")
        funcs = get_functions_for_system(n_orb)
        run_trim_func = funcs['run_trim']

        eri_flat = current_eri.reshape(-1) if current_eri.ndim == 4 else current_eri

        fe, dets, coeffs = run_trim_func(
            dets, current_h1, eri_flat, mol_name, n_elec, n_orb,
            [1], [len(dets)],
            False, False, [], 1e-3
        )

        fe = fe + nuclear_repulsion

        details = {
            'loaded_from': initial_dets_path,
            'n_dets': len(dets),
            'skipped_initial_trimci': True
        }
        run_args = args

        log_orbopt(f"Loaded {len(dets)} determinants, skipped initial TrimCI")
    else:
        fe, dets, coeffs, details, run_args = run_trimci_main_calculation(
            current_h1, current_eri, n_alpha, n_beta, n_orb, mol_name, args_initial, nuclear_repulsion, folder=orbopt_dir
        )

    log_orbopt(f"Initial Energy: {fe:.8f}")

    # Track dets path for tracking_dets feature
    tracking_dets_enabled = optimizer_options_dict.get('tracking_dets', False)
    last_dets_path = None
    if tracking_dets_enabled and not load_initial_dets_as_initial_wf:
        results_dir = details.get('results_dir', '')
        if results_dir:
            last_dets_path = os.path.join(results_dir, "dets.npz")
            log_orbopt(f"tracking_dets enabled. Initial dets saved at: {last_dets_path}")
    elif tracking_dets_enabled and load_initial_dets_as_initial_wf:
        initial_dets_save_path = os.path.join(orbopt_dir, "initial_dets.npz")
        np.savez_compressed(initial_dets_save_path, dets=dets_to_array(dets), dets_coeffs=np.array(coeffs))
        last_dets_path = initial_dets_save_path
        log_orbopt(f"tracking_dets enabled. Saved initial dets to: {last_dets_path}")

    opt_energy_list = []
    opt_energy_after_list = []

    # Best-energy tracking (enabled by default via optimizer_options_dict)
    select_best_cycle = optimizer_options_dict.get('select_best_cycle', True)

    # Candidates: initial fe is the first candidate
    best_energy = fe
    best_dets = list(dets)
    best_coeffs = list(coeffs)
    best_details = details
    best_run_args = run_args
    best_h1 = current_h1.copy()
    best_eri = current_eri.copy()
    best_U_total = U_total.copy()
    best_source = "initial"
    best_cycle_idx = 0             # 0 = before any orbopt cycle
    log_orbopt(f"  [best-tracker] Initial candidate: E={best_energy:.8f} (initial CI)")

    for cycle in range(num_cycles):
        log_orbopt(f"Orbital Optimization Cycle {cycle+1}/{num_cycles}", newline=True)

        new_h1, new_eri, opt_energy, converged, U_step = optimizer.optimize(
            current_h1, current_eri, dets, coeffs, optimizer_options_dict=optimizer_options_dict, callback_func=log_orbopt
        )

        opt_energy_list.append(opt_energy)
        U_total = U_total @ U_step

        log_orbopt(f"  Optimization finished. Energy: {opt_energy:.8f}")

        # Check if BFGS-optimized energy is the new best
        # The dets here are from the previous cycle, evaluated on the new orbitals
        if select_best_cycle and opt_energy < best_energy:
            best_energy = opt_energy
            best_dets = list(dets)
            best_coeffs = list(coeffs)
            best_h1 = new_h1.copy()
            best_eri = new_eri.copy()
            best_U_total = U_total.copy()
            best_details = {
                'source': 'bfgs_optimization',
                'cycle': cycle + 1,
                'opt_energy': opt_energy,
                'converged': converged,
            }
            best_run_args = run_args
            best_source = f"cycle_{cycle+1}_opt (BFGS)"
            best_cycle_idx = cycle + 1
            log_orbopt(f"  [best-tracker] New best: E={best_energy:.8f} from {best_source}")

        current_h1 = new_h1
        current_eri = new_eri

        # Save intermediate FCIDUMP
        fcidump_path = os.path.join(orbopt_dir, f"fcidump_cycle_{cycle+1}")
        _save_fcidump(orblab, tools, fcidump_path, current_h1, current_eri, psym, n_orb, n_elec, nuclear_repulsion, ms2)
        log_orbopt(f"  Saved intermediate FCIDUMP to {fcidump_path}")

        # Re-run CI with new integrals
        log_orbopt("  Re-running CI with new orbitals...")

        if dets_schedule is not None:
            args.max_final_dets = dets_schedule[1 + cycle]
            log_orbopt(f"  Setting max_final_dets to {args.max_final_dets}")

        if num_runs_schedule is not None:
            args.num_runs = num_runs_schedule[1 + cycle]
            log_orbopt(f"  Setting num_runs to {args.num_runs}")

        if tracking_dets_enabled and last_dets_path and os.path.exists(last_dets_path):
            args.load_initial_dets = True
            args.initial_dets_path = last_dets_path
            log_orbopt(f"  Using tracked dets from: {last_dets_path}")

        fe, dets, coeffs, details, run_args = run_trimci_main_calculation(
            current_h1, current_eri, n_alpha, n_beta, n_orb, mol_name, args, nuclear_repulsion, folder=orbopt_dir
        )

        if tracking_dets_enabled:
            results_dir = details.get('results_dir', '')
            if results_dir:
                last_dets_path = os.path.join(results_dir, "dets.npz")
                log_orbopt(f"  Updated tracked dets path to: {last_dets_path}")
        opt_energy_after_list.append(fe)
        log_orbopt(f"  Cycle {cycle+1} Final Energy: {fe:.8f}")

        # Check if CI re-run energy is the new best
        if select_best_cycle and fe < best_energy:
            best_energy = fe
            best_dets = list(dets)
            best_coeffs = list(coeffs)
            best_details = details
            best_run_args = run_args
            best_h1 = current_h1.copy()
            best_eri = current_eri.copy()
            best_U_total = U_total.copy()
            best_source = f"cycle_{cycle+1}_ci (CI re-run)"
            best_cycle_idx = cycle + 1
            log_orbopt(f"  [best-tracker] New best: E={best_energy:.8f} from {best_source}")

        # Save rotation matrix and other data
        npz_path = os.path.join(orbopt_dir, "orbital_optimization_result_cycle_{}.npz".format(cycle+1))
        details_serializable = {k: (v if isinstance(v, (int, float, str, bool, list)) else str(v))
                                for k, v in details.items()} if isinstance(details, dict) else str(details)
        run_args_serializable = {k: (v if isinstance(v, (int, float, str, bool, list)) else str(v))
                                 for k, v in vars(run_args).items()} if hasattr(run_args, '__dict__') else str(run_args)
        np.savez_compressed(npz_path,
                U=U_total,
                h1=current_h1,
                eri=current_eri,
                h1_original=h1_original,
                eri_original=eri_original,
                nuclear_repulsion=nuclear_repulsion,
                n_alpha=n_alpha,
                n_beta=n_beta,
                n_orb=n_orb,
                energy=fe,
                opt_energy_list=opt_energy_list,
                opt_energy_after_list=opt_energy_after_list,
                dets=dets,
                coeffs=coeffs,
                details=details_serializable,
                run_args=run_args_serializable)
        log_orbopt(f"  Saved rotation matrix and system data to {npz_path}")

    # Select which result to return and save
    if select_best_cycle:
        log_orbopt(f"\n[best-tracker] Final selection: E={best_energy:.8f} from {best_source}")
        if num_cycles > 0 and best_source != f"cycle_{num_cycles}_ci (CI re-run)":
            log_orbopt(f"  (Last cycle energy was {fe:.8f}, best was {best_energy - fe:.6f} Ha lower)")
        final_h1 = best_h1
        final_eri = best_eri
        final_fe = best_energy
        final_dets = best_dets
        final_U_total = best_U_total
        final_cycle_idx = best_cycle_idx
        final_coeffs = best_coeffs
        final_details = best_details
        final_run_args = best_run_args
    else:
        final_h1 = current_h1
        final_eri = current_eri
        final_fe = fe
        final_dets = dets
        final_coeffs = coeffs
        final_details = details
        final_run_args = run_args
        final_U_total = U_total
        final_cycle_idx = num_cycles

    # Save final orbitals (from the best cycle, not necessarily the last)
    final_fcidump_path = os.path.join(orbopt_dir, "fcidump_orbopt")
    _save_fcidump(orblab, tools, final_fcidump_path, final_h1, final_eri, psym, n_orb, n_elec, nuclear_repulsion, ms2)
    log_orbopt(f"Saved final optimized orbitals to {final_fcidump_path}")

    # Save the selected state's dets + rotation matrix, matching the
    # per-cycle npz schema so downstream readers (e.g. extract_p_trajectory)
    # can treat it as "cycle N" with the convention that N = chosen cycle.
    orbopt_npz = os.path.join(orbopt_dir, "orbital_optimization_result_orbopt.npz")
    details_serializable = {k: (v if isinstance(v, (int, float, str, bool, list)) else str(v))
                            for k, v in final_details.items()} if isinstance(final_details, dict) else str(final_details)
    run_args_serializable = {k: (v if isinstance(v, (int, float, str, bool, list)) else str(v))
                             for k, v in vars(final_run_args).items()} if hasattr(final_run_args, '__dict__') else str(final_run_args)
    np.savez_compressed(
        orbopt_npz,
        U=final_U_total,
        h1=final_h1,
        eri=final_eri,
        h1_original=h1_original,
        eri_original=eri_original,
        nuclear_repulsion=nuclear_repulsion,
        n_alpha=n_alpha,
        n_beta=n_beta,
        n_orb=n_orb,
        energy=final_fe,
        opt_energy_list=opt_energy_list,
        opt_energy_after_list=opt_energy_after_list,
        dets=final_dets,
        coeffs=final_coeffs,
        source=best_source if select_best_cycle else f"cycle_{num_cycles}_last",
        selected_cycle=final_cycle_idx,
        fcidump_orbopt_path=final_fcidump_path,
        orbopt_dir=orbopt_dir,
        details=details_serializable,
        run_args=run_args_serializable,
    )
    log_orbopt(f"Saved selected orbopt state to {orbopt_npz}")

    # Expose orbopt paths so downstream drivers can locate fcidump_orbopt
    # without filesystem guessing. `final_details` comes from the best sub-run
    # and only carries per-run `results_dir`; inject the orbopt-level paths.
    final_details = dict(final_details) if isinstance(final_details, dict) else {}
    final_details["orbopt_dir"] = orbopt_dir
    final_details["fcidump_orbopt_path"] = final_fcidump_path

    return final_fe, final_dets, final_coeffs, final_details, final_run_args
