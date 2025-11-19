"""
TrimCI driver script
High-level workflow for running TrimCI from FCIDUMP or molecule.
"""
import json, os, sys
import logging
import re
import time
from datetime import datetime
import math
import numpy as np
from pathlib import Path
from types import SimpleNamespace as Namespace
import numpy as np
from .. import trimci_core
from math import comb

# Use trimci high-level interface
from trimci import (
    extract_mol_name,
    load_or_create_Hij_cache,
    generate_reference_det,
    generate_excitations,
    screening,
    trim,
    setup_molecule,
    get_functions_for_system,
)

# ========== Logging Configuration ==========
LOG_FORMAT = "%(asctime)s: %(message)s"  # Removed levelname and milliseconds
BLUE_COLOR = "\033[94m"
RESET_COLOR = "\033[0m"
RED_BOLD = "\033[1;31m"
RESET = "\033[0m"

def setup_logging(verbose: bool):
    # Clear any existing handlers to avoid conflicts
    logging.getLogger().handlers.clear()
    
    level = logging.INFO if verbose else logging.WARNING  # INFO for verbose, WARNING for non-verbose
    class CleanFormatter(logging.Formatter):
        def format(self, record):
            # Custom format without milliseconds
            record.asctime = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
            formatted = f"{record.asctime}: {record.getMessage()}"
            return f"{BLUE_COLOR}{formatted}{RESET_COLOR}"
    
    logging.basicConfig(level=level, stream=sys.stdout, force=True)
    for handler in logging.root.handlers:
        handler.setFormatter(CleanFormatter())

def log_important(message: str):
    """Log important messages that should always be shown"""
    logging.warning(message)  # Use WARNING level to ensure it's always shown

def log_verbose(message: str):
    """Log verbose messages that are only shown when verbose=True"""
    logging.info(message)  # Use INFO level for verbose-only messages


# ========== FCIDUMP Reading ==========
def read_fcidump(fcidump_path: str):
    log_verbose(f"🔍 Reading FCIDUMP file: {fcidump_path}")
    with open(fcidump_path, 'r') as f:
        lines = f.readlines()

    header_lines = []
    data_start_idx = 0
    for i, line in enumerate(lines):
        header_lines.append(line)
        if "&END" in line or "/" in line:
            data_start_idx = i + 1
            break
    header_text = ''.join(header_lines)

    def extract_int(keyword):
        m = re.search(rf"{keyword}\s*=\s*(-?\d+)", header_text, re.IGNORECASE)
        return int(m.group(1)) if m else None

    n_orb = extract_int("NORB")
    n_elec = extract_int("NELEC")
    ms2 = extract_int("MS2") or 0
    psym = extract_int("PSYM") or 8

    n_alpha = int((n_elec + ms2) // 2)
    n_beta = int((n_elec - ms2) // 2)

    log_important(f"🔍 Detected NORB={n_orb}, NELEC={n_elec}, N_ALPHA={n_alpha}, N_BETA={n_beta}, MS2={ms2}, PSYM={psym}")

    h1 = np.zeros((n_orb, n_orb))
    eri = np.zeros((n_orb, n_orb, n_orb, n_orb))
    nuclear_repulsion = 0.0

    for line in lines[data_start_idx:]:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        val = float(parts[0])
        p_raw, q_raw, r_raw, s_raw = map(int, parts[1:])
        if (p_raw == q_raw == r_raw == s_raw == 0):
            nuclear_repulsion = val
            continue
        p, q, r, s = p_raw - 1, q_raw - 1, r_raw - 1, s_raw - 1
        if r == -1 and s == -1:
            h1[p, q] = val
            h1[q, p] = val
        else:
            if psym == 4:
                eri[p, q, r, s] = val
                eri[q, p, s, r] = val
                eri[r, s, p, q] = val
                eri[s, r, q, p] = val
            else:
                # assuming psym == 8:
                eri[p, q, r, s] = val
                eri[q, p, r, s] = val
                eri[p, q, s, r] = val
                eri[q, p, s, r] = val
                eri[r, s, p, q] = val
                eri[s, r, p, q] = val
                eri[r, s, q, p] = val
                eri[s, r, q, p] = val
    return h1, eri, n_elec, n_orb, nuclear_repulsion, n_alpha, n_beta


def read_fcidump2(fcidump_path: str):
    """
    Read FCIDUMP file using PySCF's built-in reader.
    Returns the same format as read_fcidump: (h1, eri, n_elec, n_orb, nuclear_repulsion)
    """
    log_verbose(f"🔍 Reading FCIDUMP file with PySCF: {fcidump_path}")
    from pyscf.tools import fcidump
    from pyscf import ao2mo
    
    # Use PySCF's fcidump reader - returns a dictionary
    fcidump_data = fcidump.read(fcidump_path)
    
    # Extract data from dictionary
    h1 = fcidump_data['H1']
    eri_compressed = fcidump_data['H2']
    n_orb = fcidump_data['NORB']
    n_elec = fcidump_data['NELEC']
    nuclear_repulsion = fcidump_data['ECORE']
    
    # Convert compressed ERI to 4D tensor format
    # PySCF returns ERI in compressed format, need to restore to (n_orb, n_orb, n_orb, n_orb)
    eri = ao2mo.restore(1, eri_compressed, n_orb)
    
    log_verbose(f"🔍 Detected NORB={n_orb}, NELEC={n_elec}")
    
    return h1, eri, n_elec, n_orb, nuclear_repulsion



# ========== Main Workflow ==========
def run_full(fcidump_path: str = None,
             molecule: str = None, basis: str = "sto-3g", spin: int = 0,
             trimci_config_path: str = None, config_dict: dict = None, **overrides):
    """
    Convenience wrapper for run_full_calculation() that forwards all arguments unchanged.
    """
    return run_full_calculation(fcidump_path=fcidump_path,
                                molecule=molecule,
                                basis=basis,
                                spin=spin,
                                trimci_config_path=trimci_config_path,
                                config_dict=config_dict,
                                **overrides)

def run_full_calculation(fcidump_path: str = None,
                         molecule: str = None, basis: str = "sto-3g", spin: int = 0,
                         trimci_config_path: str = None, config_dict: dict = None, **overrides):
    """
    Run calculation.
    """
    if fcidump_path is None and molecule is None:
        raise ValueError("Either fcidump_path or molecule must be provided")
    if fcidump_path and molecule:
        raise ValueError("fcidump_path and molecule are mutually exclusive")

    if fcidump_path:
        h1, eri, n_elec, n_orb, nuclear_repulsion, n_alpha, n_beta = read_fcidump(fcidump_path)
        args = load_configurations(str(Path(fcidump_path).parent), trimci_config_path)
        # Apply explicit config dictionary if provided (higher precedence than file)
        if config_dict:
            for key, value in config_dict.items():
                setattr(args, key, value)
        # Override parameters
        for key, value in overrides.items():
            setattr(args, key, value)
        # Setup logging based on config
        #n_alpha = n_beta = n_elec // 2
        mol_name = f"FCIDUMP_{n_elec}e_{n_orb}o"
    else:
        mol, mf, h1, eri = setup_molecule(molecule, basis, spin=0)
        n_elec = mol.nelectron
        spin = mol.spin  # difference between alpha and beta electrons (2S)
        n_orb = len(h1)
        n_alpha = (n_elec + spin) // 2
        n_beta = (n_elec - spin) // 2

        nuclear_repulsion = mol.energy_nuc()
        mol_name = extract_mol_name(molecule)
        # Apply explicit config dictionary if provided (higher precedence than file)
        if config_dict:
            args = load_configurations(".", trimci_config_path, save_if_not_exist=False)
            for key, value in config_dict.items():
                setattr(args, key, value)

        else:
            args = load_configurations(".", trimci_config_path)


        for key, value in overrides.items():
            setattr(args, key, value)

    setup_logging(args.verbose)
    # Write run_full start block to realtime_progress.out
    try:
        _ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _num_runs = getattr(args, 'num_runs', 1)
        _max_dets = getattr(args, 'max_final_dets', 'N/A')
        with open("realtime_progress.out", "a", encoding="utf-8") as f:
            f.write(f"\n==== [ {_ts} ] RUN_FULL START ====\n")
            f.write(
                f"System: {mol_name} | electrons: {n_elec} | orbitals: {n_orb} | num_runs: {_num_runs} | max_final_dets: {_max_dets}\n"
            )
    except Exception:
        # Non-fatal: ignore file writing errors
        pass
    print(args)
    n_total = int(comb(n_orb, n_alpha) * comb(n_orb, n_beta))


    # Check if multiple runs are requested
    num_runs = getattr(args, 'num_runs', 1)
    
    if num_runs > 1:
        import shutil
        log_important(f"🔄 Running {num_runs} independent calculations to find the best result...")
        
        best_energy = float('inf')
        best_result = None
        all_results = []
        all_results_dirs = []
        
        for run_idx in range(num_runs):
            start_total = time.perf_counter()
            log_important(f"📊 Starting run {run_idx + 1}/{num_runs}")
            
            # Run the calculation
            final_energy, current_dets, current_coeffs, iteration_details, run_args = iterative_workflow(
                h1, eri, n_alpha, n_beta, n_orb, mol_name, args, nuclear_repulsion, start_total
            )
            
            # Store result and results directory
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
            all_results_dirs.append(results_dir)
            
            log_important(f"✅ Run {run_idx + 1} completed with energy: {final_energy:.8f}")
            
            # Check if this is the best result so far
            if final_energy < best_energy:
                best_energy = final_energy
                best_result = run_result
                log_important(f"🏆 New best energy found: {final_energy:.8f}")
            
            log_important(f"Current best energy: {best_energy:.8f}")
        
        # Log summary of all runs
        log_important(f"📈 Summary of all {num_runs} runs:")
        for result in all_results:
            marker = "🏆" if result['run_idx'] == best_result['run_idx'] else "  "
            log_important(f"{marker} Run {result['run_idx']}: Energy = {result['final_energy']:.8f}")
        
        log_important(f"🎯 Best result from run {best_result['run_idx']} with energy: {best_result['final_energy']:.8f}")
        
        # Clean up non-best results directories
        best_results_dir = best_result['results_dir']
        for result in all_results:
            if result['run_idx'] != best_result['run_idx'] and result['results_dir']:
                try:
                    if os.path.exists(result['results_dir']):
                        shutil.rmtree(result['results_dir'])
                        log_verbose(f"🗑️ Removed non-best results directory: {result['results_dir']}")
                except Exception as e:
                    log_verbose(f"⚠️ Failed to remove directory {result['results_dir']}: {e}")
        
        # Generate multi-run report
        generate_multi_run_report(all_results, best_result, mol_name, args)
        
        # Mark RUN_FULL end with summary
        try:
            _ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open("realtime_progress.out", "a", encoding="utf-8") as f:
                f.write(f"==== [ {_ts} ] RUN_FULL END ====\n")
                f.write(f"Summary: best=run_{best_result['run_idx']} | energy: {best_result['final_energy']:.8f} | kept_dir: {best_results_dir}\n\n")
        except Exception:
            # Non-fatal: ignore file writing errors
            pass
        
        # Return the best result
        return (best_result['final_energy'], best_result['current_dets'], 
                best_result['current_coeffs'], best_result['iteration_details'], 
                best_result['run_args'])
    else:
        # Single run
        start_total = time.perf_counter()
        _fe, _cd, _cc, _id, _ra = iterative_workflow(h1, eri, n_alpha, n_beta, n_orb, mol_name,
                                  args, nuclear_repulsion, start_total)
        # Record FINAL RUN END
        try:
            _ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            _elapsed = time.perf_counter() - start_total
            with open("realtime_progress.out", "a", encoding="utf-8") as f:
                f.write(f"---- [ {_ts} ] FINAL RUN END ----\n")
                f.write(f"label: single | energy: {_fe:.8f} | ndets: {getattr(args, 'max_final_dets', 'N/A')} | elapsed_s: {_elapsed:.2f}\n")
        except Exception:
            # Non-fatal: ignore file writing errors
            pass
        # Mark RUN_FULL END
        try:
            _ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open("realtime_progress.out", "a", encoding="utf-8") as f:
                f.write(f"==== [ {_ts} ] RUN_FULL END ====\n")
                f.write(f"Summary: single | energy: {_fe:.8f} | kept_dir: {_id.get('results_dir', '')}\n\n")
        except Exception:
            # Non-fatal: ignore file writing errors
            pass
        return (_fe, _cd, _cc, _id, _ra)

# ========== Auto Runner ==========
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
        ndets_explore: Determinant budget for exploration phase (smaller and faster). If None and ndets is set, defaults to ndets*0.5.
        exploration: Whether to explore small variations around baseline params
        max_exploration: Max number of variants to try (besides baseline)
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
        h1, eri, n_elec, n_orb, nuclear_repulsion, n_alpha, n_beta = read_fcidump(fcidump_path)
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

    # Apply explicit config dict then overrides
    if config_dict:
        for k, v in config_dict.items():
            setattr(args, k, v)


    setup_logging(getattr(args, 'verbose', False))

    # Write run_auto start block to realtime_progress.out
    try:
        _ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _ndets_explore_hint = ndets_explore if ndets_explore is not None else "auto"
        with open("realtime_progress.out", "a", encoding="utf-8") as f:
            f.write(f"\n==== [ {_ts} ] RUN_AUTO START ====\n")
            f.write(
                f"System: {mol_name} | electrons: {n_elec} | orbitals: {n_orb} | goal: {goal} | ndets: {ndets} | ndets_explore: {_ndets_explore_hint}\n"
            )
    except Exception:
        # Non-fatal: ignore file writing errors
        pass

    # Helper: clone Namespace
    def clone(ns):
        return Namespace(**vars(ns))

    total_configs = comb(n_orb, n_alpha) * comb(n_orb, n_beta)
    log10_configs = math.log10(total_configs) if total_configs > 0 else 0.0


    if log10_configs < 8:   # ~ up to 1e6
        if ndets is None:
            ndets = 50
        cat = "small"
    elif log10_configs < 16: # ~ up to 1e8
        if ndets is None:
            ndets = 100
        cat = "medium"
    elif log10_configs < 24:# ~ up to 1e10
        if ndets is None:
            ndets = 200
        cat = "large"
    else:
        if ndets is None:
            ndets = 500
        cat = "xlarge"

    # Helper: auto tune baseline args based on size & goal
    def auto_tune(base: Namespace, n_orb_val: int, n_alpha_val: int, n_beta_val: int, goal_val: str, ndets_val: int = None) -> Namespace:
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
        else:  # xlarge
            tuned.initial_pool_size = 1000
            tuned.pool_core_ratio = 30
            tuned.local_trim_keep_ratio = 0.05
            tuned.threshold = 0.06
            tuned.max_final_dets = 400
            tuned.max_rounds = 4
            tuned.num_groups = 16

        tuned.core_set_ratio = [1, 1.2]
        tuned.max_final_dets = int(ndets_val)
        tuned.pool_build_strategy = "heat_bath"
        tuned.verbose = False
        tuned.load_initial_dets = False
        tuned.first_cycle_keep_size = 10

        # Adjust for goal preference
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
        else:  # xlarge
            exploration_ndets = max(1000, int(ndets ** 0.5))

    for k, v in overrides.items():
        print(k)
        if k == "max_final_dets":
            continue
        elif k == "explore_final_dets":
            exploration_ndets = v
            log_verbose(f"Overriding max_final_dets to {v} for exploration")
            setattr(baseline, "max_final_dets", v)
        else:
            setattr(baseline, k, v)

    # Exploration variants around baseline (apply exploration budget)
    baseline_explore = clone(baseline)
    baseline_explore.max_final_dets = exploration_ndets

    # edge case
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
    else:  # xlarge
        n_random = 1000
    for _ in range(nexploration):
        var = clone(baseline_explore)

        tweak = {"initial_dets_dict": {
                    "random": [
                        1,
                        n_random
                    ],
                    "hf": [
                        1, 1
                    ]
                    }
                }
        for key, value in tweak.items():
            setattr(var, key, value)
        variants.append(var)

    # Run each variant and pick the best by final energy
    # Mark explore phase start
    explore_start_time = time.perf_counter()
    try:
        _ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("realtime_progress.out", "a", encoding="utf-8") as f:
            f.write(f"---- [ {_ts} ] EXPLORE START ----\n")
            f.write(f"exploration_ndets: {exploration_ndets} | variants: {len(variants)}\n")
            f.write(f"total_configs: {int(total_configs):,} | log10_configs: {log10_configs:.2f} | scale: {cat}\n")
    except Exception:
        # Non-fatal: ignore file writing errors
        pass
    log_important(f"🔧 Explore: explore_ndets={exploration_ndets}, exploring {len(variants)-1} nexploration")

    best_energy = float('inf')
    best_result = None
    all_results = []
    all_results_dirs = []


    for idx, run_args in enumerate(variants):
        start_total = time.perf_counter()
        label = f"explore_{idx}"
        log_important(f"📊 Starting {label} ({idx+1}/{len(variants)})")


        final_energy, current_dets, current_coeffs, iteration_details, run_ns = iterative_workflow(
            h1, eri, n_alpha, n_beta, n_orb, mol_name, run_args, nuclear_repulsion, start_total
        )

        results_dir = iteration_details.get('results_dir', '')

        # Write realtime progress to file
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
            # Non-fatal: ignore file writing errors
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
        all_results_dirs.append(results_dir)

        log_important(f"✅ {label} completed with energy: {final_energy:.8f}")
        if final_energy < best_energy:
            best_energy = final_energy
            best_result = run_result
            log_important(f"🏆 New best energy: {final_energy:.8f} ({label})")



    # After exploration, run a final calculation to target ndets
    # Mark explore phase end
    try:
        _ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _explore_elapsed = time.perf_counter() - explore_start_time
        if best_result:
            _best_label = best_result['label']
            _best_energy = f"{best_result['final_energy']:.8f}"
        else:
            _best_label = "None"
            _best_energy = "N/A"
        with open("realtime_progress.out", "a", encoding="utf-8") as f:
            f.write(f"---- [ {_ts} ] EXPLORE END ----\n")
            f.write(f"best: {_best_label} | energy: {_best_energy} | elapsed_s: {_explore_elapsed:.2f} | completed_runs: {len(variants)}\n")
    except Exception:
        # Non-fatal: ignore file writing errors
        pass
    final_result = None
    if ndets is not None and best_result is not None:
        final_args = clone(best_result['run_args'])
        final_args.load_initial_dets = True
        final_args.initial_dets_path = os.path.join(best_result['results_dir'], "dets.npz")
        final_args.max_final_dets = int(ndets)
        log_important(f"🚀 Starting final_ndets run with max_final_dets={final_args.max_final_dets}")
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
        log_important(f"✅ final_ndets completed with energy: {final_energy:.8f}")

        # Record final run end to realtime_progress.out
        try:
            _ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            _final_elapsed = time.perf_counter() - final_start_time
            with open("realtime_progress.out", "a", encoding="utf-8") as f:
                f.write(f"---- [ {_ts} ] FINAL RUN END ----\n")
                f.write(f"label: {final_result['label']} | energy: {final_result['final_energy']:.8f} | ndets: {final_args.max_final_dets} | elapsed_s: {_final_elapsed:.2f}\n")
        except Exception:
            # Non-fatal: ignore file writing errors
            pass



    # Clean up: keep only the final result directory if it exists; otherwise keep the best exploration
    best_dir = best_result['results_dir'] if best_result else ''
    final_dir = final_result['results_dir'] if final_result else ''
    for result in all_results:
        if result['results_dir'] and (result['results_dir'] != final_dir and result['results_dir'] != best_dir):
            try:
                if os.path.exists(result['results_dir']):
                    import shutil
                    shutil.rmtree(result['results_dir'])
                    log_verbose(f"🗑️ Removed non-final results directory: {result['results_dir']}")
            except Exception as e:
                log_verbose(f"⚠️ Failed to remove directory {result['results_dir']}: {e}")

    # Generate report and return
    if final_result:
        generate_multi_run_report(all_results, final_result, mol_name, baseline, final_run=True)
        log_important(f"🎯 Final: {final_result['label']} with energy {final_result['final_energy']:.8f}")
        # Mark RUN_AUTO end with summary for final result
        try:
            _ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open("realtime_progress.out", "a", encoding="utf-8") as f:
                f.write(f"==== [ {_ts} ] RUN_AUTO END ====\n")
                f.write(
                    f"Summary: final={final_result['label']} | energy: {final_result['final_energy']:.8f} | kept_dir: {keep_dir}\n\n"
                )
        except Exception:
            # Non-fatal: ignore file writing errors
            pass
        return (final_result['final_energy'], final_result['current_dets'],
                final_result['current_coeffs'], final_result['iteration_details'],
                final_result['run_args'])
    elif best_result:
        generate_multi_run_report(all_results, best_result, mol_name, baseline)
        log_important(f"🎯 Best: {best_result['label']} with energy {best_result['final_energy']:.8f}")
        # Mark RUN_AUTO end with summary for best exploration result
        try:
            _ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open("realtime_progress.out", "a", encoding="utf-8") as f:
                f.write(f"==== [ {_ts} ] RUN_AUTO END ====\n")
                f.write(
                    f"Summary: best={best_result['label']} | energy: {best_result['final_energy']:.8f} | kept_dir: {keep_dir}\n\n"
                )
        except Exception:
            # Non-fatal: ignore file writing errors
            pass
        return (best_result['final_energy'], best_result['current_dets'],
                best_result['current_coeffs'], best_result['iteration_details'],
                best_result['run_args'])
    else:
        # Fallback to baseline when exploration produced no results
        _baseline_start_time = time.perf_counter()
        _fe, _cd, _cc, _id, _ra = iterative_workflow(
            h1, eri, n_alpha, n_beta, n_orb, mol_name,
            baseline, nuclear_repulsion, _baseline_start_time
        )
        # Mark RUN_AUTO end with summary for baseline fallback
        try:
            _ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open("realtime_progress.out", "a", encoding="utf-8") as f:
                f.write(f"==== [ {_ts} ] RUN_AUTO END ====\n")
                f.write(
                    f"Summary: fallback=baseline | energy: {_fe:.8f} | kept_dir: {_id.get('results_dir', '')}\n\n"
                )
        except Exception:
            # Non-fatal: ignore file writing errors
            pass
        return (_fe, _cd, _cc, _id, _ra)

# ========== Iterative Workflow ==========
def iterative_workflow(h1, eri, n_alpha, n_beta, n_orb,
                       mol_name, args, nuclear_repulsion, start_total):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3]
    results_dir = str(Path("trimci_results") / f"{mol_name}_{timestamp}")
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    log_verbose(f"📁 Results will be saved in: {results_dir}")

    # workflow parameters
    max_iterations = getattr(args, 'max_iterations', 
                             getattr(args, 'exp_max_iterations', -1))
    energy_threshold = getattr(args, 'energy_threshold',
                               getattr(args, 'exp_energy_threshold', 1e-12))
    
    initial_pool_size = getattr(args, 'initial_pool_size', 100)
    _core_set_ratio = getattr(args, 'core_set_ratio', 2)
    max_final_dets = getattr(args, 'max_final_dets', None)
    pool_core_ratio = getattr(args, 'pool_core_ratio', 10)
    max_rounds = getattr(args, 'max_rounds', 1)
    pool_build_strategy = getattr(args, 'pool_build_strategy', 'heat_bath')
    num_groups = getattr(args, 'num_groups', 10)
    save_period = getattr(args, 'save_period', 50)
    keep_pool_to_next_core_ratio = getattr(args, 'local_trim_keep_ratio',
                                           getattr(args, 'keep_pool_to_next_core_ratio', 0))
    debug = getattr(args, 'debug', False)
    save_initial = getattr(args, 'save_initial', False)
    save_pool = getattr(args, 'save_pool', False)
    threshold = getattr(args, 'threshold', False)
    keep_ratio = getattr(args, 'keep_ratio', 0.1) # not recommended; use local_trim_keep_ratio instead

    # Special case: unlimited iterations
    if max_iterations == -1:
        if max_final_dets is None:
            raise ValueError("When exp_max_iterations = -1, max_final_dets must be specified")
        max_iterations = 200000
        log_verbose(f"🧪 Workflow (controlled by max_final_dets={max_final_dets}, "
                     f"energy_thresh={energy_threshold})")
    else:
        if max_final_dets is not None:
            log_verbose(f"🧪 Workflow (max_iter={max_iterations}, "
                         f"max_final_dets={max_final_dets}, energy_thresh={energy_threshold})")
        else:
            log_verbose(f"🧪 Workflow (max_iter={max_iterations}, energy_thresh={energy_threshold})")

    # Initialize reference determinant and core set
    if getattr(args, 'load_initial_dets', False):
        # Try to load from dets.npz file
        dets_path = getattr(args, "initial_dets_path", "dets.npz")
        log_important(f"🔄 Loading initial determinants from dets.npz")
        dets_array_name = getattr(args, "dets_array_name", "dets")
        if dets_array_name == "dets":
            loaded_core_set, loaded_coeffs = load_initial_dets_from_file(dets_path, core_set=False)
        else:
            loaded_core_set, loaded_coeffs = load_initial_dets_from_file(dets_path, core_set=True)
            
        if loaded_core_set is not None and loaded_coeffs is not None:
            current_core_set = loaded_core_set
            current_coeffs = loaded_coeffs
            log_important(f"✅ Loaded {len(current_core_set)} determinants from dets.npz")
        else:
            # Fallback to reference determinant if loading fails
            det_ref = generate_reference_det(n_alpha, n_beta, n_orb)
            log_important(f"🔄 Fallback to reference determinant: {det_ref}")
            current_core_set = [det_ref]
            current_coeffs = [1.0]
        
    else:
        init_dict = getattr(args, "initial_dets_dict", None)
        if init_dict:
            current_core_set, current_coeffs = generate_initial_states(
                n_alpha, n_beta, n_orb,
                initial_dets_dict=init_dict,
                save_path=None
            )
            log_important(
                f"✅ Generated {len(current_core_set)} determinants from initial_dets_dict "
                f"(coeffs normalized, before norm={list(init_dict.values())})"
            )
            if debug:
                #print current_core_set
                #print current_coeffs
                log_verbose(f"🔄 Initial core set: {current_core_set}")
                log_verbose(f"🔄 Initial coeffs: {current_coeffs}")
        else:
            # default reference
            det_ref = generate_reference_det(n_alpha, n_beta, n_orb)
            log_verbose(f"🔄 Reference determinant: {det_ref}")
            current_core_set = [det_ref]
            current_coeffs = [1.0]


    
    pool_size = max(math.ceil(len(current_core_set) * pool_core_ratio), initial_pool_size)

    previous_energy = None
    current_dets = current_core_set
    
    # Calculate initial energy if determinants were loaded from file
    if getattr(args, 'load_initial_dets', False) and len(current_core_set) > 1:
        # Need to calculate energy for the loaded determinants
        # For now, set to 0.0 and let first iteration calculate it properly
        current_energy = 0.0
        log_verbose(f"🔋 Initial energy will be calculated in first iteration")
    else:
        current_energy = 0.0

    # Iteration info tracking
    iteration_details = {
        'max_iterations': max_iterations,
        'energy_threshold': energy_threshold,
        'initial_pool_size': pool_size,
        'iterations': []
    }

    # Save initial state if requested
    if save_initial:
        # Compose initial iter info
        iter_info_init = {
            'iteration': -1,
            'core_set_size': len(current_core_set),
            'pool_size': pool_size
        }
        # Electronic and total energy at initial stage
        total_energy_init = current_energy + nuclear_repulsion
        iter_info_init['electronic_energy'] = current_energy
        iter_info_init['total_energy'] = total_energy_init
        iter_info_init['final_dets_count'] = len(current_core_set)

        # Prepare final_coeffs sorted by magnitude
        sorted_idx = np.argsort(np.abs(current_coeffs))[::-1]
        final_coeffs_init = [current_coeffs[i] for i in sorted_idx]

        # Save using same format as iteration results
        save_iteration_results(-1, current_energy, total_energy_init,
                               current_core_set, final_coeffs_init, current_coeffs,
                               current_core_set, iter_info_init, outdir=results_dir,
                               pool=None, save_pool=save_pool)

    n_total_configs = comb(n_orb, n_alpha) * comb(n_orb, n_beta)
    for iteration in range(max_iterations):
        iteration_start = time.perf_counter()

        # Get core_set_ratio for current iteration
        if isinstance(_core_set_ratio, list):
            core_set_ratio = _core_set_ratio[iteration % len(_core_set_ratio)]
        else:
            core_set_ratio = _core_set_ratio


        log_important("="*60)
        log_important(f"{RED_BOLD}🔄 TrimCI iteration {iteration+1}/{max_iterations}{RESET}")
        log_verbose(f"🔧 Core set ratio: {core_set_ratio:.4f}")


        pool_size = min(pool_size, n_total_configs)
        iter_info = {
            'iteration': iteration+1,
            'core_set_size': len(current_core_set),
            'pool_size': pool_size
        }

        # Step 1: Pool building
        start_time = time.perf_counter()

        log_verbose(f"📦 Building pool with {len(current_core_set)} core determinants")
        if pool_build_strategy == 'heat_bath':
            norm = np.sqrt(np.sum([c**2 for c in current_coeffs]))
            current_coeffs = [c/norm for c in current_coeffs]
            pool, final_threshold = screening(current_core_set, current_coeffs, n_orb, h1, eri,
                                            threshold, pool_size, {},
                                            f"{mol_name}.bin",
                                            max_rounds=max_rounds)
        elif pool_build_strategy == 'normalized_uniform':
            coeffs = [1.0/np.sqrt(len(current_core_set))] * len(current_core_set)
            pool, final_threshold = screening(current_core_set, coeffs, n_orb, h1, eri,
                                            threshold, pool_size, {},
                                            f"{mol_name}.bin",
                                            max_rounds=max_rounds)
        elif pool_build_strategy == 'uniform':
            pool, final_threshold = screening(current_core_set, [], n_orb, h1, eri,
                                            threshold, pool_size, {},
                                            f"{mol_name}.bin",
                                            max_rounds=max_rounds)
        else:
            raise ValueError(f"Unknown pool_build_strategy: {pool_build_strategy}")

        log_verbose(f"🔍 Screening completed: {len(pool)} determinants "
                 f"in {time.perf_counter()-start_time:.1f}s, final threshold: {final_threshold:.2e}")

        iter_info['pool_time'] = time.perf_counter() - start_time
        iter_info['actual_pool_size'] = len(pool)
        iter_info['final_threshold'] = final_threshold
        
        # Update threshold for next iteration
        # 2025-10-16 21:49:59
        if iteration > 0:
            threshold = final_threshold

        if len(pool) < len(current_core_set) * pool_core_ratio:
            log_important(f"⚠️ Pool size {len(pool)} < {len(current_core_set) * pool_core_ratio}, stopping.")
            break


        # Step 2: Smart trim
        if num_groups>=1:
            trim_m = [num_groups]
            if keep_pool_to_next_core_ratio > 0:
                #keep_pool_size = math.ceil(len(current_core_set)*core_set_ratio)*keep_pool_to_next_core_ratio
                keep_pool_size = math.ceil(len(current_core_set)*keep_pool_to_next_core_ratio)
                trim_k = [math.ceil(keep_pool_size / num_groups)]
            else:
                trim_k = [math.ceil(pool_size * keep_ratio / num_groups)]
        else:
            m = int(np.power(pool_size, num_groups))
            trim_m = [m]
            if keep_pool_to_next_core_ratio > 0:
                #keep_pool_size = math.ceil(len(current_core_set)*core_set_ratio)*keep_pool_to_next_core_ratio
                keep_pool_size = math.ceil(len(current_core_set)*keep_pool_to_next_core_ratio)
                trim_k = [math.ceil(keep_pool_size / m)]
            else:
                trim_k = [math.ceil(pool_size * keep_ratio / m)]

        iter_info['trim_m'], iter_info['trim_k'] = trim_m, trim_k

        # Step 3: Run trim
        log_verbose(f"✂️  Running trim with m={trim_m}, k={trim_k}")
        current_energy, current_dets, current_coeffs = trim(
            pool=pool,
            h1=h1,
            eri=eri,
            mol_name=mol_name,
            n_elec=n_alpha+n_beta,
            n_orb=n_orb,
            group_sizes=trim_m,
            keep_sizes=trim_k,
            quantization=False,
            save_cache=False,
            verbose=getattr(args, 'verbose', False),
            external_core_dets=current_core_set 
        )

        iter_info['final_dets_count'] = len(current_dets)
        iter_info['electronic_energy'] = current_energy

        # Step 4: Energy accounting
        total_energy = current_energy + nuclear_repulsion
        iter_info['total_energy'] = total_energy
        log_important(f"⚡ Iteration {iteration+1} total energy: {total_energy:.8f}")
        log_important(f"🔤 Core set: {len(current_core_set)}, Determinants: {len(current_dets)}")

        # Step 5: Convergence
        if previous_energy is not None:
            energy_change = abs(total_energy - previous_energy)
            iter_info['energy_change'] = energy_change
            log_important(f"📊 ΔE = {-energy_change:.2e}")
            

            if energy_change < energy_threshold:
                log_important("✅ Converged")
                iter_info['converged'] = True
            else:
                iter_info['converged'] = False
        else:
            iter_info['converged'] = False
        previous_energy = total_energy

        # Record and output iteration timing details
        time_cost = time.perf_counter() - iteration_start
        iter_info['iteration_time'] = time_cost
        
        # Calculate cumulative time
        total_elapsed = time.perf_counter() - start_total
        iter_info['cumulative_time'] = total_elapsed
        
        # Output detailed timing information
        # Auto-pick best unit for time display
        if total_elapsed < 60:
            unit, factor = "s", 1
        elif total_elapsed < 3600:
            unit, factor = "min", 60
        else:
            unit, factor = "h", 3600
        log_important(f"⏱️  Iteration {iteration+1} time: {time_cost:.2f}s (Total: {total_elapsed/factor:.1f}{unit})")
        if 'pool_time' in iter_info:
            trim_time = time_cost - iter_info['pool_time']
            iter_info['trim_time'] = trim_time
            log_important(f"📊 Pool: {iter_info['pool_time']:.2f}s, Trim: {trim_time:.2f}s")

        # Save iteration results
        iteration_details['iterations'].append(iter_info)




        # Step 6: Update core set
        old_size = len(current_core_set)  # Define old_size before the if condition
        # Always create final_coeffs for potential use in save_iteration_results
        sorted_idx = np.argsort(np.abs(current_coeffs))[::-1]
        final_coeffs = [current_coeffs[i] for i in sorted_idx]
        
        if iteration < max_iterations - 1:
            new_size = min(len(current_dets), math.ceil(old_size*core_set_ratio))
            log_important(f"🔄 Updating core set size: {old_size} -> {new_size} (max: {len(current_dets)})")
            if core_set_ratio <= 0:
                new_size = 1
                log_important(f"⚠️ Core set ratio {core_set_ratio:.4f} <= 0, setting new_size to 1.")

            if iteration == 0 and getattr(args, 'first_cycle_keep_size', 0):
                new_size = min(len(current_dets), getattr(args, 'first_cycle_keep_size', 0))
                log_important(f"🔄 First cycle, setting new_size to {new_size} (max: {getattr(args, 'first_cycle_keep_size', 0)})")




            current_core_set = [current_dets[i] for i in sorted_idx[:new_size]]
            current_coeffs = [current_coeffs[i] for i in sorted_idx[:new_size]]
            # Renormalize coefficients
            norm = np.sqrt(np.sum([c**2 for c in current_coeffs]))
            current_coeffs = [c/norm for c in current_coeffs]
            iter_info['next_core_set_size'] = new_size

            if new_size == old_size and not getattr(args, 'allow_core_set_unchange', True):
                log_important("🛑 Core set unchanged, stopping.")
                break

            pool_size = math.ceil(new_size * pool_core_ratio)
            iter_info['next_pool_size'] = pool_size
            log_verbose(f"📈 Next pool size: {pool_size}")

            if (iteration + 1) % save_period == 0 or (save_initial and iteration == 0):
                save_iteration_results(iteration + 1, current_energy, total_energy,
                            current_dets, final_coeffs, current_coeffs,
                            current_core_set, iter_info, outdir=results_dir,
                            pool=pool, save_pool=save_pool)



        # Step 7: Max determinants stopping condition
        if (max_final_dets is not None and len(current_dets) >= max_final_dets and
                iteration > 0):
            log_important(f"🛑 Exceeded max_final_dets={max_final_dets}, stopping.")
            iter_info['stopped_by_max_final_dets_actual'] = True
            break

        if iter_info.get('converged', False):
            break

    final_energy = current_energy + nuclear_repulsion
    total_time = time.perf_counter() - start_total
    log_important(f"⏱️ Final energy: {final_energy:.8f}")
    log_important(f"⏱️ Workflow time: {total_time:.1f}s")

    iteration_details.update({
        'total_time': total_time,
        'final_energy': final_energy,
        'final_electronic_energy': current_energy,
        'final_dets_count': len(current_dets),
        'converged': any(it.get('converged', False) for it in iteration_details['iterations']),
        'total_iterations': len(iteration_details['iterations']),
        'results_dir': results_dir
    })

    iteration_details['n_electrons'] = n_alpha + n_beta
    iteration_details['n_orbitals'] = n_orb
    iteration_details['nuclear_repulsion'] = nuclear_repulsion
    iteration_details['results_dir'] = results_dir

    # Ensure final_coeffs is defined (in case no iterations were executed)
    if 'final_coeffs' not in locals():
        sorted_idx = np.argsort(np.abs(current_coeffs))[::-1]
        final_coeffs = [current_coeffs[i] for i in sorted_idx]

    save_final_results(final_energy, current_dets, final_coeffs, current_coeffs,
                       current_core_set, iteration_details, args, outdir=results_dir)

    return final_energy, current_dets, current_coeffs, iteration_details, args

# ========== Configuration ==========
def load_configurations(config_dir: str, trimci_config_path: str = None, save_if_not_exist: bool = True):
    default_config = {
            "threshold": 0.06,
            "local_trim_keep_ratio": 0.1,
            "verbose": False,
            "initial_pool_size": 100,
            "core_set_ratio": 1.02,
            "pool_core_ratio": 20,
            "max_final_dets": 100,
            "max_rounds": 2,
            "pool_build_strategy": "heat_bath",
            "num_groups": 10,
            "load_initial_dets": False,
            "num_runs": 1,
        }

    if trimci_config_path is None:
        trimci_config_path = os.path.join(config_dir, "trimci_config.json")
    if os.path.exists(trimci_config_path):
        with open(trimci_config_path, 'r') as f:
            default_config.update(json.load(f))
        log_verbose(f"📋 Loaded config: {trimci_config_path}")
    else:
        if save_if_not_exist:
            with open(trimci_config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            log_verbose(f"📋 Created default config: {trimci_config_path}")

    return Namespace(**default_config)

def dets_to_array(dets):
    """Convert determinants to numpy array as uint64 pairs for C++ compatibility"""
    if not dets:
        return np.array([], dtype=np.uint64).reshape(0, 6)  # Empty array with correct shape
    
    # Determine the maximum number of uint64 values needed
    max_uint64_pairs = 1  # Default for standard Determinant (64-bit)
    
    # Check if we have larger determinants
    for det in dets:
        alpha_bits = det.alpha
        if isinstance(alpha_bits, list):
            max_uint64_pairs = max(max_uint64_pairs, len(alpha_bits))
    
    result = []
    for det in dets:
        alpha_bits = det.alpha
        beta_bits = det.beta
        
        # Handle different determinant types
        if isinstance(alpha_bits, list):
            # For Determinant128/192, alpha/beta are already arrays of uint64
            alpha_vals = [int(val) & 0xFFFFFFFFFFFFFFFF for val in alpha_bits]
            beta_vals = [int(val) & 0xFFFFFFFFFFFFFFFF for val in beta_bits]
            
            # Pad to maximum size if needed
            while len(alpha_vals) < max_uint64_pairs:
                alpha_vals.append(0)
            while len(beta_vals) < max_uint64_pairs:
                beta_vals.append(0)
                
            # Interleave alpha and beta values: [alpha0, beta0, alpha1, beta1, ...]
            row = []
            for i in range(max_uint64_pairs):
                row.extend([alpha_vals[i], beta_vals[i]])
            result.append(row)
        else:
            # For standard Determinant, convert to uint64 and pad
            alpha_val = int(alpha_bits) & 0xFFFFFFFFFFFFFFFF
            beta_val = int(beta_bits) & 0xFFFFFFFFFFFFFFFF
            
            row = [alpha_val, beta_val]
            # Pad with zeros for larger determinant compatibility
            for i in range(1, max_uint64_pairs):
                row.extend([0, 0])
            result.append(row)
    
    return np.array(result, dtype=np.uint64)

def det_to_bitstring(det):
    """Convert a determinant to bitstring format like '0 1 2 5 10 11 12 13 14 15 16 17 18 19 20 21 , 0 1 2 3 4 5 6 7 8 9 22 23 24 25 26 27'"""
    # Extract occupied orbitals from alpha and beta parts
    alpha_orbs = []
    beta_orbs = []
    
    # Get alpha and beta data
    alpha_bits = det.alpha
    beta_bits = det.beta
    
    # Handle different determinant types
    if isinstance(alpha_bits, list):
        # For Determinant128/192, alpha/beta are arrays
        # Process each uint64 element in the array
        for array_idx, (alpha_val, beta_val) in enumerate(zip(alpha_bits, beta_bits)):
            for bit_idx in range(64):  # Each uint64 has 64 bits
                orbital_idx = array_idx * 64 + bit_idx
                if alpha_val & (1 << bit_idx):
                    alpha_orbs.append(orbital_idx)
                if beta_val & (1 << bit_idx):
                    beta_orbs.append(orbital_idx)
    else:
        # For standard Determinant, alpha/beta are integers
        for i in range(64):  # uint64 has 64 bits
            if alpha_bits & (1 << i):
                alpha_orbs.append(i)
            if beta_bits & (1 << i):
                beta_orbs.append(i)
    
    # Format as space-separated strings
    alpha_str = " ".join(map(str, sorted(alpha_orbs)))
    beta_str = " ".join(map(str, sorted(beta_orbs)))
    
    return f"{alpha_str} , {beta_str}"

def get_top_determinants(dets, coeffs, top_n=10):
    """Get top N determinants by coefficient magnitude"""
    if len(dets) == 0 or len(coeffs) == 0:
        return []
    
    # Get indices sorted by coefficient magnitude (descending)
    sorted_indices = np.argsort(np.abs(coeffs))[::-1]
    
    # Take top N
    top_indices = sorted_indices[:min(top_n, len(sorted_indices))]
    
    # Format as [coeff, bitstring] pairs
    top_dets = []
    for idx in top_indices:
        coeff = float(coeffs[idx])
        bitstring = det_to_bitstring(dets[idx])
        top_dets.append([coeff, bitstring])
    
    return top_dets

def load_initial_dets_from_file(dets_path: str = "dets.npz", core_set: bool = False):
    """
    Load initial determinants and coefficients from dets.npz file.
    Returns: (dets, coeffs) or (None, None) if file doesn't exist
    """
    if not os.path.exists(dets_path):
        log_verbose(f"⚠️ dets.npz file not found at {dets_path}")
        return None, None
    
    try:
        data = np.load(dets_path, allow_pickle=True)
        log_verbose(f"📂 Loading initial determinants from {dets_path}")
        
        # Determine which keys to load based on core_set flag
        dets_key = 'core_set' if core_set else 'dets'
        coeffs_key = 'core_set_coeffs' if core_set else 'dets_coeffs'
        
        if dets_key in data and coeffs_key in data:
            dets_array = data[dets_key]
            coeffs = data[coeffs_key]
            
            # Convert array back to determinant objects
            from .. import trimci_core
            from trimci import get_functions_for_system
            
            # Handle both new uint64 format and legacy object format
            if dets_array.dtype == np.uint64:
                # New format: uint64 arrays with interleaved alpha/beta pairs
                # Shape should be (n_dets, 2*n_uint64_pairs)
                n_dets, total_cols = dets_array.shape
                n_uint64_pairs = total_cols // 2
                
                # Determine number of orbitals from the array structure
                # Each pair (alpha, beta) represents 64 orbitals; use pair count directly
                n_orb_estimate = 64 * n_uint64_pairs
                functions = get_functions_for_system(n_orb_estimate)
                DeterminantClass = functions['determinant_class']
                
                # Create determinants from uint64 arrays
                dets = []
                for i in range(n_dets):
                    if n_uint64_pairs == 1:
                        # Standard 64-bit determinant
                        alpha_bits = int(dets_array[i, 0])
                        beta_bits = int(dets_array[i, 1])
                        dets.append(DeterminantClass(alpha_bits, beta_bits))
                    else:
                        # Multi-uint64 determinant (128-bit or 192-bit)
                        alpha_array = []
                        beta_array = []
                        for j in range(n_uint64_pairs):
                            alpha_array.append(int(dets_array[i, 2*j]))
                            beta_array.append(int(dets_array[i, 2*j + 1]))
                        dets.append(DeterminantClass(alpha_array, beta_array))
                        
            elif dets_array.dtype == object:
                # Legacy format: object arrays with large integers
                log_verbose("⚠️ Loading legacy object format - this may not be compatible with C++")
                
                # Determine number of orbitals from the data to get correct Determinant class
                max_bits = max(max(row[0], row[1]) for row in dets_array)
                n_orb_estimate = int(max_bits).bit_length() if max_bits > 0 else 64
                functions = get_functions_for_system(n_orb_estimate)
                DeterminantClass = functions['determinant_class']
                
                # Helper function to create Determinant objects with appropriate constructor
                def create_determinant_from_file(alpha_bits, beta_bits):
                    """Create a Determinant object with the appropriate constructor based on the class type"""
                    class_name = str(DeterminantClass)
                    
                    if 'Determinant192' in class_name:
                        # Convert to array format for Determinant192 (3 x uint64_t)
                        # Handle large integers by masking and converting properly
                        alpha_array = [(alpha_bits & 0xFFFFFFFFFFFFFFFF), 
                                      ((alpha_bits >> 64) & 0xFFFFFFFFFFFFFFFF), 
                                      ((alpha_bits >> 128) & 0xFFFFFFFFFFFFFFFF)]
                        beta_array = [(beta_bits & 0xFFFFFFFFFFFFFFFF), 
                                     ((beta_bits >> 64) & 0xFFFFFFFFFFFFFFFF), 
                                     ((beta_bits >> 128) & 0xFFFFFFFFFFFFFFFF)]
                        # Convert negative values to unsigned
                        alpha_array = [x if x >= 0 else x + (1 << 64) for x in alpha_array]
                        beta_array = [x if x >= 0 else x + (1 << 64) for x in beta_array]
                        return DeterminantClass(alpha_array, beta_array)
                    elif 'Determinant128' in class_name:
                        # Convert to array format for Determinant128 (2 x uint64_t)
                        # Handle large integers by masking and converting properly
                        alpha_array = [(alpha_bits & 0xFFFFFFFFFFFFFFFF), 
                                      ((alpha_bits >> 64) & 0xFFFFFFFFFFFFFFFF)]
                        beta_array = [(beta_bits & 0xFFFFFFFFFFFFFFFF), 
                                     ((beta_bits >> 64) & 0xFFFFFFFFFFFFFFFF)]
                        # Convert negative values to unsigned
                        alpha_array = [x if x >= 0 else x + (1 << 64) for x in alpha_array]
                        beta_array = [x if x >= 0 else x + (1 << 64) for x in beta_array]
                        return DeterminantClass(alpha_array, beta_array)
                    else:
                        # Standard Determinant (64-bit)
                        return DeterminantClass(alpha_bits, beta_bits)
                
                dets = [create_determinant_from_file(int(row[0]), int(row[1])) for row in dets_array]
            else:
                # Handle other numeric dtypes (legacy compatibility)
                max_bits = max(np.max(dets_array[:, 0]), np.max(dets_array[:, 1]))
                max_bits = int(max_bits)
                n_orb_estimate = int(max_bits).bit_length() if max_bits > 0 else 64
                functions = get_functions_for_system(n_orb_estimate)
                DeterminantClass = functions['determinant_class']
                
                dets = [DeterminantClass(int(row[0]), int(row[1])) for row in dets_array]
            
            log_verbose(f"✅ Loaded {len(dets)} determinants from {dets_key}")
            log_verbose(f"✅ Loaded {len(coeffs)} coefficients")
            
            return dets, coeffs.tolist()
        else:
            log_verbose(f"⚠️ Required keys '{dets_key}' or '{coeffs_key}' not found in {dets_path}")
            return None, None
            
    except Exception as e:
        log_verbose(f"❌ Error loading {dets_path}: {e}")
        return None, None

def save_iteration_results(iter_idx, current_energy, total_energy,
                           current_dets, final_coeffs, current_coeffs,
                           current_core_set, iter_info,
                           outdir="results", pool=None, save_pool=False):
    Path(outdir).mkdir(exist_ok=True)

    # Get top-10 determinants
    top_10_dets = get_top_determinants(current_dets, final_coeffs, top_n=10)

    # Save energy and statistical information
    json_path = os.path.join(outdir, f"iter_{iter_idx:03d}.json")
    with open(json_path, "w") as f:
        json.dump({
            "iteration": iter_idx,
            "electronic_energy": current_energy,
            "total_energy": total_energy,
            "final_dets_count": len(current_dets),
            "core_set_size": len(current_core_set),
            "iteration_info": iter_info,
            "top_10_determinants": top_10_dets
        }, f, indent=2)

    # Save determinants and coefficients
    npz_path = os.path.join(outdir, f"iter_{iter_idx:03d}.npz")
    if save_pool and pool is not None:
        np.savez_compressed(npz_path,
                            dets=dets_to_array(current_dets),
                            dets_coeffs=np.array(final_coeffs),
                            core_set_coeffs=np.array(current_coeffs),
                            core_set=dets_to_array(current_core_set),
                            pool=dets_to_array(pool))
    else:
        np.savez_compressed(npz_path,
                            dets=dets_to_array(current_dets),
                            dets_coeffs=np.array(final_coeffs),
                            core_set_coeffs=np.array(current_coeffs),
                            core_set=dets_to_array(current_core_set))
    log_verbose(f"💾 Saved iteration {iter_idx} results → {json_path}, {npz_path}")

def save_final_results(final_energy, current_dets, final_coeffs, current_coeffs,
                       current_core_set, iteration_details, args,
                       outdir="results"):
    Path(outdir).mkdir(exist_ok=True)

    # Get top-10 determinants
    top_10_dets = get_top_determinants(current_dets, final_coeffs, top_n=10)

    # Save final summary
    json_path = os.path.join(outdir, "trimci_results.json")
    with open(json_path, "w") as f:
        json.dump({
            "final_energy": final_energy,
            "final_dets_count": len(current_dets),
            "final_core_set_size": len(current_core_set),
            "config": vars(args),
            "iteration_summary": iteration_details,
            "top_10_determinants": top_10_dets
        }, f, indent=2)

    # Save final determinants
    npz_path = os.path.join(outdir, "dets.npz")
    np.savez_compressed(npz_path,
                        dets=dets_to_array(current_dets),
                        dets_coeffs=np.array(final_coeffs),
                        core_set_coeffs=np.array(current_coeffs),
                        core_set=dets_to_array(current_core_set))
    log_important(f"💾 Saved final results → {json_path}, {npz_path}")
    



# initial_dets_dict = {
#     "reference": 1.0,
#     "random_excited": {
#         "coeff": 0.2, "count": 3, "min_level": 1, "max_level": 3
#     },
#     "bitstring": [
#         [0.8, "0 1 2 3 , 0 1 2 3"],
#         [0.5, "0 1 2 4 , 0 1 2 4"]
#     ]
# }

# dets, coeffs = generate_initial_states(
#     n_alpha=4, n_beta=4, n_orb=8,
#     initial_dets_dict=initial_dets_dict,
#     save_path="dets.npz"
# )

# initial_dets_dict = {
#     "bitstring": [
#         [1.0, "0 1 2 3 4 5 6 7 8 9 10 11 70 71 , 0 1 2 3 4 5 6 7 8 9 10 11 72 73"]
#     ]
# }

def generate_initial_states(n_alpha, n_beta, n_orb,
                            initial_dets_dict=None,
                            save_path=None):
    """
    Generate initial determinants and coefficients for TrimCI.

    Parameters
    ----------
    n_alpha, n_beta : int
        Number of alpha/beta electrons.
    n_orb : int
        Number of orbitals.
    initial_dets_dict : dict or None
        Dict of {type: coeff or [coeff, count] or dict or [[coeff, state], ...]}.
        Supported keys:
          - "reference"
          - "afm"
          - "paramagnetic"
          - "stripe"
          - "random"
          - "random_excited" (dict)
          - "bitstring" ([[coeff, state], ...])
    save_path : str or None
        If given, save dets.npz to this path.

    Returns
    -------
    dets : list[Determinant]
    coeffs : list[float]
    """
    dets, coeffs = [], []
    rng = np.random.default_rng()

    # Get appropriate Determinant class based on orbital count
    functions = get_functions_for_system(n_orb)
    DeterminantClass = functions['determinant_class']
    log_important(f"Using Determinant class: {DeterminantClass}")

    # --- Helper: HF reference ---
    def hf_reference():
        return functions['generate_reference_det'](n_alpha, n_beta)

    # --- Helper: ensure uint64 wrap ---
    def to_uint64(x):
        if isinstance(x, np.integer):
            x = int(x)
        if x < 0:
            x += (1 << 64)
        return x & 0xFFFFFFFFFFFFFFFF

    # --- Robust create_determinant ---
    def create_determinant(alpha_bits, beta_bits):
        """Support both int and array input"""
        class_name = str(DeterminantClass)
        if 'Determinant192' in class_name:
            if isinstance(alpha_bits, (list, tuple)):
                alpha_array = [to_uint64(x) for x in alpha_bits]
                beta_array  = [to_uint64(x) for x in beta_bits]
            else:
                alpha_array = [to_uint64((alpha_bits >> (64*i)) & 0xFFFFFFFFFFFFFFFF) for i in range(3)]
                beta_array  = [to_uint64((beta_bits  >> (64*i)) & 0xFFFFFFFFFFFFFFFF) for i in range(3)]
            return DeterminantClass(alpha_array, beta_array)

        elif 'Determinant128' in class_name:
            if isinstance(alpha_bits, (list, tuple)):
                alpha_array = [to_uint64(x) for x in alpha_bits]
                beta_array  = [to_uint64(x) for x in beta_bits]
            else:
                alpha_array = [to_uint64((alpha_bits >> (64*i)) & 0xFFFFFFFFFFFFFFFF) for i in range(2)]
                beta_array  = [to_uint64((beta_bits  >> (64*i)) & 0xFFFFFFFFFFFFFFFF) for i in range(2)]
            return DeterminantClass(alpha_array, beta_array)

        else:
            if isinstance(alpha_bits, (list, tuple)):
                alpha_bits = alpha_bits[0]
                beta_bits  = beta_bits[0]
            return DeterminantClass(to_uint64(alpha_bits), to_uint64(beta_bits))

    # --- Preset determinants ---
    def add_preset(kind, coeff, count=1):
        for _ in range(count):
            if kind == "reference" or kind == "hf":
                dets.append(hf_reference())
            elif kind == "afm":
                alpha_bits = sum([1 << i for i in range(0, n_orb, 2)][:n_alpha])
                beta_bits  = sum([1 << i for i in range(1, n_orb, 2)][:n_beta])
                dets.append(create_determinant(alpha_bits, beta_bits))
            elif kind == "paramagnetic":
                alpha_bits = sum([1 << i for i in range(n_alpha)])
                beta_bits  = sum([1 << i for i in range(n_beta)])
                dets.append(create_determinant(alpha_bits, beta_bits))
            elif kind == "stripe":
                half = n_orb // 2
                alpha_bits = sum([1 << i for i in range(min(n_alpha, half))])
                beta_bits  = sum([1 << i for i in range(half, half + n_beta)])
                dets.append(create_determinant(alpha_bits, beta_bits))
            coeffs.append(float(coeff))

    # --- Random determinants ---
    def add_random(coeff, count=1):
        for _ in range(count):
            occ_alpha = rng.choice(n_orb, n_alpha, replace=False)
            occ_beta  = rng.choice(n_orb, n_beta, replace=False)

            class_name = str(DeterminantClass)
            if 'Determinant192' in class_name:
                n_segments = 3
            elif 'Determinant128' in class_name:
                n_segments = 2
            else:
                n_segments = 1

            alpha_array = [0] * n_segments
            beta_array  = [0] * n_segments

            for i in occ_alpha.tolist():
                seg, bit = divmod(i, 64)
                alpha_array[seg] |= (1 << bit)
            for i in occ_beta.tolist():
                seg, bit = divmod(i, 64)
                beta_array[seg]  |= (1 << bit)

            det = create_determinant(alpha_array, beta_array)
            dets.append(det)
            coeffs.append(float(coeff))

    # --- Random excited determinants ---
    def add_random_excited(config):
        coeff = float(config.get("coeff", 1.0))
        count = int(config.get("count", 1))
        min_level = int(config.get("min_level", 1))
        max_level = int(config.get("max_level", 1))
        random_coeffs = bool(config.get("random_coeffs", False))
        spin_preserve = bool(config.get("spin_preserve", False))

        ref_det = hf_reference()
        seen = set()

        for _ in range(count):
            level = rng.integers(min_level, max_level + 1)
            if spin_preserve:
                n_alpha_exc = level // 2
                n_beta_exc = level - n_alpha_exc
            else:
                n_alpha_exc = rng.integers(0, level + 1)
                n_beta_exc = level - n_alpha_exc

            alpha_occ = [i for i in range(n_orb) if (ref_det.alpha >> i) & 1]
            alpha_vir = [i for i in range(n_orb) if not (ref_det.alpha >> i) & 1]
            beta_occ  = [i for i in range(n_orb) if (ref_det.beta >> i) & 1]
            beta_vir  = [i for i in range(n_orb) if not (ref_det.beta >> i) & 1]

            if not (n_alpha_exc <= len(alpha_occ) and n_alpha_exc <= len(alpha_vir)):
                continue
            if not (n_beta_exc <= len(beta_occ) and n_beta_exc <= len(beta_vir)):
                continue

            chosen_alpha_occ = rng.choice(alpha_occ, size=n_alpha_exc, replace=False) if n_alpha_exc > 0 else []
            chosen_alpha_vir = rng.choice(alpha_vir, size=n_alpha_exc, replace=False) if n_alpha_exc > 0 else []
            chosen_beta_occ  = rng.choice(beta_occ,  size=n_beta_exc, replace=False) if n_beta_exc > 0 else []
            chosen_beta_vir  = rng.choice(beta_vir,  size=n_beta_exc, replace=False) if n_beta_exc > 0 else []

            alpha_bits, beta_bits = ref_det.alpha, ref_det.beta
            for o, v in zip(chosen_alpha_occ, chosen_alpha_vir):
                alpha_bits &= ~(1 << o)
                alpha_bits |=  (1 << v)
            for o, v in zip(chosen_beta_occ, chosen_beta_vir):
                beta_bits &= ~(1 << o)
                beta_bits |=  (1 << v)

            det = create_determinant(alpha_bits, beta_bits)
            key = (tuple(np.atleast_1d(det.alpha)), tuple(np.atleast_1d(det.beta)))
            if key in seen:
                continue
            seen.add(key)
            dets.append(det)
            coeffs.append(rng.normal(0, coeff) if random_coeffs else coeff)

    # --- Manual bitstring input ---
    def add_manual_bitstring(entries):
        if isinstance(entries, str):
            entries = [[1.0, entries]]
        elif isinstance(entries, (list, tuple)) and len(entries) == 2 and isinstance(entries[1], str):
            entries = [entries]
        elif isinstance(entries, list):
            if all(isinstance(e, str) for e in entries):
                entries = [[1.0, e] for e in entries]
            elif all(isinstance(e, (list, tuple)) and len(e) == 2 for e in entries):
                pass
            else:
                raise ValueError(f"Invalid bitstring format: {entries}")
        else:
            raise ValueError(f"Unsupported bitstring type: {type(entries)}")

        seen = set()
        for coeff_val, entry in entries:
            alpha_str, beta_str = entry.split(',', 1)
            alpha_occ = [int(x) for x in alpha_str.strip().split() if x.strip().isdigit()]
            beta_occ  = [int(x) for x in beta_str.strip().split() if x.strip().isdigit()]

            # build bitmasks (supports >64 orbitals)
            n_segments = 3 if '192' in str(DeterminantClass) else (2 if '128' in str(DeterminantClass) else 1)
            alpha_array = [0] * n_segments
            beta_array  = [0] * n_segments
            for i in alpha_occ:
                seg, bit = divmod(i, 64)
                alpha_array[seg] |= (1 << bit)
            for i in beta_occ:
                seg, bit = divmod(i, 64)
                beta_array[seg]  |= (1 << bit)

            det = create_determinant(alpha_array, beta_array)
            key = (tuple(np.atleast_1d(det.alpha)), tuple(np.atleast_1d(det.beta)))
            if key in seen:
                continue
            seen.add(key)
            dets.append(det)
            coeffs.append(float(coeff_val))

    # --- Parse input dict ---
    if initial_dets_dict:
        for kind, val in initial_dets_dict.items():
            if kind == "bitstring":
                add_manual_bitstring(val)
            elif isinstance(val, (int, float)):
                if kind == "random":
                    add_random(val, 1)
                else:
                    add_preset(kind, val, 1)
            elif isinstance(val, (list, tuple)):
                if kind == "random":
                    coeff, count = val
                    add_random(coeff, int(count))
                else:
                    coeff, count = val
                    add_preset(kind, coeff, int(count))
            elif isinstance(val, dict) and kind == "random_excited":
                add_random_excited(val)
            else:
                raise ValueError(f"Unsupported value for {kind}: {val}")

    # --- Normalize coefficients ---
    norm = np.sqrt(sum(c**2 for c in coeffs))
    if norm > 0:
        coeffs = [c / norm for c in coeffs]

    # --- Optional save ---
    if save_path:
        alpha_arr = np.array([np.atleast_1d(d.alpha) for d in dets], dtype=np.uint64)
        beta_arr  = np.array([np.atleast_1d(d.beta)  for d in dets], dtype=np.uint64)
        np.savez_compressed(save_path,
                            dets=np.column_stack([alpha_arr[:,0], beta_arr[:,0]]),
                            dets_coeffs=np.array(coeffs),
                            core_set=np.column_stack([alpha_arr[:,0], beta_arr[:,0]]),
                            core_set_coeffs=np.array(coeffs))



    return dets, coeffs


# ========== Multi-Run Report Generation ==========
def generate_multi_run_report(all_results, best_result, mol_name, args, final_run: bool = False):
    """
    Generate a comprehensive report for multiple runs in Markdown format.
    final_run: when True, highlight the final calculation and separate exploration.
    """
    from datetime import datetime
    
    report_filename = f"multi_run_report_{mol_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    
    try:
        with open(report_filename, 'w', encoding='utf-8') as f:
            # Header
            title = "# TrimCI Final Run Report" if final_run else "# TrimCI Multi-Run Report"
            f.write(title + "\n\n")
            f.write(f"**Molecule:** {mol_name}  \n")
            f.write(f"**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
            f.write(f"**Total Runs:** {len(all_results)}  \n\n")
            
            # Configuration Summary
            f.write("## Configuration Summary\n\n")
            f.write(f"- **Number of Runs:** {getattr(args, 'num_runs', 1)}\n")
            f.write(f"- **Max Iterations:** {getattr(args, 'exp_max_iterations', 'N/A')}\n")
            f.write(f"- **Energy Threshold:** {getattr(args, 'exp_energy_threshold', 'N/A')}\n")
            f.write(f"- **Initial Pool Size:** {getattr(args, 'initial_pool_size', 'N/A')}\n")
            f.write(f"- **Max Final Determinants:** {getattr(args, 'max_final_dets', 'N/A')}\n\n")
            
            # Results Summary
            f.write("## Results Summary\n\n")
            
            # Best/Final result highlight
            f.write("### 🏆 Final Result\n" if final_run else "### 🏆 Best Result\n")
            f.write(f"- **Run:** {best_result['run_idx']}\n")
            f.write(f"- **Final Energy:** {best_result['final_energy']:.8f} Hartree\n")
            f.write(f"- **Number of Determinants:** {len(best_result['current_dets'])}\n")
            best_time = best_result['iteration_details'].get('total_time', 'N/A')
            if isinstance(best_time, (int, float)):
                f.write(f"- **Calculation Time:** {best_time:.1f} seconds\n")
            else:
                f.write(f"- **Calculation Time:** {best_time}\n")
            f.write(f"- **Results Directory:** `{best_result['results_dir']}`\n\n")
            
            # All runs table
            f.write("### All Runs Comparison\n\n")
            f.write("| Run | Final Energy (Hartree) | Determinants | Time (s) | Status |\n")
            f.write("|-----|------------------------|--------------|----------|--------|\n")
            
            energies = [result['final_energy'] for result in all_results]
            times = []
            min_energy = min(energies)
            max_energy = max(energies)
            
            for result in all_results:
                energy = result['final_energy']
                n_dets = len(result['current_dets'])
                calc_time = result['iteration_details'].get('total_time', 'N/A')
                times.append(calc_time if isinstance(calc_time, (int, float)) else None)
                
                if isinstance(calc_time, (int, float)):
                    time_str = f"{calc_time:.1f}"
                else:
                    time_str = str(calc_time)
                
                if final_run:
                    status = "🚀 Final" if str(result.get('label','')) == 'final_ndets' else "Explored"
                else:
                    status = "🏆 Best" if result['run_idx'] == best_result['run_idx'] else "Completed"
                f.write(f"| {result['run_idx']} | {energy:.8f} | {n_dets} | {time_str} | {status} |\n")
            
            f.write("\n")
            
            # Add legend when final_run
            if final_run:
                f.write("Status legend: 🚀 Final = final_ndets, Explored = exploration variants.\n\n")
                
                # Run phases section
                f.write("## Run Phases\n\n")
                
                # Exploration Phase summary
                explore_runs = [r for r in all_results if str(r.get('label','')) != 'final_ndets']
                f.write("### Exploration Phase\n\n")
                f.write(f"- **Exploration Runs:** {len(explore_runs)}\n")
                f.write(f"- **Goal:** {getattr(args, 'goal', 'balanced')}\n")
                if explore_runs:
                    try:
                        min_exp_energy = min([rr['final_energy'] for rr in explore_runs])
                        f.write(f"- **Best Exploration Energy:** {min_exp_energy:.8f} Hartree\n")
                    except Exception:
                        pass
                f.write("\n")
                
                # Final Calculation summary
                f.write("### Final Calculation\n\n")
                f.write(f"- **Final Run Index:** {best_result['run_idx']}\n")
                f.write(f"- **Final Energy:** {best_result['final_energy']:.8f} Hartree\n")
                try:
                    final_args = best_result.get('run_args', None)
                    if final_args is not None:
                        f.write(f"- **Target Determinants:** {getattr(final_args, 'max_final_dets', 'N/A')}\n")
                except Exception:
                    pass
                f.write("\n")
            
            # Statistics
            f.write("### Statistical Analysis\n\n")
            energy_std = np.std(energies)
            energy_range = max_energy - min_energy
            f.write(f"- **Energy Range:** {energy_range:.8f} Hartree\n")
            f.write(f"- **Energy Standard Deviation:** {energy_std:.8f} Hartree\n")
            f.write(f"- **Average Energy:** {np.mean(energies):.8f} Hartree\n")
            f.write(f"- **Median Energy:** {np.median(energies):.8f} Hartree\n\n")
            
            # Time statistics
            valid_times = [t for t in times if t is not None]
            if valid_times:
                total_computation_time = sum(valid_times)
                f.write("#### Timing Statistics\n\n")
                f.write(f"- **Total Computation Time:** {total_computation_time:.1f} seconds\n")
                f.write(f"- **Average Time per Run:** {np.mean(valid_times):.1f} seconds\n")
                f.write(f"- **Median Time per Run:** {np.median(valid_times):.1f} seconds\n")
                f.write(f"- **Time Range:** {min(valid_times):.1f} - {max(valid_times):.1f} seconds\n")
                if len(valid_times) > 1:
                    f.write(f"- **Time Standard Deviation:** {np.std(valid_times):.1f} seconds\n")
                f.write("\n")
            
            # Convergence information for best run
            if 'iterations' in best_result['iteration_details']:
                f.write("### Best Run Convergence Details\n\n")
                iterations = best_result['iteration_details']['iterations']
                if iterations:
                    f.write(f"- **Total Iterations:** {len(iterations)}\n")
                    f.write(f"- **Final Iteration Energy:** {iterations[-1].get('energy', 'N/A')}\n")
                    f.write(f"- **Final Core Set Size:** {iterations[-1].get('core_set_size', 'N/A')}\n\n")
            
            # Footer
            f.write("---\n")
            f.write("*Report generated by TrimCI Multi-Run Analysis*\n")
        
        log_important(f"📄 Multi-run report saved as: {report_filename}")
        
    except Exception as e:
        log_verbose(f"⚠️ Failed to generate multi-run report: {e}")

