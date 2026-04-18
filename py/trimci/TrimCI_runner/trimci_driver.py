"""
TrimCI driver script
High-level workflow for running TrimCI.
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
# Colors for non-logging output
RED_BOLD = "\033[1;31m"
RESET = "\033[0m"

# Import unified logging functions
from trimci.trimci_logging import setup_logging, log_important, log_verbose, get_verbosity_from_args


def generate_unique_timestamp() -> str:
    """
    Generate a unique timestamp string for folder naming.
    
    Uses microsecond precision + PID to prevent collisions when multiple
    jobs start simultaneously (e.g., Slurm array jobs).
    
    Format: YYYYMMDD_HHMMSSffffff_PID (6 digits microseconds + PID)
    
    Returns:
        str: Unique timestamp like "20260125_033041123456_12345"
    """
    # Use full microseconds (6 digits) for uniqueness within same process
    ts = datetime.now().strftime("%Y%m%d_%H%M%S%f")  # Includes 6 digit microseconds
    # Add PID for extra uniqueness in HPC environments (different Slurm jobs)
    pid_suffix = f"_{os.getpid()}"
    return ts + pid_suffix


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

    log_verbose(f"🔍 Detected NORB={n_orb}, NELEC={n_elec}, N_ALPHA={n_alpha}, N_BETA={n_beta}, MS2={ms2}, PSYM={psym}")

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
    return h1, eri, n_elec, n_orb, nuclear_repulsion, n_alpha, n_beta, psym

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
    ms2 = fcidump_data['MS2']
    n_alpha = int((n_elec + ms2) // 2)
    n_beta = int((n_elec - ms2) // 2)
    
    
    # Convert compressed ERI to 4D tensor format
    # PySCF returns ERI in compressed format, need to restore to (n_orb, n_orb, n_orb, n_orb)
    eri = ao2mo.restore(1, eri_compressed, n_orb)
    
    log_verbose(f"🔍 Detected NORB={n_orb}, NELEC={n_elec}")
    
    return h1, eri, n_elec, n_orb, nuclear_repulsion, n_alpha, n_beta

# ========== Main Workflow ==========
def run_full(fcidump_path: str = None,
             molecule: str = None, basis: str = "sto-3g", spin: int = 0,
             trimci_config_path: str = None, config_dict: dict = None, **overrides):
    """
    MAIN ENTRY POINT for all TrimCI calculations.
    
    This function is the unified entry point that dispatches to different execution modes
    based on configuration. All external scripts should call this function.
    
    Execution Branches & Log Files:
    ================================
    
    1. PARALLEL MODE (n_parallel > 1)
       → run_trimci_main_calculation() 
         → run_trimci_main_calculation() → parallel [parallel_runner.py]
       → Log: multi_summary.log
       → Nests: single runs in subprocesses
    
    2. SEQUENTIAL MULTI-RUN MODE (n_parallel == 1, num_runs > 1)
       → run_trimci_main_calculation()
         → run_trimci_main_calculation_single() [multi-run branch]
       → Log: multi_summary.log
       → Nests: single runs sequentially
    
    3. SINGLE RUN MODE (n_parallel == 1, num_runs == 1)
       → run_trimci_main_calculation()
         → run_trimci_main_calculation_single() [single branch]
       → Log: single_summary.log
    
    Nesting Structure:
    ==================
    multi_summary.log (parallel or sequential)
    └── single_summary.log (per run, if verbose)
    
    Key Config Parameters:
    =====================
    - n_parallel: int → number of parallel workers (>1 enables parallel mode)
    - num_runs: int → number of runs (>1 enables multi-run ensemble)
    - threshold, pool_core_ratio, max_final_dets: TrimCI algorithm params
    - core_set_schedule: list → determinant schedule e.g. [10, 100, 1000]
    - initial_dets_dict: dict → initial determinant configuration
    
    Returns:
        Tuple: (final_energy, dets, coeffs, details, run_args)
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
    Internal implementation of run_full(). See run_full() for documentation.
    """
    if fcidump_path is None and molecule is None:
        raise ValueError("Either fcidump_path or molecule must be provided")
    if fcidump_path and molecule:
        raise ValueError("fcidump_path and molecule are mutually exclusive")

    if fcidump_path:
        h1, eri, n_elec, n_orb, nuclear_repulsion, n_alpha, n_beta, psym = read_fcidump(fcidump_path)

        # Check for force_spinless override
        force_spinless = overrides.get('force_spinless', False) or (config_dict and config_dict.get('force_spinless', False))
        if force_spinless:
            log_important(f"⚠️ Force spinless mode enabled: Setting n_beta=0, n_alpha={n_elec}")
            n_alpha = n_elec
            n_beta = 0

        args = load_configurations(str(Path(fcidump_path).parent), trimci_config_path)
        # if getattr(args, "debug", False):
        #     print("debug mode enabled")
        #     h1_pyscf, eri_pyscf, n_elec_pyscf, n_orb_pyscf, nuclear_repulsion_pyscf, n_alpha_pyscf, n_beta_pyscf = read_fcidump2(fcidump_path)
        #     print(f"🔍 Debug mode enabled. Using PySCF FCIDUMP reader.")
        #     print(f"🔍 Detected NORB={n_orb_pyscf}, NELEC={n_elec_pyscf}")
        #     # Compare results from the two FCIDUMP readers
        #     if not np.allclose(h1, h1_pyscf):
        #         print(f"{RED_BOLD}WARNING: H1 integrals differ between custom and PySCF FCIDUMP readers!{RESET}")
        #         print(f"  Max absolute difference in H1: {np.max(np.abs(h1 - h1_pyscf)):.6e}")
        #     if not np.allclose(eri, eri_pyscf):
        #         print(f"{RED_BOLD}WARNING: ERI integrals differ between custom and PySCF FCIDUMP readers!{RESET}")
        #         print(f"  Max absolute difference in ERI: {np.max(np.abs(eri - eri_pyscf)):.6e}")
        #     if not np.isclose(nuclear_repulsion, nuclear_repulsion_pyscf):
        #         print(f"{RED_BOLD}WARNING: Nuclear repulsion energy differs between custom and PySCF FCIDUMP readers!{RESET}")
        #         print(f"  Custom: {nuclear_repulsion:.6f}, PySCF: {nuclear_repulsion_pyscf:.6f}")

        #     # Use PySCF's parsed values for the rest of the calculation if debug is on
        #     h1, eri, n_elec, n_orb, nuclear_repulsion, n_alpha, n_beta = h1_pyscf, eri_pyscf, n_elec_pyscf, n_orb_pyscf, nuclear_repulsion_pyscf, n_alpha_pyscf, n_beta_pyscf

        # Apply explicit config dictionary if provided (higher precedence than file)
        if config_dict:
            for key, value in config_dict.items():
                setattr(args, key, value)
        # Override parameters
        for key, value in overrides.items():
            setattr(args, key, value)
        args.psym = psym
        # Add source info to args for logging
        args.fcidump_path = fcidump_path
        args.nuclear_repulsion = nuclear_repulsion
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
        # Add source info to args for logging
        args.molecule_spec = molecule
        args.nuclear_repulsion = nuclear_repulsion

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
    verbosity = getattr(args, 'verbosity', 1 if getattr(args, 'verbose', False) else 0)
    if verbosity >= 1:
        print(args)
    n_total = int(comb(n_orb, n_alpha) * comb(n_orb, n_beta))

    # Auto-determine max_final_dets if set to "auto"
    # Target: ~10s on modern laptop. Heuristic based on Davidson O(n_dets^2 * n_orb^2)
    max_final_dets = getattr(args, 'max_final_dets', None)
    if max_final_dets == "auto" or max_final_dets == -1:
        # Empirical formula: scales inversely with n_orb^1.5
        # n_orb=10 → ~300 dets, n_orb=20 → ~100 dets, n_orb=40 → ~35 dets
        auto_dets = int(3000 / (n_orb ** 1.5))
        auto_dets = max(50, min(500, auto_dets))  # Clamp to [50, 500]
        auto_dets = min(auto_dets, n_total)  # Don't exceed total possible
        log_important(f"🔧 max_final_dets='auto': n_orb={n_orb} → max_final_dets={auto_dets}")
        args.max_final_dets = auto_dets
    elif max_final_dets is not None and max_final_dets > n_total:
        # Guard: auto-adjust if exceeds maximum possible determinants
        log_important(f"⚠️ max_final_dets ({max_final_dets}) exceeds total ({n_total}). Adjusting to {n_total}.")
        args.max_final_dets = n_total

    return run_trimci_main_calculation(h1, eri, n_alpha, n_beta, n_orb, mol_name, args, nuclear_repulsion)


def run_trimci_main_calculation(h1, eri, n_alpha, n_beta, n_orb, mol_name, args, nuclear_repulsion, folder=None):
    """
    Smart wrapper that dispatches to single or parallel execution based on args.n_parallel.
    
    If args.n_parallel > 1, uses spawn subprocesses for parallel execution.
    Otherwise, uses the single-threaded version.
    """
    n_parallel = getattr(args, 'n_parallel', 1)
    
    if n_parallel > 1:
        # Use parallel version
        from .parallel_runner import run_trimci_main_calculation_parallel
        return run_trimci_main_calculation_parallel(
            h1, eri, n_alpha, n_beta, n_orb, mol_name, args, nuclear_repulsion,
            folder=folder,
            num_runs=getattr(args, 'num_runs', 50),
            n_parallel=n_parallel,
            omp_per_run=getattr(args, 'omp_per_run', None)
        )
    else:
        # Use single-threaded version
        return run_trimci_main_calculation_single(h1, eri, n_alpha, n_beta, n_orb, mol_name, args, nuclear_repulsion, folder)


def run_trimci_main_calculation_single(h1, eri, n_alpha, n_beta, n_orb, mol_name, args, nuclear_repulsion, folder=None):
    # Check if multiple runs are requested
    num_runs = getattr(args, 'num_runs', 1)
    
    if num_runs > 1:
        import shutil
        log_important(f"🔄 Running {num_runs} independent calculations to find the best result...")
        
        best_energy = float('inf')
        best_result = None
        all_results = []
        all_results_dirs = []

        unique_ts = generate_unique_timestamp()
        if folder is None:
            multi_run_folder = str(Path("trimci_multi_run_results") / f"{mol_name}_{unique_ts}")
        else:
            multi_run_folder = str(Path(folder) / f"multi_{mol_name}_{unique_ts}")
        
        # Setup file logging
        Path(multi_run_folder).mkdir(parents=True, exist_ok=True)
        log_file = os.path.join(multi_run_folder, "realtime_progress.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.WARNING)
        file_handler.setFormatter(logging.Formatter("%(asctime)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        logging.getLogger().addHandler(file_handler)
        
        # Setup unified summary log using TrimCISummaryLogger
        from .summary_logger import TrimCISummaryLogger
        args_dict = vars(args).copy() if hasattr(args, '__dict__') else dict(args)
        summary_logger = TrimCISummaryLogger(os.path.join(multi_run_folder, "multi_summary.log"), mode="multi")
        summary_logger.write_header(mol_name, n_orb, n_alpha, n_beta, args_dict, num_runs=num_runs)
        
        multi_start_time = time.perf_counter()
        
        for run_idx in range(num_runs):
            start_total = time.perf_counter()
            log_important(f"📊 Starting run {run_idx + 1}/{num_runs}")
            
            # Run the calculation
            final_energy, current_dets, current_coeffs, iteration_details, run_args = iterative_workflow(
                h1, eri, n_alpha, n_beta, n_orb, mol_name, args, nuclear_repulsion, start_total, multi_run_folder
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

            # Log to summary using TrimCISummaryLogger
            run_elapsed = time.perf_counter() - start_total
            n_iters = iteration_details.get('total_iterations')
            summary_logger.log_run_complete(run_idx + 1, final_energy, run_elapsed, n_iters)
            
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
        # Sort results by energy
        all_results.sort(key=lambda x: x['final_energy'])
        
        num_keep = getattr(args, 'num_runs_keep_top_k', 1)
        top_k_results = all_results[:num_keep]
        top_k_indices = set(r['run_idx'] for r in top_k_results)
        
        log_important(f"🧹 Keeping top {num_keep} results (Runs: {sorted(list(top_k_indices))})")

        for result in all_results:
            if result['run_idx'] not in top_k_indices and result['results_dir']:
                try:
                    if os.path.exists(result['results_dir']):
                        shutil.rmtree(result['results_dir'])
                        log_verbose(f"🗑️ Removed non-top-{num_keep} results directory: {result['results_dir']}")
                except Exception as e:
                    log_verbose(f"⚠️ Failed to remove directory {result['results_dir']}: {e}")
        
        # Combine determinants from top k results
        if num_keep > 0:
            log_important(f"🧩 Combining determinants from top {num_keep} runs...")
            
            # Dictionary to store unique determinants and their coefficients
            # Key: (alpha, beta) tuple, Value: (determinant object, coefficient)
            unique_dets_map = {}
            
            # Iterate through results (already sorted by energy)
            for result in top_k_results:
                current_dets = result['current_dets']
                current_coeffs = result['current_coeffs']
                
                for det, coeff in zip(current_dets, current_coeffs):
                    # Normalize to tuple for map key
                    if isinstance(det.alpha, list):
                        key = (tuple(det.alpha), tuple(det.beta))
                    else:
                        key = (det.alpha, det.beta)
                    
                    # If not present, add it. Since results are sorted by energy, 
                    # we keep the coefficient from the best run that had this determinant.
                    if key not in unique_dets_map:
                        unique_dets_map[key] = (det, coeff)
            
            unique_dets_list = [v[0] for v in unique_dets_map.values()]
            unique_coeffs_list = [v[1] for v in unique_dets_map.values()]
            
            log_important(f"🧩 Combined {len(unique_dets_list)} unique determinants.")
            
            # Check if renormalization is requested
            if getattr(args, 'num_runs_keep_top_renormalize', False):
                log_important("🔄 Renormalizing combined determinants using trim()...")
                try:
                    # Prepare arguments for trim
                    # We use get_functions_for_system to get the correct run_trim function
                    # and handle potential version mismatch regarding the 'tol' parameter
                    
                    funcs = get_functions_for_system(n_orb)
                    run_trim_func = funcs['run_trim']
                    
                    # We want to keep all of them, so keep_size = len(unique_dets_list)
                    # group_size = 1 as requested
                    group_sizes = [1]
                    keep_sizes = [len(unique_dets_list)]
                    
                    # Try calling with tol (new version)
                    try:
                        t_energy, t_dets, t_coeffs = run_trim_func(
                            unique_dets_list, h1, eri, mol_name, n_alpha+n_beta, n_orb,
                            group_sizes, keep_sizes, 
                            False, False, [], 1e-3
                        )
                    except TypeError:
                        # Fallback to old version without tol
                        t_energy, t_dets, t_coeffs = run_trim_func(
                            unique_dets_list, h1, eri, mol_name, n_alpha+n_beta, n_orb,
                            group_sizes, keep_sizes, 
                            False, False, []
                        )
                    
                    # Update coefficients and determinants
                    unique_dets_list = t_dets
                    unique_coeffs_list = t_coeffs
                    log_important(f"✅ Renormalization complete. Energy: {t_energy:.8f}")
                    
                except Exception as e:
                    log_important(f"⚠️ Renormalization failed: {e}")
                    # Fallback to simple normalization
                    norm = np.linalg.norm(unique_coeffs_list)
                    if norm > 1e-12:
                        unique_coeffs_list = [c/norm for c in unique_coeffs_list]
                    log_important("⚠️ Used simple normalization instead.")
            else:
                # Simple normalization if not re-optimizing
                norm = np.linalg.norm(unique_coeffs_list)
                if norm > 1e-12:
                    unique_coeffs_list = [c/norm for c in unique_coeffs_list]
            
            # Save combined determinants
            dets_combine_path = os.path.join(multi_run_folder, "dets_combine.npz")
            try:
                # Format matching save_final_results but without core_set/core_set_coeffs
                # np.savez_compressed(npz_path,
                #                     dets=dets_to_array(current_dets),
                #                     dets_coeffs=np.array(final_coeffs),
                #                     core_set_coeffs=np.array(current_coeffs),
                #                     core_set=dets_to_array(current_core_set))
                
                np.savez_compressed(dets_combine_path, 
                                    dets=dets_to_array(unique_dets_list),
                                    dets_coeffs=np.array(unique_coeffs_list))
                log_important(f"💾 Saved combined determinants to {dets_combine_path}")
            except Exception as e:
                log_important(f"⚠️ Failed to save combined determinants: {e}")
            
            # Save individual run dets/coeffs for NOCI downstream use
            dets_multi_run_path = os.path.join(multi_run_folder, "dets_multi_run.npz")
            try:
                # Build list of wavefunctions, each as a dict with dets, coeffs, energy
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
                log_important(f"💾 Saved {num_keep} wavefunctions to {dets_multi_run_path}")
            except Exception as e:
                log_important(f"⚠️ Failed to save multi-run dets: {e}")
        
        # Prepare combined results info
        combined_results_info = {
            'num_combined_dets': len(unique_dets_list),
            'renormalized': getattr(args, 'num_runs_keep_top_renormalize', False),
            'file_path': dets_combine_path
        }
        if getattr(args, 'num_runs_keep_top_renormalize', False) and 't_energy' in locals():
            combined_results_info['renormalized_energy'] = t_energy
            
        # Calculate top 10 determinants for combined results
        combined_results_info['top_10_determinants'] = get_top_determinants(unique_dets_list, unique_coeffs_list, top_n=10)
        
        # Generate multi-run report
        generate_multi_run_report(all_results, best_result, mol_name, args, 
                                  folder=multi_run_folder, 
                                  combined_results_info=combined_results_info)
        
        # Mark RUN_FULL end with summary
        try:
            _ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open("realtime_progress.out", "a", encoding="utf-8") as f:
                f.write(f"==== [ {_ts} ] RUN_FULL END ====\n")
                f.write(f"Summary: best=run_{best_result['run_idx']} | energy: {best_result['final_energy']:.8f} | kept_dir: {best_results_dir}\n\n")
        except Exception:
            # Non-fatal: ignore file writing errors
            pass
        
        # Write final summary to multi_summary.log
        total_elapsed = time.perf_counter() - multi_start_time
        # Add elapsed time to each result for summary
        for r in all_results:
            r['elapsed'] = r.get('elapsed', 0)
        summary_logger.write_multi_summary(all_results, best_result, total_elapsed)
        summary_logger.close()
        
        # Teardown file logging
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()

        # Return the best result
        return (best_result['final_energy'], best_result['current_dets'], 
                best_result['current_coeffs'], best_result['iteration_details'], 
                best_result['run_args'])
    else:
        # Single run
        start_total = time.perf_counter()
        if folder is None:
            single_run_folder = str(Path("trimci_single_run_results"))
        else:
            single_run_folder = str(Path(folder))

        # Setup file logging
        Path(single_run_folder).mkdir(parents=True, exist_ok=True)
        log_file = os.path.join(single_run_folder, "realtime_progress.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.WARNING)
        file_handler.setFormatter(logging.Formatter("%(asctime)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        logging.getLogger().addHandler(file_handler)
        
        # Setup single_summary.log
        from .summary_logger import TrimCISummaryLogger
        args_dict = vars(args).copy() if hasattr(args, '__dict__') else dict(args)
        summary_logger = TrimCISummaryLogger(os.path.join(single_run_folder, "single_summary.log"), mode="single")
        summary_logger.write_header(mol_name, n_orb, n_alpha, n_beta, args_dict)
        
        _fe, _cd, _cc, _id, _ra = iterative_workflow(h1, eri, n_alpha, n_beta, n_orb, mol_name,
                                  args, nuclear_repulsion, start_total, single_run_folder)
        
        _elapsed = time.perf_counter() - start_total
        n_dets = len(_cd) if _cd else 0
        n_iters = _id.get('total_iterations', 0)
        
        # Write single_summary.log
        summary_logger.write_single_summary(_fe, n_dets, n_iters, _elapsed, _id.get('results_dir', ''))
        summary_logger.close()
        
        # Record FINAL RUN END (legacy)
        try:
            _ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open("realtime_progress.out", "a", encoding="utf-8") as f:
                f.write(f"---- [ {_ts} ] FINAL RUN END ----\n")
                f.write(f"label: single | energy: {_fe:.8f} | ndets: {getattr(args, 'max_final_dets', 'N/A')} | elapsed_s: {_elapsed:.2f}\n")
        except Exception:
            pass
        # Mark RUN_FULL END (legacy)
        try:
            _ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open("realtime_progress.out", "a", encoding="utf-8") as f:
                f.write(f"==== [ {_ts} ] RUN_FULL END ====\n")
                f.write(f"Summary: single | energy: {_fe:.8f} | kept_dir: {_id.get('results_dir', '')}\n\n")
        except Exception:
            pass
        # Teardown file logging
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()
        return (_fe, _cd, _cc, _id, _ra)

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
        h1, eri, n_elec, n_orb, nuclear_repulsion, n_alpha, n_beta, _ = read_fcidump(fcidump_path)

        # Check for force_spinless override
        force_spinless = overrides.get('force_spinless', False) or (config_dict and config_dict.get('force_spinless', False))
        if force_spinless:
            log_important(f"⚠️ Force spinless mode enabled: Setting n_beta=0, n_alpha={n_elec}")
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

# ========== Basic Workflow ==========
def iterative_workflow_py(h1, eri, n_alpha, n_beta, n_orb,
                       system_name, args, nuclear_repulsion,
                       start_time=None, results_dir="trimci_results"):
    """
    Main TrimCI iterative workflow.
    
    This function implements the core TrimCI algorithm: an iterative method 
    that efficiently builds a compact, variationally CI-based wavefunction.
    
    Algorithm Overview:
    ===================
    STEP 0: Core Set Initialization
            Initialize starting determinants from file, dict, or HF reference.
    
    [BEGIN ITERATIVE LOOP]
        STEP 1: Pool Construction via Heat-Bath Screening
                Expand core set by generating connected determinants with H_ij*c_j weighting.
        
        STEP 2: Local Trim Parameter Setup
                Configure hierarchical trimming (m groups, k survivors per group).
        
        STEP 3: Subspace Diagonalization & Selection (TRIM Core)
                Partition → Diagonalize → Keep top-k → Merge → Re-diagonalize.
        
        STEP 4: Energy Accounting & Statistics
                Compute E_total, track ΔE (decision in STEP 6).
        
        STEP 5: Core Set Growth & Preparation for Next Iteration
                Sort by |c_i|, grow core set (via schedule or ratio), normalize.
        
        STEP 6: Termination Conditions
                Break if converged, schedule exhausted, max_iterations, or max_final_dets.
    [END ITERATIVE LOOP]
    
    FINALIZATION: Post-Processing & Result Assembly
                  Run final Davidson on core set, save results to disk.
    
    Args:
        h1: One-body integrals (n_orb x n_orb)
        eri: Two-body integrals (flattened or 4D)
        n_alpha, n_beta: Number of alpha/beta electrons
        n_orb: Number of orbitals
        system_name: System identifier for logging and file naming (e.g., "H6", "Fe4S4")
        args: Configuration namespace with workflow parameters
            Key parameters:
            - core_set_schedule: List[int], optional. Explicit core set sizes per iteration.
                Example: [10, 20, 50, 100] -> iter 1: 10 dets, iter 2: 20 dets, etc.
                When provided, overrides core_set_ratio. Workflow terminates when exhausted.
            - core_set_ratio: float or List[float]. Growth factor per iteration (default).
            - max_final_dets: int. Hard limit on core set size (ignored if schedule is provided).
            - threshold: float. Heat-bath screening threshold.
        nuclear_repulsion: Nuclear repulsion energy
        start_total: Start time for timing
        results_dir: Output directory
    
    Returns:
        (final_energy, dets, coeffs, iteration_details, args)
        
        dets:   List of determinants (sorted by |coeff|)
        coeffs: List of CI coefficients (sorted by |coeff|)
    """
    # Handle start_time = None (default to current time)
    if start_time is None:
        start_time = time.perf_counter()
    
    # Deprecation warning
    import warnings
    warnings.warn(
        "iterative_workflow_py is deprecated. Use iterative_workflow() which calls C++ backend.",
        DeprecationWarning,
        stacklevel=2
    )

    unique_ts = generate_unique_timestamp()
    results_dir = str(Path(results_dir) / f"{system_name}_{unique_ts}")
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    log_verbose(f"📁 Results will be saved in: {results_dir}")

    # Flatten ERI if needed (for C++ core compatibility)
    if hasattr(eri, "reshape") and hasattr(eri, "ndim") and eri.ndim == 4:
        eri = eri.reshape(-1)
        log_verbose("ℹ️  Flattened ERI tensor for C++ core.")

    # =========================================================================
    # Parameter Extraction from args
    # =========================================================================
    
    # --- Termination Conditions ---
    max_iterations = getattr(args, 'max_iterations', 
                             getattr(args, 'exp_max_iterations', -1))
    energy_threshold = getattr(args, 'energy_threshold',
                               getattr(args, 'exp_energy_threshold', 1e-12))
    max_final_dets = getattr(args, 'max_final_dets', None)

    # Handle max_iterations = -1 (unlimited, rely on other termination conditions)
    if max_iterations == -1:
        if max_final_dets is None and core_set_schedule is None:
            raise ValueError("max_iterations=-1 requires max_final_dets or core_set_schedule")
        max_iterations = 200000  # Safety cap

    # --- Core Set Growth ---
    _core_set_ratio = getattr(args, 'core_set_ratio', 2)
    initial_pool_size = getattr(args, 'initial_pool_size', 100)

    # Core set schedule: explicit list of core sizes per iteration
    # e.g., [10, 20, 50, 100] means: after iter 1 -> 10 dets, iter 2 -> 20 dets, etc.
    # When the schedule is exhausted, the workflow terminates.
    core_set_schedule = getattr(args, 'core_set_schedule', None)
    if core_set_schedule is not None:
        core_set_schedule = list(core_set_schedule)  # Ensure it's a mutable list
        log_important(f"📋 Using core_set_schedule: {core_set_schedule}")

    # --- Pool Building (STEP 1) ---
    pool_core_ratio = getattr(args, 'pool_core_ratio', 10)
    pool_build_strategy = getattr(args, 'pool_build_strategy', 'heat_bath')
    threshold = getattr(args, 'threshold', 0.01)  # Heat-bath screening threshold
    threshold_decay = getattr(args, 'threshold_decay', 0.9)
    max_rounds = getattr(args, 'max_rounds', 1)
    # Attentive TrimCI: restrict excitations to specific orbitals
    attentive_orbitals = getattr(args, 'attentive_orbitals', None)
    # Strategy factor: pre-filter multiplier for PT2 modes
    # -1 = automatic (1 for heat_bath, 20 for PT2 modes)
    strategy_factor = getattr(args, 'strategy_factor', -1)
    
    # DMRG-style noise injection for escaping local minima
    # noise_strength > 0 adds Gaussian noise to coefficients before sorting
    # Typical values: 0.01-0.1 for early phases, 0 for final phases
    noise_strength = getattr(args, 'noise_strength', 0.0)
    
    # --- TRIM Parameters (STEP 2-3) ---
    num_groups_base = getattr(args, 'num_groups', 10)
    # num_groups_ratio: when > 0, num_groups = max(num_groups, core_set_size * ratio)
    # This allows adaptive group count scaling with wavefunction size
    num_groups_ratio = getattr(args, 'num_groups_ratio', 0)
    # local_trim_keep_ratio: total raw_dets = core_set * ratio, then k = raw_dets / num_groups
    local_trim_keep_ratio = getattr(args, 'local_trim_keep_ratio',
                                    getattr(args, 'keep_pool_to_next_core_ratio', 0))
    keep_ratio = getattr(args, 'keep_ratio', 0.1)  # Legacy, prefer local_trim_keep_ratio

    # --- Saving & Debug ---
    save_period = getattr(args, 'save_period', 50)
    save_initial = getattr(args, 'save_initial', False)
    save_pool = getattr(args, 'save_pool', False)
    debug = getattr(args, 'debug', False)
    






    # =============================================================================
    # STEP 0: Core Set Initialization
    # =============================================================================
    # Initialize the starting determinants for the iterative TrimCI algorithm.
    # Sources (in priority order):
    #   1. Load from file if load_initial_dets=True
    #   2. Generate from initial_dets_dict (e.g., HF + random configurations)
    #   3. Fallback to single reference (HF) determinant
    load_initial_dets_num = getattr(args, 'load_initial_dets_num', None)  # None = load all
    
    if getattr(args, 'load_initial_dets', False):
        dets_path = getattr(args, "initial_dets_path", "dets.npz")
        log_important(f"🔄 Loading initial determinants from {dets_path}")
        dets_array_name = getattr(args, "dets_array_name", "dets")
        loaded_core_set, loaded_coeffs = load_initial_dets_from_file(
            dets_path, core_set=(dets_array_name != "dets"))
            
        if loaded_core_set is not None and loaded_coeffs is not None:
            # Apply top-k truncation if load_initial_dets_num is specified
            if load_initial_dets_num is not None and load_initial_dets_num < len(loaded_core_set):
                original_count = len(loaded_core_set)
                # Sort by |coefficient| descending and take top-k
                sorted_idx = np.argsort(np.abs(loaded_coeffs))[::-1][:load_initial_dets_num]
                loaded_core_set = [loaded_core_set[i] for i in sorted_idx]
                loaded_coeffs = [loaded_coeffs[i] for i in sorted_idx]
                log_important(f"✅ Loaded top-{load_initial_dets_num} determinants (truncated from {original_count})")
            else:
                log_important(f"✅ Loaded {len(loaded_core_set)} determinants")
            current_core_set = loaded_core_set
            current_core_coeffs = loaded_coeffs
        else:
            det_ref = generate_reference_det(n_alpha, n_beta, n_orb)
            log_important(f"🔄 Fallback to reference determinant: {det_ref}")
            current_core_set = [det_ref]
            current_core_coeffs = [1.0]
    else:
        init_dict = getattr(args, "initial_dets_dict", None)
        if init_dict:
            current_core_set, current_core_coeffs = generate_initial_states(
                n_alpha, n_beta, n_orb, initial_dets_dict=init_dict, save_path=None)
            log_important(f"✅ Generated {len(current_core_set)} determinants from initial_dets_dict")
            if debug:
                log_verbose(f"🔄 Initial core set: {current_core_set}")
                log_verbose(f"🔄 Initial coeffs: {current_core_coeffs}")
        else:
            det_ref = generate_reference_det(n_alpha, n_beta, n_orb)
            log_verbose(f"🔄 Reference determinant: {det_ref}")
            current_core_set = [det_ref]
            current_core_coeffs = [1.0]

    pool_size = max(math.ceil(len(current_core_set) * pool_core_ratio), initial_pool_size)
    previous_energy = None
    current_dets = current_core_set
    current_energy = 0.0

    # Iteration tracking
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
        iter_info_init['n_dets'] = len(current_core_set)

        # Sort by magnitude if more than 1 det (initial state may not be sorted)
        if len(current_core_set) > 1:
            sorted_idx = np.argsort(np.abs(current_core_coeffs))[::-1]
            save_dets = [current_core_set[i] for i in sorted_idx]
            save_coeffs = [current_core_coeffs[i] for i in sorted_idx]
        else:
            save_dets, save_coeffs = current_core_set, current_core_coeffs

        save_iteration_results(-1, save_dets, save_coeffs, iter_info_init, 
                               outdir=results_dir, pool=None, save_pool=save_pool)

    n_total_configs = comb(n_orb, n_alpha) * comb(n_orb, n_beta)
    for iteration in range(max_iterations):
        iteration_start = time.perf_counter()

        # Current iteration config
        if isinstance(_core_set_ratio, list):
            core_set_ratio = _core_set_ratio[iteration % len(_core_set_ratio)]
        else:
            core_set_ratio = _core_set_ratio

        log_important("="*60)
        log_important(f"{RED_BOLD}🔄 TrimCI iteration {iteration+1}/{max_iterations}{RESET}")
        pool_size = min(pool_size, n_total_configs)

        # for research purpose
        if getattr(args, "research_tool_dict", {}).get("minimum_pool_size", 1) > pool_size:
            pool_size = getattr(args, "research_tool_dict", {}).get("minimum_pool_size", 1)
            log_important(f"⚠️ Research_tool: minimum pool size {pool_size}")

        iter_info = {
            'iteration': iteration+1,
            'core_set_size(before_pool_building)': len(current_core_set),
            'target_pool_size': pool_size
        }

        # =====================================================================
        # STEP 1: Pool Construction via Heat-Bath Screening
        # =====================================================================
        # Expand the current core set by generating connected determinants.
        # Uses importance sampling: H_ij * c_j weighting to prioritize
        # determinants with strong coupling to the current wavefunction.
        pool_start_time = time.perf_counter()

        log_verbose(f"📦 Building pool with {len(current_core_set)} core determinants")
        # Control C++ output: pass verbosity directly
        _verbosity = getattr(args, 'verbosity', 1 if getattr(args, 'verbose', False) else 0)
        pool_verbosity = _verbosity
        if pool_build_strategy == 'heat_bath':
            norm = np.sqrt(np.sum([c**2 for c in current_core_coeffs]))
            current_core_coeffs = [c/norm for c in current_core_coeffs]
            pool, final_threshold = screening(current_core_set, current_core_coeffs, n_orb, h1, eri,
                                            threshold, pool_size, {},
                                            f"{system_name}.bin",
                                            max_rounds=max_rounds,
                                            threshold_decay=threshold_decay,
                                            attentive_orbitals=attentive_orbitals,
                                            verbosity=pool_verbosity,
                                            strategy_factor=strategy_factor)
        elif pool_build_strategy == 'normalized_uniform':
            coeffs = [1.0/np.sqrt(len(current_core_set))] * len(current_core_set)
            pool, final_threshold = screening(current_core_set, coeffs, n_orb, h1, eri,
                                            threshold, pool_size, {},
                                            f"{system_name}.bin",
                                            max_rounds=max_rounds,
                                            threshold_decay=threshold_decay,
                                            attentive_orbitals=attentive_orbitals,
                                            verbosity=pool_verbosity,
                                            strategy_factor=strategy_factor)
        elif pool_build_strategy == 'uniform':
            pool, final_threshold = screening(current_core_set, [], n_orb, h1, eri,
                                            threshold, pool_size, {},
                                            f"{system_name}.bin",
                                            max_rounds=max_rounds,
                                            threshold_decay=threshold_decay,
                                            attentive_orbitals=attentive_orbitals,
                                            verbosity=pool_verbosity,
                                            strategy_factor=strategy_factor)
        elif pool_build_strategy == 'heat_bath_pt2':
            # PT2-weighted screening: uses |H_ij * c_i| / |E_0 - H_jj|
            # This gives higher weight to determinants with smaller energy gaps
            norm = np.sqrt(np.sum([c**2 for c in current_core_coeffs]))
            current_core_coeffs = [c/norm for c in current_core_coeffs]
            
            # Compute precise E0 for current core_set using fast CI energy evaluation
            # This is more accurate than using prev_energy from a different det set
            evaluate_ci_energy = trimci_core.evaluate_ci_energy
            dets_alpha = [d.alpha for d in current_core_set]
            dets_beta = [d.beta for d in current_core_set]
            current_e0 = evaluate_ci_energy(dets_alpha, dets_beta, current_core_coeffs, 
                                           h1.tolist() if hasattr(h1, 'tolist') else h1,
                                           eri.reshape(-1).tolist() if hasattr(eri, 'reshape') else eri,
                                           n_orb)
            log_verbose(f"📐 PT2 screening with E0={current_e0:.6f}")
            
            pool, final_threshold = screening(current_core_set, current_core_coeffs, n_orb, h1, eri,
                                            threshold, pool_size, {},
                                            f"{system_name}.bin",
                                            max_rounds=max_rounds,
                                            threshold_decay=threshold_decay,
                                            attentive_orbitals=attentive_orbitals,
                                            verbosity=pool_verbosity,
                                            screening_mode="heat_bath_pt2",
                                            e0=current_e0,
                                            strategy_factor=strategy_factor)
        elif pool_build_strategy == 'pt2':
            # Full PT2 estimation (aggregates contributions from multiple parents)
            norm = np.sqrt(np.sum([c**2 for c in current_core_coeffs]))
            current_core_coeffs = [c/norm for c in current_core_coeffs]
            
            # Compute precise E0 for current core_set
            evaluate_ci_energy = trimci_core.evaluate_ci_energy
            dets_alpha = [d.alpha for d in current_core_set]
            dets_beta = [d.beta for d in current_core_set]
            current_e0 = evaluate_ci_energy(dets_alpha, dets_beta, current_core_coeffs,
                                           h1.tolist() if hasattr(h1, 'tolist') else h1,
                                           eri.reshape(-1).tolist() if hasattr(eri, 'reshape') else eri,
                                           n_orb)
            log_verbose(f"📐 PT2 screening with E0={current_e0:.6f}")
            
            pool, final_threshold = screening(current_core_set, current_core_coeffs, n_orb, h1, eri,
                                            threshold, pool_size, {},
                                            f"{system_name}.bin",
                                            max_rounds=max_rounds,
                                            threshold_decay=threshold_decay,
                                            attentive_orbitals=attentive_orbitals,
                                            verbosity=pool_verbosity,
                                            screening_mode="pt2",
                                            e0=current_e0,
                                            strategy_factor=strategy_factor)
        else:
            raise ValueError(f"Unknown pool_build_strategy: {pool_build_strategy}")

        # for research purpose
        if getattr(args, "research_tool_dict", {}).get("pool_size_control", False):
            actual_pool_size = len(pool)
            pool = pool[:pool_size]
            log_important(f"⚠️ Research_tool: Pool size {actual_pool_size} -> {pool_size}")

        log_verbose(f"🔍 Screening completed: {len(pool)} determinants "
                 f"in {time.perf_counter()-pool_start_time:.1f}s, final threshold: {final_threshold:.2e}")

        iter_info['pool_building_time'] = time.perf_counter() - pool_start_time
        iter_info['actual_pool_size'] = len(pool)
        iter_info['final_threshold'] = final_threshold

        if iteration > 0:
            threshold = final_threshold


        # =====================================================================
        # STEP 2: Local Trim Parameter Setup
        # =====================================================================
        # Configure the hierarchical trimming: divide pool into `num_groups`
        # subgroups, keep top-k from each subgroup to reduce N^3 -> N^2 scaling.
        
        # Dynamic num_groups: scale with core_set_size if num_groups_ratio > 0
        if num_groups_ratio > 0:
            num_groups = max(num_groups_base, int(len(current_core_set) * num_groups_ratio))
        else:
            num_groups = num_groups_base
        
        if num_groups>=1:
            trim_m = [num_groups]
            if local_trim_keep_ratio > 0:
                keep_pool_size = math.ceil(len(current_core_set) * local_trim_keep_ratio)
                trim_k = [math.ceil(keep_pool_size / num_groups)]
            else:
                trim_k = [math.ceil(pool_size * keep_ratio / num_groups)]
        else:
            m = int(np.power(pool_size, num_groups))
            trim_m = [m]
            if local_trim_keep_ratio > 0:
                keep_pool_size = math.ceil(len(current_core_set) * local_trim_keep_ratio)
                trim_k = [math.ceil(keep_pool_size / m)]
            else:
                trim_k = [math.ceil(pool_size * keep_ratio / m)]

        iter_info['trim_m'], iter_info['trim_k'] = trim_m, trim_k

        # =====================================================================
        # STEP 3: Subspace Diagonalization & Selection (TRIM Core)
        # =====================================================================
        # Execute the core TrimCI algorithm:
        #   1. Partition pool into m groups
        #   2. For each group: build H_ij, diagonalize, keep top-k by |c_i|
        #   3. Merge survivors, re-diagonalize for final variational energy
        log_verbose(f"✂️  Running trim with m={trim_m}, k={trim_k}")
        
        funcs = get_functions_for_system(n_orb)
        run_trim_func = funcs['run_trim']
        current_energy, current_dets, trim_coeffs = run_trim_func(
            pool, h1, eri, system_name, n_alpha+n_beta, n_orb,
            trim_m, trim_k, 
            False, False, current_core_set, 1e-3, pool_verbosity
        )

        iter_info['raw_dets_count'] = len(current_dets)
        iter_info['raw_electronic_energy'] = current_energy
        
        # TRIM returns dets and coeffs that are consistent
        # full_coeffs: saved before truncation for output (return_mode='full')
        # current_core_coeffs: will be truncated for next iteration
        full_coeffs = list(trim_coeffs)
        current_core_coeffs = list(trim_coeffs)

        # =====================================================================
        # STEP 4: Energy Accounting & Statistics
        # =====================================================================
        # Compute total energy (E_elec + E_nuc), track ΔE.
        # Note: Convergence decision is made in STEP 6.
        total_energy = current_energy + nuclear_repulsion
        iter_info['raw_energy'] = total_energy
        log_important(f"⚡ Iteration {iteration+1} total energy: {total_energy:.8f}")
        log_important(f"🔤 Core set: {len(current_core_set)}, Raw Determinants: {len(current_dets)}")

        # Compute energy change for convergence tracking
        if previous_energy is not None:
            energy_change = total_energy - previous_energy
            iter_info['energy_change'] = energy_change
            log_important(f"📊 ΔE = {energy_change:.2e}")
            # Convergence: |ΔE| < threshold (energy can go up or down)
            iter_info['converged'] = (abs(energy_change) < energy_threshold)
        else:
            iter_info['converged'] = False
        previous_energy = total_energy

        # =====================================================================
        # STEP 5: Core Set Growth & Preparation for Next Iteration
        # =====================================================================
        # Sort determinants by |coefficient|, grow core set by schedule or ratio,
        # normalize coefficients, and compute next iteration's pool size.
        old_size = len(current_core_set)
        
        # DMRG-style noise injection: add randomness to help escape local minima
        # Noise is added to |coefficients| before sorting, giving smaller dets a chance
        if noise_strength > 0:
            abs_coeffs = np.abs(full_coeffs)
            # Scale noise by max coefficient (DMRG convention: ~1e-3 of max singular value)
            noise_scale = noise_strength * np.max(abs_coeffs)
            abs_coeffs = abs_coeffs + np.random.randn(len(abs_coeffs)) * noise_scale
            sorted_idx = np.argsort(abs_coeffs)[::-1]
            log_verbose(f"🎲 Noise injection: strength={noise_strength:.3f}, scale={noise_scale:.2e}")
        else:
            sorted_idx = np.argsort(np.abs(full_coeffs))[::-1]
        sorted_dets = [current_dets[i] for i in sorted_idx]
        sorted_coeffs = [full_coeffs[i] for i in sorted_idx]
        
        # Compute new core size
        # Priority: core_set_schedule > first_cycle_keep_size > core_set_ratio
        if core_set_schedule is not None and iteration < len(core_set_schedule):
            # Use scheduled size for this iteration
            scheduled_size = core_set_schedule[iteration]
            new_size = min(len(current_dets), scheduled_size)
            log_important(f"📋 Core set (scheduled): {old_size} -> {new_size} "
                         f"(schedule[{iteration}]={scheduled_size}, max: {len(current_dets)})")
        elif iteration == 0 and getattr(args, 'first_cycle_keep_size', 0):
            new_size = min(len(current_dets), getattr(args, 'first_cycle_keep_size', 0))
            log_important(f"🔄 Core set: {old_size} -> {new_size} (max: {len(current_dets)})")
        else:
            new_size = min(len(current_dets), math.ceil(old_size * core_set_ratio))
            if core_set_ratio <= 0:
                new_size = 1
            log_important(f"🔄 Core set: {old_size} -> {new_size} (max: {len(current_dets)})")

        # Truncate and normalize
        current_core_set = [current_dets[i] for i in sorted_idx[:new_size]]
        current_core_coeffs = [sorted_coeffs[i] for i in range(new_size)]
        norm = np.linalg.norm(current_core_coeffs)
        current_core_coeffs = [c/norm for c in current_core_coeffs]
        iter_info['core_set_size(after_trimming)'] = new_size

        pool_size = math.ceil(new_size * pool_core_ratio)
        iter_info['next_pool_size'] = pool_size
        log_verbose(f"📈 Next pool size: {pool_size}")

        # Periodic save (core_set is already sorted by |coeff|)
        if (iteration + 1) % save_period == 0 or (save_initial and iteration == 0):
            save_iteration_results(iteration + 1,
                        current_core_set, current_core_coeffs, iter_info, 
                        outdir=results_dir, pool=pool, save_pool=save_pool)

        # =====================================================================
        # STEP 6: Termination Conditions
        # =====================================================================
        # Finalize iteration timing and bookkeeping, then check stopping criteria.
        
        # Timing (includes all steps 1-5)
        time_cost = time.perf_counter() - iteration_start
        total_elapsed = time.perf_counter() - start_time
        
        # Auto-pick best unit for time display
        unit, factor = ("s", 1) if total_elapsed < 60 else (("min", 60) if total_elapsed < 3600 else ("h", 3600))
        log_important(f"⏱️  Iteration {iteration+1} time: {time_cost:.2f}s (Total: {total_elapsed/factor:.1f}{unit})")
        if 'pool_time' in iter_info:
            iter_info['trim_time'] = time_cost - iter_info['pool_time']
            log_important(f"📊 Pool: {iter_info['pool_time']:.2f}s, Trim: {iter_info['trim_time']:.2f}s")

        iter_info['iteration_time'] = time_cost
        iter_info['cumulative_time'] = total_elapsed


        # Commit iteration record
        iteration_details['iterations'].append(iter_info)
        
        # Check stopping criteria (in priority order):
        #   1. Energy convergence
        #   2. core_set_schedule exhausted
        #   3. max_iterations reached
        #   4. max_final_dets reached (only if NOT using schedule)
        
        # Termination by energy convergence (DISABLED - can cause premature stopping)
        # if iter_info.get('converged', False):
        #     log_important(f"✅ Energy converged (ΔE < {energy_threshold:.2e}), stopping.")
        #     iter_info['stopped_by_convergence'] = True
        #     break
        
        # Termination by schedule exhaustion
        if core_set_schedule is not None and iteration >= len(core_set_schedule) - 1:
            log_important(f"📋 Core set schedule exhausted at iteration {iteration+1}, stopping.")
            iter_info['stopped_by_schedule'] = True
            break
        
        # Termination by max_iterations
        if iteration >= max_iterations - 1:
            log_important(f"🛑 Reached max_iterations={max_iterations}, stopping.")
            iter_info['stopped_by_max_iterations'] = True
            break
        
        # Termination by max_final_dets (skip if using schedule - let schedule control)
        if core_set_schedule is None and max_final_dets is not None and len(current_core_set) >= max_final_dets:
            log_important(f"🛑 Reached max_final_dets={max_final_dets}, stopping.")
            iter_info['stopped_by_max_final_dets'] = True
            break

    # =========================================================================
    # FINALIZATION: Post-Processing & Result Assembly
    # =========================================================================
    # After the iterative loop completes:
    #   1. Store "raw" results from the last TRIM iteration (before truncation)
    #   2. Run final Davidson diagonalization on the truncated core set
    #      (with warm-start from previous coefficients to avoid excited states)
    #   3. Assemble iteration_details and save to disk
    
    # Record the "raw" results from the last trim iteration
    final_raw_energy = current_energy + nuclear_repulsion
    final_raw_dets = current_dets
    final_raw_coeffs = full_coeffs
    final_raw_dets_count = len(current_dets)
    
    # Compute final core energy by running Davidson on the core_set
    # This gives us a cleaner energy estimate using only the most important determinants
    funcs = get_functions_for_system(n_orb)
    diag_func = funcs['diagonalize_subspace_davidson']

    # Use the core_set (already sorted and truncated) for final diagonalization
    # Pass current_core_coeffs as initial guess for warm start (avoids excited state locking)
    core_energy, core_coeffs_out = diag_func(
        current_core_set,      # core determinants
        h1,                    # 1-electron integrals
        eri,                   # 2-electron integrals (already flat)
        {},                    # Hij cache
        False,                 # quantization
        500,                   # max_iter
        1e-6,                  # tolerance
        False,                 # verbose
        n_orb,                 # n_orb
        current_core_coeffs    # initial guess from previous iteration (warm start)
    )
    final_core_energy = core_energy + nuclear_repulsion
    final_core_coeffs = list(core_coeffs_out)
    final_core_dets_count = len(current_core_set)

    
    # Use core energy as the final energy (cleaner, on the truncated variational space)
    final_energy = final_core_energy
    
    total_time = time.perf_counter() - start_time
    log_important(f"⏱️ Final energy: {final_energy:.8f}, Workflow time: {total_time:.1f}s")

    iteration_details.update({
        'total_time': total_time,
        # Final core set results (primary output)
        'final_energy': final_energy,
        'final_core_energy': final_core_energy,
        'final_core_dets_count': final_core_dets_count,
        # Raw results from last trim (for reference)
        'final_raw_energy': final_raw_energy,
        'final_raw_dets_count': final_raw_dets_count,
        # Legacy fields (deprecated but kept for compatibility)
        'final_electronic_energy': final_core_energy - nuclear_repulsion,
        'final_dets_count': final_core_dets_count,
        # Metadata
        'converged': any(it.get('converged', False) for it in iteration_details['iterations']),
        'total_iterations': len(iteration_details['iterations']),
        'n_electrons': n_alpha + n_beta,
        'n_orbitals': n_orb,
        'nuclear_repulsion': nuclear_repulsion,
        'results_dir': results_dir
    })

    # Output: core_set with fresh coefficients from Davidson
    save_final_results(final_energy, current_core_set, final_core_coeffs, 
                       iteration_details, args, outdir=results_dir)

    return final_energy, current_core_set, final_core_coeffs, iteration_details, args

# ========== Configuration ==========
DEFAULT_CONFIG = {
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
    "first_cycle_keep_size": 10,  # Default keep size for first cycle
}

def _load_config_from_py(py_path: str) -> dict:
    """Load config dict from a Python file."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("trimci_config", py_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, 'config') and isinstance(module.config, dict):
        return module.config
    raise ValueError(f"trimci_config.py must define a 'config' dict, not found in {py_path}")

def _load_config_from_json(json_path: str) -> dict:
    """Load config dict from a JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

def load_configurations(config_dir: str, trimci_config_path: str = None, save_if_not_exist: bool = True):
    """
    Load TrimCI configuration with priority:
      1. Explicit path (if provided): .py or .json based on extension
      2. Auto-detect in config_dir: trimci_config.py > trimci_config.json
      3. Default config (optionally saved as .json)
    """
    config = DEFAULT_CONFIG.copy()

    # Case 1: Explicit path provided
    if trimci_config_path is not None:
        if os.path.exists(trimci_config_path):
            if trimci_config_path.endswith('.py'):
                config.update(_load_config_from_py(trimci_config_path))
            else:
                config.update(_load_config_from_json(trimci_config_path))
            log_verbose(f"📋 Loaded config: {trimci_config_path}")
        else:
            log_verbose(f"⚠️ Config file not found: {trimci_config_path}, using defaults")
        return Namespace(**config)

    # Case 2: Auto-detect in config_dir (priority: .py > .json)
    py_path = os.path.join(config_dir, "trimci_config.py")
    json_path = os.path.join(config_dir, "trimci_config.json")

    if os.path.exists(py_path):
        config.update(_load_config_from_py(py_path))
        log_verbose(f"📋 Loaded config: {py_path}")
    elif os.path.exists(json_path):
        config.update(_load_config_from_json(json_path))
        log_verbose(f"📋 Loaded config: {json_path}")
    else:
        # Case 3: No config found, optionally create default .json
        if save_if_not_exist:
            with open(json_path, 'w') as f:
                json.dump(DEFAULT_CONFIG, f, indent=2)
            log_verbose(f"📋 Created default config: {json_path}")

    return Namespace(**config)

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

def save_iteration_results(iter_idx, dets, coeffs, iter_info,
                           outdir="results", pool=None, save_pool=False):
    """Save intermediate iteration results."""
    Path(outdir).mkdir(exist_ok=True)

    top_10_dets = get_top_determinants(dets, coeffs, top_n=10)

    # Save JSON
    json_path = os.path.join(outdir, f"iter_{iter_idx:03d}.json")
    with open(json_path, "w") as f:
        json.dump({
            "iteration": iter_idx,
            "n_dets": len(dets),
            "iteration_info": iter_info,
            "top_10_determinants": top_10_dets
        }, f, indent=2)

    # Save NPZ (with dual naming for backward compatibility)
    npz_path = os.path.join(outdir, f"iter_{iter_idx:03d}.npz")
    dets_arr = dets_to_array(dets)
    coeffs_arr = np.array(coeffs)
    save_data = {
        'dets': dets_arr, 'core_set': dets_arr,  # dual naming
        'coeffs': coeffs_arr, 'core_set_coeffs': coeffs_arr, 'dets_coeffs': coeffs_arr  # triple naming
    }
    if save_pool and pool is not None:
        save_data['pool'] = dets_to_array(pool)
    np.savez_compressed(npz_path, **save_data)
    log_verbose(f"💾 Saved iteration {iter_idx} → {npz_path}")

def save_final_results(final_energy, dets, coeffs, iteration_details, args, outdir="results"):
    """Save final TrimCI results."""
    Path(outdir).mkdir(exist_ok=True)

    top_10_dets = get_top_determinants(dets, coeffs, top_n=10)

    # Save JSON
    json_path = os.path.join(outdir, "trimci_results.json")
    with open(json_path, "w") as f:
        json.dump({
            "final_energy": final_energy,
            "n_dets": len(dets),
            "config": vars(args),
            "iteration_summary": iteration_details,
            "top_10_determinants": top_10_dets
        }, f, indent=2)

    # Save NPZ (with dual naming for backward compatibility)
    npz_path = os.path.join(outdir, "dets.npz")
    dets_arr = dets_to_array(dets)
    coeffs_arr = np.array(coeffs)
    np.savez_compressed(npz_path,
                        dets=dets_arr, core_set=dets_arr,  # dual naming
                        coeffs=coeffs_arr, core_set_coeffs=coeffs_arr, dets_coeffs=coeffs_arr)  # triple naming
    log_important(f"💾 Saved final results → {npz_path}")
    



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

    # Determine n_segments from class name
    class_name = str(DeterminantClass)
    n_segments = 1
    for bits, segs in [(512,8),(448,7),(384,6),(320,5),(256,4),(192,3),(128,2)]:
        if f'Determinant{bits}' in class_name:
            n_segments = segs
            break

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
        """Support both int and array input for all Determinant types"""
        class_name = str(DeterminantClass)
        
        # Determine number of segments based on Determinant type
        if 'Determinant512' in class_name:
            n_segments = 8
        elif 'Determinant448' in class_name:
            n_segments = 7
        elif 'Determinant384' in class_name:
            n_segments = 6
        elif 'Determinant320' in class_name:
            n_segments = 5
        elif 'Determinant256' in class_name:
            n_segments = 4
        elif 'Determinant192' in class_name:
            n_segments = 3
        elif 'Determinant128' in class_name:
            n_segments = 2
        else:
            n_segments = 1
        
        if n_segments > 1:
            # Multi-segment determinants require array input
            if isinstance(alpha_bits, (list, tuple)):
                alpha_array = [to_uint64(x) for x in alpha_bits]
                beta_array  = [to_uint64(x) for x in beta_bits]
                # Pad if needed
                while len(alpha_array) < n_segments:
                    alpha_array.append(0)
                while len(beta_array) < n_segments:
                    beta_array.append(0)
            else:
                # Convert large integer to array of uint64
                alpha_array = [to_uint64((alpha_bits >> (64*i)) & 0xFFFFFFFFFFFFFFFF) for i in range(n_segments)]
                beta_array  = [to_uint64((beta_bits  >> (64*i)) & 0xFFFFFFFFFFFFFFFF) for i in range(n_segments)]
            return DeterminantClass(alpha_array, beta_array)
        else:
            # Standard 64-bit Determinant
            if isinstance(alpha_bits, (list, tuple)):
                alpha_bits = alpha_bits[0]
                beta_bits  = beta_bits[0]
            return DeterminantClass(to_uint64(alpha_bits), to_uint64(beta_bits))

    # Track all generated dets for dedup across all sources
    seen_dets = set()

    # --- Preset determinants ---
    def add_preset(kind, coeff, count=1):
        for _ in range(count):
            if kind == "reference" or kind == "hf":
                det = hf_reference()
            elif kind == "afm":
                alpha_bits = sum([1 << i for i in range(0, n_orb, 2)][:n_alpha])
                beta_bits  = sum([1 << i for i in range(1, n_orb, 2)][:n_beta])
                det = create_determinant(alpha_bits, beta_bits)
            elif kind == "paramagnetic":
                alpha_bits = sum([1 << i for i in range(n_alpha)])
                beta_bits  = sum([1 << i for i in range(n_beta)])
                det = create_determinant(alpha_bits, beta_bits)
            elif kind == "stripe":
                half = n_orb // 2
                alpha_bits = sum([1 << i for i in range(min(n_alpha, half))])
                beta_bits  = sum([1 << i for i in range(half, half + n_beta)])
                det = create_determinant(alpha_bits, beta_bits)
            else:
                continue
            key = (det.alpha, det.beta)
            if key not in seen_dets:
                seen_dets.add(key)
                dets.append(det)
                coeffs.append(float(coeff))

    # --- Random determinants (with dedup + count preservation) ---
    def _make_bitmask_arrays(occupied_orbitals):
        arrays = [0] * n_segments
        for i in occupied_orbitals:
            seg, bit = divmod(i, 64)
            arrays[seg] |= (1 << bit)
        return arrays

    def add_random(coeff, count=1):
        from scipy.special import comb as _comb
        total_possible = int(_comb(n_orb, n_alpha)) * int(_comb(n_orb, n_beta))
        added = 0
        max_attempts = count * 10
        attempts = 0
        while added < count and attempts < max_attempts:
            attempts += 1
            occ_alpha = rng.choice(n_orb, n_alpha, replace=False)
            occ_beta  = rng.choice(n_orb, n_beta, replace=False)
            alpha_array = _make_bitmask_arrays(occ_alpha.tolist())
            beta_array  = _make_bitmask_arrays(occ_beta.tolist())
            det = create_determinant(alpha_array, beta_array)
            key = (det.alpha, det.beta)
            if key in seen_dets:
                continue
            seen_dets.add(key)
            dets.append(det)
            coeffs.append(float(coeff))
            added += 1
            if len(seen_dets) >= total_possible:
                break
        if added < count:
            log_important(f"add_random: requested {count}, generated {added} "
                          f"(space exhausted or max_attempts reached)")

    # --- Random closed shell determinants (with dedup + count preservation) ---
    def add_random_closed_shell(coeff, count=1):
        if n_alpha != n_beta:
            raise ValueError(f"random_closed_shell requires n_alpha ({n_alpha}) == n_beta ({n_beta})")
        from scipy.special import comb as _comb
        total_possible = int(_comb(n_orb, n_alpha))
        added = 0
        max_attempts = count * 10
        attempts = 0
        while added < count and attempts < max_attempts:
            attempts += 1
            occ = rng.choice(n_orb, n_alpha, replace=False)
            alpha_array = _make_bitmask_arrays(occ.tolist())
            beta_array = list(alpha_array)
            det = create_determinant(alpha_array, beta_array)
            key = (det.alpha, det.beta)
            if key in seen_dets:
                continue
            seen_dets.add(key)
            dets.append(det)
            coeffs.append(float(coeff))
            added += 1
            if len(seen_dets) >= total_possible:
                break
        if added < count:
            log_important(f"add_random_closed_shell: requested {count}, generated {added} "
                          f"(space exhausted or max_attempts reached)")

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
            class_name = str(DeterminantClass)
            if '512' in class_name:
                n_segments = 8
            elif '448' in class_name:
                n_segments = 7
            elif '384' in class_name:
                n_segments = 6
            elif '320' in class_name:
                n_segments = 5
            elif '256' in class_name:
                n_segments = 4
            elif '192' in class_name:
                n_segments = 3
            elif '128' in class_name:
                n_segments = 2
            else:
                n_segments = 1
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
                elif kind == "random_closed_shell":
                    add_random_closed_shell(val, 1)
                else:
                    add_preset(kind, val, 1)
            elif isinstance(val, (list, tuple)):
                if kind == "random":
                    coeff, count = val
                    add_random(coeff, int(count))
                elif kind == "random_closed_shell":
                    coeff, count = val
                    add_random_closed_shell(coeff, int(count))
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
# Moved to report_generator.py for cleaner organization
from .report_generator import generate_multi_run_report


# ========== C++ Backend Iterative Workflow ==========
def iterative_workflow(h1, eri, n_alpha, n_beta, n_orb,
                       system_name, args, nuclear_repulsion,
                       start_time=None, results_dir="trimci_results"):
    """
    Main TrimCI iterative workflow.
    
    This is a wrapper that calls the C++ implementation for maximum performance.
    The C++ backend eliminates the overhead of C++ -> Python object conversion
    for intermediate results (pool of millions of determinants).
    
    For the pure Python implementation (deprecated), use iterative_workflow_py().
    
    Args:
        h1: One-body integrals (n_orb x n_orb)
        eri: Two-body integrals (flattened or 4D)
        n_alpha, n_beta: Number of alpha/beta electrons
        n_orb: Number of orbitals
        system_name: System identifier for logging and file naming
        args: Configuration namespace with workflow parameters
        nuclear_repulsion: Nuclear repulsion energy
        start_time: Optional start time for timing (default: current time)
        results_dir: Output directory
    
    Returns:
        (final_energy, dets, coeffs, iteration_details, args)
    """
    import time
    from pathlib import Path
    
    if start_time is None:
        start_time = time.perf_counter()
    
    # Import C++ backend - select appropriate version based on n_orb
    from trimci.trimci_core import IterativeWorkflowParams
    from trimci.auto_selector import get_functions_for_system
    
    # Get the appropriate determinant type and functions
    functions = get_functions_for_system(n_orb)
    type_suffix = functions.get('type_suffix', '')
    
    # Import the correct iterative_workflow function
    if not type_suffix or type_suffix == '':
        # 64-bit (default)
        from trimci.trimci_core import iterative_workflow_cpp
    else:
        # Scalable types (128-512)
        import trimci.trimci_core as trimci_core
        workflow_fn_name = f'iterative_workflow_cpp{type_suffix}'
        if hasattr(trimci_core, workflow_fn_name):
            iterative_workflow_cpp = getattr(trimci_core, workflow_fn_name)
            log_important(f"Using scalable C++ workflow: {workflow_fn_name}")
        else:
            # Fallback to Python implementation if scalable C++ not available
            log_important(f"⚠️ {workflow_fn_name} not available, falling back to Python workflow")
            return iterative_workflow_py(h1, eri, n_alpha, n_beta, n_orb,
                                         system_name, args, nuclear_repulsion,
                                         start_time, results_dir)

    
    # Prepare ERI and h1 for C++ backend
    # Use numpy arrays directly with the _np versions when available (faster for large systems)
    use_numpy_version = False  # TEMPORARILY DISABLED - needs debugging
    
    if use_numpy_version:
        # Keep as numpy arrays - use _np version of workflow
        if hasattr(eri, "reshape") and hasattr(eri, "ndim") and eri.ndim == 4:
            eri_np = eri.reshape(-1).astype(np.float64, copy=False)
        else:
            eri_np = np.asarray(eri, dtype=np.float64).flatten()
        
        h1_np = np.asarray(h1, dtype=np.float64)
        
        # Try to get numpy-optimized workflow function
        import trimci.trimci_core as trimci_core
        if not type_suffix or type_suffix == '':
            numpy_workflow_fn_name = 'iterative_workflow_cpp_np'
        else:
            numpy_workflow_fn_name = f'iterative_workflow_cpp_np{type_suffix}'
        
        if hasattr(trimci_core, numpy_workflow_fn_name):
            iterative_workflow_cpp = getattr(trimci_core, numpy_workflow_fn_name)
            log_verbose(f"Using numpy-optimized C++ workflow: {numpy_workflow_fn_name}")
            eri = eri_np
            h1 = h1_np
        else:
            # Fallback to list conversion
            log_verbose(f"Numpy-optimized workflow not available, using list conversion")
            eri = eri_np.tolist()
            h1 = h1_np.tolist()
    else:
        # Original list-based path (slower for large systems)
        if hasattr(eri, "reshape") and hasattr(eri, "ndim") and eri.ndim == 4:
            eri = eri.reshape(-1).tolist()
        elif hasattr(eri, "tolist"):
            eri = eri.tolist()
        
        if hasattr(h1, "tolist"):
            h1 = h1.tolist()
    
    # Create results directory
    unique_ts = generate_unique_timestamp()
    results_dir = str(Path(results_dir) / f"{system_name}_{unique_ts}")
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # Build IterativeWorkflowParams from args
    # =========================================================================
    params = IterativeWorkflowParams()
    
    # Termination conditions
    params.max_iterations = getattr(args, 'max_iterations', 
                                   getattr(args, 'exp_max_iterations', 200000))
    params.energy_threshold = getattr(args, 'energy_threshold',
                                      getattr(args, 'exp_energy_threshold', 1e-12))
    params.max_final_dets = getattr(args, 'max_final_dets', -1) or -1
    
    # Core set growth
    _core_set_ratio = getattr(args, 'core_set_ratio', 2)
    if isinstance(_core_set_ratio, (int, float)):
        params.core_set_ratio = [float(_core_set_ratio)]
    else:
        params.core_set_ratio = [float(r) for r in _core_set_ratio]
    
    params.initial_pool_size = getattr(args, 'initial_pool_size', 100)
    params.first_cycle_keep_size = getattr(args, 'first_cycle_keep_size', 10)
    
    # Core set schedule
    _schedule = getattr(args, 'core_set_schedule', None)
    if _schedule is not None:
        params.core_set_schedule = list(_schedule)
    
    # Pool building
    params.pool_core_ratio = getattr(args, 'pool_core_ratio', 10)
    params.pool_build_strategy = getattr(args, 'pool_build_strategy', 'heat_bath')
    params.threshold = getattr(args, 'threshold', 0.01)
    params.first_cycle_threshold = getattr(args, 'first_cycle_threshold', -1.0)
    params.threshold_decay = getattr(args, 'threshold_decay', 0.9)
    params.max_rounds = getattr(args, 'max_rounds', 1)
    params.strategy_factor = getattr(args, 'strategy_factor', -1)
    params.pool_strict_target_size = getattr(args, 'pool_strict_target_size', False)
    params.max_pool_size = getattr(args, 'max_pool_size', 0)
    params.stagnation_limit = getattr(args, 'stagnation_limit', 3)
    
    _attentive = getattr(args, 'attentive_orbitals', None)
    if _attentive is not None:
        params.attentive_orbitals = list(_attentive)
    
    # DMRG-style noise
    params.noise_strength = getattr(args, 'noise_strength', 0.0)
    
    # TRIM parameters
    params.num_groups = getattr(args, 'num_groups', 10)
    params.num_groups_ratio = getattr(args, 'num_groups_ratio', 0)
    params.local_trim_keep_ratio = getattr(args, 'local_trim_keep_ratio',
                                           getattr(args, 'keep_pool_to_next_core_ratio', 0))
    params.keep_ratio = getattr(args, 'keep_ratio', 0.1)
    
    # Verbosity
    params.verbosity = getattr(args, 'verbosity', 1)
    params.davidson_init = getattr(args, 'davidson_init', 'lowest_diag_noise')
    
    # Saving parameters
    params.save_period = getattr(args, 'save_period', 1000000)  # Default: effectively disabled
    params.save_pool = getattr(args, 'save_pool', False)
    params.save_initial = getattr(args, 'save_initial', False)
    params.output_dir = results_dir  # Use the results directory
    
    # =========================================================================
    # Generate initial determinants
    # =========================================================================
    initial_dets_dict = getattr(args, 'initial_dets_dict', None)
    load_initial_dets = getattr(args, 'load_initial_dets', False)
    
    if load_initial_dets:
        # Load from file - check both dets_path and initial_dets_path (for orbopt compatibility)
        dets_path = getattr(args, 'initial_dets_path', None) or getattr(args, 'dets_path', 'dets.npz')
        dets_array_name = getattr(args, 'dets_array_name', 'dets')
        initial_dets, initial_coeffs = load_initial_dets_from_file(
            dets_path, core_set=(dets_array_name != 'dets'))
        
        # Handle load failure - fallback to HF reference (matches Python behavior)
        if initial_dets is None or initial_coeffs is None:
            log_important(f"⚠️ Failed to load from {dets_path}, falling back to HF reference")
            ref_det = generate_reference_det(n_alpha, n_beta, n_orb)
            initial_dets = [ref_det]
            initial_coeffs = [1.0]
        else:
            # Apply top-k truncation if load_initial_dets_num is specified (matches Python behavior)
            load_initial_dets_num = getattr(args, 'load_initial_dets_num', None)
            if load_initial_dets_num is not None and load_initial_dets_num < len(initial_dets):
                original_count = len(initial_dets)
                sorted_idx = np.argsort(np.abs(initial_coeffs))[::-1][:load_initial_dets_num]
                initial_dets = [initial_dets[i] for i in sorted_idx]
                initial_coeffs = [initial_coeffs[i] for i in sorted_idx]
                log_important(f"✅ Loaded top-{load_initial_dets_num} determinants (truncated from {original_count})")
            else:
                log_important(f"✅ Loaded {len(initial_dets)} determinants from {dets_path}")
    elif initial_dets_dict is not None:
        # Generate from dict
        initial_dets, initial_coeffs = generate_initial_states(
            n_alpha, n_beta, n_orb,
            initial_dets_dict
        )
    else:
        # Default: HF reference
        ref_det = generate_reference_det(n_alpha, n_beta, n_orb)
        initial_dets = [ref_det]
        initial_coeffs = [1.0]
    
    # Deduplicate initial dets (keep first occurrence, preserve order)
    n_before = len(initial_dets)
    seen_keys = set()
    unique_dets, unique_coeffs = [], []
    for d, c in zip(initial_dets, initial_coeffs):
        key = (d.alpha, d.beta)
        if key not in seen_keys:
            seen_keys.add(key)
            unique_dets.append(d)
            unique_coeffs.append(c)
    initial_dets, initial_coeffs = unique_dets, unique_coeffs
    if len(initial_dets) < n_before:
        log_important(f"Dedup: {n_before} -> {len(initial_dets)} initial dets")

    # =========================================================================
    # Call C++ backend
    # =========================================================================
    log_important(f"🚀 Starting C++ iterative_workflow with {len(initial_dets)} initial dets")
    
    cpp_result = iterative_workflow_cpp(
        h1, eri, n_alpha, n_beta, n_orb,
        system_name,
        initial_dets, initial_coeffs,
        nuclear_repulsion,
        params
    )
    
    if not cpp_result.success:
        raise RuntimeError(f"C++ iterative_workflow failed: {cpp_result.error_message}")
    
    # =========================================================================
    # Convert result to Python format
    # =========================================================================
    total_time = time.perf_counter() - start_time
    
    # Build iteration_details dict (compatible with Python version)
    iteration_details = {
        'iterations': [],
        'total_time': cpp_result.total_time,
        'final_energy': cpp_result.final_energy,
        'final_core_energy': cpp_result.final_energy,
        'final_core_dets_count': len(cpp_result.final_dets),
        'final_raw_energy': cpp_result.final_energy,
        'final_raw_dets_count': len(cpp_result.final_dets),
        'final_electronic_energy': cpp_result.final_energy - nuclear_repulsion,
        'final_dets_count': len(cpp_result.final_dets),
        'converged': any(info.converged for info in cpp_result.iteration_history),
        'total_iterations': cpp_result.total_iterations,
        'n_electrons': n_alpha + n_beta,
        'n_orbitals': n_orb,
        'nuclear_repulsion': nuclear_repulsion,
        'results_dir': results_dir
    }
    
    # Convert iteration history
    for info in cpp_result.iteration_history:
        iteration_details['iterations'].append({
            'iteration': info.iteration,
            'core_set_size_before': info.core_set_size_before,
            'target_pool_size': info.target_pool_size,
            'actual_pool_size': info.actual_pool_size,
            'final_threshold': info.final_threshold,
            'pool_building_time': info.pool_building_time,
            'trim_m': info.trim_m,
            'trim_k': info.trim_k,
            'raw_dets_count': info.raw_dets_count,
            'raw_energy': info.raw_energy,
            'energy_change': info.energy_change,
            'converged': info.converged,
            'core_set_size_after': info.core_set_size_after,
            'iteration_time': info.iteration_time,
            'cumulative_time': info.cumulative_time,
        })
    
    # Save results
    save_final_results(
        cpp_result.final_energy,
        list(cpp_result.final_dets),
        list(cpp_result.final_coeffs),
        iteration_details,
        args,
        outdir=results_dir
    )
    
    log_important(f"✅ C++ iterative_workflow complete: E={cpp_result.final_energy:.8f}, "
                  f"dets={len(cpp_result.final_dets)}, time={cpp_result.total_time:.2f}s")
    
    return (cpp_result.final_energy, 
            list(cpp_result.final_dets), 
            list(cpp_result.final_coeffs), 
            iteration_details, 
            args)
