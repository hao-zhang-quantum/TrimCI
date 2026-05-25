"""
FCIDUMP I/O, result saving, and timestamp utilities.
"""
import json
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np

from trimci.trimci_logging import log_verbose, log_important


def generate_unique_timestamp() -> str:
    """
    Generate a unique timestamp string for folder naming.

    Uses microsecond precision + PID to prevent collisions when multiple
    jobs start simultaneously (e.g., Slurm array jobs).

    Format: YYYYMMDD_HHMMSSffffff_PID (6 digits microseconds + PID)

    Returns:
        str: Unique timestamp like "20260125_033041123456_12345"
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S%f")
    pid_suffix = f"_{os.getpid()}"
    return ts + pid_suffix


def read_fcidump(fcidump_path: str):
    log_verbose(f"Reading FCIDUMP file: {fcidump_path}")
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

    log_verbose(f"Detected NORB={n_orb}, NELEC={n_elec}, N_ALPHA={n_alpha}, N_BETA={n_beta}, MS2={ms2}, PSYM={psym}")

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
    Returns the same format as read_fcidump:
        (h1, eri, n_elec, n_orb, nuclear_repulsion, n_alpha, n_beta, psym)
    """
    log_verbose(f"Reading FCIDUMP file with PySCF: {fcidump_path}")
    from pyscf.tools import fcidump
    from pyscf import ao2mo

    fcidump_data = fcidump.read(fcidump_path)

    h1 = fcidump_data['H1']
    eri_compressed = fcidump_data['H2']
    n_orb = fcidump_data['NORB']
    n_elec = fcidump_data['NELEC']
    nuclear_repulsion = fcidump_data['ECORE']
    ms2 = fcidump_data['MS2']
    psym = fcidump_data.get('PSYM', 8)
    n_alpha = int((n_elec + ms2) // 2)
    n_beta = int((n_elec - ms2) // 2)

    eri = ao2mo.restore(1, eri_compressed, n_orb)

    log_verbose(f"Detected NORB={n_orb}, NELEC={n_elec}, PSYM={psym}")

    return h1, eri, n_elec, n_orb, nuclear_repulsion, n_alpha, n_beta, psym


def save_iteration_results(iter_idx, dets, coeffs, iter_info,
                           outdir="results", pool=None, save_pool=False):
    """Save intermediate iteration results."""
    from .det_utils import dets_to_array, get_top_determinants

    Path(outdir).mkdir(exist_ok=True)

    top_10_dets = get_top_determinants(dets, coeffs, top_n=10)

    json_path = os.path.join(outdir, f"iter_{iter_idx:03d}.json")
    with open(json_path, "w") as f:
        json.dump({
            "iteration": iter_idx,
            "n_dets": len(dets),
            "iteration_info": iter_info,
            "top_10_determinants": top_10_dets
        }, f, indent=2)

    npz_path = os.path.join(outdir, f"iter_{iter_idx:03d}.npz")
    dets_arr = dets_to_array(dets)
    coeffs_arr = np.array(coeffs)
    save_data = {
        'dets': dets_arr, 'core_set': dets_arr,
        'coeffs': coeffs_arr, 'core_set_coeffs': coeffs_arr, 'dets_coeffs': coeffs_arr
    }
    if save_pool and pool is not None:
        save_data['pool'] = dets_to_array(pool)
    np.savez_compressed(npz_path, **save_data)
    log_verbose(f"Saved iteration {iter_idx} -> {npz_path}")


def save_final_results(final_energy, dets, coeffs, iteration_details, args, outdir="results"):
    """Save final TrimCI results."""
    from .det_utils import dets_to_array, get_top_determinants

    Path(outdir).mkdir(exist_ok=True)

    top_10_dets = get_top_determinants(dets, coeffs, top_n=10)

    json_path = os.path.join(outdir, "trimci_results.json")
    with open(json_path, "w") as f:
        json.dump({
            "final_energy": final_energy,
            "n_dets": len(dets),
            "config": vars(args),
            "iteration_summary": iteration_details,
            "top_10_determinants": top_10_dets
        }, f, indent=2)

    npz_path = os.path.join(outdir, "dets.npz")
    dets_arr = dets_to_array(dets)
    coeffs_arr = np.array(coeffs)
    np.savez_compressed(npz_path,
                        dets=dets_arr, core_set=dets_arr,
                        coeffs=coeffs_arr, core_set_coeffs=coeffs_arr, dets_coeffs=coeffs_arr)
    log_important(f"Saved final results -> {npz_path}")
