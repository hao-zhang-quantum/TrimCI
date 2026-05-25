"""
High-level Python interface wrapping trimci_core C++ module.

This interface automatically selects the appropriate determinant type
based on the number of orbitals in the system for optimal performance.
"""

from . import trimci_core
import sys
sys.modules.setdefault('trimci.trimci_core', trimci_core)

from typing import List, Tuple, Optional, Any
from pathlib import Path
from .auto_selector import get_functions_for_system, get_system_info, print_system_recommendation

def extract_mol_name(atom_str: str) -> str:
    """Extract molecule name from atom string."""
    return trimci_core.extract_mol_name(atom_str)

def load_or_create_Hij_cache(mol_name: str, n_elec: int, n_orb: int, cache_dir: str="cache"):
    """
    Load or create Hamiltonian cache with automatic determinant type selection.
    
    Args:
        mol_name: Name of the molecule
        n_elec: Number of electrons
        n_orb: Number of orbitals (used for automatic type selection)
        cache_dir: Cache directory path
        
    Returns:
        Tuple of (cache, cache_path)
    """
    # Get appropriate functions for this system size
    funcs = get_functions_for_system(n_orb)
    
    cache, path = funcs['load_or_create_Hij_cache'](mol_name, n_elec, n_orb, cache_dir)
    return cache, Path(path)

def generate_reference_det(n_alpha: int, n_beta: int, n_orb: Optional[int] = None):
    """
    Generate reference determinant with automatic type selection.
    
    Args:
        n_alpha: Number of alpha electrons
        n_beta: Number of beta electrons  
        n_orb: Number of orbitals (if None, estimated from electrons)
        
    Returns:
        Reference determinant of appropriate type
    """
    # Estimate n_orb if not provided (assume at least 2x electrons for reasonable basis)
    if n_orb is None:
        n_orb = max(64, max(n_alpha, n_beta))

    if n_orb > 192:
        raise ValueError(f"Number of orbitals {n_orb} exceeds maximum supported (192)")
    
    # Get appropriate functions for this system size
    funcs = get_functions_for_system(n_orb)
    
    return funcs['generate_reference_det'](n_alpha, n_beta)

def generate_excitations(det, n_orb: int):
    """
    Generate excitations with automatic function selection.
    
    Args:
        det: Input determinant
        n_orb: Number of orbitals
        
    Returns:
        List of excited determinants
    """
    funcs = get_functions_for_system(n_orb)
    return funcs['generate_excitations'](det, n_orb)

def screening(pool: List, initial_coeff: List, n_orb: int, h1, eri,
              threshold: float, target_size: int, cache, cache_file: str,
              max_rounds: int = 2, threshold_decay: float = 0.5,
              attentive_orbitals: Optional[List[int]] = None,
              verbosity: int = 1,
              screening_mode: str = "hb",
              e0: float = 0.0,
              strategy_factor: int = -1):
    """
    Perform screening with automatic function selection.

    Args:
        pool: Initial determinant pool
        initial_coeff: Initial coefficients
        n_orb: Number of orbitals (used for automatic type selection)
        h1: One-electron integrals
        eri: Two-electron integrals (flat 1D array or 4D array)
        threshold: Screening threshold
        target_size: Target pool size
        cache: Hamiltonian cache
        cache_file: Cache file path
        max_rounds: Maximum screening rounds per threshold level
        threshold_decay: Decay factor for threshold relaxation (e.g., 0.5 = halve each time)
        attentive_orbitals: If provided, excitations are restricted to these orbital indices
                           (for Attentive TrimCI). Empty list or None means use all orbitals.
        verbosity: Verbosity level (0=silent, 1=basic, 2=detailed)
        screening_mode: Screening strategy:
                       - "hb" (default): heat-bath screening, rank by MAX(|H_ij*c_i|)
                       - "hb-pt2": heat-bath screening + PT2 reranking
        e0: Current variational energy (required for PT2 modes)
        strategy_factor: Pre-filter multiplier for PT2 modes (-1=auto: 1 for heat_bath, 20 for PT2)
    Returns:
        Tuple of (result_pool, final_threshold)
    """
    funcs = get_functions_for_system(n_orb)

    # Flatten ERI if it's 4D
    if hasattr(eri, "reshape") and hasattr(eri, "ndim") and eri.ndim == 4:
        eri_flat = eri.reshape(-1)
    else:
        eri_flat = eri

    # Convert None to empty list for C++
    att_orbs = attentive_orbitals if attentive_orbitals is not None else []

    result_pool, final_threshold = funcs['pool_build'](pool, initial_coeff, n_orb, h1, eri_flat,
                                                      threshold, target_size, cache, cache_file,
                                                      max_rounds, threshold_decay, att_orbs, verbosity,
                                                      screening_mode, e0, strategy_factor)
    return result_pool, final_threshold

def trim(pool: List, h1, eri,
         mol_name: str, n_elec: int, n_orb: int,
         group_sizes: List[int], keep_sizes: List[int],
         quantization: bool=False, save_cache: bool=True,
         external_core_dets: Optional[List]=None, tol: float=1e-3,
         verbosity: int = 1):
    """
    Perform TRIM calculation with automatic function selection.
    
    Args:
        pool: Initial determinant pool
        h1: One-electron integrals
        eri: Two-electron integrals (flat 1D array or 4D array)
        mol_name: Molecule name
        n_elec: Number of electrons
        n_orb: Number of orbitals (used for automatic type selection)
        group_sizes: Group sizes for TRIM
        keep_sizes: Keep sizes for TRIM
        quantization: Enable quantization
        save_cache: Save cache to disk
        external_core_dets: External core determinants
        tol: Davidson tolerance
        verbosity: Verbosity level (0=silent, 1=basic, 2=detailed)
        
    Returns:
        TRIM calculation results
    """
    if external_core_dets is None:
        external_core_dets = []
    
    # Print system recommendation if verbose
    if verbosity >= 1:
        print_system_recommendation(n_orb)
        print()  # Add spacing
    
    funcs = get_functions_for_system(n_orb)

    # Flatten ERI if it's 4D
    if hasattr(eri, "reshape") and hasattr(eri, "ndim") and eri.ndim == 4:
        eri_flat = eri.reshape(-1)
    else:
        eri_flat = eri
    
    return funcs['run_trim'](pool, h1, eri_flat, mol_name,
                            n_elec, n_orb, group_sizes, keep_sizes,
                            quantization, save_cache, external_core_dets, tol, verbosity)

# Additional utility functions for advanced users

def get_determinant_class(n_orb: int):
    """
    Get the appropriate determinant class for the given orbital count.
    
    Args:
        n_orb: Number of orbitals
        
    Returns:
        Determinant class appropriate for the system size
    """
    funcs = get_functions_for_system(n_orb)
    return funcs['determinant_class']

def get_system_recommendation(n_orb: int) -> dict:
    """
    Get system recommendation information.
    
    Args:
        n_orb: Number of orbitals
        
    Returns:
        Dictionary with system information and recommendations
    """
    return get_system_info(n_orb)

def create_determinant(alpha_bits, beta_bits, n_orb: int):
    """
    Create a determinant with automatic type selection.
    
    Args:
        alpha_bits: Alpha electron bit pattern
        beta_bits: Beta electron bit pattern  
        n_orb: Number of orbitals
        
    Returns:
        Determinant of appropriate type
    """
    det_class = get_determinant_class(n_orb)
    return det_class(alpha_bits, beta_bits)

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
