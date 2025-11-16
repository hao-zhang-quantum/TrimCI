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
        estimated_orb = max(64, max(n_alpha, n_beta))

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
              max_rounds: int = -1):
    """
    Perform screening with automatic function selection.
    
    Args:
        pool: Initial determinant pool
        initial_coeff: Initial coefficients
        n_orb: Number of orbitals (used for automatic type selection)
        h1: One-electron integrals
        eri: Two-electron integrals
        threshold: Screening threshold
        target_size: Target pool size
        cache: Hamiltonian cache
        cache_file: Cache file path
        max_rounds: Maximum screening rounds
        
    Returns:
        Tuple of (result_pool, final_threshold)
    """
    funcs = get_functions_for_system(n_orb)
    
    result_pool, final_threshold = funcs['pool_build'](pool, initial_coeff, n_orb, h1, eri,
                                                      threshold, target_size, cache, cache_file,
                                                      max_rounds)
    return result_pool, final_threshold

def trim(pool: List, h1, eri,
         mol_name: str, n_elec: int, n_orb: int,
         group_sizes: List[int], keep_sizes: List[int],
         quantization: bool=False, save_cache: bool=True, verbose: bool=True,
         external_core_dets: Optional[List]=None):
    """
    Perform TRIM calculation with automatic function selection.
    
    Args:
        pool: Initial determinant pool
        h1: One-electron integrals
        eri: Two-electron integrals
        mol_name: Molecule name
        n_elec: Number of electrons
        n_orb: Number of orbitals (used for automatic type selection)
        group_sizes: Group sizes for TRIM
        keep_sizes: Keep sizes for TRIM
        quantization: Enable quantization
        save_cache: Save cache to disk
        verbose: Enable verbose output
        external_core_dets: External core determinants
        
    Returns:
        TRIM calculation results
    """
    if external_core_dets is None:
        external_core_dets = []
    
    # Print system recommendation if verbose
    if verbose:
        print_system_recommendation(n_orb)
        print()  # Add spacing
    
    funcs = get_functions_for_system(n_orb)
    
    return funcs['run_trim'](pool, h1, eri, mol_name,
                            n_elec, n_orb, group_sizes, keep_sizes,
                            quantization, save_cache, verbose, external_core_dets)

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
