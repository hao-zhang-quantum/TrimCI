"""
Automatic determinant type selector based on orbital count.

This module provides functionality to automatically select the appropriate
determinant type based on the number of orbitals in the system.
"""

from typing import Any, Dict, Callable, Tuple, List
import warnings

# Try to import scalable determinant types
try:
    from . import trimci_core
    SCALABLE_AVAILABLE = True
    
    # Verify that required scalable functions are available
    required_functions = [
        'Determinant128', 'Determinant192',
        'generate_reference_det_128', 'generate_reference_det_192',
        'generate_excitations_128', 'generate_excitations_192',
        'load_or_create_Hij_cache_128', 'load_or_create_Hij_cache_192',
        'compute_H_ij_128', 'compute_H_ij_192',
        'pool_build_128', 'pool_build_192',
        'diagonalize_subspace_davidson_128', 'diagonalize_subspace_davidson_192',
        'select_top_k_dets_128', 'select_top_k_dets_192',
        'run_trim_128', 'run_trim_192'
    ]
    
    missing_functions = []
    for func_name in required_functions:
        if not hasattr(trimci_core, func_name):
            missing_functions.append(func_name)
    
    if missing_functions:
        SCALABLE_AVAILABLE = False
        warnings.warn(f"Scalable determinant types partially available but missing functions: {missing_functions}. "
                     "Using fallback to standard types.")
    
except ImportError:
    SCALABLE_AVAILABLE = False
    warnings.warn("Scalable determinant types not available. Using fallback to standard types. "
                 "To enable scalable types, ensure the C++ extension was built with scalable support.")

class DeterminantTypeSelector:
    """
    Automatic selector for determinant types based on system size.
    
    Provides optimal determinant type selection and corresponding function mappings
    based on the number of orbitals in the quantum system.
    """
    
    def __init__(self):
        self._type_thresholds = [
            (64, "64"),    # Up to 64 orbitals: use Determinant64
            (128, "128"),  # 65-128 orbitals: use Determinant128  
            (192, "192"),  # 129-192 orbitals: use Determinant192
        ]
        
    def select_determinant_type(self, n_orb: int) -> str:
        """
        Select the appropriate determinant type based on orbital count.
        
        Args:
            n_orb: Number of orbitals in the system
            
        Returns:
            String identifier for the determinant type ("64", "128", or "192")
            
        Raises:
            ValueError: If n_orb exceeds maximum supported orbitals (192)
        """
        if n_orb <= 0:
            raise ValueError(f"Number of orbitals must be positive, got {n_orb}")
            
        for threshold, type_id in self._type_thresholds:
            if n_orb <= threshold:
                return type_id
                
        raise ValueError(f"Number of orbitals {n_orb} exceeds maximum supported (192)")
    
    def get_function_mapping(self, n_orb: int) -> Dict[str, Callable]:
        """
        Get the appropriate function mapping for the given orbital count.
        
        Args:
            n_orb: Number of orbitals in the system
            
        Returns:
            Dictionary mapping function names to their implementations
        """
        det_type = self.select_determinant_type(n_orb)
        
        if not SCALABLE_AVAILABLE or det_type == "64":
            # Use standard functions for 64-bit or when scalable not available
            if det_type != "64" and not SCALABLE_AVAILABLE:
                warnings.warn(f"System requires Determinant{det_type} but scalable types not available. "
                            f"Falling back to Determinant64. Performance may be suboptimal.")
            
            return {
                'determinant_class': trimci_core.Determinant,
                'generate_reference_det': trimci_core.generate_reference_det,
                'generate_excitations': trimci_core.generate_excitations,
                'load_or_create_Hij_cache': trimci_core.load_or_create_Hij_cache,
                'pair_key': trimci_core.pair_key,
                'compute_H_ij': trimci_core.compute_H_ij,
                'pool_build': trimci_core.pool_build,
                'diagonalize_subspace_davidson': trimci_core.diagonalize_subspace_davidson,
                'select_top_k_dets': trimci_core.select_top_k_dets,
                'run_trim': trimci_core.run_trim,
                'type_suffix': ''
            }
        else:
            # Use scalable functions with error handling
            suffix = f"_{det_type}"
            try:
                return {
                    'determinant_class': getattr(trimci_core, f'Determinant{det_type}'),
                    'generate_reference_det': getattr(trimci_core, f'generate_reference_det{suffix}'),
                    'generate_excitations': getattr(trimci_core, f'generate_excitations{suffix}'),
                    'load_or_create_Hij_cache': getattr(trimci_core, f'load_or_create_Hij_cache{suffix}'),
                    'pair_key': trimci_core.pair_key,
                    'compute_H_ij': getattr(trimci_core, f'compute_H_ij{suffix}'),
                    'pool_build': getattr(trimci_core, f'pool_build{suffix}'),
                    'diagonalize_subspace_davidson': getattr(trimci_core, f'diagonalize_subspace_davidson{suffix}'),
                    'select_top_k_dets': getattr(trimci_core, f'select_top_k_dets{suffix}'),
                    'run_trim': getattr(trimci_core, f'run_trim{suffix}'),
                    'type_suffix': suffix
                }
            except AttributeError as e:
                warnings.warn(f"Failed to access scalable function for Determinant{det_type}: {e}. "
                            f"Falling back to Determinant64.")
                # Fallback to standard functions
                return {
                    'determinant_class': trimci_core.Determinant,
                    'generate_reference_det': trimci_core.generate_reference_det,
                    'generate_excitations': trimci_core.generate_excitations,
                    'load_or_create_Hij_cache': trimci_core.load_or_create_Hij_cache,
                    'pair_key': trimci_core.pair_key,
                    'compute_H_ij': trimci_core.compute_H_ij,
                    'pool_build': trimci_core.pool_build,
                    'diagonalize_subspace_davidson': trimci_core.diagonalize_subspace_davidson,
                    'select_top_k_dets': trimci_core.select_top_k_dets,
                    'run_trim': trimci_core.run_trim,
                    'type_suffix': ''
                }
    
    def get_recommended_type_info(self, n_orb: int) -> Dict[str, Any]:
        """
        Get detailed information about the recommended determinant type.
        
        Args:
            n_orb: Number of orbitals in the system
            
        Returns:
            Dictionary with type information including memory usage estimates
        """
        det_type = self.select_determinant_type(n_orb)
        
        type_info = {
            "64": {
                "name": "Determinant64",
                "max_orbitals": 64,
                "memory_per_det": 16,  # bytes
                "description": "Standard 64-bit determinant for small systems"
            },
            "128": {
                "name": "Determinant128", 
                "max_orbitals": 128,
                "memory_per_det": 32,  # bytes
                "description": "128-bit determinant for medium systems (recommended for 78 orbitals)"
            },
            "192": {
                "name": "Determinant192",
                "max_orbitals": 192, 
                "memory_per_det": 48,  # bytes
                "description": "192-bit determinant for large systems"
            }
        }
        
        info = type_info[det_type].copy()
        info["selected_for_orbitals"] = n_orb
        info["scalable_available"] = SCALABLE_AVAILABLE
        
        return info

# Global selector instance
_selector = DeterminantTypeSelector()

def select_determinant_type(n_orb: int) -> str:
    """
    Select the appropriate determinant type based on orbital count.
    
    Args:
        n_orb: Number of orbitals in the system
        
    Returns:
        String identifier for the determinant type
    """
    return _selector.select_determinant_type(n_orb)

def get_functions_for_system(n_orb: int) -> Dict[str, Callable]:
    """
    Get the appropriate function set for the given system size.
    
    Args:
        n_orb: Number of orbitals in the system
        
    Returns:
        Dictionary mapping function names to their implementations
    """
    return _selector.get_function_mapping(n_orb)

def get_system_info(n_orb: int) -> Dict[str, Any]:
    """
    Get detailed information about the recommended setup for the system.
    
    Args:
        n_orb: Number of orbitals in the system
        
    Returns:
        Dictionary with system information and recommendations
    """
    return _selector.get_recommended_type_info(n_orb)

def print_system_recommendation(n_orb: int) -> None:
    """
    Print a human-readable recommendation for the system.
    
    Args:
        n_orb: Number of orbitals in the system
    """
    info = get_system_info(n_orb)
    
    print(f"=== TrimCI System Recommendation ===")
    print(f"Orbitals: {n_orb}")
    print(f"Recommended type: {info['name']}")
    print(f"Description: {info['description']}")
    print(f"Memory per determinant: {info['memory_per_det']} bytes")
    print(f"Maximum orbitals supported: {info['max_orbitals']}")
    print(f"Scalable types available: {'Yes' if info['scalable_available'] else 'No'}")
    
    if n_orb > 64 and not info['scalable_available']:
        print("\n⚠️  WARNING: Your system has >64 orbitals but scalable types are not available.")
        print("   Consider rebuilding with scalable determinant support for optimal performance.")