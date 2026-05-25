import sys
from pathlib import Path
from unittest.mock import MagicMock

# Mock netket to avoid import error in wavelab if not installed
try:
    import netket
except ImportError:
    sys.modules['netket'] = MagicMock()

# Add code directory to sys.path to allow importing wavelab
# Path to jac_impl.py: .../code/dev_version_v0.1.1_202511/py/trimci/jac_impl.py
# We want to add .../code to sys.path
current_file = Path(__file__).resolve()
# trimci -> py -> dev_version -> code
code_dir = current_file.parent.parent.parent.parent
if str(code_dir) not in sys.path:
    sys.path.append(str(code_dir))

import numpy as np
# scipy.linalg is imported lazily inside _compute_orbital_gradient_fd which uses it.
try:
    from wavelab.wavefunction import Wavefunction
    from wavelab.transforms.rdm_transforms import compute_1rdm, compute_2rdm
    HAS_WAVELAB = True
except ImportError:
    HAS_WAVELAB = False
    Wavefunction = None
    compute_1rdm = None
    compute_2rdm = None

try:
    import trimci
    HAS_TRIMCI_CORE = True
except ImportError:
    HAS_TRIMCI_CORE = False

def det_to_int_tuple(det):
    """Convert trimci Determinant to (alpha, beta) integer tuple."""
    if isinstance(det.alpha, list):
        alpha = 0
        for i, val in enumerate(det.alpha):
            alpha |= (val << (64 * i))
        beta = 0
        for i, val in enumerate(det.beta):
            beta |= (val << (64 * i))
        return (alpha, beta)
    else:
        return (det.alpha, det.beta)

def _compute_orbital_gradient_cpp(h1, eri, dets, coeffs, n_orb, n_elec, attentive_orbitals=None):
    """Compute orbital gradient using high-performance C++ kernel.
    
    If attentive_orbitals is provided, only computes gradient for pairs (p,q) 
    where both p and q are in attentive_orbitals. This provides O(k^2*N^3) 
    speedup compared to O(N^5) for full gradient.
    
    Optimization: Accepts pre-converted lists to avoid O(N^4) conversion overhead.
    """
    if not HAS_TRIMCI_CORE:
        raise ImportError("TrimCI Core not found, cannot use C++ kernel.")
    
    # Ensure inputs are in correct format for C++
    # Skip conversion if already a list (optimization)
    if isinstance(h1, list):
        h1_list = h1
    elif isinstance(h1, np.ndarray):
        h1_list = h1.tolist()
    else:
        h1_list = list(h1)
    
    # C++ expects flattened ERI
    if isinstance(eri, list):
        eri_flat = eri  # Already a flat list
    elif isinstance(eri, np.ndarray):
        eri_flat = eri.reshape(-1).tolist()
    else:
        eri_flat = list(eri)
    
    # Convert attentive_orbitals to list for C++
    att_orb_list = list(attentive_orbitals) if attentive_orbitals else []
    
    funcs = trimci.get_functions_for_system(n_orb)
    if 'compute_orbital_gradient' not in funcs:
        raise NotImplementedError(f"compute_orbital_gradient not implemented in C++ core for {n_orb} orbitals.")

    grad_cpp, f_diag_cpp = funcs['compute_orbital_gradient'](
        dets, 
        coeffs, 
        h1_list, 
        eri_flat, 
        n_orb, 
        n_elec,
        att_orb_list
    )
    return np.array(grad_cpp), np.array(f_diag_cpp)

def _compute_orbital_gradient_py(h1, eri, dets, coeffs, n_orb, n_elec):
    """
    Compute orbital gradient using Python/Wavelab reference implementation.
    
    This computes the gradient of the CI energy with respect to orbital rotation 
    parameters under the FIXED CI COEFFICIENTS assumption.
    
    The gradient is derived from the generalized Fock matrix:
        g[p,q] = 2 * (F[q,p] - F[p,q]) = 2 * (F.T - F)[p,q]
    
    where F is the generalized Fock matrix:
        F[p,q] = F^(1)[p,q] + F^(2)[p,q]
                = sum_r h[p,r] * Gamma1[r,q] - sum_{rst} (pr|st) * Gamma2_eff[q,r,s,t]
    
    Note: This gradient assumes FIXED CI coefficients. When used in an optimization
    that re-diagonalizes at each step, this is NOT the correct gradient - use
    finite difference on the objective function instead (see orblab.py jac_wrapper).
    
    Returns
    -------
    grad : ndarray (n_orb, n_orb)
        Anti-symmetric gradient matrix
    f_diag : ndarray (n_orb,)
        Diagonal of the Fock matrix
    """
    if not HAS_WAVELAB:
        raise ImportError("wavelab package is required to compute orbital gradients in Python mode.")

    # ========== PHASE 1: Construct Wavefunction ==========
    wf_dict = {}
    for det, coeff in zip(dets, coeffs):
        key = det_to_int_tuple(det)
        wf_dict[key] = coeff
    
    wf = Wavefunction(wf_dict, basis="spinful")
    
    # ========== PHASE 2: Compute RDMs ==========
    # 1-RDM: Gamma1[p,q] = <Psi| a^dag_p a_q |Psi>  (summed over spin)
    Gamma_up, Gamma_dn = compute_1rdm(wf, n_orb)
    
    # 2-RDM components (in wavelab convention):
    # G2_uuuu[i,j,k,l] = <Psi| a^dag_{i,up} a^dag_{j,up} a_{k,up} a_{l,up} |Psi>
    # G2_dddd similar for down-spin
    # G2_udud[i,j,k,l] = <Psi| a^dag_{i,up} a^dag_{j,dn} a_{k,up} a_{l,dn} |Psi>
    G2_uuuu, G2_dddd, G2_udud = compute_2rdm(wf, n_orb)
    
    # ========== PHASE 3: Construct Effective 2-RDM ==========
    # The effective 2-RDM for the orbital gradient is:
    # Gamma2_eff[p,q,r,s] = sum_{sigma,tau} G2[p,r,q,s; sigma,tau]
    # 
    # Need to permute from wavelab convention to this convention.
    # Permutation P_{0,2,1,3} maps [i0,i1,i2,i3] -> [i0,i2,i1,i3]
    # Permutation P_{1,3,0,2} maps [i0,i1,i2,i3] -> [i1,i3,i0,i2]
    Gamma2_eff = np.zeros((n_orb, n_orb, n_orb, n_orb), dtype=complex)
    
    # Same-spin: alpha-alpha and beta-beta, plus alpha-beta
    Gamma2_eff += G2_uuuu.transpose(0, 2, 1, 3)  # P_{0,2,1,3}
    Gamma2_eff += G2_dddd.transpose(0, 2, 1, 3)  # P_{0,2,1,3}
    Gamma2_eff += G2_udud.transpose(0, 2, 1, 3)  # P_{0,2,1,3} for (up,dn)
    # Cross term: beta-alpha (from alpha-beta by symmetry)
    Gamma2_eff += G2_udud.transpose(1, 3, 0, 2)  # P_{1,3,0,2}

    # ========== PHASE 4: Compute Generalized Fock Matrix ==========
    # F[p,q] = F^(1)[p,q] + F^(2)[p,q]
    
    # 1-body: F^(1)[p,q] = sum_r h[p,r] * Gamma1[r,q] = (h @ Gamma1)[p,q]
    Gamma1 = Gamma_up + Gamma_dn
    F_1body = h1 @ Gamma1
    
    # 2-body: F^(2)[p,q] = -sum_{rst} (pr|st) * Gamma2_eff[q,r,s,t]
    # The negative sign arises from the specific 2-RDM convention used
    F_2body = -1.0 * np.einsum('prst,qrst->pq', eri, Gamma2_eff)
    
    F = F_1body + F_2body

    # ========== PHASE 5: Compute Gradient ==========
    # Gradient formula: g[p,q] = 2 * (F[q,p] - F[p,q])
    # In matrix form: g = 2 * (F.T - F)
    # This is ANTI-SYMMETRIC: g[p,q] = -g[q,p]
    grad_mat = 2 * (F.T - F)
    
    return grad_mat.real, F.diagonal()

def _solve_energy_single_point(h1, eri_flat, dets, coeffs, n_orb):
    """Helper to compute energy for FD."""
    if not HAS_TRIMCI_CORE:
        raise ImportError("TrimCI Core required for energy calculation in FD mode.")
        
    funcs = trimci.get_functions_for_system(n_orb)
    compute_H_ij = funcs['compute_H_ij']
    
    E = 0.0
    n_dets = len(dets)
    # Simple double loop
    for i in range(n_dets):
        # Diagonal
        elem = compute_H_ij(dets[i], dets[i], h1, eri_flat)
        E += coeffs[i] * coeffs[i] * elem
        
        # Off-diagonal
        for j in range(i+1, n_dets):
            elem = compute_H_ij(dets[i], dets[j], h1, eri_flat)
            E += 2.0 * coeffs[i] * coeffs[j] * elem
    return E

def _compute_orbital_gradient_fd(h1, eri, dets, coeffs, n_orb, n_elec, delta=1e-5):
    """Compute orbital gradient using Finite Difference (Forensic Verification)."""
    # Requires scipy.linalg.expm
    from scipy.linalg import expm
    
    grad_fd = np.zeros((n_orb, n_orb))
    eri_shape = (n_orb, n_orb, n_orb, n_orb)
    if eri.shape != eri_shape:
        eri_full = eri.reshape(eri_shape)
    else:
        eri_full = eri
        
    # Helper to transform integrals
    def transform_np(h1, eri, U):
        # h1' = U.T h1 U
        h1_T = U.T @ h1 @ U
        # eri' transform: (pr|st)' = sum U_pi U_rj U_sk U_tl (ij|kl)
        # Optimized chain:
        tmp = np.einsum('ijkl,tl->ijkt', eri, U)
        tmp = np.einsum('ijkt,sk->ijst', tmp, U)
        tmp = np.einsum('ijst,rj->irst', tmp, U)
        eri_T = np.einsum('irst,pi->prst', tmp, U)
        return h1_T, eri_T

    for p in range(n_orb):
        for q in range(p+1, n_orb):
            # K_pq = delta -> U = exp(-K)
            K = np.zeros((n_orb, n_orb))
            K[p,q] = delta
            K[q,p] = -delta
            
            U_plus = expm(-K)
            U_minus = expm(K)
            
            h1_p, eri_p = transform_np(h1, eri_full, U_plus)
            h1_m, eri_m = transform_np(h1, eri_full, U_minus)
            
            E_plus = _solve_energy_single_point(h1_p, eri_p.reshape(-1).tolist(), dets, coeffs, n_orb)
            E_minus = _solve_energy_single_point(h1_m, eri_m.reshape(-1).tolist(), dets, coeffs, n_orb)
            
            g_val = (E_plus - E_minus) / (2.0 * delta)
            grad_fd[p,q] = g_val
            grad_fd[q,p] = -g_val
            
    return grad_fd, np.zeros(n_orb) # FD doesn't give F_diag easily

def compute_orbital_gradient(h1, eri, dets, coeffs, n_orb, n_elec, mode='cpp', attentive_orbitals=None):
    """
    Compute orbital gradient for the given integrals and wavefunction.
    
    Parameters
    ----------
    h1 : ndarray (n, n) or list
        1-electron integrals.
    eri : ndarray (n, n, n, n) or flattened or list
        2-electron integrals.
    dets : list
        List of determinant objects.
    coeffs : list
        List of CI coefficients.
    n_orb : int
        Number of spatial orbitals.
    n_elec : int or tuple
        Number of electrons.
    mode : str, optional
        Computation mode: 'cpp', 'py', or 'fd'. Default is 'cpp'.
        - 'cpp': Use optimized C++ kernel (fastest).
        - 'py': Use Wavelab/Python reference (verified correct).
        - 'fd': Use Finite Difference (slow, for verification).
    attentive_orbitals : list, optional
        If provided, only compute gradient for pairs (p,q) where both p and q 
        are in this list. Provides O(k^2*N^3) speedup vs O(N^5) for cpp mode.

    Returns
    -------
    grad : ndarray (n, n)
        Gradient matrix (anti-symmetric).
    f_diag : ndarray (n,)
        Diagonal of Generalized Fock Matrix (zeros for FD mode).
    """
    if mode == 'cpp':
        # Default behavior: Try C++, fallback to Python if unavailable
        if HAS_TRIMCI_CORE:
            try:
                return _compute_orbital_gradient_cpp(h1, eri, dets, coeffs, n_orb, n_elec, attentive_orbitals)
            except Exception:
                # If specifically requested C++ fails, we might want to warn, 
                # but for compatibility, we fallback to Py if enabled.
                pass
        
        # Fallback to Python if C++ failed or not available
        return _compute_orbital_gradient_py(h1, eri, dets, coeffs, n_orb, n_elec)

    elif mode == 'py':
        return _compute_orbital_gradient_py(h1, eri, dets, coeffs, n_orb, n_elec)
    
    elif mode == 'fd':
        return _compute_orbital_gradient_fd(h1, eri, dets, coeffs, n_orb, n_elec)
        
    else:
        raise ValueError(f"Unknown mode: {mode}")

