"""
TrimCI Orbital Optimizer with Active Gradient Acceleration.

See knowledge_docs/orblab_benchmark_and_usage.md for benchmark results and usage examples.
"""

import numpy as np
import scipy.linalg
from scipy.optimize import minimize
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from .auto_selector import get_functions_for_system
from . import trimci_core # Bindings
from pyscf import ao2mo
from .TrimCI_runner.io_utils import read_fcidump
import os
from pathlib import Path
from .jac_impl import compute_orbital_gradient


# Import unified logging (with OrbLab prefix wrappers)
from .trimci_logging import log_important as _log_important, log_verbose as _log_verbose

def log_verbose(msg):
    _log_verbose(msg, prefix="[OrbLab] ")

def log_important(msg):
    _log_important(msg, prefix="[OrbLab] ")


class OrbitalOptimizer:
    """
    Orbital Optimizer for TrimCI with Active Gradient Acceleration.
    
    Gradient Modes:
    ---------------
    - 'none': No explicit gradient; let scipy compute numerical gradient (most accurate)
    - 'full': Full FD gradient on all parameters (same accuracy as 'none', explicit)
    - 'active_l1': Adaptive selection using L1 norm threshold (RECOMMENDED: 8x speedup)
    - 'active_l2': Adaptive selection using L2² energy threshold
    - 'active_topk': Fixed Top-K parameter selection
    
    The Active Gradient methods use the updated_ci gradient to identify important
    parameters, then compute FD gradient only for those parameters.

    """
    
    def __init__(self, n_orb, n_elec, mol_name="molecule", verbose=False, warm_initial=False, 
                 attentive_orbitals=None):
        """
        Initialize OrbitalOptimizer.
        
        Args:
            n_orb: Number of orbitals
            n_elec: Number of electrons
            mol_name: Molecule name for caching
            verbose: Verbose logging
            warm_initial: Use warm-start for Davidson
            attentive_orbitals: Optional list of orbital indices to optimize.
                               If None, optimize all orbital pairs (full optimization).
                               If provided, only optimize rotations BETWEEN these orbitals.
                               This reduces parameters from C(n_orb,2) to C(len(attentive),2).
        """
        self.n_orb = n_orb
        self.n_elec = n_elec
        self.mol_name = mol_name
        self.verbose = verbose
        self.warm_initial = warm_initial
        self.last_x = None
        self.last_h1_rot = None
        self.last_eri_rot = None
        self.last_dets = None
        self.last_coeffs = None
        self.last_energy = None
        self.optimization_curve = []
        self.nuclear_repulsion = 0.0
        
        # Attentive orbital optimization
        self.attentive_orbitals = attentive_orbitals
        
        # Full index map: all C(n_orb, 2) pairs (always needed for C++ interface)
        self.full_idx_map = [(i, j) for i in range(n_orb) for j in range(i+1, n_orb)]
        # Dict for O(1) pair -> full_idx lookup (used for large systems)
        self._pair_to_full_idx = {pair: idx for idx, pair in enumerate(self.full_idx_map)}
        
        # Index mapping for gradient packing
        if attentive_orbitals is None:
            # Full optimization: all C(n_orb, 2) pairs
            self.idx_map = self.full_idx_map
            self.reduced_to_full = None  # No mapping needed
        else:
            # Attentive optimization: only pairs within attentive set
            att_sorted = sorted(attentive_orbitals)
            self.idx_map = [(i, j) for i in att_sorted for j in att_sorted if j > i]
            
            # Precompute reduced -> full index mapping (used by C++ interface)
            # Use dict lookup O(1) instead of list.index() O(n²)
            self.reduced_to_full = {
                reduced_idx: self._pair_to_full_idx[pair]
                for reduced_idx, pair in enumerate(self.idx_map)
            }
            
            if verbose:
                log_important(f"Attentive OrbOpt: {len(self.idx_map)} parameters "
                             f"(from {len(att_sorted)} orbitals, vs {n_orb*(n_orb-1)//2} full)")

    def optimize(self, h1, eri, dets, coeffs, optimizer_options_dict=None, callback_func=None):
        """
        Optimize orbitals to minimize energy.
        Uses trimci.trim to evaluate energy (re-optimizing CI coefficients at each step).
        
        Args:
            h1, eri: Integrals
            dets: List of determinants (the subspace)
            coeffs: Initial coefficients (not strictly used if we re-diagonalize, but good for reference)
            optimizer_options_dict: Dictionary of options for the optimizer
                - gradient_mode: 'none', 'full', 'active_l1', 'active_l2', 'active_topk'
                    - 'none': Let scipy compute numerical gradient (most accurate, slowest)
                    - 'full': Full FD gradient on all parameters
                    - 'active_l1': L1 norm adaptive selection (RECOMMENDED: 8x speedup)
                    - 'active_l2': L2² energy adaptive selection
                    - 'active_topk': Fixed Top-K parameter selection
                - active_parameter: target percentage (0-1) for L1/L2, or integer for topk
                - optimizer: 'L-BFGS-B' (default)
                - maxiter, maxfun, ftol, gtol: scipy minimize options
                - davidson_tol, davidson_max_iter: Davidson solver options
                
        Returns:
            new_h1, new_eri, final_energy, converged, U_final
        """
        if optimizer_options_dict is None:
            optimizer_options_dict = {}

        # === C++ Optimizer Routing ===
        optimizer_name = optimizer_options_dict.get('optimizer', 'L-BFGS-B').lower()

        # "auto_bfgs": resolve CPU vs GPU based on n_orb (4-index transform
        # N^5 work dwarfs GPU overhead for N >= 16; below that, CPU wins).
        if optimizer_name == 'auto_bfgs':
            try:
                import trimci.trimci_core as _tc
                gpu_ok = _tc.has_gpu() and hasattr(_tc, 'compute_orbital_gradient_gpu_128')
            except (ImportError, AttributeError):
                gpu_ok = False
            if gpu_ok and self.n_orb >= 16:
                optimizer_name = 'gpu_bfgs'
                log_important(f"[optimizer] auto_bfgs → gpu_bfgs (n_orb={self.n_orb})")
            else:
                optimizer_name = 'cpp_bfgs'
                log_important(f"[optimizer] auto_bfgs → cpp_bfgs "
                              f"(n_orb={self.n_orb}, gpu_ok={gpu_ok})")
            optimizer_options_dict = dict(optimizer_options_dict)
            optimizer_options_dict['optimizer'] = optimizer_name

        cpp_optimizers = {'approx_newton', 'cpp_newton', 'bfgs', 'cpp_bfgs'}
        if optimizer_name in cpp_optimizers:
            if self.attentive_orbitals is not None:
                log_important("C++ optimizers do not support attentive_orbitals; using full rotation.")
            if optimizer_name in {'bfgs', 'cpp_bfgs'}:
                return self._optimize_cpp_bfgs(h1, eri, dets, coeffs, optimizer_options_dict, callback_func)
            else:
                return self._optimize_cpp_approx_newton(h1, eri, dets, coeffs, optimizer_options_dict, callback_func)

        # === GPU Routing ===
        # gpu_newton : simple steepest-descent + line search + GPU grad/xform
        # gpu_bfgs   : true L-BFGS via scipy, energy+grad evaluated on GPU
        if optimizer_name == 'gpu_newton':
            if self.attentive_orbitals is not None:
                log_important("gpu_newton does not support attentive_orbitals; using full rotation.")
            return self._optimize_gpu_newton(h1, eri, dets, coeffs,
                                             optimizer_options_dict, callback_func)
        if optimizer_name == 'gpu_bfgs':
            if self.attentive_orbitals is not None:
                log_important("gpu_bfgs does not support attentive_orbitals; using full rotation.")
            return self._optimize_gpu_bfgs(h1, eri, dets, coeffs,
                                           optimizer_options_dict, callback_func)

        record_optimization = optimizer_options_dict.get('record_optimization', False)
        
        current_h1 = h1.copy()
        current_eri = eri.copy()
        
        # Reset cache for new optimization run
        self.last_x = None
        self.last_energy = None
        self.last_h1_rot = None
        self.last_eri_rot = None
        self.last_dets = dets
        
        if record_optimization:
            self.optimization_curve = []
        
        # Initial energy
        e_init = self._evaluate_energy(current_h1, current_eri, dets)
        log_important(f"Initial Energy: {e_init:.8f}")

        n = self.n_orb
        n_params = len(self.idx_map)  # Respects attentive_orbitals
        x0 = np.zeros(n_params)
        
        # Optional: add small random perturbation to break symmetry
        # This helps avoid getting stuck in saddle points or symmetric local minima
        initial_perturbation = optimizer_options_dict.get('initial_perturbation', 0.0)
        if initial_perturbation > 0:
            x0 += np.random.randn(n_params) * initial_perturbation
            if callback_func:
                callback_func(f"Initial perturbation: {initial_perturbation:.2e}")

        # Gradient configuration
        gradient_mode = optimizer_options_dict.get('gradient_mode', 'full')  # Default to FAST mode
        active_parameter = optimizer_options_dict.get('active_parameter', 0.90)  # Default 90% for L1/L2
        if gradient_mode == "full" or gradient_mode == "analytical":
            active_parameter = None
        
        if callback_func:
            if active_parameter is not None:
                callback_func(f"Gradient mode: {gradient_mode}, active_parameter: {active_parameter}")
            else:
                callback_func(f"Gradient mode: {gradient_mode}")


        start_time = time.time()
        
        # Callback to log progress
        def callback(x):
            if self.last_x is not None and np.allclose(x, self.last_x, atol=1e-14):
                e = self.last_energy
            else:
                e = objective(x)

            if record_optimization:
                self.optimization_curve.append(e)
            if callback_func:
                callback_func(f"  Optimizer step: Energy = {e:.8f}")

        # Extract optimizer method
        method = optimizer_options_dict.get('optimizer', 'L-BFGS-B')

        # Prepare options for minimize
        exclude_keys = {'gradient_mode', 'active_parameter', 'cycles', 'jac', 
                       'record_optimization', 'optimizer', 'threshold', 
                       'max_micro_cycles', 'davidson_tol', 'davidson_max_iter',
                       'initial_perturbation'}
        minimize_options = {k: v for k, v in optimizer_options_dict.items() if k not in exclude_keys}
        
        if 'maxiter' not in minimize_options:
            minimize_options['maxiter'] = optimizer_options_dict.get('max_micro_cycles', 
                                                                      optimizer_options_dict.get('maxiter', 100))
        if 'maxfun' not in minimize_options:
            minimize_options['maxfun'] = optimizer_options_dict.get('maxfun', 50000)
        if 'ftol' not in minimize_options:
            minimize_options['ftol'] = optimizer_options_dict.get('threshold', 
                                                                   optimizer_options_dict.get('ftol', 1e-10))
        if 'gtol' not in minimize_options:
            minimize_options['gtol'] = optimizer_options_dict.get('gtol', 1e-8)
        if 'disp' not in minimize_options:
            minimize_options['disp'] = self.verbose or optimizer_options_dict.get('verbose', False)

        tol = optimizer_options_dict.get('davidson_tol', 1e-6)
        davidson_max_iter = optimizer_options_dict.get('davidson_max_iter', 500)
        
        # Pre-convert integrals to list format for C++ (avoids O(N^4) conversion per gradient call)
        h1_list_cached = current_h1.tolist()
        if current_eri.ndim == 4:
            eri_list_cached = current_eri.reshape(-1).tolist()
        else:
            eri_list_cached = current_eri.tolist() if hasattr(current_eri, 'tolist') else list(current_eri)
        
        # Pre-load cache for gradient computation
        funcs = get_functions_for_system(n)
        load_cache_func = funcs['load_or_create_Hij_cache']
        grad_cache, _ = load_cache_func(self.mol_name, self.n_elec, n, "cache_orblab")
        # Timing accumulators
        timing_stats = {
            'objective_calls': 0,
            'unpack_kappa': 0.0,
            'transform_integrals': 0.0,
            'evaluate_energy': 0.0,
            'jac_calls': 0,
            'jac_gradient_selection': 0.0,
            'jac_fd_compute': 0.0,
        }
        
        def objective(x):
            timing_stats['objective_calls'] += 1
            
            t0 = time.time()
            U = self.unpack_kappa(x)
            timing_stats['unpack_kappa'] += time.time() - t0
            
            t0 = time.time()
            h1_rot, eri_rot = self.transform_integrals(current_h1, current_eri, U)
            timing_stats['transform_integrals'] += time.time() - t0
            
            t0 = time.time()
            e, _, coeffs_out = self._evaluate_energy_full(h1_rot, eri_rot, dets, tol, davidson_max_iter=davidson_max_iter)
            timing_stats['evaluate_energy'] += time.time() - t0
            
            # Cache for gradient computation
            self.last_x = x.copy()
            self.last_energy = e
            self.last_h1_rot = h1_rot
            self.last_eri_rot = eri_rot
            self.last_coeffs = list(coeffs_out)
            
            return e

        def jac_wrapper(x):
            """
            Compute gradient using the configured gradient_mode.
            
            Gradient Modes:
            ---------------
            - 'analytical': Analytical gradient from RDM (fastest, ~500x vs FD)
            - 'full': Full FD on all parameters (n_params evaluations)
            - 'active_l1': Select params capturing active_parameter fraction of L1 norm
            - 'active_l2': Select params capturing active_parameter fraction of L2² energy
            - 'active_topk': Select top active_parameter parameters
            
            Attentive Mode:
            ---------------
            When attentive_orbitals is set, the optimizer works in "reduced space":
            - n_params = C(k, 2) where k = len(attentive_orbitals), not C(n_orb, 2)
            - self.idx_map contains only pairs within attentive set
            - x and active_indices are in reduced space
            
            C++ Interface Note:
            -------------------
            The C++ compute_fd_gradient_parallel expects FULL-length x (n*(n-1)/2).
            In attentive mode, we expand x to full length before calling C++,
            then map the gradient back to reduced space.
            
            Python fallback works directly in reduced space via self.unpack_kappa.
            """
            timing_stats['jac_calls'] += 1
            # Adaptive FD step size based on davidson_tol
            # h_opt ~ 100 * tol balances truncation error (h*f'') and roundoff error (δE/h)
            # Upper bound 1e-5: prevent large truncation error
            # Lower bound 1e-7: prevent roundoff explosion even with very tight tol
            eps = max(1e-7, min(1e-5, 100 * tol))
            
            # Ensure we have cached state
            if self.last_x is None or not np.allclose(x, self.last_x, atol=1e-14):
                objective(x)
            
            e_center = self.last_energy
            h1_rot = self.last_h1_rot
            eri_rot = self.last_eri_rot
            coeffs = self.last_coeffs
            
            # ========== ANALYTICAL GRADIENT MODE ==========
            if gradient_mode == 'analytical':
                # Use RDM-based analytical gradient (no FD needed!)
                t0_analytical = time.time()
                
                # Pre-convert to lists for C++
                h1_rot_list = h1_rot.tolist()
                eri_rot_flat_list = eri_rot.reshape(-1).tolist() if eri_rot.ndim == 4 else eri_rot.tolist()
                
                # Compute analytical gradient using RDM
                G_analytical, _ = compute_orbital_gradient(
                    h1_rot_list, eri_rot_flat_list, dets, coeffs, n, self.n_elec,
                    attentive_orbitals=self.attentive_orbitals
                )
                
                # Extract gradient for our parameter space
                grad_vec = np.array([G_analytical[i, j] for i, j in self.idx_map])
                
                timing_stats['jac_gradient_selection'] += 0.0  # No selection needed
                timing_stats['jac_fd_compute'] += time.time() - t0_analytical
                
                return grad_vec
            
            # ========== FINITE DIFFERENCE MODES ==========
            t0_sel = time.time()
            if gradient_mode == 'full':
                # Full finite difference
                active_indices = list(range(n_params))
            else:
                # Pre-convert h1_rot and eri_rot to lists for compute_orbital_gradient
                # This avoids O(N^4) conversion inside the function
                h1_rot_list = h1_rot.tolist()
                eri_rot_flat_list = eri_rot.reshape(-1).tolist() if eri_rot.ndim == 4 else eri_rot.tolist()
                
                # Use updated_ci gradient for parameter selection
                # Pass attentive_orbitals for potential speedup in attentive mode
                G_upd, _ = compute_orbital_gradient(h1_rot_list, eri_rot_flat_list, dets, coeffs, n, self.n_elec,
                                                     attentive_orbitals=self.attentive_orbitals)
                grad_upd = np.array([G_upd[i, j] for i, j in self.idx_map])
                mag = np.abs(grad_upd)
                sorted_idx = np.argsort(mag)[::-1]
                sorted_mag = mag[sorted_idx]
                
                if gradient_mode == 'active_topk':
                    # Fixed top-K selection
                    k = min(int(active_parameter), n_params)
                    active_indices = sorted_idx[:k].tolist()
                    
                elif gradient_mode == 'active_l1':
                    # L1 norm threshold
                    total_l1 = np.sum(mag)
                    if total_l1 < 1e-20:
                        active_indices = list(range(n_params))
                    else:
                        target = active_parameter * total_l1
                        cumsum = np.cumsum(sorted_mag)
                        k = int(np.searchsorted(cumsum, target)) + 1
                        k = min(k, n_params)
                        active_indices = sorted_idx[:k].tolist()
                        
                elif gradient_mode == 'active_l2':
                    # L2² energy threshold
                    total_l2_sq = np.sum(mag**2)
                    if total_l2_sq < 1e-20:
                        active_indices = list(range(n_params))
                    else:
                        target = active_parameter * total_l2_sq
                        cumsum = np.cumsum(sorted_mag**2)
                        k = int(np.searchsorted(cumsum, target)) + 1
                        k = min(k, n_params)
                        active_indices = sorted_idx[:k].tolist()
                else:
                    raise ValueError(f"Unknown gradient_mode: {gradient_mode}")
            timing_stats['jac_gradient_selection'] += time.time() - t0_sel
            
            # ========== COMPUTE FD GRADIENT FOR ACTIVE PARAMETERS ==========
            grad_vec = np.zeros(n_params)  # Result in reduced space
            
            # Detect attentive mode: reduced parameter count indicates attentive optimization
            n_full_params = n * (n - 1) // 2
            is_attentive_mode = (n_params < n_full_params)
            
            # C++ OpenMP-based parallel FD computation
            # Using pre-cached h1_list_cached, eri_list_cached, and grad_cache
            t0_fd = time.time()
            try:
                
                if is_attentive_mode:
                    # ----- ATTENTIVE MODE: Index Space Mapping -----
                    # C++ expects full-length x, but we have reduced x.
                    # Use precomputed self.reduced_to_full mapping from __init__
                    
                    # Step 1: Expand x from reduced to full space
                    x_full = np.zeros(n_full_params)
                    for reduced_idx, full_idx in self.reduced_to_full.items():
                        x_full[full_idx] = x[reduced_idx]
                    
                    # Step 2: Map active_indices from reduced to full space
                    active_indices_full = [self.reduced_to_full[idx] for idx in active_indices]
                    
                    # Step 3: Call C++ with full-space parameters (using pre-cached data)
                    grad_vec_cpp = trimci_core.compute_fd_gradient_parallel(
                        dets, h1_list_cached, eri_list_cached, grad_cache,
                        x_full.tolist(), active_indices_full,  # Full space
                        n, self.n_elec, eps, tol, davidson_max_iter
                    )
                    
                    # Step 4: Map gradient from full back to reduced space
                    grad_full = np.array(grad_vec_cpp)
                    for reduced_idx, full_idx in self.reduced_to_full.items():
                        grad_vec[reduced_idx] = grad_full[full_idx]
                else:
                    # ----- FULL MODE: Direct call (no mapping needed) -----
                    grad_vec_cpp = trimci_core.compute_fd_gradient_parallel(
                        dets, h1_list_cached, eri_list_cached, grad_cache,
                        x.tolist(), active_indices,
                        n, self.n_elec, eps, tol, davidson_max_iter
                    )
                    grad_vec = np.array(grad_vec_cpp)
                
            except Exception as e:
                raise RuntimeError(f"C++ FD gradient computation failed: {e}")
            timing_stats['jac_fd_compute'] += time.time() - t0_fd
            
            return grad_vec

        # Run optimization
        # gradient_mode='none' means let scipy handle numerical gradient
        use_jac = (gradient_mode != 'none')
        
        # Check if using ML optimizer
        ml_optimizers = {'adam', 'sgd', 'rmsprop', 'adamw'}
        if method.lower() in ml_optimizers:
            # Use ML optimizer
            res = self._run_ml_optimizer(
                method.lower(), x0, objective, jac_wrapper if use_jac else None,
                optimizer_options_dict, callback if (record_optimization or callback_func) else None
            )
        else:
            # gradient_mode='none' means let scipy handle numerical gradient
            res = minimize(objective, x0, method=method, 
                           jac=jac_wrapper if use_jac else None,
                           options=minimize_options,
                           callback=callback if (record_optimization or callback_func) else None)
        
        elapsed = time.time() - start_time
        
        if callback_func:
            callback_func(f"  Optimization finished. Success: {res.success}, Message: {res.message}")
            callback_func(f"  Final Energy: {res.fun:.8f} (Change: {res.fun - e_init:.8f})")
            callback_func(f"  Time: {elapsed:.1f}s, Iterations: {res.nit}")
        
        # Print timing statistics
        if callback_func or self.verbose:
            log_func = callback_func if callback_func else print
            log_func(f"  [TIMING] objective calls: {timing_stats['objective_calls']}")
            log_func(f"  [TIMING]   unpack_kappa:       {timing_stats['unpack_kappa']*1000:.1f} ms")
            log_func(f"  [TIMING]   transform_integrals: {timing_stats['transform_integrals']*1000:.1f} ms")
            log_func(f"  [TIMING]   evaluate_energy:     {timing_stats['evaluate_energy']*1000:.1f} ms")
            log_func(f"  [TIMING] jac calls: {timing_stats['jac_calls']}")
            log_func(f"  [TIMING]   gradient_selection:  {timing_stats['jac_gradient_selection']*1000:.1f} ms")
            log_func(f"  [TIMING]   fd_compute:          {timing_stats['jac_fd_compute']*1000:.1f} ms")
        
        # Store timing stats for analysis
        self.timing_stats = timing_stats
        
        U_final = self.unpack_kappa(res.x)
        new_h1, new_eri = self.transform_integrals(current_h1, current_eri, U_final)
        
        return new_h1, new_eri, res.fun, res.success, U_final

    def _prepare_cpp_integrals(self, h1, eri):
        """Prepare integrals for C++ interface."""
        h1_list = h1.tolist()
        if eri.ndim == 4:
            eri_flat = eri.reshape(-1).tolist()
        else:
            eri_flat = list(eri) if hasattr(eri, '__iter__') else eri.tolist()
        return h1_list, eri_flat

    def _dets_to_128_bit(self, dets):
        """Convert Determinant64 or Determinant128 list to (alpha_list, beta_list)
        of length-2 uint64 arrays (128-bit format required by GPU orbopt primitives)."""
        import numpy as np
        alpha_list = []
        beta_list = []
        for d in dets:
            a = d.alpha
            b = d.beta
            if isinstance(a, (list, tuple)) or (hasattr(a, '__iter__') and not isinstance(a, (int, np.integer))):
                arr_a = list(a)
                arr_b = list(b)
                if len(arr_a) < 2: arr_a.append(0)
                if len(arr_b) < 2: arr_b.append(0)
                alpha_list.append([int(arr_a[0]) & 0xFFFFFFFFFFFFFFFF,
                                   int(arr_a[1]) & 0xFFFFFFFFFFFFFFFF])
                beta_list.append([int(arr_b[0]) & 0xFFFFFFFFFFFFFFFF,
                                  int(arr_b[1]) & 0xFFFFFFFFFFFFFFFF])
            else:
                alpha_list.append([int(a) & 0xFFFFFFFFFFFFFFFF, 0])
                beta_list.append([int(b) & 0xFFFFFFFFFFFFFFFF, 0])
        return alpha_list, beta_list

    def _optimize_gpu_newton(self, h1, eri, dets, coeffs, options, callback_func):
        """
        GPU-accelerated approximate-Newton orbital optimizer.

        Chains the Phase 1-3 GPU primitives in a Python outer loop:
          1. compute_orbital_gradient_gpu_128 (2-RDM → Fock → grad)
          2. Steepest-descent step scaled by step_scale, capped at max_param
          3. Matrix exponential rotation U = exp(X - X^T) on host (scipy.linalg.expm)
          4. transform_integrals_gpu (cuBLAS 4-index transform)
          5. Line search over step scales until energy improves
          6. CPU Davidson to refresh CI coefficients on rotated integrals
          7. Convergence on grad norm or energy change

        Same return tuple as the CPU approx_newton path:
          (new_h1, new_eri, final_energy, converged, U_final)

        Hessian-diagonal step is deferred — uses steepest-descent direction with
        line search for now, which is the simplest ||grad||-reducing update. Full
        approx-Newton with Hessian diagonal is a straightforward extension once
        the pipeline is proven.
        """
        import numpy as np
        from scipy.linalg import expm
        import trimci.trimci_core as tc
        if not hasattr(tc, 'compute_orbital_gradient_gpu_128'):
            raise RuntimeError("GPU orbopt requires trimci built with TRIMCI_HAS_GPU + Phase 3 gradient binding.")

        opts = options or {}
        max_iter = int(opts.get('maxiter', opts.get('max_micro_cycles', 50)))
        grad_tol = float(opts.get('grad_tol', opts.get('threshold', 1e-6)))
        energy_tol = float(opts.get('energy_tol', opts.get('ftol', 1e-8)))
        step_scale = float(opts.get('step_scale', 1.0))
        max_param = float(opts.get('max_param', 0.2))
        line_search = bool(opts.get('line_search', True))
        ls_max_iter = int(opts.get('ls_max_iter', 6))
        ls_shrink = float(opts.get('ls_shrink', 0.5))
        ls_energy_tol = float(opts.get('ls_energy_tol', 0.0))
        davidson_tol = float(opts.get('davidson_tol', 1e-3))
        davidson_max_iter = int(opts.get('davidson_max_iter', 200))
        record_trace = bool(opts.get('trace', opts.get('cpp_trace', False)))

        n_orb = int(self.n_orb)
        # Infer n_alpha / n_beta from first det
        def _popcount(v):
            if isinstance(v, (list, tuple)) or (hasattr(v, '__iter__') and not isinstance(v, int)):
                return sum(bin(int(w) & 0xFFFFFFFFFFFFFFFF).count('1') for w in v)
            return bin(int(v) & 0xFFFFFFFFFFFFFFFF).count('1')
        n_alpha = _popcount(dets[0].alpha)
        n_beta  = _popcount(dets[0].beta)

        dets_a_128, dets_b_128 = self._dets_to_128_bit(dets)

        # Force warm-start Davidson: line-search MUST trial against the SAME
        # eigenvalue root as the current state. Without warm-start, cold Davidson
        # on rotated integrals may converge to a different root → E_new not
        # comparable to E_prev, line search spuriously fails.
        _saved_warm = self.warm_initial
        self.warm_initial = True

        # Working integrals + rotation accumulator
        current_h1 = np.ascontiguousarray(h1, dtype=np.float64)
        if current_h1.ndim == 1:
            current_h1 = current_h1.reshape(n_orb, n_orb)
        current_eri = np.ascontiguousarray(eri, dtype=np.float64)
        if current_eri.ndim == 1:
            current_eri = current_eri.reshape(n_orb, n_orb, n_orb, n_orb)
        U_total = np.eye(n_orb)

        current_coeffs = np.asarray(coeffs, dtype=np.float64)
        norm = np.linalg.norm(current_coeffs)
        if norm > 1e-15:
            current_coeffs = current_coeffs / norm

        # Initial energy via existing CPU Davidson helper
        E_prev, _, coeffs_out = self._evaluate_energy_full(
            current_h1, current_eri, dets,
            tol=davidson_tol, initial_guess=list(current_coeffs),
            davidson_max_iter=davidson_max_iter)
        current_coeffs = np.asarray(coeffs_out)
        nm = np.linalg.norm(current_coeffs)
        if nm > 1e-15: current_coeffs = current_coeffs / nm

        if callback_func:
            callback_func(f"  [GPU Newton] init E={E_prev:.8f}")

        trace = []
        converged = False
        n_iter_done = 0
        final_grad_norm = float('inf')

        for it in range(max_iter):
            # 1. GPU gradient
            h1_flat = current_h1.flatten().tolist()
            eri_flat = current_eri.flatten().tolist()
            grad_flat = tc.compute_orbital_gradient_gpu_128(
                dets_a_128, dets_b_128, current_coeffs.tolist(),
                h1_flat, eri_flat, n_orb, n_alpha, n_beta)
            grad = np.asarray(grad_flat).reshape(n_orb, n_orb)
            grad_norm = float(np.linalg.norm(grad))
            final_grad_norm = grad_norm
            if record_trace:
                trace.append((E_prev, grad_norm))

            if callback_func:
                callback_func(f"  [GPU Newton] iter {it+1}: E={E_prev:.8f} |grad|={grad_norm:.3e}")

            if grad_norm < grad_tol:
                converged = True
                n_iter_done = it + 1
                break

            # 2. Steepest-descent direction with cap
            kappa_base = -step_scale * grad
            mx = float(np.max(np.abs(kappa_base)))
            if mx > max_param:
                kappa_base = kappa_base * (max_param / mx)

            # Compute 1-RDM and 2-RDM ONCE at current coeffs — used for fixed-C
            # line search evaluation (<C|H'|C> via RDM × integrals contraction).
            # This avoids Davidson root-switching during line search.
            rdm1_flat, rdm2_tri = tc.compute_rdm_gpu_128(
                dets_a_128, dets_b_128, current_coeffs.tolist(),
                n_orb, n_alpha, n_beta)
            rdm1_mat = np.asarray(rdm1_flat).reshape(n_orb, n_orb)

            def energy_at_fixed_C(h1_mat, eri_4d):
                # E = tr(rdm1 · h1) + 0.5 * Σ_pqrs rdm2_tri[combine4(p,q,r,s)] * eri[p,q,r,s]
                e1 = float(np.einsum('pq,pq->', rdm1_mat, h1_mat))
                e2 = 0.0
                N = n_orb
                # Unpack triangular rdm2 into full 4D once (fixed per iter)
                # O(N^4) extra memory + ops — negligible vs the N^5 gradient.
                rdm2_full = np.zeros((N, N, N, N))
                for p in range(N):
                    for q in range(N):
                        for r in range(N):
                            for s in range(N):
                                a = p*N + s; b = q*N + r
                                hi = max(a, b); lo = min(a, b)
                                rdm2_full[p, q, r, s] = rdm2_tri[(hi*(hi+1))//2 + lo]
                e2 = 0.5 * float(np.einsum('pqrs,pqrs->', rdm2_full, eri_4d))
                return e1 + e2

            # Precompute rdm2_full ONCE this iter (it doesn't depend on scale).
            _N = n_orb
            rdm2_full = np.zeros((_N, _N, _N, _N))
            for p in range(_N):
                for q in range(_N):
                    for r in range(_N):
                        for s in range(_N):
                            a = p*_N + s; b = q*_N + r
                            hi = max(a, b); lo = min(a, b)
                            rdm2_full[p, q, r, s] = rdm2_tri[(hi*(hi+1))//2 + lo]
            def _E_fixed_C(h1_mat, eri_4d):
                e1 = float(np.einsum('pq,pq->', rdm1_mat, h1_mat))
                e2 = 0.5 * float(np.einsum('pqrs,pqrs->', rdm2_full, eri_4d))
                return e1 + e2

            # 3. Line search using fixed-C energy estimator
            scale = 1.0
            accepted = False
            best_h1 = None; best_eri = None
            best_U_step = None
            iters_ls = ls_max_iter if line_search else 1
            for _ls in range(iters_ls):
                X = scale * kappa_base   # already anti-symmetric
                U_step = expm(X)
                h1_new_flat, eri_new_flat = tc.transform_integrals_gpu(
                    list(current_h1.flatten()), list(current_eri.flatten()),
                    list(U_step.flatten()), n_orb)
                h1_new = np.asarray(h1_new_flat).reshape(n_orb, n_orb)
                eri_new = np.asarray(eri_new_flat).reshape(n_orb, n_orb, n_orb, n_orb)
                # Restore exact 8-fold symmetries (FP drift over many cuBLAS
                # chains breaks them; Hubbard cannot tolerate it).
                h1_new = symmetrize_h1(h1_new)
                eri_new = symmetrize_eri(eri_new, psym=8)

                # Fixed-C energy estimator — no Davidson.
                E_trial = _E_fixed_C(h1_new, eri_new)

                if callback_func:
                    callback_func(f"    [GPU Newton]   LS scale={scale:.4f}: "
                                  f"E_trial(fixed C)={E_trial:.8f} "
                                  f"{'accept' if E_trial < E_prev - ls_energy_tol else 'reject'}")

                if E_trial < E_prev - ls_energy_tol:
                    accepted = True
                    best_h1 = h1_new; best_eri = eri_new
                    best_U_step = U_step
                    break
                scale *= ls_shrink

            if not accepted:
                if callback_func:
                    callback_func(f"  [GPU Newton] line search failed at iter {it+1}; stopping")
                n_iter_done = it + 1
                break

            # Commit rotation; then ONE Davidson refresh of CI coefficients.
            current_h1 = best_h1
            current_eri = best_eri
            U_total = U_total @ best_U_step
            E_refresh, _, coeffs_out = self._evaluate_energy_full(
                current_h1, current_eri, dets, tol=davidson_tol,
                initial_guess=list(current_coeffs), davidson_max_iter=davidson_max_iter)
            current_coeffs = np.asarray(coeffs_out)
            nm = np.linalg.norm(current_coeffs)
            if nm > 1e-15: current_coeffs = current_coeffs / nm
            dE = E_refresh - E_prev
            E_prev = E_refresh
            n_iter_done = it + 1
            if callback_func:
                callback_func(f"    [GPU Newton]   post-Davidson E={E_prev:.8f} (dE={dE:+.3e})")
            if abs(dE) < energy_tol and it > 0:
                converged = True
                break

        if callback_func:
            callback_func(f"  [GPU Newton] done: iters={n_iter_done} E={E_prev:.8f} "
                          f"|grad|={final_grad_norm:.3e} converged={converged}")

        # Restore saved warm_initial state before returning.
        self.warm_initial = _saved_warm
        return current_h1, current_eri, E_prev, converged, U_total

    def _optimize_gpu_bfgs(self, h1, eri, dets, coeffs, options, callback_func):
        """GPU BFGS orbital optimizer — direct port of C++ orbital_bfgs_optimize_t.

        Explicit BFGS outer loop with owned N_param x N_param Hessian, first-iter
        analytic Hessian diagonal from (1-RDM, 2-RDM, Fock), rank-2 BFGS update,
        accept/reject line search. GPU primitives replace the hot numerics:
          - compute_rdm_gpu_128            : 1-/2-RDM at iter 0 (for hess_diag)
          - compute_orbital_gradient_gpu_128: analytic gradient every iter
          - transform_integrals_gpu        : cuBLAS 4-index transform per LS trial

        CRITICAL: coeffs_cur (Davidson warm-start anchor) updates ONLY on
        accepted line-search steps, mirroring the CPU loop. Earlier scipy-based
        implementation moved the anchor on every trial, polluting warm-start.

        Returns (h1_rot, eri_rot, total_energy, converged, U_total) — same
        shape as _optimize_cpp_bfgs.
        """
        import numpy as np
        from scipy.linalg import expm
        import trimci.trimci_core as tc
        if not hasattr(tc, 'compute_orbital_gradient_gpu_128'):
            raise RuntimeError("GPU BFGS requires trimci built with Phase 3 gradient binding.")
        if not hasattr(tc, 'compute_rdm_gpu_128'):
            raise RuntimeError("GPU BFGS requires trimci built with compute_rdm_gpu_128 binding.")

        opts = options or {}
        # Defaults mirror _optimize_cpp_bfgs.
        max_iter = int(opts.get('maxiter', opts.get('max_micro_cycles', 50)))
        grad_tol = float(opts.get('grad_tol', opts.get('threshold', 1e-6)))
        energy_tol = float(opts.get('energy_tol', opts.get('ftol', 1e-8)))
        step_scale = float(opts.get('step_scale', 1.0))
        min_hess = float(opts.get('min_hess', 1e-5))
        max_param = float(opts.get('max_param', 0.2))
        line_search = bool(opts.get('line_search', True))
        ls_max_iter = int(opts.get('ls_max_iter', 6))
        ls_shrink = float(opts.get('ls_shrink', 0.5))
        ls_energy_tol = float(opts.get('ls_energy_tol', 0.0))
        davidson_tol = float(opts.get('davidson_tol', 1e-3))
        davidson_max_iter = int(opts.get('davidson_max_iter', 200))
        bfgs_damp = float(opts.get('bfgs_damp', 1e-3))
        bfgs_update_tol = float(opts.get('bfgs_update_tol', 1e-6))
        bfgs_step_factor = float(opts.get('bfgs_step_factor', 5.0))
        bfgs_step_decay = float(opts.get('bfgs_step_decay', 0.97))
        bfgs_step_min = float(opts.get('bfgs_step_min', 0.25))
        bfgs_init_div = float(opts.get('bfgs_init_div', 4.0))

        n_orb = int(self.n_orb)

        def _popcount(v):
            if isinstance(v, (list, tuple)) or (hasattr(v, '__iter__') and not isinstance(v, int)):
                return sum(bin(int(w) & 0xFFFFFFFFFFFFFFFF).count('1') for w in v)
            return bin(int(v) & 0xFFFFFFFFFFFFFFFF).count('1')
        n_alpha = _popcount(dets[0].alpha)
        n_beta = _popcount(dets[0].beta)
        dets_a_128, dets_b_128 = self._dets_to_128_bit(dets)

        # Warm-start Davidson is essential: trial integrals are close to the
        # current point, so the coeffs_cur guess keeps Davidson on the same root.
        _saved_warm = self.warm_initial
        self.warm_initial = True

        h1_cur = np.ascontiguousarray(h1, dtype=np.float64)
        if h1_cur.ndim == 1:
            h1_cur = h1_cur.reshape(n_orb, n_orb)
        else:
            h1_cur = h1_cur.copy()
        eri_cur = np.ascontiguousarray(eri, dtype=np.float64)
        if eri_cur.ndim == 1:
            eri_cur = eri_cur.reshape(n_orb, n_orb, n_orb, n_orb)
        else:
            eri_cur = eri_cur.copy()
        U_total = np.eye(n_orb)

        coeffs_cur = np.asarray(coeffs, dtype=np.float64).copy()
        nm = np.linalg.norm(coeffs_cur)
        if nm > 1e-15: coeffs_cur /= nm

        # Initial Davidson on the input integrals.
        E_total, _, c_out = self._evaluate_energy_full(
            h1_cur, eri_cur, dets, tol=davidson_tol,
            initial_guess=list(coeffs_cur), davidson_max_iter=davidson_max_iter)
        coeffs_cur = np.asarray(c_out)
        nm = np.linalg.norm(coeffs_cur)
        if nm > 1e-15: coeffs_cur /= nm
        energy = E_total  # track total energy throughout
        prev_energy = energy
        best_energy = energy
        best_h1 = h1_cur.copy()
        best_eri = eri_cur.copy()
        best_U = U_total.copy()
        best_grad_norm = 0.0

        if callback_func:
            callback_func(f"  [GPU BFGS] init E={energy:.8f}")

        if max_iter <= 0:
            self.warm_initial = _saved_warm
            return h1_cur, eri_cur, energy, False, U_total

        idx_map = [(i, j) for i in range(n_orb) for j in range(i + 1, n_orb)]
        dim = len(idx_map)
        grad = np.zeros(dim)
        grad_prev = np.zeros(dim)
        update_prev = np.zeros(dim)
        hess = np.zeros((dim, dim))

        first_iter = True
        initial_update_norm = 0.0
        step_size_factor = bfgs_step_factor
        converged = False
        fell_back = False
        iter_done = 0
        grad_norm = 0.0

        for it in range(max_iter):
            # 1. GPU gradient at current (h1, eri, coeffs).
            grad_flat = tc.compute_orbital_gradient_gpu_128(
                dets_a_128, dets_b_128, coeffs_cur.tolist(),
                list(h1_cur.flatten()), list(eri_cur.flatten()),
                n_orb, n_alpha, n_beta)
            grad_mat = np.asarray(grad_flat).reshape(n_orb, n_orb)
            for k, (p, q) in enumerate(idx_map):
                grad[k] = grad_mat[p, q]

            grad_norm = float(np.linalg.norm(grad))
            iter_done = it + 1
            if not np.isfinite(grad_norm):
                fell_back = True
                break

            if it == 0 or energy < best_energy:
                best_energy = energy
                best_grad_norm = grad_norm
                best_U = U_total.copy()
                best_h1 = h1_cur.copy()
                best_eri = eri_cur.copy()

            # 2. Build or update Hessian.
            if first_iter:
                # Iter-0 analytic Hessian diagonal — port of hessian_part() in
                # orbital_opt.cpp:1219-1237. Needs 1-RDM, 2-RDM, Fock.
                rdm1_flat, rdm2_tri = tc.compute_rdm_gpu_128(
                    dets_a_128, dets_b_128, coeffs_cur.tolist(),
                    n_orb, n_alpha, n_beta)
                rdm1 = np.asarray(rdm1_flat).reshape(n_orb, n_orb)
                rdm2_tri = np.asarray(rdm2_tri)
                N = n_orb
                # Unpack triangular rdm2 (index = hi*(hi+1)/2 + lo, a=p*N+s, b=q*N+r).
                pi, qi, ri, si = np.indices((N, N, N, N))
                a = pi * N + si
                b = qi * N + ri
                hi = np.maximum(a, b)
                lo = np.minimum(a, b)
                tri_idx = (hi * (hi + 1)) // 2 + lo
                rdm2_full = rdm2_tri[tri_idx]
                # Fock: F[m,n] = Σ_q γ[m,q] h1[n,q] + Σ_qrs Γ[m,r,s,q] eri[n,q,r,s]
                fock = rdm1 @ h1_cur.T + np.einsum(
                    'mrsq,nqrs->mn', rdm2_full, eri_cur)

                def hess_part(p0, q0, r0, s0):
                    elem = 2.0 * rdm1[p0, r0] * h1_cur[q0, s0]
                    if q0 == s0:
                        elem -= (fock[p0, r0] + fock[r0, p0])
                    # Σ_{m,n} [(Γ[p0,r0,n,m] + Γ[p0,n,r0,m]) * eri[q0,m,n,s0]
                    #        + Γ[p0,m,n,r0] * eri[q0,s0,m,n]]
                    A = rdm2_full[p0, r0, :, :] + rdm2_full[p0, :, r0, :]  # (n, m)
                    B = eri_cur[q0, :, :, s0]                              # (m, n)
                    y1 = float(np.einsum('nm,mn->', A, B))
                    C = rdm2_full[p0, :, :, r0]                            # (m, n)
                    D = eri_cur[q0, s0, :, :]                              # (m, n)
                    y2 = float(np.einsum('mn,mn->', C, D))
                    return elem + 2.0 * (y1 + y2)

                hess_diag = np.zeros(dim)
                for k, (p, q) in enumerate(idx_map):
                    hdiag = (hess_part(p, q, p, q)
                             - 2.0 * hess_part(p, q, q, p)
                             + hess_part(q, p, q, p))
                    hess_diag[k] = max(hdiag / bfgs_init_div, min_hess)
                hess = np.diag(hess_diag)
                grad_prev = np.zeros(dim)
                update_prev = np.zeros(dim)
                # Release large temporaries.
                del rdm2_full, rdm2_tri, tri_idx, hi, lo, a, b, pi, qi, ri, si
            else:
                y = grad - grad_prev
                hs = hess @ update_prev
                ys = float(y @ update_prev)
                shs = float(hs @ update_prev)
                if ys > bfgs_update_tol and abs(shs) > bfgs_update_tol:
                    hess = hess + np.outer(y, y) / ys - np.outer(hs, hs) / shs

            # 3. Solve hess_shift @ new_param = -grad.
            hess_shift = hess + bfgs_damp * np.eye(dim)
            try:
                new_param = np.linalg.solve(hess_shift, -grad)
            except np.linalg.LinAlgError:
                new_param = np.linalg.lstsq(hess_shift, -grad, rcond=None)[0]
            if not np.all(np.isfinite(new_param)):
                new_param = np.zeros(dim)
            if step_scale != 1.0:
                new_param = new_param * step_scale

            # 4. Step bounds.
            update_norm = float(np.linalg.norm(new_param))
            if first_iter:
                initial_update_norm = update_norm
                first_iter = False
            elif initial_update_norm > 0.0:
                max_norm = step_size_factor * initial_update_norm
                if update_norm > max_norm:
                    new_param = new_param * (max_norm / update_norm)
                    update_norm = max_norm
                step_size_factor = max(bfgs_step_min,
                                       step_size_factor * bfgs_step_decay)
            if max_param > 0.0 and new_param.size > 0:
                max_abs = float(np.max(np.abs(new_param)))
                if max_abs > max_param:
                    new_param = new_param * (max_param / max_abs)

            # 5. Gradient convergence check.
            if grad_norm < grad_tol:
                converged = True
                break

            # 6. Line search — trial integrals from CURRENT (h1_cur, eri_cur),
            # Davidson warm-started from coeffs_cur (anchor, unchanged on reject).
            accepted = False
            alpha_ls = 1.0
            energy_slack = max(ls_energy_tol,
                               1e-12 * max(1.0, abs(prev_energy)))
            ls_iters = max(1, ls_max_iter) if line_search else 1
            accepted_param = np.zeros(dim)
            for ls in range(ls_iters):
                params = new_param * alpha_ls
                if not np.all(np.isfinite(params)):
                    alpha_ls *= ls_shrink
                    continue
                # Anti-symmetric X with CPU sign convention (orbital_opt.cpp:589-591):
                #   X[p,q] = -params[k],  X[q,p] = +params[k]
                X = np.zeros((n_orb, n_orb))
                for k, (p, q) in enumerate(idx_map):
                    X[p, q] = -params[k]
                    X[q, p] = params[k]
                U_step = expm(X)

                h1_rot_flat, eri_rot_flat = tc.transform_integrals_gpu(
                    list(h1_cur.flatten()), list(eri_cur.flatten()),
                    list(U_step.flatten()), n_orb)
                h1_rot = np.asarray(h1_rot_flat).reshape(n_orb, n_orb)
                eri_rot = np.asarray(eri_rot_flat).reshape(
                    n_orb, n_orb, n_orb, n_orb)

                # Restore exact integral symmetries after the cuBLAS DGEMM
                # chain. transform_integrals_gpu does not enforce these and
                # FP rounding accumulates over many cycles. For Hubbard's
                # super-sparse + symmetric ERI this drift breaks structural
                # zeros and causes Davidson to lock onto wrong roots in
                # later cycles (multi-cycle energy regression observed,
                # 2026-04-25). Uses the same routine FCIDUMP loading uses,
                # so cycle k+1 sees integrals with identical symmetry to
                # the initial input.
                h1_rot = symmetrize_h1(h1_rot)
                eri_rot = symmetrize_eri(eri_rot, psym=8)

                E_try, _, c_try = self._evaluate_energy_full(
                    h1_rot, eri_rot, dets, tol=davidson_tol,
                    initial_guess=list(coeffs_cur),
                    davidson_max_iter=davidson_max_iter)

                if not np.isfinite(E_try):
                    alpha_ls *= ls_shrink
                    continue

                accept = (not line_search) or (E_try <= prev_energy + energy_slack)
                if callback_func:
                    callback_func(
                        f"    [GPU BFGS]   LS alpha={alpha_ls:.4f}: "
                        f"E_try={E_try:.8f} dE={E_try - prev_energy:+.3e} "
                        f"{'accept' if accept else 'reject'}")

                if accept:
                    accepted = True
                    accepted_param = params.copy()
                    U_total = U_total @ U_step
                    h1_cur = h1_rot
                    eri_cur = eri_rot
                    energy = E_try
                    coeffs_cur = np.asarray(c_try)
                    nm = np.linalg.norm(coeffs_cur)
                    if nm > 1e-15: coeffs_cur /= nm
                    break
                alpha_ls *= ls_shrink

            if not accepted:
                if callback_func:
                    callback_func(
                        f"  [GPU BFGS] line search failed at iter {iter_done}; "
                        f"falling back to best prior step")
                fell_back = True
                break

            if callback_func:
                callback_func(
                    f"  [GPU BFGS] iter {iter_done}: E={energy:.8f} "
                    f"|grad|={grad_norm:.3e}")

            # 7. BFGS secant update prep, convergence check.
            grad_prev = grad.copy()
            update_prev = accepted_param.copy()
            delta_energy = abs(energy - prev_energy)
            if energy_tol > 0.0 and delta_energy < energy_tol:
                converged = True
                break
            prev_energy = energy

        if fell_back:
            U_total = best_U
            h1_cur = best_h1
            eri_cur = best_eri
            energy = best_energy
            grad_norm = best_grad_norm
            converged = False

        self.warm_initial = _saved_warm
        if callback_func:
            callback_func(
                f"  [GPU BFGS] done: iters={iter_done} E={energy:.8f} "
                f"|grad|={grad_norm:.3e} converged={converged}")
        return h1_cur, eri_cur, float(energy), converged, U_total

    def _optimize_cpp_approx_newton(self, h1, eri, dets, coeffs, options, callback_func):
        """
        C++ Gradient-Based Orbital Optimization using Approximate Newton.

        Uses analytic gradients from RDM, updates CI each iteration.
        """
        opts = options or {}
        rotate_integrals = opts.get('cpp_rotate_integrals', True)
        step_scale = float(opts.get('step_scale', 1.0))
        min_hess = float(opts.get('min_hess', 1e-5))
        max_param = float(opts.get('max_param', 0.2))
        line_search = bool(opts.get('line_search', True))
        ls_max_iter = int(opts.get('ls_max_iter', 6))
        ls_shrink = float(opts.get('ls_shrink', 0.5))
        ls_energy_tol = float(opts.get('ls_energy_tol', 0.0))
        max_iter = int(opts.get('maxiter', opts.get('max_micro_cycles', 50)))
        grad_tol = float(opts.get('grad_tol', opts.get('threshold', 1e-6)))
        energy_tol = float(opts.get('energy_tol', opts.get('ftol', 1e-8)))
        davidson_tol = float(opts.get('davidson_tol', 1e-3))
        davidson_max_iter = int(opts.get('davidson_max_iter', 200))
        trace = bool(opts.get('trace', opts.get('cpp_trace', False)))

        h1_list, eri_flat = self._prepare_cpp_integrals(h1, eri)
        coeffs_list = list(coeffs) if hasattr(coeffs, '__iter__') else [coeffs]

        funcs = get_functions_for_system(self.n_orb)
        U, h1_rot, eri_rot, grad_norm, energy, converged, n_iter, trace_data = \
            funcs['orbital_approx_newton_optimize'](
                dets, coeffs_list, h1_list, eri_flat, self.n_orb, -1, -1,
                max_iter, grad_tol, energy_tol, step_scale, min_hess, max_param,
                line_search, ls_max_iter, ls_shrink, ls_energy_tol,
                davidson_tol, davidson_max_iter, rotate_integrals, trace
            )

        if rotate_integrals and len(h1_rot) > 0:
            new_h1 = np.array(h1_rot, dtype=float)
            new_eri = np.array(eri_rot, dtype=float).reshape((self.n_orb,) * 4)
        else:
            new_h1, new_eri = self.transform_integrals(h1, eri, np.array(U))

        energy_total = energy + self.nuclear_repulsion
        if callback_func:
            if trace and trace_data:
                for idx, (e_step, g_step) in enumerate(trace_data, start=1):
                    e_total = e_step + self.nuclear_repulsion
                    callback_func(f"  CPP approx_newton iter {idx}: grad={g_step:.3e} E={e_total:.8f}")
            callback_func(f"  CPP approx_newton: iter={n_iter} grad={grad_norm:.3e} E={energy_total:.8f} conv={converged}")

        return new_h1, new_eri, energy_total, converged, np.array(U)

    def _optimize_cpp_bfgs(self, h1, eri, dets, coeffs, options, callback_func):
        """
        C++ Gradient-Based Orbital Optimization using BFGS.
        
        Uses analytic gradients from RDM, updates CI each iteration.
        """
        opts = options or {}
        rotate_integrals = opts.get('cpp_rotate_integrals', True)
        step_scale = float(opts.get('step_scale', 1.0))
        min_hess = float(opts.get('min_hess', 1e-5))
        max_param = float(opts.get('max_param', 0.2))
        max_iter = int(opts.get('maxiter', opts.get('max_micro_cycles', 50)))
        grad_tol = float(opts.get('grad_tol', opts.get('threshold', 1e-6)))
        energy_tol = float(opts.get('energy_tol', opts.get('ftol', 1e-8)))
        davidson_tol = float(opts.get('davidson_tol', 1e-3))
        davidson_max_iter = int(opts.get('davidson_max_iter', 200))
        trace = bool(opts.get('trace', opts.get('cpp_trace', False)))
        line_search = bool(opts.get('line_search', True))
        ls_max_iter = int(opts.get('ls_max_iter', 6))
        ls_shrink = float(opts.get('ls_shrink', 0.5))
        ls_energy_tol = float(opts.get('ls_energy_tol', 0.0))
        # BFGS-specific
        bfgs_damp = float(opts.get('bfgs_damp', 1e-3))
        bfgs_update_tol = float(opts.get('bfgs_update_tol', 1e-6))
        bfgs_step_factor = float(opts.get('bfgs_step_factor', 5.0))
        bfgs_step_decay = float(opts.get('bfgs_step_decay', 0.97))
        bfgs_step_min = float(opts.get('bfgs_step_min', 0.25))
        bfgs_init_div = float(opts.get('bfgs_init_div', 4.0))

        h1_list, eri_flat = self._prepare_cpp_integrals(h1, eri)
        coeffs_list = list(coeffs) if hasattr(coeffs, '__iter__') else [coeffs]

        funcs = get_functions_for_system(self.n_orb)
        U, h1_rot, eri_rot, grad_norm, energy, converged, n_iter, trace_data = \
            funcs['orbital_bfgs_optimize'](
                dets, coeffs_list, h1_list, eri_flat, self.n_orb, -1, -1,
                max_iter, grad_tol, energy_tol, step_scale, min_hess, max_param,
                bfgs_damp, bfgs_update_tol, bfgs_step_factor, bfgs_step_decay,
                bfgs_step_min, bfgs_init_div, line_search, ls_max_iter,
                ls_shrink, ls_energy_tol, davidson_tol, davidson_max_iter,
                rotate_integrals, trace
            )

        if rotate_integrals and len(h1_rot) > 0:
            new_h1 = np.array(h1_rot, dtype=float)
            new_eri = np.array(eri_rot, dtype=float).reshape((self.n_orb,) * 4)
        else:
            new_h1, new_eri = self.transform_integrals(h1, eri, np.array(U))

        energy_total = energy + self.nuclear_repulsion
        if callback_func:
            if trace and trace_data:
                for idx, (e_step, g_step) in enumerate(trace_data, start=1):
                    e_total = e_step + self.nuclear_repulsion
                    callback_func(f"  CPP bfgs iter {idx}: grad={g_step:.3e} E={e_total:.8f}")
            callback_func(f"  CPP bfgs: iter={n_iter} grad={grad_norm:.3e} E={energy_total:.8f} conv={converged}")

        return new_h1, new_eri, energy_total, converged, np.array(U)

    def _run_ml_optimizer(self, method, x0, objective, jac_func, options, callback):
        """
        Run ML-style optimizer (Adam, SGD, RMSprop, AdamW).
        
        Returns a scipy-compatible OptimizeResult object.
        """
        from types import SimpleNamespace
        
        # Extract options
        lr = options.get('learning_rate', 0.01)
        maxiter = options.get('maxiter', 1000)
        gtol = options.get('gtol', 1e-6)
        beta1 = options.get('beta1', 0.9)  # Adam momentum
        beta2 = options.get('beta2', 0.999)  # Adam RMSprop factor
        eps = options.get('eps', 1e-8)  # Numerical stability
        momentum = options.get('momentum', 0.9)  # SGD momentum
        weight_decay = options.get('weight_decay', 0.0)  # AdamW
        
        n_params = len(x0)
        x = x0.copy()
        
        # Initialize state
        if method == 'adam' or method == 'adamw':
            m = np.zeros(n_params)  # First moment
            v = np.zeros(n_params)  # Second moment
        elif method == 'rmsprop':
            v = np.zeros(n_params)  # Moving average of squared gradients
        elif method == 'sgd':
            velocity = np.zeros(n_params)  # Momentum
        
        best_x = x.copy()
        best_energy = float('inf')
        converged = False
        message = "Maximum iterations reached"
        
        for i in range(maxiter):
            # Compute objective and gradient
            energy = objective(x)
            
            if jac_func is not None:
                grad = jac_func(x)
            else:
                # Numerical gradient
                grad = np.zeros(n_params)
                eps_fd = 1e-5
                for k in range(n_params):
                    x_plus = x.copy()
                    x_plus[k] += eps_fd
                    grad[k] = (objective(x_plus) - energy) / eps_fd
            
            # Track best
            if energy < best_energy:
                best_energy = energy
                best_x = x.copy()
            
            # Check convergence
            grad_norm = np.linalg.norm(grad)
            if grad_norm < gtol:
                converged = True
                message = f"Converged: gradient norm {grad_norm:.2e} < {gtol:.2e}"
                break
            
            # Update step
            if method == 'adam':
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad**2
                m_hat = m / (1 - beta1**(i+1))
                v_hat = v / (1 - beta2**(i+1))
                x = x - lr * m_hat / (np.sqrt(v_hat) + eps)
                
            elif method == 'adamw':
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad**2
                m_hat = m / (1 - beta1**(i+1))
                v_hat = v / (1 - beta2**(i+1))
                x = x - lr * (m_hat / (np.sqrt(v_hat) + eps) + weight_decay * x)
                
            elif method == 'rmsprop':
                v = beta2 * v + (1 - beta2) * grad**2
                x = x - lr * grad / (np.sqrt(v) + eps)
                
            elif method == 'sgd':
                velocity = momentum * velocity + grad
                x = x - lr * velocity
            
            # Callback
            if callback is not None:
                callback(x)
        
        # Return scipy-compatible result
        result = SimpleNamespace(
            x=best_x,
            fun=best_energy,
            success=converged,
            message=message,
            nit=i + 1,
            nfev=i + 1,
            jac=grad if jac_func else None
        )
        
        return result


    def _evaluate_energy(self, h1, eri, dets):
        e, _, _ = self._evaluate_energy_full(h1, eri, dets)
        return e

    def _evaluate_energy_full(self, h1, eri, dets, tol=1e-3, initial_guess=None, davidson_max_iter=500):
        """
        Evaluate energy by diagonalizing H in the basis of dets using Davidson.
        Uses initial_guess for warm-start acceleration.
        
        Args:
            h1, eri: Integrals
            dets: List of determinants
            tol: Davidson convergence tolerance
            initial_guess: Optional initial CI coefficients for warm-start
            
        Returns:
            (total_energy, dets, coeffs)
        """
        n = self.n_orb
        n_dets = len(dets)
        
        try:
            # Get system-specific C++ functions
            funcs = get_functions_for_system(n)
            diag_func = funcs['diagonalize_subspace_davidson']
            load_cache_func = funcs['load_or_create_Hij_cache']
            
            # Load or create cache
            cache, _ = load_cache_func(self.mol_name, self.n_elec, n, "cache_orblab")
            
            # Ensure ERI is flat for C++
            if eri.ndim == 4:
                eri_flat = eri.reshape(-1)
            else:
                eri_flat = eri
            
            # Prepare initial guess based on warm_initial setting
            if self.warm_initial:
                if initial_guess is None:
                    if self.last_coeffs is not None and len(self.last_coeffs) == n_dets:
                        initial_guess = self.last_coeffs
                    else:
                        initial_guess = []
                elif len(initial_guess) != n_dets:
                    initial_guess = []
            else:
                initial_guess = []
            
            # Call Davidson diagonalizer
            energy, coeffs_out = diag_func(
                dets,           # determinants
                h1,             # 1-electron integrals
                eri_flat,       # flattened 2-electron integrals
                cache,          # Hij cache
                False,          # quantization
                davidson_max_iter,  # max_iter (configurable)
                tol,            # tolerance
                False,          # verbose
                n,              # n_orb
                initial_guess   # initial guess (empty = use C++ default)
            )
            
            # Store coefficients for next warm-start
            self.last_coeffs = np.array(coeffs_out)
            
            total_energy = energy + self.nuclear_repulsion
            return total_energy, dets, np.array(coeffs_out)
            
        except Exception as e:
            log_important(f"Error in energy evaluation: {e}")
            return 0.0, [], np.array([])

    def transform_integrals(self, h1, eri, U):
        """
        Transform integrals using optimized block-diagonal algorithm.
        
        Performance notes (N=36, k=12 benchmark, 2026-01-20):
        - Python BD:        ~10 ms  (uses NumPy + Apple Accelerate BLAS)
        - C++ Pure BD:      ~18 ms  (Eigen without optimized BLAS)
        - C++ via Python:   ~89 ms  (includes Python↔C++ data conversion)
        
        For attentive mode with block-diagonal U, Python BD is the fastest option
        when called from Python. C++ BD is preferred for internal C++ loops
        (e.g., compute_fd_gradient_parallel) where data is already in C++.
        
        Args:
            h1: (N, N) one-electron integrals
            eri: (N, N, N, N) or (N^4,) two-electron integrals
            U: (N, N) orbital rotation matrix
            
        Returns:
            h1_new, eri_new: Transformed integrals
        """
        n = h1.shape[0]
        
        # 1-body transform: h' = U^T h U
        h1_new = U.T @ h1 @ U
        
        # Ensure ERI is 4D
        if eri.ndim == 1:
            eri_4d = eri.reshape(n, n, n, n)
        else:
            eri_4d = eri

        # # Call C++
        # # Expecting h1 list-of-lists, eri list, U list-of-lists
        # h1_new_list, eri_new_flat_list = trimci_core.transform_integrals(
        #     h1.tolist(),
        #     eri_flat.tolist(),
        #     U.tolist()
        # )
        
        # # Convert back to numpy
        # h1_new = np.array(h1_new_list)
        # eri_new_flat = np.array(eri_new_flat_list)
        # eri_new = eri_new_flat.reshape(n, n, n, n)

        # 2-body transform
        att_orbs = self.attentive_orbitals if self.attentive_orbitals is not None else []
        
        if att_orbs and len(att_orbs) < n:
            # Block-diagonal optimized transform (Python BD)
            # Uses U_att^T for all 4 passes - this is correct!
            # Each pass does: sum_x U[x,y] * tensor[x,...] = U^T @ tensor.reshape(n, n^3)
            eri_new = self._transform_eri_block_diagonal(eri_4d, U, att_orbs)
        else:
            # Full transform using O(n^5) step-by-step einsum chain
            # Much faster than O(n^8) single einsum for n > 10
            # Gold standard: g'[i,j,k,l] = sum_{pqrs} U[p,i] U[q,j] U[r,k] U[s,l] g[p,q,r,s]
            # Each step contracts one index: g[...,old] @ U[old, new] -> g[...,new]
            tmp = np.einsum('ijkl,lt->ijkt', eri_4d, U)  # contract 4th index
            tmp = np.einsum('ijkt,ks->ijst', tmp, U)     # contract 3rd index
            tmp = np.einsum('ijst,jr->irst', tmp, U)     # contract 2nd index
            eri_new = np.einsum('irst,ip->prst', tmp, U) # contract 1st index
        
        return h1_new, eri_new
    
    def _transform_eri_block_diagonal(self, eri, U, attentive_orbitals):
        """
        Block-diagonal ERI transform: O(k*N^3) instead of O(N^5).
        
        When U is block-diagonal with U_att on attentive block and I elsewhere,
        only the attentive sub-block needs transformation.
        
        Key insight: All 4 passes use U_att^T (not a mix of U_att^T and U_att).
        This was validated against the gold standard np.einsum on 2026-01-20.
        """
        n = U.shape[0]
        k = len(attentive_orbitals)
        att = np.array(sorted(attentive_orbitals))
        U_att = U[np.ix_(att, att)]
        
        g = eri.copy()
        
        # All 4 passes use U_att^T
        # Pass 1: transform first index
        g[att, :, :, :] = (U_att.T @ g[att, :, :, :].reshape(k, -1)).reshape(k, n, n, n)
        
        # Pass 2: transform second index
        g_slice = g[:, att, :, :].transpose(1, 0, 2, 3).reshape(k, -1)
        g[:, att, :, :] = (U_att.T @ g_slice).reshape(k, n, n, n).transpose(1, 0, 2, 3)
        
        # Pass 3: transform third index
        g_slice = g[:, :, att, :].transpose(2, 0, 1, 3).reshape(k, -1)
        g[:, :, att, :] = (U_att.T @ g_slice).reshape(k, n, n, n).transpose(1, 2, 0, 3)
        
        # Pass 4: transform fourth index
        g_slice = g[:, :, :, att].transpose(3, 0, 1, 2).reshape(k, -1)
        g[:, :, :, att] = (U_att.T @ g_slice).reshape(k, n, n, n).transpose(1, 2, 3, 0)
        
        return g

    def unpack_kappa(self, x):
        """
        Convert flat parameter vector to unitary rotation matrix.
        
        For attentive optimization, x only contains parameters for attentive pairs.
        The resulting K matrix has zeros for non-attentive rotations.
        """
        K = np.zeros((self.n_orb, self.n_orb))
        
        # Use idx_map which respects attentive_orbitals
        for idx, (i, j) in enumerate(self.idx_map):
            val = x[idx]
            K[i, j] = val
            K[j, i] = -val
        
        return scipy.linalg.expm(K)


# Utility functions

def symmetrize_eri(eri, psym=8):
    """
    Symmetrize ERI according to psym level:
    psym = 8 : full 8-fold symmetry
    psym = 4 : (pq|rs) = (rs|pq)
    psym = 1 : no symmetrization
    """
    if psym == 1:
        return eri

    # Swap (pq) <-> (rs)
    eri = 0.5 * (eri + eri.transpose(2, 3, 0, 1))

    if psym == 4:
        # 4-fold only enforces (pq|rs) = (rs|pq)
        return eri

    if psym == 8:
        # enforce p<->q
        eri = 0.5 * (eri + eri.transpose(1, 0, 2, 3))
        # enforce r<->s
        eri = 0.5 * (eri + eri.transpose(0, 1, 3, 2))
        return eri

    raise ValueError(f"Unknown psym={psym}, must be 1, 4, or 8")

def symmetrize_h1(h1):
    return 0.5 * (h1 + h1.T)


def write_fcidump_with_psym(path, h1, eri, n_orb, n_elec, e_nuc, ms2, psym=8, tol=1e-15):
    """Write FCIDUMP. For psym=8, delegate to pyscf (canonical 8-fold).
    For psym!=8, write all entries (1-fold) with PSYM=psym in header so the
    TrimCI reader (entry/fcidump_reader.py) honors the requested symmetry.

    Reason: pyscf's tools.fcidump.from_integrals always writes 8-fold canonical;
    feeding it 4-fold ERI silently drops information. K-space / complex-orbital
    SCI workflows need PSYM=4 markers preserved across Phase 0 → Phase 1 → 2.
    """
    if psym == 8:
        from pyscf import tools as _pyscf_tools
        _pyscf_tools.fcidump.from_integrals(path, h1, eri, n_orb, n_elec, e_nuc, ms=ms2)
        return
    n = n_orb
    eri_4d = eri if eri.ndim == 4 else eri.reshape(n, n, n, n)
    h1_2d = h1 if h1.ndim == 2 else h1.reshape(n, n)
    with open(path, "w") as f:
        f.write(f" &FCI NORB= {n_orb}, NELEC= {n_elec}, MS2= {ms2},\n")
        f.write("  ORBSYM=" + ",".join(["1"] * n_orb) + ",\n")
        f.write(f"  ISYM=1, PSYM={psym},\n")
        f.write(" &END\n")
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        v = float(eri_4d[i, j, k, l])
                        if abs(v) > tol:
                            f.write(f" {v:24.16e}  {i+1:4d}  {j+1:4d}  {k+1:4d}  {l+1:4d}\n")
        for i in range(n):
            for j in range(n):
                v = float(h1_2d[i, j])
                if abs(v) > tol:
                    f.write(f" {v:24.16e}  {i+1:4d}  {j+1:4d}     0     0\n")
        f.write(f" {float(e_nuc):24.16e}     0     0     0     0\n")


def symmetrize_fcidump(fcidump_path, psym=8, verbosity=1):
    h1, eri, n_elec, n_orb, nuclear_repulsion, n_alpha, n_beta = read_fcidump(fcidump_path)
    h1 = symmetrize_h1(h1)
    eri = symmetrize_eri(eri, psym=psym)
    final_fcidump_path = os.path.join(Path(fcidump_path).parent, "fcidump_symmetrized")
    write_fcidump_with_psym(final_fcidump_path, h1, eri, n_orb, n_elec,
                            nuclear_repulsion, n_alpha - n_beta, psym=psym)
    if verbosity >= 1:
        print(f"FCIDump saved to {final_fcidump_path}")
