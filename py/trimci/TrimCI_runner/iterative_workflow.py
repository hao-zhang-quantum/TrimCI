"""
C++ backend iterative workflow wrapper.
"""
import time
from pathlib import Path

import numpy as np

from trimci.trimci_logging import log_important, log_verbose
from trimci import generate_reference_det, get_functions_for_system

from .io_utils import generate_unique_timestamp, save_final_results
from .det_utils import generate_initial_states, load_initial_dets_from_file, perturb_determinants


def _spin_key(bits):
    if isinstance(bits, np.ndarray):
        bits = bits.tolist()
    if isinstance(bits, (list, tuple)):
        return tuple(int(x) & 0xFFFFFFFFFFFFFFFF for x in bits)
    return (int(bits) & 0xFFFFFFFFFFFFFFFF,)


def _det_key(det):
    return (_spin_key(det.alpha), _spin_key(det.beta))


def iterative_workflow(h1, eri, n_alpha, n_beta, n_orb,
                       system_name, args, nuclear_repulsion,
                       start_time=None, results_dir="trimci_results"):
    """
    Main TrimCI iterative workflow (C++ backend).

    This wrapper calls the C++ implementation for maximum performance.
    The C++ backend eliminates overhead of C++ -> Python object conversion
    for intermediate results (pool of millions of determinants).

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
    if start_time is None:
        start_time = time.perf_counter()

    from trimci.trimci_core import IterativeWorkflowParams

    # Get the appropriate determinant type and functions
    functions = get_functions_for_system(n_orb)
    type_suffix = functions.get('type_suffix', '')

    # Import the correct iterative_workflow function
    # Prefer _np variant (accepts numpy arrays directly, avoids 1.6 GB .tolist() copy)
    use_np_variant = False
    if not type_suffix or type_suffix == '':
        import trimci.trimci_core as trimci_core
        if hasattr(trimci_core, 'iterative_workflow_cpp_np'):
            iterative_workflow_cpp = trimci_core.iterative_workflow_cpp_np
            use_np_variant = True
            log_important("Using numpy-optimized C++ workflow: iterative_workflow_cpp_np")
        else:
            from trimci.trimci_core import iterative_workflow_cpp
    else:
        import trimci.trimci_core as trimci_core
        np_fn_name = f'iterative_workflow_cpp_np{type_suffix}'
        fn_name = f'iterative_workflow_cpp{type_suffix}'
        if hasattr(trimci_core, np_fn_name):
            iterative_workflow_cpp = getattr(trimci_core, np_fn_name)
            use_np_variant = True
            log_important(f"Using scalable C++ workflow: {np_fn_name}")
        elif hasattr(trimci_core, fn_name):
            iterative_workflow_cpp = getattr(trimci_core, fn_name)
            log_important(f"Using scalable C++ workflow: {fn_name}")
        else:
            raise RuntimeError(
                f"{fn_name} not available. "
                f"Rebuild C++ module with scalable support for {type_suffix}."
            )

    # Prepare ERI and h1 for C++ backend
    if use_np_variant:
        # _np variant accepts numpy arrays directly — no .tolist() needed
        # Ensure h1 is 2D contiguous float64, eri is 1D flat contiguous float64
        import numpy as np
        if hasattr(eri, "reshape") and hasattr(eri, "ndim") and eri.ndim == 4:
            eri = eri.reshape(-1)
        elif not isinstance(eri, np.ndarray):
            eri = np.asarray(eri, dtype=np.float64).ravel()
        eri = np.ascontiguousarray(eri.ravel(), dtype=np.float64)

        if not isinstance(h1, np.ndarray):
            h1 = np.asarray(h1, dtype=np.float64)
        if h1.ndim == 1:
            n = int(np.sqrt(len(h1)))
            h1 = h1.reshape(n, n)
        h1 = np.ascontiguousarray(h1, dtype=np.float64)
    else:
        # Legacy list-based path
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
    _steps = getattr(args, 'core_set_steps', None)
    if _schedule is not None:
        params.core_set_schedule = list(_schedule)
    elif _steps is not None and _steps > 1:
        # Auto-generate schedule: first_cycle_keep_size → max_final_dets
        # core_set_steps = total number of points (including first and last)
        # e.g. steps=5: [10, 56, 316, 1778, 10000]
        start = params.first_cycle_keep_size
        end = params.max_final_dets if params.max_final_dets > 0 else start
        if end > start:
            ratio = (end / start) ** (1.0 / (_steps - 1))
            schedule = []
            for i in range(_steps):
                val = min(int(round(start * ratio ** i)), end)
                schedule.append(val)
            schedule[-1] = end
            params.core_set_schedule = schedule

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
    params.overshoot_factor = getattr(args, 'overshoot_factor', 1.8)
    params.stagnation_limit = getattr(args, 'stagnation_limit', 3)

    _attentive = getattr(args, 'attentive_orbitals', None)
    if _attentive is not None:
        params.attentive_orbitals = list(_attentive)

    # DMRG-style noise
    params.noise_strength = getattr(args, 'noise_strength', 0.0)
    params.noise_decay = getattr(args, 'noise_decay', 1.0)
    params.random_seed = int(getattr(args, 'random_seed', 0) or 0) & 0xFFFFFFFFFFFFFFFF

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
    params.save_period = getattr(args, 'save_period', 1000000)
    params.save_pool = getattr(args, 'save_pool', False)
    params.save_initial = getattr(args, 'save_initial', False)
    params.output_dir = results_dir

    # =========================================================================
    # Generate initial determinants
    # =========================================================================
    initial_dets_dict = getattr(args, 'initial_dets_dict', None)
    load_initial_dets = getattr(args, 'load_initial_dets', False)

    initial_dets, initial_coeffs = [], []

    if load_initial_dets:
        dets_path = getattr(args, 'initial_dets_path', None) or getattr(args, 'dets_path', 'dets.npz')
        dets_array_name = getattr(args, 'dets_array_name', 'dets')
        loaded_dets, loaded_coeffs = load_initial_dets_from_file(
            dets_path,
            core_set=(dets_array_name != 'dets'),
            expected_n_alpha=n_alpha,
            expected_n_beta=n_beta,
        )

        if loaded_dets is None or loaded_coeffs is None:
            log_important(f"Failed to load from {dets_path}, falling back to HF reference")
        else:
            load_initial_dets_num = getattr(args, 'load_initial_dets_num', None)
            if load_initial_dets_num is not None and load_initial_dets_num < len(loaded_dets):
                original_count = len(loaded_dets)
                sorted_idx = np.argsort(np.abs(loaded_coeffs))[::-1][:load_initial_dets_num]
                loaded_dets = [loaded_dets[i] for i in sorted_idx]
                loaded_coeffs = [loaded_coeffs[i] for i in sorted_idx]
                log_important(f"Loaded top-{load_initial_dets_num} determinants (truncated from {original_count})")
            else:
                log_important(f"Loaded {len(loaded_dets)} determinants from {dets_path}")
            initial_dets.extend(loaded_dets)
            initial_coeffs.extend(loaded_coeffs)

    # Tracking randomness: random-walk perturbation of tracked dets.
    # Each tracked det undergoes n_steps = loaded_dets_randomness * n_elec sequential
    # random single excitations (random spin channel, random occ→vir).
    # 0 = no perturbation, 1.0 = each electron moved once on average (fully randomized).
    loaded_dets_randomness = getattr(args, 'loaded_dets_randomness', 0.0)
    if loaded_dets_randomness > 0 and load_initial_dets and initial_dets:
        from .det_utils import _get_n_segments
        functions = get_functions_for_system(n_orb)
        DetClass = functions['determinant_class']
        n_seg = _get_n_segments(DetClass)
        n_elec = n_alpha + n_beta
        n_steps = max(1, int(round(loaded_dets_randomness * n_elec)))
        # Derive walk RNG from args.random_seed when set; else fresh entropy.
        # Mirrors the run_full convention: random_seed=0/None → system time.
        _walk_seed = int(getattr(args, 'random_seed', 0) or 0)
        rng = (np.random.default_rng(_walk_seed) if _walk_seed
               else np.random.default_rng())

        def _to_uint64(x):
            if isinstance(x, np.integer): x = int(x)
            return int(x) & 0xFFFFFFFFFFFFFFFF

        def _get_occ_vir(det):
            a_bits, b_bits = det.alpha, det.beta
            a_occ, b_occ = [], []
            if isinstance(a_bits, (list, tuple)):
                for seg, (a, b) in enumerate(zip(a_bits, b_bits)):
                    for bit in range(64):
                        orb = seg * 64 + bit
                        if orb >= n_orb: break
                        if int(a) & (1 << bit): a_occ.append(orb)
                        if int(b) & (1 << bit): b_occ.append(orb)
            else:
                for i in range(n_orb):
                    if int(a_bits) & (1 << i): a_occ.append(i)
                    if int(b_bits) & (1 << i): b_occ.append(i)
            all_orbs = set(range(n_orb))
            return a_occ, sorted(all_orbs - set(a_occ)), b_occ, sorted(all_orbs - set(b_occ))

        def _make_det(alpha_occ, beta_occ):
            a_arr = [0] * n_seg
            b_arr = [0] * n_seg
            for i in alpha_occ:
                seg, bit = divmod(i, 64)
                a_arr[seg] |= (1 << bit)
            for i in beta_occ:
                seg, bit = divmod(i, 64)
                b_arr[seg] |= (1 << bit)
            if n_seg > 1:
                return DetClass([_to_uint64(x) for x in a_arr],
                                [_to_uint64(x) for x in b_arr])
            return DetClass(_to_uint64(a_arr[0]), _to_uint64(b_arr[0]))

        n_tracked = len(initial_dets)
        walked_dets = []
        for det in initial_dets[:n_tracked]:
            a_occ, a_vir, b_occ, b_vir = _get_occ_vir(det)
            cur_a, cur_b = list(a_occ), list(b_occ)
            for _ in range(n_steps):
                if rng.random() < 0.5 and cur_a and a_vir:
                    idx = rng.integers(len(cur_a))
                    vidx = rng.integers(len(a_vir))
                    old, new = cur_a[idx], a_vir[vidx]
                    cur_a[idx] = new
                    a_vir[vidx] = old
                elif cur_b and b_vir:
                    idx = rng.integers(len(cur_b))
                    vidx = rng.integers(len(b_vir))
                    old, new = cur_b[idx], b_vir[vidx]
                    cur_b[idx] = new
                    b_vir[vidx] = old
            walked_dets.append(_make_det(sorted(cur_a), sorted(cur_b)))
        initial_dets.extend(walked_dets)
        initial_coeffs.extend([1.0] * len(walked_dets))
        log_important(f"Tracking randomness: {n_tracked} dets × {n_steps} steps "
                      f"(randomness={loaded_dets_randomness:.2f}) → {len(walked_dets)} walked dets")

    # Perturbation: generate nearby dets from loaded ones (quantum annealing Bx style)
    perturb_strength = getattr(args, 'perturb_strength', 0.0)
    perturb_count = getattr(args, 'perturb_count', 0)
    if perturb_strength > 0 and perturb_count > 0 and initial_dets:
        p_dets, p_coeffs = perturb_determinants(
            initial_dets, initial_coeffs, n_alpha, n_beta, n_orb,
            perturb_strength, perturb_count)
        initial_dets.extend(p_dets)
        initial_coeffs.extend(p_coeffs)

    if initial_dets_dict is not None:
        gen_dets, gen_coeffs = generate_initial_states(
            n_alpha, n_beta, n_orb, initial_dets_dict)
        initial_dets.extend(gen_dets)
        initial_coeffs.extend(gen_coeffs)

    if not initial_dets:
        ref_det = generate_reference_det(n_alpha, n_beta, n_orb)
        initial_dets = [ref_det]
        initial_coeffs = [1.0]

    # Deduplicate initial dets (keep first occurrence, preserve order)
    n_before = len(initial_dets)
    seen_keys = set()
    unique_dets, unique_coeffs = [], []
    for d, c in zip(initial_dets, initial_coeffs):
        key = _det_key(d)
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
    log_important(f"Starting C++ iterative_workflow with {len(initial_dets)} initial dets")

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

    save_final_results(
        cpp_result.final_energy,
        list(cpp_result.final_dets),
        list(cpp_result.final_coeffs),
        iteration_details,
        args,
        outdir=results_dir
    )

    log_important(f"C++ iterative_workflow complete: E={cpp_result.final_energy:.8f}, "
                  f"dets={len(cpp_result.final_dets)}, time={cpp_result.total_time:.2f}s")

    return (cpp_result.final_energy,
            list(cpp_result.final_dets),
            list(cpp_result.final_coeffs),
            iteration_details,
            args)
