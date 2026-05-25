"""
TrimCI Parallel Runner

Provides parallel execution of multiple TrimCI runs using spawn subprocesses.
This module MUST NOT import trimci at the top level to allow proper OMP control.

Multi-GPU: when ``args.n_gpus > 0``, workers are pinned to individual GPUs
via ``CUDA_VISIBLE_DEVICES`` in a Pool initializer callback (runs ONCE per
worker process, BEFORE any trimci import, so CUDA context initializes on the
right device). Workers are assigned GPUs round-robin from a queue.
"""
import os
import time
from pathlib import Path
from multiprocessing import get_context

from .summary_logger import TrimCISummaryLogger


def _init_worker_gpu(gpu_queue):
    """Pool initializer: claim a GPU from the shared queue.

    Called ONCE per worker process at startup. Sets CUDA_VISIBLE_DEVICES so
    the subsequent trimci import (in _worker_single_run) creates the CUDA
    context on the pinned device. After CUDA init, CUDA_VISIBLE_DEVICES can't
    be changed effectively — that's why we do this in the initializer, not
    per-task.
    """
    # Timeout turns a Pool spawn-fanout stall into a detectable failure.
    # If n_gpu_workers = N but only M < N GPU IDs were enqueued (should not
    # happen with correct dispatcher code, but defensively), an under-filled
    # queue would otherwise block forever on subsequent worker init.
    try:
        gpu_id = gpu_queue.get(timeout=30)
    except Exception as e:
        raise RuntimeError(
            "GPU worker init: failed to claim a GPU id from the dispatcher queue"
            " (timeout 30s). Check n_gpu_workers ≤ number of queued IDs."
        ) from e
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    # NVIDIA-recommended: also ensures CUDA driver honors the mask even in
    # libraries that cache device info.
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'


def _worker_single_run(args_tuple):
    """
    Worker function executed in a spawned subprocess.

    IMPORTANT: This function sets OMP_NUM_THREADS BEFORE importing trimci,
    ensuring the C++ OpenMP runtime respects the thread count.

    If a GPU has been pinned via the Pool initializer, CUDA_VISIBLE_DEVICES
    is already set before this runs — the subsequent trimci import will
    create the CUDA context on the pinned device.
    """
    (omp_threads, run_idx, integrals_path, n_alpha, n_beta, n_orb,
     mol_name, args_dict, nuclear_repulsion, worker_folder) = args_tuple

    # 1. Set OMP_NUM_THREADS BEFORE any trimci import
    os.environ['OMP_NUM_THREADS'] = str(omp_threads)

    # 1b. Load integrals from shared file (avoids pickling 228 MB per worker)
    import numpy as np
    data = np.load(integrals_path)
    h1 = data['h1']
    eri = data['eri']
    
    # 2. Now import trimci (C++ module loads here with correct OMP setting;
    #    if this is a GPU worker, CUDA_VISIBLE_DEVICES is already pinned by
    #    the Pool initializer, so CUDA context binds to the right device).
    from types import SimpleNamespace

    # 3. Reconstruct args namespace
    args = SimpleNamespace(**args_dict)

    # 4. Create worker-specific folder
    run_folder = os.path.join(worker_folder, f"run_{run_idx:03d}")
    Path(run_folder).mkdir(parents=True, exist_ok=True)

    # 5. Execute single run. GPU workers dispatch through _run_multi_gpu
    #    with num_runs=1; CPU workers use the C++ iterative_workflow.
    # Use args.n_gpu_workers (set by the dispatcher) rather than
    # CUDA_VISIBLE_DEVICES because CPU workers may inherit the env var from
    # the parent HTCondor job even when they shouldn't use GPU.
    worker_uses_gpu = int(getattr(args, 'n_gpu_workers', 0)) > 0
    start_time = time.perf_counter()
    try:
        if worker_uses_gpu:
            from trimci.TrimCI_runner.trimci_driver import _run_multi_gpu
            # _run_multi_gpu expects num_runs; feed 1 (this worker's slice).
            # Seed derivation MUST be per-worker even when the user passes
            # a base random_seed — otherwise each of N workers mixes with
            # r=0 internally and gets the identical trajectory, silently
            # defeating multi-run diversity. Mix the user's base seed with
            # this worker's run_idx using a Weyl-style constant.
            base_seed = int(getattr(args, 'random_seed', 0) or 0)
            args.random_seed = (base_seed + (run_idx + 1) * 0x9e3779b97f4a7c15) \
                & 0xFFFFFFFFFFFFFFFF
            final_energy, dets, coeffs, details, run_args = _run_multi_gpu(
                h1, eri, n_alpha, n_beta, n_orb, mol_name,
                args, nuclear_repulsion, run_folder, num_runs=1,
            )
        else:
            from trimci.TrimCI_runner.iterative_workflow import iterative_workflow
            final_energy, dets, coeffs, details, run_args = iterative_workflow(
                h1, eri, n_alpha, n_beta, n_orb, mol_name,
                args, nuclear_repulsion, start_time, run_folder
            )
        elapsed = time.perf_counter() - start_time
        
        # Return serializable result
        return {
            'run_idx': run_idx,
            'success': True,
            'final_energy': final_energy,
            'dets': dets,
            'coeffs': coeffs,
            'details': details,
            'run_args': run_args,
            'elapsed': elapsed,
            'results_dir': run_folder
        }
    except Exception as e:
        return {
            'run_idx': run_idx,
            'success': False,
            'error': str(e),
            'results_dir': run_folder
        }


def run_trimci_main_calculation_parallel(
    h1, eri, n_alpha, n_beta, n_orb, mol_name, args, nuclear_repulsion,
    folder=None,
    num_runs=None,
    n_parallel=None,
    omp_per_run=None
):
    """
    Parallel version of run_trimci_main_calculation.
    
    Executes multiple independent TrimCI runs in parallel using spawn subprocesses.
    Each subprocess has its own OMP_NUM_THREADS setting.
    
    Args:
        h1, eri, n_alpha, n_beta, n_orb, mol_name, args, nuclear_repulsion:
            Same as run_trimci_main_calculation
        folder: Output directory (optional)
        num_runs: Total number of runs. Defaults to args.num_runs or 50.
        n_parallel: Number of parallel workers. Defaults to 8.
        omp_per_run: OMP threads per worker. If None, auto-calculated from
                     OMP_NUM_THREADS environment variable.
    
    Returns:
        Same as run_trimci_main_calculation:
        (best_energy, best_dets, best_coeffs, best_details, best_args)
    
    Example:
        # 128 cores total: 8 parallel workers × 16 OMP threads each
        result = run_trimci_main_calculation_parallel(
            h1, eri, n_alpha, n_beta, n_orb, mol_name, args, nuclear_repulsion,
            num_runs=50, n_parallel=8, omp_per_run=16
        )
    """
    # Get parameters.
    #
    # Naming:
    #   n_gpu_workers — spawn this many GPU-pinned workers (each claims one
    #                   GPU via CUDA_VISIBLE_DEVICES in the Pool initializer)
    #   n_cpu_workers — spawn this many CPU-only workers (legacy name:
    #                   n_parallel; kept for backwards compat)
    # Mutually exclusive in this call: if n_gpu_workers > 0 we run a pure
    # GPU pool; otherwise a pure CPU pool. Heterogeneous mixing (some CPU,
    # some GPU in the same Pool) is not supported — imap_unordered can't
    # guarantee a GPU task lands on a GPU-initialized worker.
    if num_runs is None:
        num_runs = getattr(args, 'num_runs', 50)

    n_gpu_workers = int(getattr(args, 'n_gpu_workers', 0))
    if n_gpu_workers > 0:
        n_workers = n_gpu_workers
        worker_mode = 'gpu'
    else:
        # CPU path: accept n_cpu_workers (new) or n_parallel (legacy).
        if n_parallel is None:
            n_parallel = getattr(args, 'n_cpu_workers',
                                 getattr(args, 'n_parallel', 8))
        n_workers = int(n_parallel)
        worker_mode = 'cpu'

    # Legacy field kept in scope for logging.
    n_parallel = n_workers

    if omp_per_run is None:
        # Auto-calculate: divide total OMP threads by number of parallel workers
        total_omp = int(os.environ.get('OMP_NUM_THREADS', os.cpu_count() or 1))
        omp_per_run = max(1, total_omp // n_workers)
    
    # Setup output folder (use unique timestamp with PID to avoid Slurm array job collisions)
    from .io_utils import generate_unique_timestamp
    unique_ts = generate_unique_timestamp()
    if folder is None:
        parallel_folder = str(Path("trimci_parallel_results") / f"{mol_name}_{unique_ts}")
    else:
        parallel_folder = str(Path(folder) / f"parallel_{mol_name}_{unique_ts}")
    Path(parallel_folder).mkdir(parents=True, exist_ok=True)
    
    # Convert args to dict for pickling
    if hasattr(args, '__dict__'):
        args_dict = vars(args).copy()
    else:
        args_dict = dict(args)
    # Force single run per worker
    args_dict['num_runs'] = 1
    
    # Initialize summary logger (unified multi_summary.log for both parallel and sequential multi-run)
    summary_log_path = os.path.join(parallel_folder, "multi_summary.log")
    logger = TrimCISummaryLogger(summary_log_path, mode="parallel")
    logger.write_header(mol_name, n_orb, n_alpha, n_beta, args_dict, num_runs, n_parallel, omp_per_run)
    
    verbosity = args_dict.get('verbosity', 1 if args_dict.get('verbose', False) else 0)
    
    if verbosity >= 1:
        print(f"🚀 Starting parallel TrimCI: {num_runs} runs, {n_parallel} workers, {omp_per_run} OMP threads each")
    
    # Save integrals to shared file (workers load by path, avoids pickling per worker)
    import numpy as np
    integrals_path = os.path.join(parallel_folder, "_shared_integrals.npz")
    np.savez(integrals_path,
             h1=np.asarray(h1, dtype=np.float64),
             eri=np.asarray(eri, dtype=np.float64).ravel())

    # Prepare task list (pass path instead of arrays)
    tasks = [
        (omp_per_run, i, integrals_path, n_alpha, n_beta, n_orb,
         mol_name, args_dict, nuclear_repulsion, parallel_folder)
        for i in range(num_runs)
    ]
    
    # Execute in parallel using spawn
    ctx = get_context('spawn')

    # Pool initializer: GPU workers get CUDA_VISIBLE_DEVICES pinned once at
    # spawn (before trimci import). CPU workers don't need an initializer.
    #
    # ECC-resilient queue: pre-fill with one GPU id PER TASK (round-robin
    # across n_workers). When a worker dies (CUDA ECC, OOM, segfault), Pool
    # spawns a replacement that calls _init_worker_gpu and grabs the next
    # queued GPU id — no starvation. Pair with maxtasksperchild=1 so each
    # task gets a fresh CUDA context (no inherited state from prior task's
    # potentially-corrupted GPU memory).
    pool_initializer = None
    pool_initargs = ()
    pool_max_tasks = None
    if worker_mode == 'gpu':
        gpu_queue = ctx.Queue()
        for i in range(num_runs):
            gpu_queue.put(i % n_workers)
        pool_initializer = _init_worker_gpu
        pool_initargs = (gpu_queue,)
        pool_max_tasks = 1   # fresh process per task → no GPU state leakage
        if verbosity >= 1:
            print(f"🎮 Multi-GPU mode: {n_workers} GPU workers, "
                  f"1 task per process (ECC-resilient)")

    if verbosity >= 1:
        print(f"📊 Launching {n_parallel} parallel workers...")

    start_all = time.perf_counter()
    successful = []
    failed = []

    try:
        with ctx.Pool(processes=n_parallel,
                      initializer=pool_initializer,
                      initargs=pool_initargs,
                      maxtasksperchild=pool_max_tasks) as pool:
            # Use imap_unordered for real-time logging as each run completes
            for r in pool.imap_unordered(_worker_single_run, tasks):
                if r['success']:
                    successful.append(r)
                    n_iters = r.get('details', {}).get('total_iterations')
                    logger.log_run_complete(r['run_idx'], r['final_energy'], r['elapsed'], n_iters)
                    if verbosity >= 1:
                        print(f"  ✓ Run {r['run_idx']:03d}: E={r['final_energy']:.8f} ({r['elapsed']:.1f}s)")
                else:
                    failed.append(r)
                    logger.log_run_failed(r['run_idx'], r.get('error', 'Unknown error'))
                    if verbosity >= 1:
                        print(f"  ✗ Run {r['run_idx']:03d}: FAILED - {r.get('error', 'Unknown')}")
    finally:
        # Cleanup shared integrals file even on Ctrl-C / SIGTERM. ERI dumps
        # for Fe4S4-class systems are 200+ MB; orphaned files accumulate
        # in /scratch over many job retries.
        try:
            os.remove(integrals_path)
        except OSError:
            pass

    total_time = time.perf_counter() - start_all

    if failed:
        if verbosity >= 1:
            print(f"⚠️ {len(failed)} runs failed:")
            for r in failed:
                print(f"   Run {r['run_idx']}: {r.get('error', 'Unknown error')}")
    
    if not successful:
        logger.close()
        raise RuntimeError("All parallel runs failed!")
    
    # Find best result
    best = min(successful, key=lambda x: x['final_energy'])
    
    # Write final summary
    logger.write_parallel_summary(successful, failed, total_time, best)
    logger.close()
    
    # Generate Markdown Report (trimci_report.md)
    try:
        from .report_generator import generate_report
        # Convert args_dict back to object-like for compatibility
        class ArgsWrapper:
            pass
        args_obj = ArgsWrapper()
        for k, v in args_dict.items():
            setattr(args_obj, k, v)
        
        generate_report(successful, best, mol_name, args_obj, folder=parallel_folder)
        if verbosity >= 1:
            print(f"📄 Report generated: {parallel_folder}/trimci_report.md")
    except Exception as e:
        if verbosity >= 1:
            print(f"⚠️ Failed to generate Markdown report: {e}")
    
    if verbosity >= 1:
        print(f"✅ Completed in {total_time:.2f}s. Best energy: {best['final_energy']:.8f} (run {best['run_idx']})")
        print(f"📁 Results saved to: {parallel_folder}")
    
    # Return in same format as run_trimci_main_calculation
    return (
        best['final_energy'],
        best['dets'],
        best['coeffs'],
        best['details'],
        best['run_args']
    )
