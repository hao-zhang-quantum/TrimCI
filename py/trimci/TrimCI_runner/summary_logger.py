"""
TrimCI Summary Logger

Unified summary logger for all TrimCI run modes, inspired by orbopt_summary.log format.
Provides consistent logging with timestamps, configuration header, progress tracking, and final summary.
"""
import os
import time
from datetime import datetime


class TrimCISummaryLogger:
    """Unified summary logger for TrimCI runs, similar to orbopt_summary.log format."""
    
    def __init__(self, log_path: str, mode: str = "single"):
        """
        Initialize the summary logger.
        
        Args:
            log_path: Path to the summary log file (trimci_summary.log)
            mode: One of 'single', 'multi', 'parallel'
        """
        self.log_path = log_path
        self.mode = mode
        self.start_time = time.perf_counter()
        self._file = open(log_path, 'w')
    
    def log(self, message: str, newline: bool = True):
        """Log message with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"{timestamp}: {message}"
        self._file.write(line + ("\n" if newline else ""))
        self._file.flush()
    
    def write_header(self, mol_name: str, n_orb: int, n_alpha: int, n_beta: int,
                    args_dict: dict, num_runs: int = 1, n_parallel: int = 1, omp_per_run: int = None,
                    fcidump_path: str = None, nuclear_repulsion: float = None):
        """Write detailed configuration header."""
        import socket
        
        # Auto-extract from args_dict if not provided
        if fcidump_path is None:
            fcidump_path = args_dict.get('fcidump_path')
        if nuclear_repulsion is None:
            nuclear_repulsion = args_dict.get('nuclear_repulsion')
        
        mode_desc = {
            'single': 'SINGLE (one run)',
            'multi': 'MULTI-RUN (sequential ensemble)',
            'parallel': 'PARALLEL (multi-run ensemble)'
        }.get(self.mode, self.mode.upper())
        
        self.log("🚀 TrimCI Run Starting")
        self.log("═" * 60)
        self.log(f"Mode: {mode_desc}")
        self.log(f"Host: {socket.gethostname()}")
        self.log(f"System: {mol_name}")
        if fcidump_path:
            self.log(f"  FCIDUMP: {fcidump_path}")
        elif args_dict.get('molecule_spec'):
            self.log(f"  Molecule: {args_dict.get('molecule_spec')}")
        self.log(f"  Orbitals: {n_orb}, Electrons: {n_alpha + n_beta} ({n_alpha}α, {n_beta}β)")
        if nuclear_repulsion is not None:
            self.log(f"  Nuclear Repulsion: {nuclear_repulsion:.8f} Ha")
        self.log(f"  Output Dir: {os.path.dirname(self.log_path)}")
        self.log("─" * 60)
        
        if self.mode in ('multi', 'parallel'):
            self.log("Run Configuration:")
            self.log(f"  num_runs: {num_runs}")
            if self.mode == 'parallel':
                self.log(f"  n_parallel: {n_parallel}")
                if omp_per_run:
                    self.log(f"  omp_per_run: {omp_per_run}")
                    self.log(f"  Total cores: {n_parallel * omp_per_run}")
            self.log("─" * 60)
        
        self.log("TrimCI Configuration:")
        # Print all config items, filter out internal/unimportant keys
        skip_keys = {'verbose', 'debug', 'num_runs', 'n_parallel', 'omp_per_run',
                    'h1', 'eri', 'h2', 'integrals', 'mo_coeff', 'coefficients'}  # skip large arrays
        import numpy as np
        for key, val in sorted(args_dict.items()):
            if key.startswith('_') or key in skip_keys:
                continue
            if val is None:
                continue
            # Skip numpy arrays (they can be huge)
            if isinstance(val, np.ndarray):
                self.log(f"  {key}: <ndarray shape={val.shape}>")
                continue
            # Format lists nicely
            if isinstance(val, (list, tuple)) and len(val) > 15:
                val = f"[{val[0]}, ..., {val[-1]}] ({len(val)} items)"
            self.log(f"  {key}: {val}")
        self.log("═" * 60)
        self.log("")
    
    def log_run_start(self, run_idx: int):
        """Log when a run starts."""
        self.log(f"📍 Run {run_idx} started")
    
    def log_run_complete(self, run_idx: int, energy: float, elapsed: float, n_iters: int = None, n_dets: int = None):
        """Log when a run completes."""
        info_parts = [f"E={energy:.8f} Ha", f"time={elapsed:.2f}s"]
        if n_iters:
            info_parts.append(f"iters={n_iters}")
        if n_dets:
            info_parts.append(f"n_dets={n_dets}")
        self.log(f"📊 Run {run_idx} completed: {', '.join(info_parts)}")
    
    def log_run_failed(self, run_idx: int, error: str):
        """Log when a run fails."""
        self.log(f"❌ Run {run_idx} FAILED: {error}")
    
    def log_iteration(self, iter_idx: int, energy: float, n_dets: int, elapsed: float = None):
        """Log iteration progress (for single run mode)."""
        time_str = f", time={elapsed:.2f}s" if elapsed else ""
        self.log(f"  Iter {iter_idx}: E={energy:.8f} Ha, n_dets={n_dets}{time_str}")
    
    def write_single_summary(self, final_energy: float, n_dets: int, n_iters: int, total_time: float, results_dir: str = ""):
        """Write summary for single run mode."""
        self.log("")
        self.log("═" * 60)
        self.log("✅ SINGLE RUN COMPLETE")
        self.log("═" * 60)
        self.log("[TIMING] Summary:")
        self.log(f"  Total time: {total_time:.2f}s")
        self.log("─" * 60)
        self.log("[RESULTS]")
        self.log(f"  Final Energy: {final_energy:.10f} Ha")
        self.log(f"  Final N_dets: {n_dets}")
        self.log(f"  Total Iterations: {n_iters}")
        self.log("═" * 60)
        if results_dir:
            self.log(f"📁 Results saved to: {results_dir}")
    
    def write_iteration_table(self, iteration_details: dict):
        """Write per-iteration details table (from iterative workflow results)."""
        iterations = iteration_details.get('iterations', [])
        if not iterations:
            return

        self.log("")
        self.log("Iteration Details:")
        self.log(f"  {'Iter':<6} {'Core':<8} {'Pool':<10} {'Raw Dets':<10} {'Energy (Ha)':<18} {'dE':<12} {'Time (s)':<10}")
        self.log("  " + "-" * 72)
        for it in iterations:
            idx = it.get('iteration', '?')
            core = it.get('core_set_size_before', it.get('core_set_size', '-'))
            pool = it.get('actual_pool_size', '-')
            raw = it.get('raw_dets_count', '-')
            energy = it.get('raw_energy', it.get('total_energy', None))
            de = it.get('energy_change', None)
            t = it.get('iteration_time', None)

            e_str = f"{energy:<18.8f}" if isinstance(energy, float) else f"{energy!s:<18}"
            de_str = f"{de:<12.2e}" if isinstance(de, float) else f"{de!s:<12}"
            t_str = f"{t:<10.2f}" if isinstance(t, (int, float)) else f"{t!s:<10}"

            self.log(f"  {idx:<6} {core:<8} {pool:<10} {raw:<10} {e_str} {de_str} {t_str}")

        e_nuclear = iteration_details.get('nuclear_repulsion', 0)
        e_final = iteration_details.get('final_energy', None)
        if e_final is not None and e_nuclear:
            self.log(f"  Electronic Energy: {e_final - e_nuclear:.10f} Ha")

    def write_multi_summary(self, all_results: list, best: dict, total_time: float):
        """Write summary for multi-run mode (sequential)."""
        self.log("")
        self.log("═" * 60)
        self.log("✅ MULTI-RUN COMPLETE")
        self.log("═" * 60)
        
        # Timing statistics
        times = [r.get('elapsed', 0) for r in all_results if r.get('elapsed')]
        self.log("[TIMING] Summary:")
        self.log(f"  Total wall time: {total_time:.2f}s")
        if times:
            self.log(f"  Avg run time: {sum(times)/len(times):.2f}s")
            self.log(f"  Min run time: {min(times):.2f}s")
            self.log(f"  Max run time: {max(times):.2f}s")
        
        self.log("─" * 60)
        self.log("[RESULTS]")
        self.log(f"  Total Runs: {len(all_results)}")
        self.log(f"  Best Energy: {best['final_energy']:.10f} Ha (run {best['run_idx']})")
        
        # Energy statistics
        if len(all_results) > 1:
            energies = [r['final_energy'] for r in all_results]
            self.log(f"  Energy Range: [{min(energies):.8f}, {max(energies):.8f}] Ha")
            self.log(f"  Energy Spread: {max(energies) - min(energies):.6f} Ha")
        
        self.log("─" * 60)
        self.log("Run Leaderboard:")
        self.log(f"  {'Rank':<5} {'Run':<5} {'Energy (Ha)':<18} {'Time (s)':<10}")
        self.log("  " + "-" * 40)
        for rank, r in enumerate(sorted(all_results, key=lambda x: x['final_energy']), 1):
            marker = "🏆" if r['run_idx'] == best['run_idx'] else "  "
            t = r.get('elapsed', 0)
            self.log(f"  {marker}{rank:<3} {r['run_idx']:<5} {r['final_energy']:<18.10f} {t:<10.2f}")
        
        self.log("═" * 60)
        # Log best run results directory if available
        if best.get('results_dir'):
            self.log(f"🏆 Best run dets: {best['results_dir']}/dets.npz")
        self.log(f"📁 Results saved to: {os.path.dirname(self.log_path)}")
    
    def write_parallel_summary(self, successful: list, failed: list, total_time: float, best: dict):
        """Write summary for parallel mode."""
        self.log("")
        self.log("═" * 60)
        if failed:
            self.log(f"⚠️ PARALLEL RUN COMPLETE (with {len(failed)} failures)")
        else:
            self.log("✅ PARALLEL RUN COMPLETE")
        self.log("═" * 60)
        
        # Timing statistics
        self.log("[TIMING] Summary:")
        if successful:
            times = [r['elapsed'] for r in successful]
            self.log(f"  Total wall time: {total_time:.2f}s")
            self.log(f"  Avg run time: {sum(times)/len(times):.2f}s")
            self.log(f"  Min run time: {min(times):.2f}s")
            self.log(f"  Max run time: {max(times):.2f}s")
        
        self.log("─" * 60)
        self.log("[RESULTS]")
        self.log(f"  Successful: {len(successful)}/{len(successful) + len(failed)}")
        self.log(f"  Best Energy: {best['final_energy']:.10f} Ha (run {best['run_idx']})")
        
        # Energy statistics
        if len(successful) > 1:
            energies = [r['final_energy'] for r in successful]
            self.log(f"  Energy Range: [{min(energies):.8f}, {max(energies):.8f}] Ha")
            self.log(f"  Energy Spread: {max(energies) - min(energies):.6f} Ha")
        
        self.log("─" * 60)
        self.log("Run Leaderboard:")
        self.log(f"  {'Rank':<5} {'Run':<5} {'Energy (Ha)':<18} {'Time (s)':<10}")
        self.log("  " + "-" * 40)
        for rank, r in enumerate(sorted(successful, key=lambda x: x['final_energy']), 1):
            marker = "🏆" if r['run_idx'] == best['run_idx'] else "  "
            self.log(f"  {marker}{rank:<3} {r['run_idx']:<5} {r['final_energy']:<18.10f} {r['elapsed']:<10.2f}")
        
        if failed:
            self.log("─" * 60)
            self.log("Failed Runs:")
            for r in failed:
                self.log(f"  Run {r['run_idx']}: {r.get('error', 'Unknown error')}")
        
        self.log("═" * 60)
        # Log best run results directory if available
        if best.get('results_dir'):
            self.log(f"🏆 Best run dets: {best['results_dir']}/dets.npz")
        self.log(f"📁 Results saved to: {os.path.dirname(self.log_path)}")
    
    def close(self):
        """Close the log file with end timestamp."""
        self.log(f"🏁 Run ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._file.close()
