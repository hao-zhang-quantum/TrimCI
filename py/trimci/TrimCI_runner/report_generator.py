"""
TrimCI Report Generator

Generates trimci_report.md for multi-run analysis.
"""
import os
import socket
from datetime import datetime
from pathlib import Path


def generate_report(all_results: list, best_result: dict, mol_name: str, 
                   args, folder: str = None, combined_results_info: dict = None,
                   final_run: bool = False) -> str:
    """
    Generate trimci_report.md for multi-run results.
    
    Args:
        all_results: List of run result dicts, each with:
            - run_idx, final_energy, elapsed (optional), 
            - results_dir (optional), dets/current_dets (optional)
        best_result: The best result dict (lowest energy)
        mol_name: System name
        args: Configuration object with attributes like threshold, schedule, etc.
        folder: Output folder (default: current directory)
    
    Returns:
        Path to generated report
    """
    report_path = Path(folder or ".") / "trimci_report.md"
    
    # Extract energies and times
    energies = [r['final_energy'] for r in all_results]
    times = []
    for r in all_results:
        t = _get_time(r)
        if t and t > 0:
            times.append(t)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        # ===== Header =====
        f.write("# TrimCI Multi-Run Report\n\n")
        f.write(f"- **System:** {mol_name}\n")
        f.write(f"- **Host:** {socket.gethostname()}\n")
        f.write(f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **Runs:** {len(all_results)}\n")
        fcidump = getattr(args, 'fcidump_path', None)
        if fcidump:
            f.write(f"- **FCIDUMP:** `{fcidump}`\n")
        f.write("\n---\n\n")
        
        # ===== Configuration =====
        f.write("## Configuration\n\n")
        f.write("| Parameter | Value |\n")
        f.write("|-----------|-------|\n")
        
        # Key parameters to show
        params = [
            ('threshold', 'Threshold'),
            ('pool_core_ratio', 'Pool/Core Ratio'),
            ('max_final_dets', 'Max Final Dets'),
            ('core_set_schedule', 'Schedule'),
            ('initial_dets_dict', 'Initial Dets'),
            ('num_runs', 'Num Runs'),
            ('n_parallel', 'N Parallel'),
            ('nuclear_repulsion', 'Nuclear Repulsion'),
        ]
        for attr, label in params:
            val = getattr(args, attr, None)
            if val is not None:
                if attr == 'nuclear_repulsion':
                    f.write(f"| {label} | {val:.8f} Ha |\n")
                elif isinstance(val, list) and len(val) > 10:
                    f.write(f"| {label} | `[{val[0]}, ..., {val[-1]}]` ({len(val)} items) |\n")
                else:
                    f.write(f"| {label} | `{val}` |\n")
        f.write("\n")
        
        # ===== Best Result =====
        f.write("## 🏆 Best Result\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| **Run** | {best_result.get('run_idx', 'N/A')} |\n")
        f.write(f"| **Energy** | **{best_result['final_energy']:.10f} Ha** |\n")
        
        best_time = _get_time(best_result)
        if best_time:
            f.write(f"| Time | {best_time:.2f} s |\n")
        
        dets = best_result.get('current_dets') or best_result.get('dets')
        if dets is not None:
            f.write(f"| Determinants | {len(dets)} |\n")
        
        results_dir = best_result.get('results_dir')
        if results_dir:
            f.write(f"| Output | `{results_dir}` |\n")
        f.write("\n")
        
        # ===== Statistics =====
        f.write("## Statistics\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| Best Energy | {min(energies):.8f} Ha |\n")
        f.write(f"| Worst Energy | {max(energies):.8f} Ha |\n")
        f.write(f"| Spread | {max(energies) - min(energies):.6f} Ha |\n")
        f.write(f"| Average | {sum(energies)/len(energies):.8f} Ha |\n")
        if times:
            f.write(f"| Total Time | {sum(times):.1f} s |\n")
            f.write(f"| Avg Time/Run | {sum(times)/len(times):.1f} s |\n")
        f.write("\n")
        
        # ===== Leaderboard with Top-1 Det =====
        f.write("## Leaderboard\n\n")
        f.write("| Rank | Run | Energy (Ha) | Time (s) | Top-1 Det (α, β) |\n")
        f.write("|------|-----|-------------|----------|------------------|\n")
        
        sorted_results = sorted(all_results, key=lambda x: x['final_energy'])
        min_e = min(energies)
        
        for rank, r in enumerate(sorted_results, 1):
            run_idx = r.get('run_idx', rank-1)
            energy = r['final_energy']
            t = _get_time(r)
            time_str = f"{t:.1f}" if t else "-"
            
            # Get top-1 determinant
            top1_str = "-"
            dets = r.get('current_dets') or r.get('dets')
            coeffs = r.get('current_coeffs') or r.get('coeffs')
            if dets is not None and coeffs is not None and len(dets) > 0:
                top1_str = _format_det(dets[0])
            
            # Bold for best
            if energy == min_e:
                f.write(f"| **{rank}** | **{run_idx}** 🏆 | **{energy:.8f}** | {time_str} | `{top1_str}` |\n")
            else:
                f.write(f"| {rank} | {run_idx} | {energy:.8f} | {time_str} | `{top1_str}` |\n")
        
        f.write("\n")
        
        # ===== Best Run Top-20 Determinants =====
        best_dets = best_result.get('current_dets') or best_result.get('dets')
        best_coeffs = best_result.get('current_coeffs') or best_result.get('coeffs')
        if best_dets is not None and best_coeffs is not None and len(best_dets) >= 1:
            f.write("## Best Run: Top Determinants\n\n")
            f.write("| Rank | Coeff | Det (α, β) |\n")
            f.write("|------|-------|------------|\n")
            
            # Sort by absolute coefficient value
            import numpy as np
            indices = np.argsort(-np.abs(best_coeffs))[:20]  # top 20
            
            for i, idx in enumerate(indices, 1):
                coeff = best_coeffs[idx]
                det_str = _format_det(best_dets[idx])
                f.write(f"| {i} | {coeff:+.6f} | `{det_str}` |\n")
            
            f.write("\n")
        
        f.write("---\n")
        f.write("*Generated by TrimCI*\n")
    
    return str(report_path)


def _get_time(result: dict) -> float:
    """Extract time from result dict, checking multiple locations."""
    t = result.get('elapsed')
    if t:
        return t
    
    # Try iteration_details
    details = result.get('iteration_details')
    if isinstance(details, dict):
        t = details.get('total_time')
        if t:
            return t
    
    # Try details
    details = result.get('details')
    if isinstance(details, dict):
        t = details.get('total_time')
        if t:
            return t
    
    return None


def _format_det(det) -> str:
    """Format a determinant as (alpha_bits, beta_bits) string."""
    # C++ Determinant objects (Determinant64, Determinant128, etc.)
    if hasattr(det, 'alpha') and hasattr(det, 'beta'):
        alpha, beta = det.alpha, det.beta
        if isinstance(alpha, list):
            return f"({alpha}, {beta})"
        return f"({bin(alpha)}, {bin(beta)})"
    elif isinstance(det, (list, tuple)) and len(det) == 2:
        alpha, beta = det
        if isinstance(alpha, int) and isinstance(beta, int):
            return f"({bin(alpha)}, {bin(beta)})"
        return f"({alpha}, {beta})"
    else:
        return str(det)


# Alias for backward compatibility
generate_multi_run_report = generate_report

