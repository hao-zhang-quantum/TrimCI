#!/usr/bin/env python
"""
Executable script to run TrimCI calculation from local FCIDUMP.
By default, reads trimci_config.json in the same folder as FCIDUMP.
Command-line args can override values in the config.
"""

import os
from datetime import datetime
import json

import argparse
from pathlib import Path
import shutil
from .trimci_driver import run_full_calculation, run_auto

# Parser for --goal that supports abbreviations b/s/a
def parse_goal(value: str) -> str:
    v = str(value).strip().lower()
    mapping = {
        'b': 'balanced',
        'balanced': 'balanced',
        's': 'speed',
        'speed': 'speed',
        'a': 'accuracy',
        'accuracy': 'accuracy',
    }
    if v in mapping:
        return mapping[v]
    raise argparse.ArgumentTypeError(
        f"Invalid goal: {value}. Use balanced|speed|accuracy or abbreviations b|s|a."
    )

# Coerce CLI value to appropriate Python type (supports JSON lists/dicts)
def _coerce_cli_value(v: str):
    s = str(v).strip()

    # Try JSON for lists/dicts first
    if (s.startswith('[') and s.endswith(']')) or (s.startswith('{') and s.endswith('}')):
        try:
            return json.loads(s)
        except Exception:
            pass

    ls = s.lower()
    # Booleans
    if ls in ('true', 'false'):
        return ls == 'true'
    # None/null
    if ls in ('none', 'null'):
        return None

    # Numbers
    try:
        if '.' in s or 'e' in ls:
            return float(s)
        return int(s)
    except ValueError:
        # Quoted JSON string
        if (s.startswith('"') and s.endswith('"')):
            try:
                return json.loads(s)
            except Exception:
                return s[1:-1]
        return s

# Consume multi-token JSON-like values until closing bracket
# Joins tokens like [1, 2, 3] or {"a": 1, "b": 2}
# Returns (combined_string, next_index)
def _consume_json_like(tokens, start_index: int):
    if start_index >= len(tokens):
        return "", start_index
    depth = 0
    acc = []
    i = start_index
    first = tokens[i]
    if not (first.startswith('[') or first.startswith('{')):
        return first, i + 1
    while i < len(tokens):
        t = tokens[i]
        acc.append(t)
        for ch in t:
            if ch in '[{':
                depth += 1
            elif ch in ']}':
                depth -= 1
        i += 1
        if depth <= 0:
            break
    return ' '.join(acc), i


def generate_markdown_report(result_data: dict, report_path: str, fcidump_path: str, trimci_config: dict):
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 📝 TrimCI Calculation Report\n\n")

        # ===== Basic Info =====
        system_info = result_data['system_info']
        config_info = result_data['configuration']
        results = result_data['results']
        timing = result_data['timing']

        f.write("## 📊 Summary\n\n")
        f.write(f"- **Calculation Timestamp**: {result_data['calculation_timestamp']}\n")
        f.write(f"- **Final Determinants**: {results['final_determinants']:,}\n")

        final_energy = results['final_energy']
        nuclear_repulsion = system_info['nuclear_repulsion']

        f.write(f"- **Total Energy**: {final_energy:.8f} Ha\n")
        f.write(f"- **Electronic Energy**: {final_energy - nuclear_repulsion:.8f} Ha\n")

        if timing['total_time'] != 'N/A' and isinstance(timing['total_time'], (int, float)):
            f.write(f"- **Wall Time**: {timing['total_time']:.1f} seconds\n\n")
        else:
            f.write(f"- **Wall Time**: {timing['total_time']}\n\n")

        # ===== Molecular Info =====
        f.write("## 🧬 System Info\n\n")
        f.write(f"- **Electrons**: {system_info['n_electrons']}\n")
        f.write(f"- **Orbitals**: {system_info['n_orbitals']}\n")
        f.write(f"- **Core (Nuclear Repulsion) Energy**: {system_info['nuclear_repulsion']:.8f} Ha\n\n")

        # ===== Parameters =====
        f.write("## ⚙️ Configuration Parameters\n\n")
        for key, value in trimci_config.items():
            f.write(f"- {key}: {value}\n")
        f.write("\n")

        # ===== Workflow Details =====
        if 'experimental_workflow' in result_data:
            exp_data = result_data['experimental_workflow']
            f.write("## 🔄 Iterative Workflow\n\n")

            if 'iterations' in exp_data:
                f.write("| Iter | Core Size | Pool Size | Final Dets | Energy (Ha) | Time (s) |\n")
                f.write("|------|-----------|-----------|-------------|-------------|----------|\n")
                for i, iteration in enumerate(exp_data['iterations'], 1):
                    core_size = iteration.get('core_set_size', 'N/A')
                    pool_size = iteration.get('actual_pool_size', 'N/A')
                    final_dets = iteration.get('final_dets_count', 'N/A')
                    energy = iteration.get('total_energy', 'N/A')
                    time_val = iteration.get('iteration_time', 'N/A')

                    if isinstance(core_size, int): core_size = f"{core_size:,}"
                    if isinstance(pool_size, int): pool_size = f"{pool_size:,}"
                    if isinstance(final_dets, int): final_dets = f"{final_dets:,}"
                    if isinstance(energy, float): energy = f"{energy:.8f}"
                    if isinstance(time_val, float): time_val = f"{time_val:.1f}"

                    f.write(f"| {i} | {core_size} | {pool_size} | {final_dets} | {energy} | {time_val} |\n")

            if 'convergence_achieved' in exp_data:
                convergence = "✅ Converged" if exp_data['convergence_achieved'] else "❌ Not Converged"
                f.write(f"\n**Convergence Status**: {convergence}\n\n")

        # ===== File Info =====
        f.write("## 📁 Related Files\n\n")
        f.write(f"- **FCIDUMP**: `{os.path.basename(fcidump_path)}`\n")
        f.write(f"- **Configuration**: `trimci_config.json`\n")
        f.write(f"- **Detailed Log**: `trimci_calculation.out`\n")
        f.write(f"- **JSON Result**: "
                f"`trimci_result_{result_data['calculation_timestamp'].replace('-', '').replace(':', '').replace(' ', '_')}.json`\n\n")

        # ===== Footer =====
        f.write("---\n\n")
        f.write(f"*Report generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")


def main():
    parser = argparse.ArgumentParser(description="Run TrimCI calculation.")
    parser.add_argument("--fcidump", type=str, default="FCIDUMP",
                        help="Path to FCIDUMP file (default: ./FCIDUMP)")
    parser.add_argument("--trimci_config", type=str, default=None,
                        help="Path to trimci_config.json (default: ./trimci_config.json)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--auto", action="store_true",
                        help="Enable auto mode; call run_auto when set")
    parser.add_argument("--goal", type=parse_goal, choices=["balanced", "speed", "accuracy"], default="balanced",
                        help="Tuning goal for auto mode: balanced | speed | accuracy (b/s/a supported)")

    # Optional override parameters
    parser.add_argument("-n", "--max_final_dets", type=int, help="Max final determinants override")
    parser.add_argument("--orbopt", action="store_true", help="Copy dets.npz to working directory as dets.npz")
    parser.add_argument("--fixed_initial_mode", action="store_true", help="No change of dets.npz")  


    args, unknown_args = parser.parse_known_args()

    # Resolve FCIDUMP path
    fcidump_path = Path(args.fcidump).resolve()

    if not fcidump_path.exists():
        # Resolve FCIDUMP path
        fcidump_candidates = list(Path.cwd().glob("fcidump*")) + list(Path.cwd().glob("FCIDUMP*"))
        if not fcidump_candidates:
            raise FileNotFoundError("❌ No FCIDUMP file found in current directory")
        fcidump_path = fcidump_candidates[0].resolve()
        print(f"⚠️ Using FCIDUMP file: {fcidump_path}")

    # Collect overrides
    overrides = {}
    if args.verbose:
        overrides["verbose"] = True
    if args.max_final_dets is not None:
        overrides["max_final_dets"] = args.max_final_dets

    # Parse unregistered CLI arguments into overrides
    # Supports forms: --key=value, --key value, --flag (True), and short -k value
    if 'unknown_args' in locals() and unknown_args:
        i = 0
        while i < len(unknown_args):
            token = unknown_args[i]
            if token.startswith('--'):
                keyval = token[2:]
                if '=' in keyval:
                    key, val = keyval.split('=', 1)
                    overrides[key] = _coerce_cli_value(val)
                    i += 1
                else:
                    key = keyval
                    if i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith('-'):
                        next_tok = unknown_args[i + 1]
                        if next_tok.startswith('[') or next_tok.startswith('{'):
                            combined, new_i = _consume_json_like(unknown_args, i + 1)
                            overrides[key] = _coerce_cli_value(combined)
                            i = new_i
                        else:
                            overrides[key] = _coerce_cli_value(next_tok)
                            i += 2
                    else:
                        overrides[key] = True
                        i += 1
            elif token.startswith('-'):
                key = token.lstrip('-')
                if i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith('-'):
                    next_tok = unknown_args[i + 1]
                    if next_tok.startswith('[') or next_tok.startswith('{'):
                        combined, new_i = _consume_json_like(unknown_args, i + 1)
                        overrides[key] = _coerce_cli_value(combined)
                        i = new_i
                    else:
                        overrides[key] = _coerce_cli_value(next_tok)
                        i += 2
                else:
                    overrides[key] = True
                    i += 1
            else:
                i += 1

    # Run calculation
    if args.auto:
        if args.max_final_dets is not None:
            final_energy, final_dets, final_coeffs, iteration_details, config = run_auto(
                fcidump_path=str(fcidump_path),
                #trimci_config_path=args.trimci_config,
                goal=args.goal,
                ndets=args.max_final_dets,
                **overrides,
            )
        else:
            final_energy, final_dets, final_coeffs, iteration_details, config = run_auto(
                fcidump_path=str(fcidump_path),
                #trimci_config_path=args.trimci_config,
                goal=args.goal,
                **overrides,
            )
    else:
        final_energy, final_dets, final_coeffs, iteration_details, config = run_full_calculation(
            fcidump_path=str(fcidump_path),
            trimci_config_path=args.trimci_config,
            **overrides,
        )

    # ----- Build result_data -----
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result_data = {
        "calculation_timestamp": timestamp,
        "system_info": {
            "n_electrons": iteration_details.get("n_electrons", "N/A"),
            "n_orbitals": iteration_details.get("n_orbitals", "N/A"),
            "nuclear_repulsion": iteration_details.get("nuclear_repulsion", 0.0),
        },
        "configuration": vars(config),   # dict
        "results": {
            "final_determinants": len(final_dets),
            "final_energy": final_energy
        },
        "timing": {
            "total_time": iteration_details.get("total_time", "N/A")
        },
        "experimental_workflow": iteration_details
    }

    # Save markdown report
    report_path = fcidump_path.parent / f"trimci_report_{timestamp.replace('-', '').replace(':', '').replace(' ', '_')}.md"
    generate_markdown_report(result_data, str(report_path), str(fcidump_path), vars(config))
    print(f"📄 Markdown report generated: {report_path}")

    # Save JSON result
    json_path = fcidump_path.parent / f"trimci_result_{timestamp.replace('-', '').replace(':', '').replace(' ', '_')}.json"
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(result_data, jf, indent=2)
    print(f"💾 JSON result saved: {json_path}")

    # Handle --orbopt flag
    if args.orbopt:
        results_dir = iteration_details.get("results_dir", "results")
        final_dets_path = Path(results_dir) / "dets.npz"
        target_path = Path.cwd() / "dets.npz"
        
        if not args.fixed_initial_mode:
            # Copy dets.npz if exists
            if final_dets_path.exists():
                shutil.copy2(final_dets_path, target_path)
                print(f"📋 Copied {final_dets_path} → {target_path}")
            else:
                print(f"⚠️  Warning: {final_dets_path} not found, skipping copy")
            
        # Backup FCIDUMP to results directory
        results_dir_path = Path(results_dir)
        if not results_dir_path.exists():
            results_dir_path.mkdir(parents=True)
        fcidump_backup = results_dir_path / "FCIDUMP"
        shutil.copy2(fcidump_path, fcidump_backup)
        print(f"💾 Backed up FCIDUMP to {fcidump_backup}")


if __name__ == "__main__":
    main()
