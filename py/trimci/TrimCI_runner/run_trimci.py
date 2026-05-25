#!/usr/bin/env python
"""
CLI entry point for TrimCI calculations.

Reads FCIDUMP from disk and runs TrimCI with parameters from
trimci_config.json (or .py) in the same directory. Command-line
arguments override config-file values.

Usage:
    tci --fcidump FCIDUMP                     # basic run
    tci --fcidump FCIDUMP -n 200              # cap at 200 determinants
    tci --auto --goal accuracy -n 500         # auto-tuning mode
    tci --fcidump FCIDUMP --threshold 0.03    # override any config param
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import argparse

from .trimci_driver import run_full_calculation
from .run_auto import run_auto
from .config import load_configurations


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _parse_goal(value: str) -> str:
    """Parse --goal with abbreviation support (b/s/a)."""
    mapping = {
        'b': 'balanced', 'balanced': 'balanced',
        's': 'speed',    'speed': 'speed',
        'a': 'accuracy', 'accuracy': 'accuracy',
    }
    v = str(value).strip().lower()
    if v in mapping:
        return mapping[v]
    raise argparse.ArgumentTypeError(
        f"Invalid goal: {value}. Use balanced|speed|accuracy or b|s|a."
    )


def _coerce_cli_value(v: str):
    """Coerce a CLI string to the appropriate Python type (int/float/bool/list/dict)."""
    s = str(v).strip()

    # JSON containers
    if (s.startswith('[') and s.endswith(']')) or (s.startswith('{') and s.endswith('}')):
        try:
            return json.loads(s)
        except Exception:
            pass

    ls = s.lower()
    if ls in ('true', 'false'):
        return ls == 'true'
    if ls in ('none', 'null'):
        return None

    try:
        if '.' in s or 'e' in ls:
            return float(s)
        return int(s)
    except ValueError:
        if s.startswith('"') and s.endswith('"'):
            try:
                return json.loads(s)
            except Exception:
                return s[1:-1]
        return s


def _consume_json_like(tokens, start_index: int):
    """Join multi-token JSON values like [1, 2, 3] or {"a": 1}."""
    if start_index >= len(tokens):
        return "", start_index
    first = tokens[start_index]
    if not (first.startswith('[') or first.startswith('{')):
        return first, start_index + 1
    depth, acc, i = 0, [], start_index
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


def _parse_unknown_args(unknown_args):
    """Parse unregistered CLI arguments into an overrides dict.

    Supports:  --key=value, --key value, --flag (bool True), -k value,
    and multi-token JSON values like --schedule '[10, 20, 50]'.
    """
    overrides = {}
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
    return overrides


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run TrimCI calculation from FCIDUMP.",
        epilog="Any unknown --key value pairs are forwarded as config overrides.",
    )
    parser.add_argument("--fcidump", type=str, default="FCIDUMP",
                        help="Path to FCIDUMP file (default: ./FCIDUMP)")
    parser.add_argument("--trimci_config", type=str, default=None,
                        help="Path to trimci_config (.json or .py)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--auto", action="store_true",
                        help="Use auto-tuning mode (run_auto)")
    parser.add_argument("--goal", type=_parse_goal,
                        choices=["balanced", "speed", "accuracy"], default="balanced",
                        help="Auto-mode tuning goal (b/s/a accepted)")
    parser.add_argument("-n", "--max_final_dets", type=int,
                        help="Max final determinants")
    parser.add_argument("--orbopt", action="store_true",
                        help="Copy final dets.npz to cwd and backup FCIDUMP")
    parser.add_argument("--fixed_initial_mode", action="store_true",
                        help="With --orbopt, skip copying dets.npz (keep existing)")

    args, unknown_args = parser.parse_known_args()

    # --- Resolve FCIDUMP path ---
    args_config = load_configurations(str(Path(args.fcidump).parent), args.trimci_config)

    if getattr(args_config, 'fcidump_name', None) is not None:
        fcidump_path = Path(args_config.fcidump_name).resolve()
    else:
        fcidump_path = Path(args.fcidump).resolve()

    if not fcidump_path.exists():
        candidates = list(Path.cwd().glob("fcidump*")) + list(Path.cwd().glob("FCIDUMP*"))
        if not candidates:
            raise FileNotFoundError("No FCIDUMP file found in current directory")
        fcidump_path = candidates[0].resolve()
        print(f"Using FCIDUMP file: {fcidump_path}")

    # --- Build overrides ---
    overrides = _parse_unknown_args(unknown_args)
    if args.verbose:
        overrides["verbose"] = True
    if args.max_final_dets is not None:
        overrides["max_final_dets"] = args.max_final_dets

    # --- Run calculation ---
    if args.auto:
        # For auto mode, pass max_final_dets as ndets (the target budget),
        # and filter it from overrides (run_auto explicitly skips it).
        auto_overrides = {k: v for k, v in overrides.items() if k != 'max_final_dets'}
        final_energy, final_dets, final_coeffs, iteration_details, config = run_auto(
            fcidump_path=str(fcidump_path),
            trimci_config_path=args.trimci_config,
            goal=args.goal,
            ndets=args.max_final_dets,
            **auto_overrides,
        )
    else:
        final_energy, final_dets, final_coeffs, iteration_details, config = run_full_calculation(
            fcidump_path=str(fcidump_path),
            trimci_config_path=args.trimci_config,
            **overrides,
        )

    # --- Save JSON result ---
    results_dir = iteration_details.get("results_dir", ".")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = os.path.join(results_dir, f"trimci_cli_result_{timestamp}.json")
    result_data = {
        "calculation_timestamp": timestamp,
        "system_info": {
            "n_electrons": iteration_details.get("n_electrons", "N/A"),
            "n_orbitals": iteration_details.get("n_orbitals", "N/A"),
            "nuclear_repulsion": iteration_details.get("nuclear_repulsion", 0.0),
        },
        "results": {
            "final_determinants": len(final_dets),
            "final_energy": final_energy,
        },
        "timing": {
            "total_time": iteration_details.get("total_time", "N/A"),
        },
        "configuration": vars(config),
    }
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(result_data, jf, indent=2, default=str)
    print(f"JSON result saved: {json_path}")

    # --- Handle --orbopt ---
    if args.orbopt:
        final_dets_path = Path(results_dir) / "dets.npz"
        target_path = Path.cwd() / "dets.npz"

        if not args.fixed_initial_mode:
            if final_dets_path.exists():
                shutil.copy2(final_dets_path, target_path)
                print(f"Copied {final_dets_path} -> {target_path}")
            else:
                print(f"Warning: {final_dets_path} not found, skipping copy")

        results_dir_path = Path(results_dir)
        results_dir_path.mkdir(parents=True, exist_ok=True)
        fcidump_backup = results_dir_path / "FCIDUMP"
        shutil.copy2(fcidump_path, fcidump_backup)
        print(f"Backed up FCIDUMP to {fcidump_backup}")


if __name__ == "__main__":
    main()
