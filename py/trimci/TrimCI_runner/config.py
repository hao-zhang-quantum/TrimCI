"""
TrimCI configuration loading and default parameters.
"""
import json
import os
from types import SimpleNamespace as Namespace

from trimci.trimci_logging import log_verbose


# Defaults for Phase 0 (the iterative core-discovery loop driven by
# run_full / run_trimci_main_calculation). Consumed in trimci_driver.py
# and iterative_workflow.py. Additional params routed through
# optimizer_options_dict are documented in orbital_optimization.py.
DEFAULT_CONFIG = {
    # ==== Pool-build screening ====
    "threshold":             0.06,   # |H_ij·c_j| cutoff for pool inclusion; ↑=speed, ↓=accuracy
    "initial_pool_size":     100,    # candidates sampled at round 1
    "pool_core_ratio":       20,     # pool size = ratio × current core size
    "pool_build_strategy":   "hb",   # "hb" (heat-bath) | "uniform" | "quantum"
    # ==== Core growth schedule ====
    "max_final_dets":        100,    # target det count at end of Phase 0; "auto" = 3000/n_orb^1.5
    "max_rounds":            2,      # TrimCI pool-build rounds per Phase-0 cycle
    "core_set_ratio":        1.02,   # per-round growth cap; list [lo, hi] also accepted
    "first_cycle_keep_size": 10,     # dets kept after the very first trim
    # ==== Parallel trimming ====
    "num_groups":            10,     # parallel trim groups (↑=more accurate, ↑=slower)
    "local_trim_keep_ratio": 0.1,    # retention ratio within each group after trim
    # ==== Run orchestration ====
    "num_runs":              1,      # independent stochastic restarts; best-energy wins
    "load_initial_dets":     False,  # True → read initial dets from initial_dets_path
    "verbose":               False,  # coarse verbosity flag (detailed: args.verbosity)
}


def _load_config_from_py(py_path: str) -> dict:
    """Load config dict from a Python file."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("trimci_config", py_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, 'config') and isinstance(module.config, dict):
        return module.config
    raise ValueError(f"trimci_config.py must define a 'config' dict, not found in {py_path}")


def _load_config_from_json(json_path: str) -> dict:
    """Load config dict from a JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def load_configurations(config_dir: str, trimci_config_path: str = None, save_if_not_exist: bool = False):
    """
    Load TrimCI configuration with priority:
      1. Explicit path (if provided): .py or .json based on extension
      2. Auto-detect in config_dir: trimci_config.py > trimci_config.json
      3. Default config (optionally saved as .json)
    """
    config = DEFAULT_CONFIG.copy()

    # Case 1: Explicit path provided
    if trimci_config_path is not None:
        if os.path.exists(trimci_config_path):
            if trimci_config_path.endswith('.py'):
                config.update(_load_config_from_py(trimci_config_path))
            else:
                config.update(_load_config_from_json(trimci_config_path))
            log_verbose(f"Loaded config: {trimci_config_path}")
        else:
            log_verbose(f"Config file not found: {trimci_config_path}, using defaults")
        return Namespace(**config)

    # Case 2: Auto-detect in config_dir (priority: .py > .json)
    py_path = os.path.join(config_dir, "trimci_config.py")
    json_path = os.path.join(config_dir, "trimci_config.json")

    if os.path.exists(py_path):
        config.update(_load_config_from_py(py_path))
        log_verbose(f"Loaded config: {py_path}")
    elif os.path.exists(json_path):
        config.update(_load_config_from_json(json_path))
        log_verbose(f"Loaded config: {json_path}")
    else:
        # Case 3: No config found, optionally create default .json
        if save_if_not_exist:
            with open(json_path, 'w') as f:
                json.dump(DEFAULT_CONFIG, f, indent=2)
            log_verbose(f"Created default config: {json_path}")

    return Namespace(**config)
