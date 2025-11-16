"""TrimCI: Truncated Configuration Interaction package.

A high-performance quantum chemistry package for truncated CI calculations
with automatic determinant type selection for optimal performance.
"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path


def _read_pyproject_version():
    here = Path(__file__).resolve()
    for parent in [here.parent, here.parent.parent, here.parent.parent.parent]:
        candidate = parent / "pyproject.toml"
        if candidate.exists():
            try:
                try:
                    import tomllib as _toml
                except Exception:
                    import tomli as _toml  # type: ignore
                data = _toml.loads(candidate.read_text(encoding="utf-8"))
                v = data.get("project", {}).get("version")
                if isinstance(v, str) and v:
                    return v
            except Exception:
                pass
    return None

try:
    __version__ = _pkg_version("trimci")
except PackageNotFoundError:
    __version__ = _read_pyproject_version() or "0.0.0"

__author__ = "TrimCI Development Team"

from .interface import (
    extract_mol_name,
    load_or_create_Hij_cache,
    generate_reference_det,
    generate_excitations,
    screening,
    trim,
    get_determinant_class,
    get_system_recommendation,
    create_determinant,
)

from .molecule_setup import setup_molecule

from .auto_selector import (
    select_determinant_type,
    get_functions_for_system,
    get_system_info,
    print_system_recommendation,
)

from .TrimCI_runner.trimci_driver import run_full_calculation, read_fcidump

__all__ = [
    "extract_mol_name",
    "load_or_create_Hij_cache",
    "generate_reference_det",
    "generate_excitations",
    "screening",
    "trim",
    "setup_molecule",
    "get_determinant_class",
    "get_system_recommendation",
    "create_determinant",
    "select_determinant_type",
    "get_functions_for_system",
    "get_system_info",
    "print_system_recommendation",
    "run_full_calculation",
    "run_auto",
    "cli_main",
    "read_fcidump",
]