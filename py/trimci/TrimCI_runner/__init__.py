"""TrimCI runner subpackage.

Provides high-level helpers to run TrimCI calculations
from FCIDUMP or molecule definitions.
"""

from .trimci_driver import run_full_calculation
from .run_trimci import main as cli_main

__all__ = [
    "run_full_calculation",
    "run_auto",
    "cli_main",
    "read_fcidump",
]