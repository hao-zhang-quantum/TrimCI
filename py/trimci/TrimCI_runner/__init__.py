"""TrimCI runner subpackage.

Provides high-level helpers to run TrimCI calculations
from FCIDUMP or molecule definitions.
"""

from .trimci_driver import run_full_calculation
from .run_auto import run_auto
from .run_trimci import main as cli_main
from .parallel_runner import run_trimci_main_calculation_parallel
from .io_utils import read_fcidump
from .run_expansion import run_expansion

__all__ = [
    "run_full_calculation",
    "run_trimci_main_calculation_parallel",
    "run_auto",
    "run_expansion",
    "cli_main",
    "read_fcidump",
]
