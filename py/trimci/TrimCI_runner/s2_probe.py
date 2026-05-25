"""Diagnostic measurement of ⟨S²⟩ and the inferred total spin S for a
TrimCI wavefunction.

The underlying C++ kernel lives in ``trimci_core.evaluate_S2`` /
``evaluate_S2_128``. This module is a thin wrapper that accepts either
raw ``(alphas, betas, coeffs)`` arrays or a list of Determinant objects.

Usage::

    from trimci.TrimCI_runner.s2_probe import measure_s2
    s2, s = measure_s2(alphas=a, betas=b, coeffs=c)
    print(f"⟨S²⟩ = {s2:.4f}   S ≈ {s:.4f}")
"""
from __future__ import annotations

import numpy as np

from trimci import trimci_core


def _s_from_s2(s2: float) -> float:
    """Invert S(S+1) = ⟨S²⟩, clamped to 0 to tolerate tiny negatives from
    numerical noise."""
    return 0.5 * (-1.0 + np.sqrt(max(0.0, 1.0 + 4.0 * s2)))


def measure_s2(
    *,
    alphas=None,
    betas=None,
    coeffs=None,
    dets=None,
    use_128: bool | None = None,
) -> tuple[float, float]:
    """Compute ⟨Ψ|S²|Ψ⟩ and the inferred S = (-1 + √(1+4⟨S²⟩))/2.

    Either (``alphas``, ``betas``, ``coeffs``) or (``dets``, ``coeffs``) must
    be provided. ``dets`` is a list of ``trimci_core.Determinant`` (64-bit)
    or ``Determinant128`` objects.

    Returns
    -------
    (s2, s): floats
    """
    if coeffs is None:
        raise ValueError("coeffs must be provided")
    coeffs = np.asarray(coeffs, dtype=np.float64)

    if dets is not None:
        if len(dets) != len(coeffs):
            raise ValueError(
                f"len(dets)={len(dets)} does not match len(coeffs)={len(coeffs)}")
        if len(dets) == 0:
            return 0.0, 0.0

        # Extract α/β from Determinant objects
        first = dets[0]
        alpha_bits = first.alpha
        if isinstance(alpha_bits, int):
            # 64-bit determinants
            alpha_arr = np.fromiter(
                (int(d.alpha) & 0xFFFFFFFFFFFFFFFF for d in dets),
                dtype=np.uint64, count=len(dets))
            beta_arr = np.fromiter(
                (int(d.beta) & 0xFFFFFFFFFFFFFFFF for d in dets),
                dtype=np.uint64, count=len(dets))
            s2 = trimci_core.evaluate_S2(alpha_arr, beta_arr, coeffs)
        elif len(alpha_bits) == 2:
            # 128-bit determinants
            alpha_list = [tuple(int(x) & 0xFFFFFFFFFFFFFFFF for x in d.alpha)
                          for d in dets]
            beta_list = [tuple(int(x) & 0xFFFFFFFFFFFFFFFF for x in d.beta)
                         for d in dets]
            s2 = trimci_core.evaluate_S2_128(alpha_list, beta_list, coeffs)
        else:
            raise NotImplementedError(
                f"No S² binding for {len(alpha_bits) * 64}-bit determinants; "
                f"only 64-bit and 128-bit are wired to Python.")
    else:
        if alphas is None or betas is None:
            raise ValueError("Provide either dets or (alphas, betas).")
        alphas = np.asarray(alphas)
        betas = np.asarray(betas)
        if use_128 is None:
            use_128 = alphas.ndim == 2 and alphas.shape[1] == 2
        if use_128:
            alpha_list = [tuple(int(x) for x in row) for row in alphas]
            beta_list = [tuple(int(x) for x in row) for row in betas]
            s2 = trimci_core.evaluate_S2_128(alpha_list, beta_list, coeffs)
        else:
            s2 = trimci_core.evaluate_S2(
                alphas.astype(np.uint64),
                betas.astype(np.uint64),
                coeffs)

    s = _s_from_s2(s2)
    return float(s2), float(s)
