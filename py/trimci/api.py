"""TrimCI friendly API layer.

Provides a clean, agent-friendly interface on top of TrimCI's internals.
All entry points normalize to FCIDUMP + config dicts, then call the
existing three-phase workflow (run_full + run_expansion).

Usage:
    import trimci
    result = trimci.ground_state_from_fcidump("FCIDUMP")
    print(result.energy)
    print(result.wavefunction.leading(5))
    result.save("/tmp/run.json")
"""
from __future__ import annotations

import json
import os
import tempfile
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


@dataclass
class Config:
    """TrimCI calculation configuration with sensible defaults.

    All parameters have production-ready defaults. Modify only what you need.
    The phase*_overrides dicts are escape hatches: any key-value pair placed
    there is merged (last-wins) into the internal config dict for that phase.

    Defaults track ``trimci.TrimCI_skill`` (the production single-file skill).
    For knobs not first-class here (pool_core_ratio, num_groups, max_rounds,
    core_set_ratio, first_cycle_keep_size, pool_build_strategy,
    local_trim_keep_ratio), use the corresponding ``phase*_overrides`` dict.
    """

    active: tuple[int, int] | None = None

    enable_phase1: bool = True
    enable_phase2: bool = True

    # ── Phase 0: Core + OrbOpt Discovery ──────────────────────────────
    num_runs: int = 64
    # Primary worker fields. Legacy alias `n_parallel` still accepted and
    # routed to `n_cpu_workers` when only the legacy field is set.
    n_cpu_workers: int = 1     # concurrent CPU subprocesses, each its own OMP pool
    n_gpu_workers: int = 0     # 1:1 to GPUs; mutually exclusive with n_cpu_workers > 1
    n_parallel: int = 1        # DEPRECATED alias for n_cpu_workers
    orbopt: bool = True
    optimizer: str = "auto_bfgs"   # auto_bfgs | cpp_bfgs | bfgs | gpu_bfgs
    orbopt_cycles: int = 10
    # Phase 0 orbopt cycle behavior:
    #   tracking_dets=True carries dets between cycles (instead of fresh random
    #   stochastic exploration each cycle). loaded_dets_randomness > 0 adds a
    #   random single-excitation walk to the inherited dets, controlling how
    #   much the tracking explores vs exploits. Together they let cycles refine
    #   a basin rather than restart from scratch.
    tracking_dets: bool = False
    loaded_dets_randomness: float = 0.0
    max_dets_phase0: int = 100
    threshold: float = 1e-2
    backend_phase0: str = "auto"   # auto | cpu | gpu (CI workflow execution backend)

    # ── Phase 1: Core + OrbOpt Refinement (Expansion) ─────────────────
    max_dets_phase1: int = 1_000_000
    growth_factor_phase1: float = 1.1
    orbopt_phase1: bool = True
    orbopt_max_iter: int = 50
    orbopt_grad_tol: float = 1e-3
    backend_phase1: str = "cpu"             # cpu | gpu | auto (Davidson matvec backend)
    use_connection_cache_phase1: bool = True
    davidson_energy_tol_phase1: float = 1e-4

    # ── Phase 2: Core Expansion Only ──────────────────────────────────
    max_dets_phase2: int = 100_000_000
    growth_factor_phase2: float = 2.0
    pt2_correction: bool = True
    # When True (and pt2_correction is True), Phase 2 expansion runs WITHOUT
    # per-round PT2; a final 1-round evaluate_only phase computes PT2 just at
    # the final det count. Saves wall when Phase 2 has many rounds — per-round
    # PT2 can cost 10-50% of each round. Default False for skill compat.
    pt2_only_last: bool = False
    backend_phase2: str = "cpu"             # cpu | gpu | auto
    davidson_energy_tol_phase2: float = 1e-5

    # ── Reproducibility / multi-seed ──────────────────────────────────
    # Master seed for all stochastic components: Phase 0 random-init draws,
    # PT2 sketch + stochastic sampling, and tracking_dets random walks.
    # None (default) → fresh entropy per run (each call gives a different
    # stochastic trajectory; matches the legacy unseeded behavior).
    # int → fully reproducible: same seed → bitwise-identical Result.
    # Sub-seeds for individual components are derived deterministically.
    seed: int | None = None

    # ── Escape hatches ────────────────────────────────────────────────
    phase0_overrides: dict = field(default_factory=dict)
    phase1_overrides: dict = field(default_factory=dict)
    phase2_overrides: dict = field(default_factory=dict)

    work_dir: str | None = None
    verbosity: int = 1


# ---------------------------------------------------------------------------
#  Result helpers
# ---------------------------------------------------------------------------

def _bits_to_int(segs: np.ndarray) -> int:
    """Reassemble (K,) uint64 array into a single Python int (low-order first)."""
    out = 0
    for k in range(len(segs)):
        out |= int(segs[k]) << (64 * k)
    return out


def _bits_to_str(segs: np.ndarray, n_orb: int) -> str:
    return format(_bits_to_int(segs), f"0{n_orb}b")


def _coef_to_python(c) -> float | complex:
    """Convert a numpy scalar coefficient to a JSON-friendly Python scalar."""
    if np.iscomplexobj(c):
        return complex(c)
    return float(c)


# ---------------------------------------------------------------------------
#  Result sub-objects
# ---------------------------------------------------------------------------

@dataclass
class Energies:
    """Energy decomposition. All values in Hartree."""
    var: float                          # variational total energy
    pt2_correction: float | None        # ΔE_PT2 (≤ 0); None if PT2 not computed
    nuclear: float                      # nuclear repulsion (constant for given geometry)
    history: list[float] = field(default_factory=list)         # E_var across all rounds
    pt2_history: list[float] = field(default_factory=list)     # ΔE_PT2 per expansion round

    @property
    def pt2(self) -> float | None:
        """Variational energy + PT2 correction (single number), or None."""
        if self.pt2_correction is None:
            return None
        return self.var + self.pt2_correction

    @property
    def electronic(self) -> float:
        """Electronic energy (var minus nuclear repulsion)."""
        return self.var - self.nuclear


@dataclass
class Wavefunction:
    """|Ψ⟩ = Σ c_i |α_i, β_i⟩.

    Behaves like a dict for bitstring lookup (backwards-compatible with the
    old Result.wavefunction dict): ``wf["110011|110011"]`` returns the
    coefficient. New accessors: ``coefficients``, ``alpha_bits``, ``beta_bits``,
    ``leading(k)``, ``occupation_numbers()``.
    """
    alpha_bits: np.ndarray              # (N, K) uint64 — α occupation bitstrings
    beta_bits: np.ndarray               # (N, K) uint64 — β occupation bitstrings
    coefficients: np.ndarray            # (N,) float64 / complex128
    n_orb: int

    _dict_cache: dict | None = field(default=None, repr=False, compare=False)

    def __len__(self) -> int:
        return len(self.coefficients)

    def __getitem__(self, key: str):
        return self.to_dict()[key]

    def __contains__(self, key: str) -> bool:
        return key in self.to_dict()

    def __iter__(self):
        return iter(self.to_dict())

    def keys(self):
        return self.to_dict().keys()

    def values(self):
        return self.to_dict().values()

    def items(self):
        return self.to_dict().items()

    @property
    def n_dets(self) -> int:
        return len(self.coefficients)

    @property
    def coeffs(self) -> np.ndarray:
        """Short alias for ``.coefficients``."""
        return self.coefficients

    @property
    def dets(self) -> np.ndarray:
        """Stacked (N, 2K) uint64 array: ``[alpha_segs..., beta_segs...]``.

        Same layout as run_expansion's ``dets_phase0.npz``. Builds a fresh
        column_stack on each access; cache to a local variable in tight loops.
        """
        return np.column_stack([self.alpha_bits, self.beta_bits])

    @classmethod
    def from_dets(cls, dets: np.ndarray, coefficients, n_orb: int) -> "Wavefunction":
        """Construct from a stacked (N, 2K) ``dets`` array (e.g. loaded from npz)."""
        dets = np.asarray(dets, dtype=np.uint64)
        if dets.ndim != 2 or dets.shape[1] % 2 != 0:
            raise ValueError(
                f"dets must be (N, 2K); got shape {dets.shape}"
            )
        K = dets.shape[1] // 2
        return cls(
            alpha_bits=dets[:, :K].copy(),
            beta_bits=dets[:, K:].copy(),
            coefficients=np.asarray(coefficients),
            n_orb=n_orb,
        )

    @property
    def is_normalized(self) -> bool:
        norm = float(np.sum(np.abs(self.coefficients) ** 2))
        return abs(norm - 1.0) < 1e-9

    @property
    def is_complex(self) -> bool:
        return self.coefficients.dtype.kind == "c"

    def leading(self, k: int = 10) -> list[tuple[str, str, float | complex]]:
        """Top-k determinants by |c_i|. Returns [(α_bitstring, β_bitstring, coef), ...]."""
        idx = np.argsort(-np.abs(self.coefficients))[:k]
        return [
            (
                _bits_to_str(self.alpha_bits[i], self.n_orb),
                _bits_to_str(self.beta_bits[i], self.n_orb),
                _coef_to_python(self.coefficients[i]),
            )
            for i in idx
        ]

    def occupation_numbers(self) -> np.ndarray:
        """⟨n_p⟩ per orbital, weighted by |c_i|². Returns shape (n_orb, 2) for [α, β]."""
        n_orb = self.n_orb
        weights = np.abs(self.coefficients) ** 2
        occ = np.zeros((n_orb, 2), dtype=np.float64)
        for i in range(self.n_dets):
            a_int = _bits_to_int(self.alpha_bits[i])
            b_int = _bits_to_int(self.beta_bits[i])
            for p in range(n_orb):
                if (a_int >> p) & 1:
                    occ[p, 0] += weights[i]
                if (b_int >> p) & 1:
                    occ[p, 1] += weights[i]
        return occ

    def to_dict(self) -> dict[str, float | complex]:
        """Bitstring-keyed coefficient dict. Cached lazily."""
        if self._dict_cache is None:
            d: dict[str, float | complex] = {}
            for i in range(self.n_dets):
                a_str = _bits_to_str(self.alpha_bits[i], self.n_orb)
                b_str = _bits_to_str(self.beta_bits[i], self.n_orb)
                d[f"{a_str}|{b_str}"] = _coef_to_python(self.coefficients[i])
            self._dict_cache = d
        return self._dict_cache


@dataclass
class Diagnostics:
    """Per-phase wall, n_dets, and the Config used."""
    n_dets_phase0: int                  # actual n_dets after Phase 0
    n_dets_final: int                   # actual n_dets after expansion
    wall_phase0: float                  # seconds
    wall_expansion: float               # seconds (Phase 1 + Phase 2 combined)
    expansion_rounds: int               # total rounds across Phase 1+2
    config: Config
    # Full Hilbert-space dimension = C(n_orb, n_alpha) × C(n_orb, n_beta).
    # When the user-requested n_dets exceeds n_fci, expansion caps are
    # silently clamped to n_fci.
    n_fci: int | None = None
    # Filesystem pointers — let Result lazy-load the final orbitals/integrals.
    # None when orbopt didn't run or expansion was skipped.
    orbopt_dir: str | None = None       # Phase 0 orbopt npz dir (has h1/eri/U at each cycle)
    checkpoint_dir: str | None = None   # Phase 1+2 root checkpoint dir
                                        # (contains phase{1,2,...}/checkpoint_integrals.bin)

    @property
    def total_wall(self) -> float:
        return self.wall_phase0 + self.wall_expansion


# ---------------------------------------------------------------------------
#  Result
# ---------------------------------------------------------------------------

@dataclass
class Result:
    """TrimCI ground-state result.

    Three logical groups (each is its own dataclass):
      - ``.energies``      — variational, PT2, nuclear, history
      - ``.wavefunction``  — dets, coefficients, leading(), occupation_numbers()
      - ``.diagnostics``   — convergence, per-phase wall, config

    Top-level shortcuts for the most common quantities:
      ``.energy`` / ``.energy_pt2`` / ``.n_dets`` / ``.wall_time``

    Save/load JSON for cross-session persistence:
      ``.save(path)``  /  ``Result.load(path)``
    """
    energies: Energies
    wavefunction: Wavefunction
    diagnostics: Diagnostics
    fcidump_path: str | None = None

    # Lazy caches for orbital-rotation properties.
    _orbopt_npz_cache: dict | None = field(default=None, repr=False, compare=False)
    _checkpoint_integrals_cache: dict | None = field(default=None, repr=False, compare=False)

    # ── Top-level convenience shortcuts ───────────────────────────────
    @property
    def energy(self) -> float:
        """Variational total energy (Ha)."""
        return self.energies.var

    @property
    def energy_pt2(self) -> float | None:
        """Variational + PT2 correction (Ha), or None if PT2 not computed."""
        return self.energies.pt2

    @property
    def n_dets(self) -> int:
        """Final determinant count."""
        return self.wavefunction.n_dets

    @property
    def wall_time(self) -> float:
        """Total wall (Phase 0 + expansion), seconds."""
        return self.diagnostics.total_wall

    # ── Orbital rotation accessors (lazy, from disk) ──────────────────
    #
    # Conceptual map of where the integrals/rotations live:
    #   user input (h1_in, eri_in)  ─── reference orbitals
    #         │  Phase 0 orbopt        — produces U_phase0, h1_p0, eri_p0
    #         ▼
    #   orbopt_dir/orbital_optimization_result_orbopt.npz
    #         │  Phase 1 orbopt rounds — produces U_phase1, h1_final, eri_final
    #         ▼
    #   checkpoint_dir/phase{N}/checkpoint_integrals.bin
    #
    # ``orbital_rotation`` composes U_phase0 @ U_phase1 to give the full
    # rotation from user input to final basis.
    # ``final_h1`` / ``final_eri`` give the integrals in the final basis
    # directly — usually what users want for downstream analysis.

    def _load_phase0_orbopt(self) -> dict | None:
        if self._orbopt_npz_cache is not None:
            return self._orbopt_npz_cache or None
        d = self.diagnostics.orbopt_dir
        if not d or not os.path.isdir(d):
            self._orbopt_npz_cache = {}
            return None
        primary = os.path.join(d, "orbital_optimization_result_orbopt.npz")
        npz_path = None
        if os.path.exists(primary):
            npz_path = primary
        else:
            try:
                cycles = sorted([f for f in os.listdir(d)
                                 if f.startswith("orbital_optimization_result_cycle_")])
                if cycles:
                    npz_path = os.path.join(d, cycles[-1])
            except OSError:
                pass
        if npz_path is None:
            self._orbopt_npz_cache = {}
            return None
        try:
            data = np.load(npz_path)
            self._orbopt_npz_cache = {k: data[k] for k in data.files
                                      if k in ("U", "h1", "eri")}
        except (OSError, KeyError):
            self._orbopt_npz_cache = {}
        return self._orbopt_npz_cache or None

    def _load_checkpoint_integrals(self) -> dict | None:
        if self._checkpoint_integrals_cache is not None:
            return self._checkpoint_integrals_cache or None
        cd = self.diagnostics.checkpoint_dir
        if not cd or not os.path.isdir(cd):
            self._checkpoint_integrals_cache = {}
            return None
        # Look for the latest phase subdir (phase1, phase2, ...) that has
        # checkpoint_integrals.bin. All phases share the same final integrals
        # so any one with the file works; pick latest to be safe.
        candidates = [os.path.join(cd, "checkpoint_integrals.bin")]
        try:
            sub = sorted([f for f in os.listdir(cd)
                          if f.startswith("phase") and os.path.isdir(os.path.join(cd, f))])
            for s in reversed(sub):  # latest phase first
                candidates.insert(0, os.path.join(cd, s, "checkpoint_integrals.bin"))
        except OSError:
            pass
        existing = [p for p in candidates if os.path.exists(p)]
        if not existing:
            self._checkpoint_integrals_cache = {}
            return None
        try:
            from trimci.checkpoint import load_integrals_checkpoint
            self._checkpoint_integrals_cache = load_integrals_checkpoint(existing[0])
        except Exception:
            self._checkpoint_integrals_cache = {}
        return self._checkpoint_integrals_cache or None

    @property
    def final_h1(self) -> np.ndarray | None:
        """One-body integrals (n_orb, n_orb) in the FINAL orbital basis.

        "Final" means after both Phase 0 and Phase 1 orbital rotations.
        Returns None if no expansion checkpoint exists (Phase 1 disabled,
        or work_dir cleaned up after the run).
        """
        ck = self._load_checkpoint_integrals()
        if ck is not None:
            return np.asarray(ck["h1"])
        # Fall back to Phase 0 result if expansion didn't run
        p0 = self._load_phase0_orbopt()
        return np.asarray(p0["h1"]) if p0 and "h1" in p0 else None

    @property
    def final_eri(self) -> np.ndarray | None:
        """Two-body integrals (n_orb,)*4 in the FINAL orbital basis."""
        ck = self._load_checkpoint_integrals()
        if ck is not None:
            return np.asarray(ck["eri"])
        p0 = self._load_phase0_orbopt()
        return np.asarray(p0["eri"]) if p0 and "eri" in p0 else None

    @property
    def orbital_rotation(self) -> np.ndarray | None:
        """Cumulative orbital rotation matrix (n_orb, n_orb) from user input
        orbitals to the final basis: ``C_final = C_input @ U``.

        Composes ``U_phase0`` × ``U_phase1`` when both are available.
        Returns None if no orbital optimization ran.
        """
        p0 = self._load_phase0_orbopt()
        ck = self._load_checkpoint_integrals()
        U0 = np.asarray(p0["U"]) if p0 and "U" in p0 else None
        U1 = np.asarray(ck["U_total"]) if ck and "U_total" in ck else None
        if U0 is None and U1 is None:
            return None
        if U0 is None:
            return U1
        if U1 is None:
            return U0
        return U0 @ U1

    # ── Pretty printing ───────────────────────────────────────────────
    def summary(self) -> str:
        d = self.diagnostics
        lines = [
            "TrimCI Calculation Summary",
            f"  Energy (var):     {self.energy:.10f} Ha",
        ]
        if self.energy_pt2 is not None:
            dpt2 = self.energies.pt2_correction
            dpt2_str = (f"{dpt2:+.6f}" if abs(dpt2) >= 1e-5
                        else f"{dpt2:+.3e}")
            lines.append(f"  Energy (PT2):     {self.energy_pt2:.10f} Ha"
                         f"  (ΔPT2 = {dpt2_str})")
        lines.extend([
            f"  Determinants:     {self.n_dets:,}  (Phase 0: {d.n_dets_phase0:,})",
            f"  Wall (Phase 0):   {d.wall_phase0:.1f} s",
            f"  Wall (Expansion): {d.wall_expansion:.1f} s "
            f"({d.expansion_rounds} rounds)",
            f"  Wall (total):     {self.wall_time:.1f} s",
        ])
        return "\n".join(lines)

    def __repr__(self) -> str:
        pt2 = (f", E_PT2={self.energy_pt2:.8f}"
               if self.energy_pt2 is not None else "")
        return f"TrimCI Result: E={self.energy:.8f}{pt2}, N_dets={self.n_dets:,}"

    # ── Serialization ─────────────────────────────────────────────────
    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dict."""
        from dataclasses import asdict
        wf = self.wavefunction
        return {
            "energies": {
                "var": self.energies.var,
                "pt2_correction": self.energies.pt2_correction,
                "nuclear": self.energies.nuclear,
                "history": list(self.energies.history),
                "pt2_history": list(self.energies.pt2_history),
            },
            "wavefunction": {
                "alpha_bits": wf.alpha_bits.tolist(),
                "beta_bits": wf.beta_bits.tolist(),
                "coefficients": (
                    [[c.real, c.imag] for c in wf.coefficients]
                    if wf.is_complex else wf.coefficients.tolist()
                ),
                "n_orb": wf.n_orb,
                "is_complex": wf.is_complex,
            },
            "diagnostics": {
                "n_dets_phase0": self.diagnostics.n_dets_phase0,
                "n_dets_final": self.diagnostics.n_dets_final,
                "wall_phase0": self.diagnostics.wall_phase0,
                "wall_expansion": self.diagnostics.wall_expansion,
                "expansion_rounds": self.diagnostics.expansion_rounds,
                "n_fci": self.diagnostics.n_fci,
                "orbopt_dir": self.diagnostics.orbopt_dir,
                "checkpoint_dir": self.diagnostics.checkpoint_dir,
                "config": asdict(self.diagnostics.config),
            },
            "fcidump_path": self.fcidump_path,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Result":
        e, w, diag = d["energies"], d["wavefunction"], d["diagnostics"]
        is_complex = w.get("is_complex", False)
        if is_complex:
            coef = np.asarray([complex(re, im) for re, im in w["coefficients"]],
                              dtype=np.complex128)
        else:
            coef = np.asarray(w["coefficients"], dtype=np.float64)
        return cls(
            energies=Energies(
                var=float(e["var"]),
                pt2_correction=(None if e["pt2_correction"] is None
                                else float(e["pt2_correction"])),
                nuclear=float(e["nuclear"]),
                history=list(e["history"]),
                pt2_history=list(e["pt2_history"]),
            ),
            wavefunction=Wavefunction(
                alpha_bits=np.asarray(w["alpha_bits"], dtype=np.uint64),
                beta_bits=np.asarray(w["beta_bits"], dtype=np.uint64),
                coefficients=coef,
                n_orb=int(w["n_orb"]),
            ),
            diagnostics=Diagnostics(
                n_dets_phase0=int(diag["n_dets_phase0"]),
                n_dets_final=int(diag["n_dets_final"]),
                wall_phase0=float(diag["wall_phase0"]),
                wall_expansion=float(diag["wall_expansion"]),
                expansion_rounds=int(diag["expansion_rounds"]),
                config=Config(**diag["config"]),
                n_fci=(int(diag["n_fci"]) if diag.get("n_fci") is not None else None),
                orbopt_dir=diag.get("orbopt_dir"),
                checkpoint_dir=diag.get("checkpoint_dir"),
            ),
            fcidump_path=d.get("fcidump_path"),
        )

    # ── Save / load (NPZ recommended for large dets, JSON for inspection) ─

    def save(self, path: str) -> str:
        """Save Result. Format auto-detected from extension:

          - ``.npz`` (default, recommended for large dets) — binary, compressed,
            preserves dtypes exactly. ~10× smaller than JSON for 1M+ dets.
          - ``.json`` — human-readable; only practical for small dets.

        If ``path`` has no extension, ``.npz`` is appended.
        """
        if path.endswith(".json"):
            return self.save_json(path)
        if not path.endswith(".npz"):
            path = path + ".npz"
        return self.save_npz(path)

    @classmethod
    def load(cls, path: str) -> "Result":
        """Load Result. Format auto-detected from extension (.npz or .json)."""
        if path.endswith(".json"):
            return cls.load_json(path)
        if not path.endswith(".npz") and not os.path.isfile(path):
            # Try .npz suffix if user passed a stem
            if os.path.isfile(path + ".npz"):
                path = path + ".npz"
            elif os.path.isfile(path + ".json"):
                return cls.load_json(path + ".json")
        return cls.load_npz(path)

    def save_npz(self, path: str) -> str:
        """Write Result to NPZ (compact binary)."""
        from dataclasses import asdict
        wf = self.wavefunction
        meta = {
            "format_version": 1,
            "energies": {
                "var": self.energies.var,
                "pt2_correction": self.energies.pt2_correction,
                "nuclear": self.energies.nuclear,
            },
            "wavefunction": {
                "n_orb": wf.n_orb,
                "is_complex": wf.is_complex,
            },
            "diagnostics": {
                "n_dets_phase0": self.diagnostics.n_dets_phase0,
                "n_dets_final": self.diagnostics.n_dets_final,
                "wall_phase0": self.diagnostics.wall_phase0,
                "wall_expansion": self.diagnostics.wall_expansion,
                "expansion_rounds": self.diagnostics.expansion_rounds,
                "n_fci": self.diagnostics.n_fci,
                "orbopt_dir": self.diagnostics.orbopt_dir,
                "checkpoint_dir": self.diagnostics.checkpoint_dir,
                "config": asdict(self.diagnostics.config),
            },
            "fcidump_path": self.fcidump_path,
        }
        np.savez_compressed(
            path,
            alpha_bits=wf.alpha_bits,
            beta_bits=wf.beta_bits,
            coefficients=wf.coefficients,
            history=np.asarray(self.energies.history, dtype=np.float64),
            pt2_history=np.asarray(self.energies.pt2_history, dtype=np.float64),
            _meta=np.array(json.dumps(meta)),
        )
        return path if path.endswith(".npz") else path + ".npz"

    @classmethod
    def load_npz(cls, path: str) -> "Result":
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["_meta"]))
            coef = data["coefficients"]
            if meta["wavefunction"]["is_complex"] and not np.iscomplexobj(coef):
                coef = coef.astype(np.complex128)
            return cls(
                energies=Energies(
                    var=float(meta["energies"]["var"]),
                    pt2_correction=(None if meta["energies"]["pt2_correction"] is None
                                    else float(meta["energies"]["pt2_correction"])),
                    nuclear=float(meta["energies"]["nuclear"]),
                    history=data["history"].tolist(),
                    pt2_history=data["pt2_history"].tolist(),
                ),
                wavefunction=Wavefunction(
                    alpha_bits=np.asarray(data["alpha_bits"], dtype=np.uint64),
                    beta_bits=np.asarray(data["beta_bits"], dtype=np.uint64),
                    coefficients=coef,
                    n_orb=int(meta["wavefunction"]["n_orb"]),
                ),
                diagnostics=Diagnostics(
                    n_dets_phase0=int(meta["diagnostics"]["n_dets_phase0"]),
                    n_dets_final=int(meta["diagnostics"]["n_dets_final"]),
                    wall_phase0=float(meta["diagnostics"]["wall_phase0"]),
                    wall_expansion=float(meta["diagnostics"]["wall_expansion"]),
                    expansion_rounds=int(meta["diagnostics"]["expansion_rounds"]),
                    config=Config(**meta["diagnostics"]["config"]),
                    n_fci=(int(meta["diagnostics"]["n_fci"])
                           if meta["diagnostics"].get("n_fci") is not None else None),
                    orbopt_dir=meta["diagnostics"].get("orbopt_dir"),
                    checkpoint_dir=meta["diagnostics"].get("checkpoint_dir"),
                ),
                fcidump_path=meta.get("fcidump_path"),
            )

    def save_json(self, path: str) -> str:
        """Write Result to JSON (human-readable; OK for small dets)."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load_json(cls, path: str) -> "Result":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    # ── Deprecated backward-compat properties ─────────────────────────
    @property
    def coefficients(self) -> np.ndarray:
        warnings.warn(
            "Result.coefficients is deprecated; use .wavefunction.coefficients",
            DeprecationWarning, stacklevel=2,
        )
        return self.wavefunction.coefficients

    @property
    def determinants(self) -> np.ndarray:
        warnings.warn(
            "Result.determinants is deprecated; "
            "use .wavefunction.alpha_bits / .beta_bits",
            DeprecationWarning, stacklevel=2,
        )
        return np.column_stack([self.wavefunction.alpha_bits,
                                self.wavefunction.beta_bits])

    @property
    def convergence(self) -> list[float]:
        warnings.warn(
            "Result.convergence is deprecated; use .energies.history",
            DeprecationWarning, stacklevel=2,
        )
        return self.energies.history

    @property
    def pt2_history(self) -> list[float]:
        warnings.warn(
            "Result.pt2_history is deprecated; use .energies.pt2_history",
            DeprecationWarning, stacklevel=2,
        )
        return self.energies.pt2_history

    @property
    def config(self) -> Config:
        warnings.warn(
            "Result.config is deprecated; use .diagnostics.config",
            DeprecationWarning, stacklevel=2,
        )
        return self.diagnostics.config

    @property
    def n_orb(self) -> int:
        warnings.warn(
            "Result.n_orb is deprecated; use .wavefunction.n_orb",
            DeprecationWarning, stacklevel=2,
        )
        return self.wavefunction.n_orb

    @property
    def e_nuclear(self) -> float:
        warnings.warn(
            "Result.e_nuclear is deprecated; use .energies.nuclear",
            DeprecationWarning, stacklevel=2,
        )
        return self.energies.nuclear


def _apply_config_overrides(config: Config | None, overrides: dict) -> Config:
    """Create a Config, optionally applying keyword overrides."""
    if config is None:
        config = Config(**overrides)
    elif overrides:
        from dataclasses import asdict
        d = asdict(config)
        d.update(overrides)
        config = Config(**d)
    return config


def _find_orbopt_fcidump(details, work_dir=None):
    """Locate fcidump_orbopt from run_full details (or by filesystem scan).

    Three defensive layers (matches trimci.TrimCI_skill):
      1. details['fcidump_orbopt_path'] — authoritative, set by
         run_orbital_optimization since trimci 0.2.0.
      2. details['orbopt_dir'] / details['results_dir'] — derive
         `fcidump_orbopt` from the orbopt directory; covers resumed jobs.
      3. filesystem scan of work_dir/trimci_orbopt_results/* — last-resort
         safety net for older result_dirs or unusual calling patterns.
    """
    if details and isinstance(details, dict):
        # Layer 1: explicit path
        p = details.get("fcidump_orbopt_path")
        if p and os.path.exists(p):
            return p
        # Layer 2: orbopt_dir + 'fcidump_orbopt' (or latest fcidump_cycle_*)
        orbopt_dir = details.get("orbopt_dir") or details.get("results_dir")
        if orbopt_dir and os.path.isdir(orbopt_dir):
            p = os.path.join(orbopt_dir, "fcidump_orbopt")
            if os.path.exists(p):
                return p
            try:
                cycles = sorted([f for f in os.listdir(orbopt_dir)
                                 if f.startswith("fcidump_cycle_")])
                if cycles:
                    return os.path.join(orbopt_dir, cycles[-1])
            except OSError:
                pass

    # Layer 3: filesystem scan (legacy / sub-run-only details)
    if work_dir and os.path.isdir(work_dir):
        for d in sorted(os.listdir(work_dir), reverse=True):
            if not d.startswith("trimci_orbopt_results"):
                continue
            base = os.path.join(work_dir, d)
            for sub in sorted(os.listdir(base), reverse=True):
                sub_path = os.path.join(base, sub)
                if not os.path.isdir(sub_path):
                    continue
                p = os.path.join(sub_path, "fcidump_orbopt")
                if os.path.exists(p):
                    return p
                try:
                    cycles = sorted([f for f in os.listdir(sub_path)
                                     if f.startswith("fcidump_cycle_")])
                    if cycles:
                        return os.path.join(sub_path, cycles[-1])
                except OSError:
                    pass
    return None


def _extract_dets_arrays(dets):
    """Convert list of det objects/tuples to (alpha, beta) uint64 arrays.

    Returns shapes:
      - 64-bit dets:        alpha (N, 1), beta (N, 1)
      - 128/256-bit dets:   alpha (N, K), beta (N, K)  where K = ceil(n_orb/64)

    The downstream npz format used by run_expansion expects
    column_stack([alpha, beta]) → (N, 2K).
    """
    if hasattr(dets[0], "alpha"):
        a0 = dets[0].alpha
        if isinstance(a0, int) or not hasattr(a0, "__len__"):
            # Determinant64: scalar uint64
            alpha = np.array([d.alpha for d in dets], dtype=np.uint64).reshape(-1, 1)
            beta = np.array([d.beta for d in dets], dtype=np.uint64).reshape(-1, 1)
        else:
            # Determinant128+: array of uint64 segments per spin
            alpha = np.array([list(d.alpha) for d in dets], dtype=np.uint64)
            beta = np.array([list(d.beta) for d in dets], dtype=np.uint64)
    else:
        alpha = np.array([d[0] for d in dets], dtype=np.uint64).reshape(-1, 1)
        beta = np.array([d[1] for d in dets], dtype=np.uint64).reshape(-1, 1)
    return alpha, beta


def _compute_n_fci(alpha_bits: np.ndarray, beta_bits: np.ndarray,
                   n_orb: int) -> int | None:
    """Return full Hilbert dim C(n_orb, n_α) × C(n_orb, n_β).

    Infers n_α and n_β by popcount on the first stored determinant. Returns
    None if popcount fails (e.g. empty array). Used to silently clamp the
    user-requested n_dets when it exceeds the actual Hilbert space.
    """
    from math import comb
    if alpha_bits.size == 0 or beta_bits.size == 0:
        return None
    try:
        a_int = _bits_to_int(alpha_bits[0])
        b_int = _bits_to_int(beta_bits[0])
        n_alpha = bin(a_int).count("1")
        n_beta = bin(b_int).count("1")
        return int(comb(n_orb, n_alpha)) * int(comb(n_orb, n_beta))
    except Exception:
        return None


def _clamp_expansion_phases_to_fci(expansion_phases: list[dict],
                                   n_fci: int) -> None:
    """In-place silent clamp of each phase's ``max_n_dets`` to ``n_fci``.

    Phase 0 already self-clamps in trimci_driver. This handles the Phase 1+2
    expansion list constructed by _config_to_phase_dicts.
    """
    for ph in expansion_phases:
        cap = ph.get("max_n_dets")
        if cap is not None and cap > n_fci:
            ph["max_n_dets"] = n_fci


def _peek_fcidump_dims(path: str) -> tuple[int, int, int]:
    """Parse NORB, NELEC, MS2 from the FCIDUMP header line.

    Returns (n_orb, n_alpha, n_beta). Used to compute n_fci upfront, so
    the (Phase 0/1/2) n_dets split can be sized to the actual Hilbert
    dimension rather than the (possibly over-large) user request.
    """
    import re
    with open(path) as f:
        for line in f:
            up = line.upper()
            m_norb = re.search(r"NORB\s*=\s*(\d+)", up)
            m_nelec = re.search(r"NELEC\s*=\s*(\d+)", up)
            m_ms2 = re.search(r"MS2\s*=\s*(-?\d+)", up)
            if m_norb and m_nelec:
                n_orb = int(m_norb.group(1))
                n_elec = int(m_nelec.group(1))
                ms2 = int(m_ms2.group(1)) if m_ms2 else 0
                n_alpha = (n_elec + ms2) // 2
                n_beta = (n_elec - ms2) // 2
                return n_orb, n_alpha, n_beta
            # Stop after the first non-header line
            if line.strip() and not up.lstrip().startswith("&"):
                break
    raise ValueError(f"Could not parse NORB/NELEC from FCIDUMP header: {path}")


def _peek_mf_dims(mf, active: tuple[int, int] | None) -> tuple[int, int, int]:
    """Extract (n_orb, n_alpha, n_beta) from a PySCF mean-field object.

    With ``active=(n_e, n_o)``: use those values; spin from ``mf.mol.spin``.
    Otherwise: take ``n_orb`` from ``mo_coeff`` (handling both RHF 2D and
    UHF 3D / tuple layouts) and (n_alpha, n_beta) from ``mf.mol.nelec``.
    """
    if active is not None:
        n_orb = int(active[1])
        n_elec = int(active[0])
        spin = int(getattr(mf.mol, "spin", 0))
        return n_orb, (n_elec + spin) // 2, (n_elec - spin) // 2
    n_alpha, n_beta = mf.mol.nelec
    mo = mf.mo_coeff
    if isinstance(mo, (tuple, list)):
        n_orb = int(np.asarray(mo[0]).shape[1])
    else:
        mo_arr = np.asarray(mo)
        n_orb = int(mo_arr.shape[-1])  # works for both (nao, nmo) and (2, nao, nmo)
    return n_orb, int(n_alpha), int(n_beta)


def _read_e_nuc_from_fcidump(path: str) -> float:
    """Read nuclear repulsion energy from FCIDUMP (last line: value 0 0 0 0)."""
    e_nuc = 0.0
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 5 and parts[1] == "0" and parts[2] == "0":
                e_nuc = float(parts[0])
    return e_nuc


def _read_n_orb_from_fcidump(path: str) -> int:
    """Read NORB from FCIDUMP header."""
    import re
    with open(path) as f:
        for line in f:
            m = re.search(r"NORB\s*=\s*(\d+)", line.strip().upper())
            if m:
                return int(m.group(1))
    raise ValueError(f"Cannot find NORB in FCIDUMP header: {path}")


def _config_to_phase_dicts(cfg: Config) -> tuple[dict, list[dict]]:
    """Translate a Config dataclass into (phase0_dict, expansion_phases).

    ``phase0_dict`` is passed to ``run_full`` (Phase 0 stochastic + orbopt).
    ``expansion_phases`` is a list of dicts for ``run_expansion`` honoring
    ``cfg.enable_phase1`` / ``cfg.enable_phase2`` / ``cfg.pt2_only_last``.

    When ``pt2_only_last=True`` and PT2 is enabled, Phase 2 expansion runs
    without per-round PT2, and a final ``evaluate_only=True`` round computes
    PT2 just at the final det count. Per-round PT2 can cost 10-50% of round
    time, so this saves wall when Phase 2 has many growth rounds.
    """
    # Back-compat: if user only set the legacy n_parallel field, route it to
    # n_cpu_workers; otherwise prefer the explicit n_cpu_workers value.
    n_cpu = cfg.n_cpu_workers if cfg.n_cpu_workers != 1 else cfg.n_parallel

    # Derive sub-seeds from master cfg.seed (None → all randomized via system
    # entropy, matching legacy behavior). When seed is set, sub-seeds come
    # from a deterministic stream so the full run is reproducible.
    phase0_seed = None
    pt2_sketch_seed = None
    pt2_stochastic_seed = None
    if cfg.seed is not None:
        sub_rng = np.random.default_rng(int(cfg.seed))
        phase0_seed = int(sub_rng.integers(1, 2**63 - 1))
        pt2_sketch_seed = int(sub_rng.integers(1, 2**63 - 1))
        pt2_stochastic_seed = int(sub_rng.integers(1, 2**63 - 1))

    phase0 = {
        "initial_dets_dict": {"reference": 1, "random": [1, 10000]},
        "backend": cfg.backend_phase0,
        "orbital_optimization": cfg.orbopt,
        "loaded_dets_randomness": cfg.loaded_dets_randomness,
        "optimizer_options_dict": {
            "optimizer": cfg.optimizer,
            "cycles": cfg.orbopt_cycles,
            "maxiter": 100,
            "tracking_dets": cfg.tracking_dets,
            "davidson_tol": 1e-7,
            "ftol": 1e-8,
        },
        "threshold": cfg.threshold,
        "pool_core_ratio": 40,
        "num_groups": 20,
        "max_rounds": 4,
        "core_set_ratio": [1, 1.1],
        "max_final_dets": cfg.max_dets_phase0,
        "first_cycle_keep_size": 10,
        "pool_build_strategy": "heat_bath",
        "local_trim_keep_ratio": 4,
        "num_runs": cfg.num_runs,
        "n_cpu_workers": n_cpu,
        "n_gpu_workers": cfg.n_gpu_workers,
        "verbosity": cfg.verbosity,
        # If phase0_seed is None, run_full falls back to time.time() (legacy).
        # If set, drives Phase 0 stochastic CI + loaded_dets random walks.
        **({"random_seed": phase0_seed} if phase0_seed is not None else {}),
        **cfg.phase0_overrides,
    }
    phase1 = {
        "backend": cfg.backend_phase1,
        "max_n_dets": cfg.max_dets_phase1,
        "growth_factor": cfg.growth_factor_phase1,
        "orbital_optimization": cfg.orbopt_phase1,
        "orbital_opt_max_iter": cfg.orbopt_max_iter,
        "orbital_opt_grad_tol": cfg.orbopt_grad_tol,
        "use_connection_cache": cfg.use_connection_cache_phase1,
        "davidson": {"energy_tol": cfg.davidson_energy_tol_phase1},
        **cfg.phase1_overrides,
    }

    expansion_phases: list[dict] = []
    if cfg.enable_phase1:
        expansion_phases.append(phase1)

    if cfg.enable_phase2:
        # Base Phase 2 dict.  PT2 seeds threaded via "pt2" override when
        # cfg.seed is set; otherwise PT2_DEFAULTS' sketch_seed=-1 keeps
        # legacy "random per run" behavior.
        pt2_seed_override = {}
        if pt2_sketch_seed is not None:
            pt2_seed_override["sketch_seed"] = pt2_sketch_seed
            pt2_seed_override["stochastic_seed"] = pt2_stochastic_seed

        phase2_base = {
            "backend": cfg.backend_phase2,
            "max_n_dets": cfg.max_dets_phase2,
            "growth_factor": cfg.growth_factor_phase2,
            "davidson": {"energy_tol": cfg.davidson_energy_tol_phase2},
            **({"pt2": pt2_seed_override} if pt2_seed_override else {}),
        }
        if cfg.pt2_only_last and cfg.pt2_correction:
            # Phase 2a: expand to target without PT2
            phase2a = {
                **phase2_base,
                "pt2_correction": False,
                **cfg.phase2_overrides,
            }
            # Phase 2b: 1 evaluate_only round on existing dets, with PT2
            phase2b = {
                "backend": cfg.backend_phase2,
                "max_n_dets": cfg.max_dets_phase2,
                "evaluate_only": True,
                "pt2_correction": True,
                "davidson": {"energy_tol": cfg.davidson_energy_tol_phase2},
                **({"pt2": pt2_seed_override} if pt2_seed_override else {}),
                **cfg.phase2_overrides,
                # phase2_overrides may force pt2_correction=False;
                # explicitly re-assert here so the final eval-only round
                # always runs PT2 regardless of user overrides.
                "pt2_correction": True,
            }
            expansion_phases.append(phase2a)
            expansion_phases.append(phase2b)
        else:
            phase2 = {
                **phase2_base,
                "pt2_correction": cfg.pt2_correction,
                **cfg.phase2_overrides,
            }
            expansion_phases.append(phase2)

    return phase0, expansion_phases


# ---------------------------------------------------------------------------
#  Entry point: FCIDUMP
# ---------------------------------------------------------------------------

def ground_state_from_fcidump(
    path: str,
    config: Config | None = None,
    **config_overrides,
) -> Result:
    """Run TrimCI on an existing FCIDUMP file.

    Args:
        path: Path to FCIDUMP file.
        config: TrimCI configuration. If None, uses defaults.
        **config_overrides: Shorthand for Config field overrides.

    Returns:
        Result object with energy, wavefunction, and lazy post-processing.
    """
    import shutil
    import time

    if not os.path.isfile(path):
        raise FileNotFoundError(f"FCIDUMP not found: {path}")

    cfg = _apply_config_overrides(config, config_overrides)
    p0_dict, expansion_phases = _config_to_phase_dicts(cfg)
    n_orb = _read_n_orb_from_fcidump(path)

    work_dir = cfg.work_dir or tempfile.mkdtemp(prefix="trimci_api_")
    os.makedirs(work_dir, exist_ok=True)

    e_nuc = _read_e_nuc_from_fcidump(path)
    energy_history: list[float] = []     # total energies across all rounds
    pt2_history_all: list[float] = []

    # ── Phase 0 ───────────────────────────────────────────────────────
    from trimci import run_full

    t_p0 = time.time()
    energy, dets, coeffs, details, _ = run_full(
        fcidump_path=path, config_dict=p0_dict,
    )
    wall_phase0 = time.time() - t_p0

    if not dets:
        raise RuntimeError("Phase 0 returned no determinants")

    energy_history.append(float(energy))
    n_dets_phase0 = len(dets)
    alpha, beta = _extract_dets_arrays(dets)
    coeffs_arr = np.array(coeffs, dtype=np.float64)

    # Compute full Hilbert dim from the first det's α/β popcount, then
    # silently clamp any expansion cap above n_fci. (Phase 0 already
    # self-clamps in trimci_driver.run_full_calculation.)
    n_fci = _compute_n_fci(alpha, beta, n_orb)
    if n_fci is not None:
        _clamp_expansion_phases_to_fci(expansion_phases, n_fci)

    dets_path = os.path.join(work_dir, "dets_phase0.npz")
    np.savez_compressed(
        dets_path,
        dets=np.column_stack([alpha, beta]),
        dets_coeffs=coeffs_arr,
    )

    orbopt_fcidump = _find_orbopt_fcidump(details, work_dir=work_dir)
    if orbopt_fcidump:
        fcidump_for_expansion = os.path.join(work_dir, "fcidump_phase0")
        shutil.copy2(orbopt_fcidump, fcidump_for_expansion)
    else:
        fcidump_for_expansion = path

    # ── Phase 1 + Phase 2 (combined call) ─────────────────────────────
    final_alpha, final_beta, final_coeffs = alpha, beta, coeffs_arr
    n_dets_final = n_dets_phase0
    wall_expansion = 0.0
    expansion_rounds = 0

    if expansion_phases:
        from trimci.TrimCI_runner.run_expansion import run_expansion

        t_exp = time.time()
        exp_result = run_expansion(
            fcidump_path=fcidump_for_expansion,
            dets_path=dets_path,
            phases=expansion_phases,
            checkpoint_dir=os.path.join(work_dir, "checkpoints"),
            output_prefix=os.path.join(work_dir, "result"),
        )
        wall_expansion = time.time() - t_exp

        exp_e_nuc = float(exp_result.get("e_nuc", 0.0))
        energy = float(exp_result["energy_var"]) + exp_e_nuc
        final_alpha = np.asarray(exp_result["alphas"])
        final_beta = np.asarray(exp_result["betas"])
        final_coeffs = np.asarray(exp_result["coeffs"])
        for e_elec in exp_result.get("energy_history", []):
            energy_history.append(float(e_elec) + exp_e_nuc)
        pt2_history_all.extend(float(x) for x in exp_result.get("pt2_history", []))
        n_dets_final = int(exp_result.get("n_final", n_dets_phase0))
        expansion_rounds = len(exp_result.get("ndets_history", []))

    # Last non-trivial PT2 correction (skip leading zeros from rounds without PT2)
    pt2_correction: float | None = None
    for v in reversed(pt2_history_all):
        if abs(v) > 0:
            pt2_correction = v
            break

    # Ensure (N, K) shape for alpha/beta
    if final_alpha.ndim == 1:
        final_alpha = final_alpha.reshape(-1, 1)
    if final_beta.ndim == 1:
        final_beta = final_beta.reshape(-1, 1)

    return Result(
        energies=Energies(
            var=float(energy),
            pt2_correction=pt2_correction,
            nuclear=e_nuc,
            history=energy_history,
            pt2_history=pt2_history_all,
        ),
        wavefunction=Wavefunction(
            alpha_bits=final_alpha.astype(np.uint64, copy=False),
            beta_bits=final_beta.astype(np.uint64, copy=False),
            coefficients=final_coeffs,
            n_orb=n_orb,
        ),
        diagnostics=Diagnostics(
            n_dets_phase0=n_dets_phase0,
            n_dets_final=n_dets_final,
            wall_phase0=wall_phase0,
            wall_expansion=wall_expansion,
            expansion_rounds=expansion_rounds,
            config=cfg,
            n_fci=n_fci,
            orbopt_dir=(details.get("orbopt_dir") if isinstance(details, dict) else None),
            checkpoint_dir=os.path.join(work_dir, "checkpoints"),
        ),
        fcidump_path=fcidump_for_expansion if orbopt_fcidump else path,
    )


# ---------------------------------------------------------------------------
#  Entry point: Raw integrals (h1, eri, n_elec, ms2)
# ---------------------------------------------------------------------------

def ground_state_from_integrals(
    h1,
    eri,
    n_elec: int,
    *,
    ms2: int = 0,
    e_nuc: float = 0.0,
    config: Config | None = None,
    **config_overrides,
) -> Result:
    """Run TrimCI directly on raw integral arrays — no FCIDUMP roundtrip.

    Args:
        h1: one-body Hamiltonian, (n_orb, n_orb) numpy array
        eri: two-body integrals, any shape that flattens to (n_orb^4,)
        n_elec: total electron count
        ms2: 2·S_z = n_alpha - n_beta. Default 0 (closed-shell singlet).
        e_nuc: nuclear repulsion energy
        config: optional Config; defaults applied if None.

    Mirrors ``ground_state_from_fcidump`` but uses ``run_full`` and
    ``run_expansion``'s direct-integrals paths, avoiding the FCIDUMP file
    write/read cycle. Especially relevant for non-FCIDUMP-friendly inputs
    (lattice models, non-PySCF integrals, open-shell ms2 ≠ 0).
    """
    import time
    import shutil

    cfg = _apply_config_overrides(config, config_overrides)
    p0_dict, expansion_phases = _config_to_phase_dicts(cfg)

    h1_arr = np.asarray(h1, dtype=np.float64)
    eri_arr = np.asarray(eri, dtype=np.float64)
    n_orb = int(h1_arr.shape[0])
    # Normalize ERI to 4-index (n_orb, n_orb, n_orb, n_orb) form.
    # PySCF's ao2mo.full returns S4/S8-packed (typically 2D); convert
    # back to full 4D so downstream solvers see a consistent layout.
    if eri_arr.ndim != 4:
        try:
            from pyscf import ao2mo
            eri_arr = np.asarray(
                ao2mo.restore(1, eri_arr, n_orb), dtype=np.float64
            )
        except ImportError:
            if eri_arr.size != n_orb ** 4:
                raise ValueError(
                    f"eri shape {eri_arr.shape} not 4D and PySCF unavailable "
                    f"to unpack — pass eri as ({n_orb},)*4 array."
                )
            eri_arr = eri_arr.reshape(n_orb, n_orb, n_orb, n_orb)
    e_nuc_f = float(e_nuc)
    n_elec_i = int(n_elec)
    ms2_i = int(ms2)
    n_alpha = (n_elec_i + ms2_i) // 2
    n_beta = (n_elec_i - ms2_i) // 2
    if n_alpha + n_beta != n_elec_i:
        raise ValueError(
            f"n_elec={n_elec_i} + ms2={ms2_i} → n_alpha={n_alpha}, "
            f"n_beta={n_beta} (sum {n_alpha + n_beta} ≠ {n_elec_i})"
        )

    work_dir = cfg.work_dir or tempfile.mkdtemp(prefix="trimci_int_")
    os.makedirs(work_dir, exist_ok=True)

    energy_history: list[float] = []
    pt2_history_all: list[float] = []

    # ── Phase 0 — direct integrals ────────────────────────────────────
    from trimci import run_full

    t_p0 = time.time()
    energy, dets, coeffs, details, _ = run_full(
        h1=h1_arr, eri=eri_arr, n_elec=n_elec_i, ms2=ms2_i, e_nuc=e_nuc_f,
        config_dict=p0_dict,
    )
    wall_phase0 = time.time() - t_p0

    if not dets:
        raise RuntimeError("Phase 0 returned no determinants")

    energy_history.append(float(energy))
    n_dets_phase0 = len(dets)
    alpha, beta = _extract_dets_arrays(dets)
    coeffs_arr = np.array(coeffs, dtype=np.float64)

    # Compute full Hilbert dim, silently clamp expansion caps to n_fci.
    n_fci = _compute_n_fci(alpha, beta, n_orb)
    if n_fci is not None:
        _clamp_expansion_phases_to_fci(expansion_phases, n_fci)

    # If orbopt ran, load rotated h1/eri from its npz output;
    # otherwise expansion proceeds on the original integrals.
    h1_for_exp, eri_for_exp = h1_arr, eri_arr
    orbopt_dir = details.get("orbopt_dir") if isinstance(details, dict) else None
    if orbopt_dir and os.path.isdir(orbopt_dir):
        npz_candidates = []
        primary = os.path.join(orbopt_dir, "orbital_optimization_result_orbopt.npz")
        if os.path.exists(primary):
            npz_candidates.append(primary)
        else:
            cycles = sorted([f for f in os.listdir(orbopt_dir)
                             if f.startswith("orbital_optimization_result_cycle_")])
            if cycles:
                npz_candidates.append(os.path.join(orbopt_dir, cycles[-1]))
        if npz_candidates:
            try:
                data = np.load(npz_candidates[0])
                h1_for_exp = np.asarray(data["h1"], dtype=np.float64)
                eri_for_exp = np.asarray(data["eri"], dtype=np.float64)
            except (OSError, KeyError):
                pass  # fall back to originals

    # ── Phase 1 + Phase 2 — direct integrals ──────────────────────────
    final_alpha, final_beta, final_coeffs = alpha, beta, coeffs_arr
    n_dets_final = n_dets_phase0
    wall_expansion = 0.0
    expansion_rounds = 0

    if expansion_phases:
        from trimci.TrimCI_runner.run_expansion import run_expansion

        # run_expansion expects (N,) for 64-bit or (N, K) for 128-bit.
        # _extract_dets_arrays always returns (N, K). For 64-bit (K=1),
        # squeeze to 1D so the wrapper picks the correct codepath.
        alpha_init = alpha[:, 0] if alpha.shape[1] == 1 else alpha
        beta_init = beta[:, 0] if beta.shape[1] == 1 else beta

        t_exp = time.time()
        exp_result = run_expansion(
            h1=h1_for_exp, eri=eri_for_exp, e_nuc=e_nuc_f,
            alpha_init=alpha_init, beta_init=beta_init,
            coeffs_init=coeffs_arr,
            phases=expansion_phases,
            checkpoint_dir=os.path.join(work_dir, "checkpoints"),
            output_prefix=os.path.join(work_dir, "result"),
        )
        wall_expansion = time.time() - t_exp

        exp_e_nuc = float(exp_result.get("e_nuc", e_nuc_f))
        energy = float(exp_result["energy_var"]) + exp_e_nuc
        final_alpha = np.asarray(exp_result["alphas"])
        final_beta = np.asarray(exp_result["betas"])
        final_coeffs = np.asarray(exp_result["coeffs"])
        for e_elec in exp_result.get("energy_history", []):
            energy_history.append(float(e_elec) + exp_e_nuc)
        pt2_history_all.extend(float(x) for x in exp_result.get("pt2_history", []))
        n_dets_final = int(exp_result.get("n_final", n_dets_phase0))
        expansion_rounds = len(exp_result.get("ndets_history", []))

    pt2_correction: float | None = None
    for v in reversed(pt2_history_all):
        if abs(v) > 0:
            pt2_correction = v
            break

    if final_alpha.ndim == 1:
        final_alpha = final_alpha.reshape(-1, 1)
    if final_beta.ndim == 1:
        final_beta = final_beta.reshape(-1, 1)

    return Result(
        energies=Energies(
            var=float(energy),
            pt2_correction=pt2_correction,
            nuclear=e_nuc_f,
            history=energy_history,
            pt2_history=pt2_history_all,
        ),
        wavefunction=Wavefunction(
            alpha_bits=final_alpha.astype(np.uint64, copy=False),
            beta_bits=final_beta.astype(np.uint64, copy=False),
            coefficients=final_coeffs,
            n_orb=n_orb,
        ),
        diagnostics=Diagnostics(
            n_dets_phase0=n_dets_phase0,
            n_dets_final=n_dets_final,
            wall_phase0=wall_phase0,
            wall_expansion=wall_expansion,
            expansion_rounds=expansion_rounds,
            config=cfg,
            n_fci=n_fci,
            orbopt_dir=(details.get("orbopt_dir") if isinstance(details, dict) else None),
            checkpoint_dir=os.path.join(work_dir, "checkpoints"),
        ),
        fcidump_path=None,
    )


# ---------------------------------------------------------------------------
#  Entry point: PySCF
# ---------------------------------------------------------------------------

def _pyscf_mf_to_fcidump(mf, active, work_dir: str) -> str:
    """Generate FCIDUMP from a PySCF mean-field object."""
    try:
        from pyscf import ao2mo
        from pyscf.tools import fcidump as pyscf_fcidump
    except ImportError:
        raise ImportError(
            "PySCF is required for molecule/mean-field input. "
            "Install with: pip install pyscf"
        )

    mol = mf.mol
    mo_coeff = mf.mo_coeff

    if active is not None:
        n_elec, n_orb = active
        # Select active orbitals (last n_orb occupied + virtual)
        n_core = (mol.nelectron - n_elec) // 2
        mo_active = mo_coeff[:, n_core:n_core + n_orb]
    else:
        n_elec = mol.nelectron
        n_orb = mo_coeff.shape[1]
        mo_active = mo_coeff
        if n_orb > 20:
            print(f"  WARNING: Full orbital space has {n_orb} orbitals. "
                  f"Consider setting config.active=(n_elec, n_orb).")

    # Transform integrals to MO basis
    h1_mo = mo_active.T @ mf.get_hcore() @ mo_active
    eri_mo = ao2mo.full(mol, mo_active)

    fcidump_path = os.path.join(work_dir, "FCIDUMP")
    pyscf_fcidump.from_integrals(
        fcidump_path, h1_mo, eri_mo, n_orb, n_elec,
        nuc=mol.energy_nuc(), ms=mol.spin,
    )
    print(f"  Active space: ({n_elec}e, {n_orb}o)")
    return fcidump_path


def ground_state_from_pyscf(
    mf,
    active: tuple[int, int] | None = None,
    config: Config | None = None,
    **config_overrides,
) -> Result:
    """Run TrimCI from a converged PySCF mean-field object.

    Args:
        mf: Converged PySCF mean-field object (RHF, ROHF, UHF).
        active: (n_electrons, n_orbitals) for active space. None = full space.
        config: TrimCI configuration.
        **config_overrides: Shorthand for Config field overrides.
    """
    cfg = _apply_config_overrides(config, config_overrides)
    active = active or cfg.active

    work_dir = cfg.work_dir or tempfile.mkdtemp(prefix="trimci_pyscf_")
    os.makedirs(work_dir, exist_ok=True)
    # Update work_dir in config so ground_state_from_fcidump uses same dir
    from dataclasses import asdict
    cfg = Config(**{**asdict(cfg), "work_dir": work_dir})

    fcidump_path = _pyscf_mf_to_fcidump(mf, active, work_dir)
    return ground_state_from_fcidump(fcidump_path, config=cfg)


# ---------------------------------------------------------------------------
#  Main entry point: ground_state() — 3-parameter dummy-friendly API
# ---------------------------------------------------------------------------

def _split_n_dets(n_dets: int) -> tuple[int, int, int]:
    """Log-scale split of a total n_dets target into per-phase caps.

    Anchor: n_dets=1e8 → (1000, 1e6, 1e8). Power-law p0 ∝ n^0.375,
    p1 ∝ n^0.75, with floors so very small targets still leave Phase 0
    enough room for stable orbopt.

    Examples (rounded):
        n=100        → (10, 32, 100)
        n=10_000     → (32, 1_000, 10_000)
        n=1_000_000  → (316, 31_623, 1_000_000)
        n=100_000_000 → (1000, 1_000_000, 100_000_000)
    """
    n = max(1, int(n_dets))
    if n < 10:
        return n, n, n
    p2 = n
    p0 = max(10, min(int(round(n ** 0.375)), 10_000))
    p1 = max(p0 * 3, min(int(round(n ** 0.75)), p2))
    p1 = min(p1, p2)
    return p0, p1, p2


def _config_from_effort(effort: float, p0: int, p1: int, p2: int,
                        work_dir: str) -> Config:
    """Build a Config from a (n_dets split, effort multiplier).

    Scales **num_runs** and **BFGS maxiter** with effort. **orbopt_cycles is
    fixed at 5** (not effort-scaled): the heatmap diagnostic showed that
    cycle ≥ 2 contributes near-zero further improvement under default
    settings — the dominant variance is in Cycle 1's stochastic exploration,
    and stretching cycles just wastes wall.

    Also enables ``tracking_dets=True`` + ``loaded_dets_randomness=0.1`` by
    default. Tracking carries dets between cycles (turning cycles into
    iterative refinement instead of fresh random restarts); the small
    randomness perturbs each carried det by a single-excitation random walk,
    keeping the basis exploring rather than locking on Cycle 1's basin.

    For drop-in compatibility with ``trimci.TrimCI_skill`` defaults, pass a
    ``Config`` directly to ``ground_state_from_fcidump``; this helper only
    sets defaults for the dummy-friendly ``ground_state`` entry point.
    """
    e = max(0.05, float(effort))
    return Config(
        max_dets_phase0=p0,
        max_dets_phase1=p1,
        max_dets_phase2=p2,
        num_runs=max(1, int(round(64 * e))),
        orbopt_cycles=5,
        # Phase 1 BFGS step cap. Fixed at 100 (not effort-scaled): BFGS
        # exits when converged, so the cap rarely binds. A safe upper bound
        # avoids accidental truncation of stiff cycles; cost is near-zero
        # for already-converged cycles.
        orbopt_max_iter=100,
        tracking_dets=True,
        loaded_dets_randomness=0.1,
        pt2_only_last=True,
        work_dir=work_dir,
    )


def ground_state(
    problem,
    n_dets: int = 100_000,
    effort: float = 1.0,
    *,
    ms2: int = 0,
    work_dir: str | None = None,
    config: Config | None = None,
    active: tuple[int, int] | None = None,
) -> Result:
    """Run a TrimCI ground-state calculation. Unified entry point.

    Accepts three problem types and an optional pre-built ``Config``:

    Args:
        problem:
            - FCIDUMP file path (``str``) — MS2 read from the file header.
            - ``(h1, eri, n_elec)`` / ``(h1, eri, n_elec, e_nuc)`` tuple of
              raw integrals — pass spin via ``ms2``.
            - PySCF mean-field object (RHF/ROHF/UHF, duck-typed via
              ``mo_coeff`` + ``mol``) — spin taken from ``mf.mol.spin``.
              Pass ``active=(n_e, n_o)`` for an active space.
        n_dets: target final determinant count. Internally split into
            Phase 0/1/2 caps on a log scale (see ``_split_n_dets``).
            **Ignored when ``config`` is provided.**
        effort: workload multiplier; effort=1.0 reproduces production
            defaults of ``trimci.TrimCI_skill``. Scales num_runs and BFGS
            iterations. **Ignored when ``config`` is provided.**
        ms2: 2·S_z = n_alpha - n_beta. Only used for the tuple input.
            FCIDUMP and PySCF mf paths take spin from their own sources;
            passing non-zero ``ms2`` there warns.
        work_dir: working directory; auto-created if None. When ``config``
            is provided, this overrides ``config.work_dir``.
        config: pre-built ``Config``. Use this for fine-grained tuning;
            ``n_dets`` / ``effort`` are ignored when set.
        active: ``(n_electrons, n_orbitals)`` for PySCF mf active space.
            Only valid when ``problem`` is an mf object.

    n_fci-aware splitting (when ``config`` is None): the requested
    ``n_dets`` is clamped to the actual Hilbert dimension
    ``C(n_orb, n_α) × C(n_orb, n_β)`` before splitting, so each phase is
    right-sized when n_dets exceeds the physical space (e.g.
    ``n_dets=10⁸`` on H4 → 36).
    """
    from math import comb

    # ── Detect input type ─────────────────────────────────────────────
    is_str = isinstance(problem, str)
    is_tuple = isinstance(problem, (tuple, list)) and len(problem) in (3, 4)
    is_mf = (
        not is_str and not is_tuple
        and hasattr(problem, "mo_coeff") and hasattr(problem, "mol")
    )
    if not (is_str or is_tuple or is_mf):
        raise ValueError(
            "problem must be a FCIDUMP path (str), an "
            "(h1, eri, n_elec [, e_nuc]) tuple, or a PySCF mean-field "
            f"object — got {type(problem).__name__}"
        )
    if active is not None and not is_mf:
        raise ValueError("active= is only valid for a PySCF mean-field problem")

    # ── Warn on ignored ms2 ───────────────────────────────────────────
    if is_str and ms2 != 0:
        warnings.warn(
            f"ms2={ms2} ignored: FCIDUMP file's MS2 field is authoritative. "
            "For non-zero ms2 with raw integrals, use the (h1, eri, n_elec) "
            "tuple form instead.",
            UserWarning, stacklevel=2,
        )
    elif is_mf and ms2 != 0:
        warnings.warn(
            f"ms2={ms2} ignored: PySCF mf path takes spin from mf.mol.spin. "
            "Modify mf.mol.spin before passing to change it.",
            UserWarning, stacklevel=2,
        )

    if is_str and not os.path.isfile(problem):
        raise FileNotFoundError(f"FCIDUMP not found: {problem}")

    # ── Build Config (n_fci-aware) unless user provided one ──────────
    if config is None:
        if is_str:
            n_orb_p, n_alpha_p, n_beta_p = _peek_fcidump_dims(problem)
        elif is_tuple:
            h1 = problem[0]
            n_orb_p = int(np.asarray(h1).shape[0])
            n_elec = int(problem[2])
            n_alpha_p = (n_elec + int(ms2)) // 2
            n_beta_p = (n_elec - int(ms2)) // 2
            if n_alpha_p + n_beta_p != n_elec:
                raise ValueError(f"n_elec={n_elec} and ms2={ms2} inconsistent")
        else:  # is_mf
            n_orb_p, n_alpha_p, n_beta_p = _peek_mf_dims(problem, active)

        n_fci = int(comb(n_orb_p, n_alpha_p)) * int(comb(n_orb_p, n_beta_p))
        effective_n_dets = min(int(n_dets), max(1, n_fci))

        wd = work_dir or tempfile.mkdtemp(prefix="trimci_gs_")
        os.makedirs(wd, exist_ok=True)
        p0, p1, p2 = _split_n_dets(effective_n_dets)
        cfg = _config_from_effort(effort, p0, p1, p2, wd)
    else:
        cfg = config
        if work_dir is not None:
            from dataclasses import asdict
            cfg = Config(**{**asdict(cfg), "work_dir": work_dir})

    # ── Dispatch ──────────────────────────────────────────────────────
    if is_str:
        return ground_state_from_fcidump(problem, config=cfg)
    elif is_tuple:
        h1 = problem[0]
        eri = problem[1]
        n_elec = int(problem[2])
        e_nuc = float(problem[3]) if len(problem) == 4 else 0.0
        return ground_state_from_integrals(
            h1, eri, n_elec, ms2=ms2, e_nuc=e_nuc, config=cfg,
        )
    else:  # is_mf
        return ground_state_from_pyscf(problem, active=active, config=cfg)
