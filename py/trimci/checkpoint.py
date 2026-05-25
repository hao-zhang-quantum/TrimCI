"""Checkpoint utilities for TrimCI expansion.

Determinant checkpoint (C++ save_checkpoint):
  "TCPT" (4B) | version uint32 (4B) | n_dets uint64 (8B) |
  round int32 (4B) | pad (4B) | energy float64 (8B) |
  alpha[n_dets] uint64 | beta[n_dets] uint64 | coeffs[n_dets] float64

Integrals checkpoint (C++ save_integrals_checkpoint):
  "TCPI" (4B) | version uint32 (4B) | n_orb int32 (4B) | round int32 (4B) |
  h1[n_orb²] float64 | eri[n_orb⁴] float64 | U_total[n_orb²] float64
"""
import struct
import numpy as np
from pathlib import Path


MAGIC_DETS = b"TCPT"
MAGIC_INTEGRALS = b"TCPI"
HEADER_SIZE = 4 + 4 + 8 + 4 + 4 + 8  # 32 bytes (dets checkpoint)


def load_checkpoint(path):
    """Load a checkpoint file written by C++ save_checkpoint.

    Auto-detects 64-bit vs 128-bit determinant format from payload size:
      bytes/det = 24 (64-bit: 8+8+8) or 40 (128-bit: 16+16+8).
    128-bit alpha/beta are returned as shape (n_dets, 2) uint64 so that
    run_expansion's use_128 detection (alpha.ndim==2 and shape[1]==2)
    triggers automatically downstream.

    Returns:
        dict with keys: alpha, beta, coeffs, energy, round, n_dets, use_128
    """
    path = Path(path)
    file_size = path.stat().st_size
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != MAGIC_DETS:
            raise ValueError(f"Bad magic: {magic!r}, expected {MAGIC_DETS!r}")
        version = struct.unpack("<I", f.read(4))[0]
        if version != 1:
            raise ValueError(f"Unsupported version: {version}")
        n_dets = struct.unpack("<Q", f.read(8))[0]
        round_num = struct.unpack("<i", f.read(4))[0]
        f.read(4)  # padding
        energy = struct.unpack("<d", f.read(8))[0]

        payload = file_size - HEADER_SIZE
        if n_dets == 0:
            bytes_per_det = 24
        elif payload % n_dets != 0:
            raise ValueError(
                f"Corrupt checkpoint: payload {payload} not divisible by n_dets {n_dets}"
            )
        else:
            bytes_per_det = payload // n_dets

        if bytes_per_det == 24:
            use_128 = False
            alpha = np.frombuffer(f.read(n_dets * 8), dtype=np.uint64).copy()
            beta = np.frombuffer(f.read(n_dets * 8), dtype=np.uint64).copy()
        elif bytes_per_det == 40:
            use_128 = True
            alpha = np.frombuffer(f.read(n_dets * 16), dtype=np.uint64).copy().reshape(n_dets, 2)
            beta = np.frombuffer(f.read(n_dets * 16), dtype=np.uint64).copy().reshape(n_dets, 2)
        else:
            raise ValueError(
                f"Unknown checkpoint det width: {bytes_per_det} bytes/det "
                f"(expected 24 for 64-bit or 40 for 128-bit)"
            )
        coeffs = np.frombuffer(f.read(n_dets * 8), dtype=np.float64).copy()

    return {
        "alpha": alpha,
        "beta": beta,
        "coeffs": coeffs,
        "energy": energy,
        "round": round_num,
        "n_dets": n_dets,
        "use_128": use_128,
    }


def save_checkpoint(path, alpha, beta, coeffs, energy, round_num=0):
    """Save a determinant checkpoint in TCPT v1 format.

    Compatible with C++ save_checkpoint / load_checkpoint.

    Args:
        path: output file path
        alpha: uint64 array of alpha bitstrings [n_dets]
        beta: uint64 array of beta bitstrings [n_dets]
        coeffs: float64 array of CI coefficients [n_dets]
        energy: Davidson eigenvalue (float64)
        round_num: expansion round number (int32)
    """
    path = Path(path)
    n_dets = len(alpha)
    with open(path, "wb") as f:
        f.write(MAGIC_DETS)
        f.write(struct.pack("<I", 1))       # version
        f.write(struct.pack("<Q", n_dets))
        f.write(struct.pack("<i", round_num))
        f.write(b"\x00" * 4)               # padding
        f.write(struct.pack("<d", energy))
        f.write(np.asarray(alpha, dtype="<u8").tobytes())
        f.write(np.asarray(beta, dtype="<u8").tobytes())
        f.write(np.asarray(coeffs, dtype="<f8").tobytes())


def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint file in a directory.

    Returns:
        Path to the latest checkpoint, or None if no checkpoints found.
    """
    d = Path(checkpoint_dir)
    files = sorted(d.glob("checkpoint_round_*.bin"),
                   key=lambda p: int(p.stem.split("_")[-1]))
    return files[-1] if files else None


def list_checkpoints(checkpoint_dir):
    """List all checkpoint files with their metadata (without loading full arrays).

    Returns:
        list of dicts with keys: path, round, n_dets, energy, size_mb
    """
    d = Path(checkpoint_dir)
    results = []
    for p in sorted(d.glob("checkpoint_round_*.bin")):
        with open(p, "rb") as f:
            magic = f.read(4)
            if magic != MAGIC_DETS:
                continue
            f.read(4)  # version
            n_dets = struct.unpack("<Q", f.read(8))[0]
            round_num = struct.unpack("<i", f.read(4))[0]
            f.read(4)  # padding
            energy = struct.unpack("<d", f.read(8))[0]
        results.append({
            "path": p,
            "round": round_num,
            "n_dets": n_dets,
            "energy": energy,
            "size_mb": p.stat().st_size / (1024 * 1024),
        })
    return results


def load_integrals_checkpoint(path):
    """Load integrals checkpoint written by C++ save_integrals_checkpoint.

    Returns:
        dict with keys: h1 (n_orb, n_orb), eri (n_orb, n_orb, n_orb, n_orb),
        U_total (n_orb, n_orb), n_orb, round
    """
    path = Path(path)
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != MAGIC_INTEGRALS:
            raise ValueError(f"Bad magic: {magic!r}, expected {MAGIC_INTEGRALS!r}")
        version = struct.unpack("<I", f.read(4))[0]
        if version != 1:
            raise ValueError(f"Unsupported version: {version}")
        n_orb = struct.unpack("<i", f.read(4))[0]
        round_num = struct.unpack("<i", f.read(4))[0]

        h1 = np.frombuffer(f.read(n_orb * n_orb * 8), dtype=np.float64).copy()
        h1 = h1.reshape(n_orb, n_orb)

        eri = np.frombuffer(f.read(n_orb**4 * 8), dtype=np.float64).copy()
        eri = eri.reshape(n_orb, n_orb, n_orb, n_orb)

        U_total = np.frombuffer(f.read(n_orb * n_orb * 8), dtype=np.float64).copy()
        U_total = U_total.reshape(n_orb, n_orb)

    return {
        "h1": h1,
        "eri": eri,
        "U_total": U_total,
        "n_orb": n_orb,
        "round": round_num,
    }
