"""
Determinant utilities: serialization, deserialization, and initial state generation.
"""
import os

import numpy as np

from trimci.trimci_logging import log_verbose, log_important


def dets_to_array(dets):
    """Convert determinants to numpy array as uint64 pairs for C++ compatibility"""
    if not dets:
        return np.array([], dtype=np.uint64).reshape(0, 6)

    max_uint64_pairs = 1

    for det in dets:
        alpha_bits = det.alpha
        if isinstance(alpha_bits, list):
            max_uint64_pairs = max(max_uint64_pairs, len(alpha_bits))

    result = []
    for det in dets:
        alpha_bits = det.alpha
        beta_bits = det.beta

        if isinstance(alpha_bits, list):
            alpha_vals = [int(val) & 0xFFFFFFFFFFFFFFFF for val in alpha_bits]
            beta_vals = [int(val) & 0xFFFFFFFFFFFFFFFF for val in beta_bits]

            while len(alpha_vals) < max_uint64_pairs:
                alpha_vals.append(0)
            while len(beta_vals) < max_uint64_pairs:
                beta_vals.append(0)

            row = []
            for i in range(max_uint64_pairs):
                row.extend([alpha_vals[i], beta_vals[i]])
            result.append(row)
        else:
            alpha_val = int(alpha_bits) & 0xFFFFFFFFFFFFFFFF
            beta_val = int(beta_bits) & 0xFFFFFFFFFFFFFFFF

            row = [alpha_val, beta_val]
            for i in range(1, max_uint64_pairs):
                row.extend([0, 0])
            result.append(row)

    return np.array(result, dtype=np.uint64)


def det_to_bitstring(det):
    """Convert a determinant to bitstring format like '0 1 2 5 , 0 1 2 3'"""
    alpha_orbs = []
    beta_orbs = []

    alpha_bits = det.alpha
    beta_bits = det.beta

    if isinstance(alpha_bits, list):
        for array_idx, (alpha_val, beta_val) in enumerate(zip(alpha_bits, beta_bits)):
            for bit_idx in range(64):
                orbital_idx = array_idx * 64 + bit_idx
                if alpha_val & (1 << bit_idx):
                    alpha_orbs.append(orbital_idx)
                if beta_val & (1 << bit_idx):
                    beta_orbs.append(orbital_idx)
    else:
        for i in range(64):
            if alpha_bits & (1 << i):
                alpha_orbs.append(i)
            if beta_bits & (1 << i):
                beta_orbs.append(i)

    alpha_str = " ".join(map(str, sorted(alpha_orbs)))
    beta_str = " ".join(map(str, sorted(beta_orbs)))

    return f"{alpha_str} , {beta_str}"


def get_top_determinants(dets, coeffs, top_n=10):
    """Get top N determinants by coefficient magnitude"""
    if len(dets) == 0 or len(coeffs) == 0:
        return []

    sorted_indices = np.argsort(np.abs(coeffs))[::-1]
    top_indices = sorted_indices[:min(top_n, len(sorted_indices))]

    top_dets = []
    for idx in top_indices:
        coeff = float(coeffs[idx])
        bitstring = det_to_bitstring(dets[idx])
        top_dets.append([coeff, bitstring])

    return top_dets


def load_initial_dets_from_file(
    dets_path: str = "dets.npz",
    core_set: bool = False,
    expected_n_alpha: int = None,
    expected_n_beta: int = None,
):
    """
    Load initial determinants and coefficients from dets.npz file.
    Returns: (dets, coeffs) or (None, None) if file doesn't exist
    """
    if not os.path.exists(dets_path):
        log_verbose(f"dets.npz file not found at {dets_path}")
        return None, None

    try:
        data = np.load(dets_path, allow_pickle=True)
        log_verbose(f"Loading initial determinants from {dets_path}")

        dets_key = 'core_set' if core_set else 'dets'
        coeffs_key = 'core_set_coeffs' if core_set else 'dets_coeffs'

        if dets_key in data and coeffs_key in data:
            dets_array = data[dets_key]
            coeffs = data[coeffs_key]
            dets_layout = "interleaved"
            has_explicit_layout = False
            if "dets_layout" in data:
                has_explicit_layout = True
                layout_value = data["dets_layout"]
                if getattr(layout_value, "shape", ()) == ():
                    dets_layout = str(layout_value.item())
                else:
                    dets_layout = str(layout_value)
                dets_layout = dets_layout.strip().lower()

            from trimci import get_functions_for_system

            if dets_array.dtype == np.uint64:
                n_dets, total_cols = dets_array.shape
                n_uint64_pairs = total_cols // 2

                if (
                    not has_explicit_layout
                    and n_uint64_pairs > 1
                    and expected_n_alpha is not None
                    and expected_n_beta is not None
                ):
                    sample = dets_array[:min(n_dets, 256)]

                    def _counts_for_layout(row, layout):
                        if layout == "block":
                            alpha_cols = row[:n_uint64_pairs]
                            beta_cols = row[n_uint64_pairs:]
                        else:
                            alpha_cols = row[0::2]
                            beta_cols = row[1::2]
                        alpha_count = sum(int(x).bit_count() for x in alpha_cols)
                        beta_count = sum(int(x).bit_count() for x in beta_cols)
                        return alpha_count, beta_count

                    def _layout_matches(layout):
                        for row in sample:
                            alpha_count, beta_count = _counts_for_layout(row, layout)
                            if (
                                alpha_count != int(expected_n_alpha)
                                or beta_count != int(expected_n_beta)
                            ):
                                return False
                        return True

                    interleaved_matches = _layout_matches("interleaved")
                    block_matches = _layout_matches("block")
                    if block_matches and not interleaved_matches:
                        dets_layout = "block"
                        log_important(
                            "Inferred dets_layout='block' from expected electron "
                            f"counts ({expected_n_alpha}α, {expected_n_beta}β) "
                            f"for {dets_path}"
                        )
                    elif interleaved_matches and not block_matches:
                        dets_layout = "interleaved"
                    elif not interleaved_matches and not block_matches:
                        log_important(
                            "Warning: could not infer determinant layout from "
                            f"expected electron counts ({expected_n_alpha}α, "
                            f"{expected_n_beta}β); using default interleaved for "
                            f"{dets_path}"
                        )

                n_orb_estimate = 64 * n_uint64_pairs
                functions = get_functions_for_system(n_orb_estimate)
                DeterminantClass = functions['determinant_class']

                dets = []
                for i in range(n_dets):
                    if n_uint64_pairs == 1:
                        alpha_bits = int(dets_array[i, 0])
                        beta_bits = int(dets_array[i, 1])
                        dets.append(DeterminantClass(alpha_bits, beta_bits))
                    else:
                        alpha_array = []
                        beta_array = []
                        if dets_layout == "block":
                            for j in range(n_uint64_pairs):
                                alpha_array.append(int(dets_array[i, j]))
                                beta_array.append(int(dets_array[i, n_uint64_pairs + j]))
                        else:
                            for j in range(n_uint64_pairs):
                                alpha_array.append(int(dets_array[i, 2*j]))
                                beta_array.append(int(dets_array[i, 2*j + 1]))
                        dets.append(DeterminantClass(alpha_array, beta_array))

            elif dets_array.dtype == object:
                log_verbose("Loading legacy object format - this may not be compatible with C++")

                max_bits = max(max(row[0], row[1]) for row in dets_array)
                n_orb_estimate = int(max_bits).bit_length() if max_bits > 0 else 64
                functions = get_functions_for_system(n_orb_estimate)
                DeterminantClass = functions['determinant_class']

                def create_determinant_from_file(alpha_bits, beta_bits):
                    class_name = str(DeterminantClass)

                    if 'Determinant192' in class_name:
                        alpha_array = [(alpha_bits & 0xFFFFFFFFFFFFFFFF),
                                      ((alpha_bits >> 64) & 0xFFFFFFFFFFFFFFFF),
                                      ((alpha_bits >> 128) & 0xFFFFFFFFFFFFFFFF)]
                        beta_array = [(beta_bits & 0xFFFFFFFFFFFFFFFF),
                                     ((beta_bits >> 64) & 0xFFFFFFFFFFFFFFFF),
                                     ((beta_bits >> 128) & 0xFFFFFFFFFFFFFFFF)]
                        alpha_array = [x if x >= 0 else x + (1 << 64) for x in alpha_array]
                        beta_array = [x if x >= 0 else x + (1 << 64) for x in beta_array]
                        return DeterminantClass(alpha_array, beta_array)
                    elif 'Determinant128' in class_name:
                        alpha_array = [(alpha_bits & 0xFFFFFFFFFFFFFFFF),
                                      ((alpha_bits >> 64) & 0xFFFFFFFFFFFFFFFF)]
                        beta_array = [(beta_bits & 0xFFFFFFFFFFFFFFFF),
                                     ((beta_bits >> 64) & 0xFFFFFFFFFFFFFFFF)]
                        alpha_array = [x if x >= 0 else x + (1 << 64) for x in alpha_array]
                        beta_array = [x if x >= 0 else x + (1 << 64) for x in beta_array]
                        return DeterminantClass(alpha_array, beta_array)
                    else:
                        return DeterminantClass(alpha_bits, beta_bits)

                dets = [create_determinant_from_file(int(row[0]), int(row[1])) for row in dets_array]
            else:
                max_bits = max(np.max(dets_array[:, 0]), np.max(dets_array[:, 1]))
                max_bits = int(max_bits)
                n_orb_estimate = int(max_bits).bit_length() if max_bits > 0 else 64
                functions = get_functions_for_system(n_orb_estimate)
                DeterminantClass = functions['determinant_class']

                dets = [DeterminantClass(int(row[0]), int(row[1])) for row in dets_array]

            log_verbose(f"Loaded {len(dets)} determinants from {dets_key}")
            log_verbose(f"Loaded {len(coeffs)} coefficients")

            return dets, coeffs.tolist()
        else:
            log_verbose(f"Required keys '{dets_key}' or '{coeffs_key}' not found in {dets_path}")
            return None, None

    except Exception as e:
        log_verbose(f"Error loading {dets_path}: {e}")
        return None, None


def _get_n_segments(DeterminantClass):
    """Get the number of uint64 segments from a Determinant class."""
    class_name = str(DeterminantClass)
    for bits, segs in [('512', 8), ('448', 7), ('384', 6), ('320', 5),
                       ('256', 4), ('192', 3), ('128', 2)]:
        if bits in class_name:
            return segs
    return 1


def generate_initial_states(n_alpha, n_beta, n_orb,
                            initial_dets_dict=None,
                            save_path=None):
    """
    Generate initial determinants and coefficients for TrimCI.

    Parameters
    ----------
    n_alpha, n_beta : int
        Number of alpha/beta electrons.
    n_orb : int
        Number of orbitals.
    initial_dets_dict : dict or None
        Dict of {type: coeff or [coeff, count] or dict or [[coeff, state], ...]}.
        Supported keys: "reference", "hf", "afm", "paramagnetic", "stripe",
                        "random", "random_closed_shell", "random_excited", "bitstring"
    save_path : str or None
        If given, save dets.npz to this path.

    Returns
    -------
    dets : list[Determinant]
    coeffs : list[float]
    """
    from trimci import get_functions_for_system

    dets, coeffs = [], []
    rng = np.random.default_rng()

    functions = get_functions_for_system(n_orb)
    DeterminantClass = functions['determinant_class']
    n_segments = _get_n_segments(DeterminantClass)
    log_important(f"Using Determinant class: {DeterminantClass}")

    def hf_reference():
        return functions['generate_reference_det'](n_alpha, n_beta)

    def to_uint64(x):
        if isinstance(x, np.integer):
            x = int(x)
        if x < 0:
            x += (1 << 64)
        return x & 0xFFFFFFFFFFFFFFFF

    def create_determinant(alpha_bits, beta_bits):
        """Support both int and array input for all Determinant types"""
        if n_segments > 1:
            if isinstance(alpha_bits, (list, tuple)):
                alpha_array = [to_uint64(x) for x in alpha_bits]
                beta_array = [to_uint64(x) for x in beta_bits]
                while len(alpha_array) < n_segments:
                    alpha_array.append(0)
                while len(beta_array) < n_segments:
                    beta_array.append(0)
            else:
                alpha_array = [to_uint64((alpha_bits >> (64*i)) & 0xFFFFFFFFFFFFFFFF) for i in range(n_segments)]
                beta_array = [to_uint64((beta_bits >> (64*i)) & 0xFFFFFFFFFFFFFFFF) for i in range(n_segments)]
            return DeterminantClass(alpha_array, beta_array)
        else:
            if isinstance(alpha_bits, (list, tuple)):
                alpha_bits = alpha_bits[0]
                beta_bits = beta_bits[0]
            return DeterminantClass(to_uint64(alpha_bits), to_uint64(beta_bits))

    def _make_bitmask_arrays(occupied_orbitals):
        """Convert a list of occupied orbital indices to segment arrays."""
        arrays = [0] * n_segments
        for i in occupied_orbitals:
            seg, bit = divmod(i, 64)
            arrays[seg] |= (1 << bit)
        return arrays

    def add_preset(kind, coeff, count=1):
        for _ in range(count):
            if kind == "reference" or kind == "hf":
                det = hf_reference()
            elif kind == "afm":
                alpha_bits = sum([1 << i for i in range(0, n_orb, 2)][:n_alpha])
                beta_bits = sum([1 << i for i in range(1, n_orb, 2)][:n_beta])
                det = create_determinant(alpha_bits, beta_bits)
            elif kind == "paramagnetic":
                alpha_bits = sum([1 << i for i in range(n_alpha)])
                beta_bits = sum([1 << i for i in range(n_beta)])
                det = create_determinant(alpha_bits, beta_bits)
            elif kind == "stripe":
                half = n_orb // 2
                alpha_bits = sum([1 << i for i in range(min(n_alpha, half))])
                beta_bits = sum([1 << i for i in range(half, half + n_beta)])
                det = create_determinant(alpha_bits, beta_bits)
            else:
                continue
            key = (tuple(np.atleast_1d(det.alpha)), tuple(np.atleast_1d(det.beta)))
            if key not in seen_dets:
                seen_dets.add(key)
                dets.append(det)
                coeffs.append(float(coeff))

    # Track all generated dets for dedup across all sources
    seen_dets = set()
    def _det_key(alpha_array, beta_array):
        """Compute det key from bitmask arrays without creating Determinant object."""
        if n_segments == 1:
            return (to_uint64(alpha_array[0]), to_uint64(beta_array[0]))
        return tuple(to_uint64(x) for x in alpha_array) + tuple(to_uint64(x) for x in beta_array)

    def add_random(coeff, count=1):
        from scipy.special import comb
        total_possible = int(comb(n_orb, n_alpha)) * int(comb(n_orb, n_beta))
        added = 0
        max_attempts = count * 10  # safety cap to avoid infinite loop
        attempts = 0
        while added < count and attempts < max_attempts:
            attempts += 1
            occ_alpha = rng.choice(n_orb, n_alpha, replace=False)
            occ_beta = rng.choice(n_orb, n_beta, replace=False)
            alpha_array = _make_bitmask_arrays(occ_alpha.tolist())
            beta_array = _make_bitmask_arrays(occ_beta.tolist())
            key = _det_key(alpha_array, beta_array)
            if key in seen_dets:
                continue
            seen_dets.add(key)
            dets.append(create_determinant(alpha_array, beta_array))
            coeffs.append(float(coeff))
            added += 1
            if len(seen_dets) >= total_possible:
                break  # exhausted all possible dets
        if added < count:
            log_important(f"add_random: requested {count}, generated {added} "
                          f"(space exhausted or max_attempts reached)")

    def add_random_closed_shell(coeff, count=1):
        if n_alpha != n_beta:
            raise ValueError(f"random_closed_shell requires n_alpha ({n_alpha}) == n_beta ({n_beta})")
        from scipy.special import comb
        total_possible = int(comb(n_orb, n_alpha))
        added = 0
        max_attempts = count * 10
        attempts = 0
        while added < count and attempts < max_attempts:
            attempts += 1
            occ = rng.choice(n_orb, n_alpha, replace=False)
            alpha_array = _make_bitmask_arrays(occ.tolist())
            beta_array = list(alpha_array)
            key = _det_key(alpha_array, beta_array)
            if key in seen_dets:
                continue
            seen_dets.add(key)
            dets.append(create_determinant(alpha_array, beta_array))
            coeffs.append(float(coeff))
            added += 1
            if len(seen_dets) >= total_possible:
                break
        if added < count:
            log_important(f"add_random_closed_shell: requested {count}, generated {added} "
                          f"(space exhausted or max_attempts reached)")

    def add_random_excited(config):
        coeff = float(config.get("coeff", 1.0))
        count = int(config.get("count", 1))
        min_level = int(config.get("min_level", 1))
        max_level = int(config.get("max_level", 1))
        random_coeffs = bool(config.get("random_coeffs", False))
        spin_preserve = bool(config.get("spin_preserve", False))

        ref_det = hf_reference()
        seen = set()

        # Normalize ref_det.alpha/beta to python-int bitmasks (works for both
        # Determinant64 (uint64) and Determinant128+ (list/array of uint64 per
        # 64-bit segment).
        def _to_bigint(v):
            if isinstance(v, (list, tuple)) or hasattr(v, '__iter__'):
                out = 0
                for seg, w in enumerate(v):
                    out |= (int(w) & 0xFFFFFFFFFFFFFFFF) << (64 * seg)
                return out
            return int(v)

        ref_alpha = _to_bigint(ref_det.alpha)
        ref_beta  = _to_bigint(ref_det.beta)

        for _ in range(count):
            level = rng.integers(min_level, max_level + 1)
            if spin_preserve:
                n_alpha_exc = level // 2
                n_beta_exc = level - n_alpha_exc
            else:
                n_alpha_exc = rng.integers(0, level + 1)
                n_beta_exc = level - n_alpha_exc

            alpha_occ = [i for i in range(n_orb) if (ref_alpha >> i) & 1]
            alpha_vir = [i for i in range(n_orb) if not (ref_alpha >> i) & 1]
            beta_occ  = [i for i in range(n_orb) if (ref_beta >> i) & 1]
            beta_vir  = [i for i in range(n_orb) if not (ref_beta >> i) & 1]

            if not (n_alpha_exc <= len(alpha_occ) and n_alpha_exc <= len(alpha_vir)):
                continue
            if not (n_beta_exc <= len(beta_occ) and n_beta_exc <= len(beta_vir)):
                continue

            chosen_alpha_occ = rng.choice(alpha_occ, size=n_alpha_exc, replace=False) if n_alpha_exc > 0 else []
            chosen_alpha_vir = rng.choice(alpha_vir, size=n_alpha_exc, replace=False) if n_alpha_exc > 0 else []
            chosen_beta_occ = rng.choice(beta_occ, size=n_beta_exc, replace=False) if n_beta_exc > 0 else []
            chosen_beta_vir = rng.choice(beta_vir, size=n_beta_exc, replace=False) if n_beta_exc > 0 else []

            alpha_bits, beta_bits = ref_alpha, ref_beta
            for o, v in zip(chosen_alpha_occ, chosen_alpha_vir):
                alpha_bits &= ~(1 << int(o))
                alpha_bits |= (1 << int(v))
            for o, v in zip(chosen_beta_occ, chosen_beta_vir):
                beta_bits &= ~(1 << int(o))
                beta_bits |= (1 << int(v))

            det = create_determinant(alpha_bits, beta_bits)
            key = (tuple(np.atleast_1d(det.alpha)), tuple(np.atleast_1d(det.beta)))
            if key in seen:
                continue
            seen.add(key)
            dets.append(det)
            coeffs.append(rng.normal(0, coeff) if random_coeffs else coeff)

    def add_manual_bitstring(entries):
        if isinstance(entries, str):
            entries = [[1.0, entries]]
        elif isinstance(entries, (list, tuple)) and len(entries) == 2 and isinstance(entries[1], str):
            entries = [entries]
        elif isinstance(entries, list):
            if all(isinstance(e, str) for e in entries):
                entries = [[1.0, e] for e in entries]
            elif all(isinstance(e, (list, tuple)) and len(e) == 2 for e in entries):
                pass
            else:
                raise ValueError(f"Invalid bitstring format: {entries}")
        else:
            raise ValueError(f"Unsupported bitstring type: {type(entries)}")

        seen = set()
        for coeff_val, entry in entries:
            alpha_str, beta_str = entry.split(',', 1)
            alpha_occ = [int(x) for x in alpha_str.strip().split() if x.strip().isdigit()]
            beta_occ = [int(x) for x in beta_str.strip().split() if x.strip().isdigit()]

            alpha_array = _make_bitmask_arrays(alpha_occ)
            beta_array = _make_bitmask_arrays(beta_occ)

            det = create_determinant(alpha_array, beta_array)
            key = (tuple(np.atleast_1d(det.alpha)), tuple(np.atleast_1d(det.beta)))
            if key in seen:
                continue
            seen.add(key)
            dets.append(det)
            coeffs.append(float(coeff_val))

    # --- Parse input dict ---
    if initial_dets_dict:
        for kind, val in initial_dets_dict.items():
            if kind == "bitstring":
                add_manual_bitstring(val)
            elif isinstance(val, (int, float)):
                if kind == "random":
                    add_random(val, 1)
                elif kind == "random_closed_shell":
                    add_random_closed_shell(val, 1)
                else:
                    add_preset(kind, val, 1)
            elif isinstance(val, (list, tuple)):
                if kind == "random":
                    coeff, count = val
                    add_random(coeff, int(count))
                elif kind == "random_closed_shell":
                    coeff, count = val
                    add_random_closed_shell(coeff, int(count))
                else:
                    coeff, count = val
                    add_preset(kind, coeff, int(count))
            elif isinstance(val, dict) and kind == "random_excited":
                add_random_excited(val)
            else:
                raise ValueError(f"Unsupported value for {kind}: {val}")

    # --- Normalize coefficients ---
    norm = np.sqrt(sum(c**2 for c in coeffs))
    if norm > 0:
        coeffs = [c / norm for c in coeffs]

    # --- Optional save ---
    if save_path:
        alpha_arr = np.array([np.atleast_1d(d.alpha) for d in dets], dtype=np.uint64)
        beta_arr = np.array([np.atleast_1d(d.beta) for d in dets], dtype=np.uint64)
        np.savez_compressed(save_path,
                            dets=np.column_stack([alpha_arr[:, 0], beta_arr[:, 0]]),
                            dets_coeffs=np.array(coeffs),
                            core_set=np.column_stack([alpha_arr[:, 0], beta_arr[:, 0]]),
                            core_set_coeffs=np.array(coeffs))

    return dets, coeffs


def perturb_determinants(source_dets, source_coeffs, n_alpha, n_beta, n_orb,
                         perturb_strength, perturb_count):
    """Generate perturbed determinants by random excitations from source dets.

    Analogous to transverse field Bx in quantum annealing: each perturbation
    applies random occupied->virtual electron hops, preserving n_alpha/n_beta.

    Parameters
    ----------
    perturb_strength : float
        Average excitation level (Poisson lambda). Single parameter controlling
        how far from source dets the perturbations explore.
        1.0 = mostly single excitations, 2.0 = single+double mix, etc.
    perturb_count : int
        Total number of perturbed determinants to generate.
    """
    from trimci import get_functions_for_system

    rng = np.random.default_rng()
    functions = get_functions_for_system(n_orb)
    DeterminantClass = functions['determinant_class']
    n_segments = _get_n_segments(DeterminantClass)

    def to_uint64(x):
        if isinstance(x, np.integer):
            x = int(x)
        if x < 0:
            x += (1 << 64)
        return x & 0xFFFFFFFFFFFFFFFF

    def make_bitmask(occupied_orbitals):
        arrays = [0] * n_segments
        for i in occupied_orbitals:
            seg, bit = divmod(i, 64)
            arrays[seg] |= (1 << bit)
        return arrays

    def create_det(alpha_array, beta_array):
        if n_segments > 1:
            a = [to_uint64(x) for x in alpha_array]
            b = [to_uint64(x) for x in beta_array]
            while len(a) < n_segments: a.append(0)
            while len(b) < n_segments: b.append(0)
            return DeterminantClass(a, b)
        return DeterminantClass(to_uint64(alpha_array[0]), to_uint64(beta_array[0]))

    def get_occupied(det):
        alpha_bits, beta_bits = det.alpha, det.beta
        alpha_occ, beta_occ = [], []
        if isinstance(alpha_bits, (list, tuple)):
            for seg_idx, (a, b) in enumerate(zip(alpha_bits, beta_bits)):
                for bit in range(64):
                    orb = seg_idx * 64 + bit
                    if orb >= n_orb:
                        break
                    if int(a) & (1 << bit): alpha_occ.append(orb)
                    if int(b) & (1 << bit): beta_occ.append(orb)
        else:
            for i in range(n_orb):
                if int(alpha_bits) & (1 << i): alpha_occ.append(i)
                if int(beta_bits) & (1 << i): beta_occ.append(i)
        return alpha_occ, beta_occ

    all_orbitals = set(range(n_orb))
    max_alpha_exc = min(n_alpha, n_orb - n_alpha)
    max_beta_exc = min(n_beta, n_orb - n_beta)
    n_source = len(source_dets)

    perturbed_dets, perturbed_coeffs = [], []
    seen = set()

    # Exclude source dets from results
    for det in source_dets:
        key = (tuple(np.atleast_1d(det.alpha)), tuple(np.atleast_1d(det.beta)))
        seen.add(key)

    max_attempts = perturb_count * 3
    attempts = 0

    while len(perturbed_dets) < perturb_count and attempts < max_attempts:
        attempts += 1

        src_idx = rng.integers(0, n_source)
        alpha_occ, beta_occ = get_occupied(source_dets[src_idx])
        alpha_vir = sorted(all_orbitals - set(alpha_occ))
        beta_vir = sorted(all_orbitals - set(beta_occ))

        # Poisson-sampled excitation level, clamp to [1, max_possible]
        k = max(1, int(rng.poisson(perturb_strength)))
        k = min(k, max_alpha_exc + max_beta_exc)

        # Split excitations between alpha and beta
        lo = max(0, k - max_beta_exc)
        hi = min(k, max_alpha_exc)
        k_alpha = int(rng.integers(lo, hi + 1))
        k_beta = k - k_alpha

        # Random occupied->virtual swaps
        cho_a_occ = rng.choice(alpha_occ, size=k_alpha, replace=False).tolist() if k_alpha > 0 else []
        cho_a_vir = rng.choice(alpha_vir, size=k_alpha, replace=False).tolist() if k_alpha > 0 else []
        cho_b_occ = rng.choice(beta_occ, size=k_beta, replace=False).tolist() if k_beta > 0 else []
        cho_b_vir = rng.choice(beta_vir, size=k_beta, replace=False).tolist() if k_beta > 0 else []

        new_alpha = sorted((set(alpha_occ) - set(cho_a_occ)) | set(cho_a_vir))
        new_beta = sorted((set(beta_occ) - set(cho_b_occ)) | set(cho_b_vir))

        det = create_det(make_bitmask(new_alpha), make_bitmask(new_beta))
        key = (tuple(np.atleast_1d(det.alpha)), tuple(np.atleast_1d(det.beta)))
        if key in seen:
            continue
        seen.add(key)

        perturbed_dets.append(det)
        perturbed_coeffs.append(1.0)

    log_important(f"Perturbed {len(perturbed_dets)}/{perturb_count} unique dets "
                  f"from {n_source} sources (strength={perturb_strength})")

    return perturbed_dets, perturbed_coeffs
