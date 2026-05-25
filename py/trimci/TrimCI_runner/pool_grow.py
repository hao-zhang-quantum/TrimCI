"""Pool-only determinant expansion: grow a checkpoint without Davidson.

Uses pool_build_only=True to skip ABIndex/Diagonals/Davidson.
Only does: PoolBuild → Sort → Checkpoint.

Typical use case: the last expansion step where you only need a larger
determinant pool (e.g. for distributed Davidson on a cluster).

Usage as CLI:
    python -m trimci.TrimCI_runner.pool_grow \
        --checkpoint checkpoint_round_3.bin \
        --integrals checkpoint_integrals.bin \
        --target-dets 1280000000 \
        --output-dir checkpoints \
        --tmp-dir /tmp/pool_grow

Usage as library:
    from trimci.TrimCI_runner.pool_grow import pool_grow
    pool_grow(
        checkpoint_path="checkpoint_round_3.bin",
        integrals_path="checkpoint_integrals.bin",
        target_dets=1_280_000_000,
        output_dir="checkpoints",
    )

Notes:
    - pool_build_only produces coeffs=[1,0,0,...] (not Davidson-converged),
      so screening quality degrades. Use only for the final expansion step.
    - When --tmp-dir is set, checkpoint is written to fast local disk first,
      then verified and copied to --output-dir (avoids NFS quota issues).
"""
import argparse
import os
import shutil
import sys


def pool_grow(
    checkpoint_path,
    integrals_path,
    target_dets,
    output_dir,
    *,
    output_name=None,
    growth_factor=2.0,
    threshold=1.5e-7,
    threshold_decay=0.9,
    tmp_dir=None,
    verbose=2,
):
    """Expand determinant pool without Davidson diagonalization.

    Args:
        checkpoint_path: Path to checkpoint_round_N.bin.
        integrals_path: Path to checkpoint_integrals.bin.
        target_dets: Target number of determinants.
        output_dir: Directory for output checkpoint.
        output_name: If set, rename output to this filename (e.g. "checkpoint_1280m.bin").
            By default uses the auto-generated checkpoint_round_N.bin name.
        growth_factor: Expansion factor per round (default 2.0).
        threshold: Screening threshold (default 1.5e-7).
        threshold_decay: Threshold decay per round (default 0.9).
        tmp_dir: If set, write checkpoint here first then copy to output_dir.
        verbose: Verbosity level (0=silent, 1=summary, 2=progress).

    Returns:
        Path to the output checkpoint file.
    """
    # Set environment before importing trimci (affects OMP threads, malloc)
    os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 1))
    os.environ.setdefault("MALLOC_ARENA_MAX", "4")
    os.environ.setdefault("MALLOC_MMAP_THRESHOLD_", "131072")

    from trimci.checkpoint import load_checkpoint, load_integrals_checkpoint
    from trimci.TrimCI_runner.run_expansion import run_expansion

    # Load data
    print(f"Loading integrals: {integrals_path}")
    intg = load_integrals_checkpoint(integrals_path)
    print(f"  n_orb={intg['n_orb']}")

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = load_checkpoint(checkpoint_path)
    input_round = ckpt["round"]
    n_dets = ckpt["n_dets"]
    energy = ckpt["energy"]
    print(f"  n_dets={n_dets}, round={input_round}, energy={energy:.8f}")

    # Auto-compute round offset so output continues from input round
    round_offset = input_round + 1

    # Determine write directory
    write_dir = tmp_dir if tmp_dir else output_dir
    os.makedirs(write_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Snapshot existing files before expansion (to detect new checkpoint)
    existing_files = set(os.listdir(write_dir))

    print(f"Target: {n_dets:,} → {target_dets:,} dets")
    print(f"Checkpoint will be written to: {write_dir}")
    if tmp_dir:
        print(f"Then copied to: {output_dir}")
    sys.stdout.flush()

    # Run pool-only expansion
    run_expansion(
        h1=intg["h1"],
        eri=intg["eri"],
        e_nuc=0.0,
        alpha_init=ckpt["alpha"],
        beta_init=ckpt["beta"],
        coeffs_init=ckpt["coeffs"],
        phases=[
            {
                "max_n_dets": target_dets,
                "growth_factor": growth_factor,
                "pool_build_only": True,
                "checkpoint_round_offset": round_offset,
                "threshold": threshold,
                "threshold_decay": threshold_decay,
                "verbose": verbose,
            },
        ],
        checkpoint_dir=write_dir,
        output_prefix=None,
        label=f"Pool-only: {n_dets:,} → {target_dets:,} dets",
    )

    # Detect new checkpoint file(s) written by expansion
    new_files = sorted(
        f for f in os.listdir(write_dir)
        if f not in existing_files and f.endswith(".bin")
    )
    if not new_files:
        print("ERROR: no new checkpoint file detected after expansion")
        sys.stdout.flush()
        return None

    output_filename = new_files[-1]  # latest new .bin file
    final_name = output_name if output_name else output_filename
    src = os.path.join(write_dir, output_filename)
    dst = os.path.join(output_dir, final_name)

    if tmp_dir and tmp_dir != output_dir:
        if not os.path.isfile(src):
            print(f"ERROR: checkpoint not found at {src}")
            sys.stdout.flush()
            return None

        src_size = os.path.getsize(src)
        print(f"\nCopying checkpoint: {src} ({src_size / (1024**3):.1f} GB) → {dst}")
        sys.stdout.flush()

        shutil.copy2(src, dst)

        # Verify copy
        dst_size = os.path.getsize(dst)
        if dst_size != src_size:
            print(f"ERROR: size mismatch after copy! src={src_size}, dst={dst_size}")
            sys.stdout.flush()
            return None

        print(f"Copy verified ({dst_size / (1024**3):.1f} GB). Removing tmp file.")
        os.remove(src)
        result_path = dst
    elif output_name and output_filename != output_name:
        # Same directory, just rename
        print(f"Renaming: {output_filename} → {final_name}")
        os.rename(src, dst)
        result_path = dst
    else:
        result_path = src

    if os.path.isfile(result_path):
        print(f"Output checkpoint: {result_path}")
    else:
        print(f"WARNING: output checkpoint not found at {result_path}")
        result_path = None

    sys.stdout.flush()
    return result_path


def main():
    parser = argparse.ArgumentParser(
        description="Pool-only determinant expansion (no Davidson).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Example:
  python -m trimci.TrimCI_runner.pool_grow \\
    --checkpoint checkpoints/checkpoint_round_3.bin \\
    --integrals checkpoints/checkpoint_integrals.bin \\
    --target-dets 1280000000 \\
    --output-dir checkpoints \\
    --tmp-dir /tmp/pool_grow
""",
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to input checkpoint_round_N.bin")
    parser.add_argument("--integrals", required=True,
                        help="Path to checkpoint_integrals.bin")
    parser.add_argument("--target-dets", type=int, required=True,
                        help="Target number of determinants")
    parser.add_argument("--output-dir", required=True,
                        help="Directory for output checkpoint")
    parser.add_argument("--output-name", default=None,
                        help="Rename output to this filename (default: auto checkpoint_round_N.bin)")
    parser.add_argument("--growth-factor", type=float, default=2.0,
                        help="Expansion factor per round (default: 2.0)")
    parser.add_argument("--threshold", type=float, default=1.5e-7,
                        help="Screening threshold (default: 1.5e-7)")
    parser.add_argument("--threshold-decay", type=float, default=0.9,
                        help="Threshold decay (default: 0.9)")
    parser.add_argument("--tmp-dir", default=None,
                        help="Write checkpoint to tmp first, then copy to output-dir")
    parser.add_argument("--verbose", type=int, default=2,
                        help="Verbosity: 0=silent, 1=summary, 2=progress (default: 2)")
    parser.add_argument("--screening-mode", default="hb", choices=["hb", "hb-pt2"],
                        help="Screening strategy: hb (heat-bath, default) or hb-pt2 (heat-bath + PT2 reranking)")
    args = parser.parse_args()

    pool_grow(
        checkpoint_path=args.checkpoint,
        integrals_path=args.integrals,
        target_dets=args.target_dets,
        output_dir=args.output_dir,
        output_name=args.output_name,
        growth_factor=args.growth_factor,
        threshold=args.threshold,
        threshold_decay=args.threshold_decay,
        tmp_dir=args.tmp_dir,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
