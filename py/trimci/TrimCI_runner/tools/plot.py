import json
import matplotlib.pyplot as plt
import sys
from pathlib import Path

def plot_trimci_energy(json_path: str, use_total=False):
    json_path = Path(json_path)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    workflow = data.get("experimental_workflow", {})
    iterations_data = workflow.get("iterations", [])

    if not iterations_data:
        raise ValueError("❌ 没有找到 iterations 数据！")

    iterations, energies, dets = [], [], []

    for entry in iterations_data:
        it = entry.get("iteration")
        energy = entry.get("total_energy") if use_total else entry.get("electronic_energy")
        ndets = entry.get("final_dets_count")
        if it is not None and energy is not None:
            iterations.append(it)
            energies.append(energy)
            dets.append(ndets if ndets is not None else 0)

    # --- 图1：能量随迭代 & determinants 数量 ---
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(iterations, energies, marker=".", linestyle="-", color="royalblue", label="Energy")
    ax1.axhline(y=0, color="black", linestyle="--", linewidth=0.8)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Energy (Ha)", color="royalblue")
    ax1.tick_params(axis='y', labelcolor="royalblue")

    ax2 = ax1.twinx()
    ax2.plot(iterations, dets, linestyle="--", color="darkorange", label="Determinants")
    ax2.set_ylabel("Final determinants", color="darkorange")
    ax2.tick_params(axis='y', labelcolor="darkorange")

    plt.title("TrimCI Energy Convergence with Determinant Growth")
    fig.tight_layout()

    save_path1 = json_path.with_name(json_path.stem + "_convergence.png")
    plt.savefig(save_path1, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Convergence figure saved to {save_path1}")

    # --- 图2：能量 vs determinants 散点图 ---
    plt.figure(figsize=(8, 6))
    plt.scatter(dets, energies, c=iterations, cmap="viridis", s=30, alpha=0.8)
    plt.colorbar(label="Iteration")
    plt.xlabel("Number of determinants")
    plt.ylabel("Energy (Ha)")
    plt.title("TrimCI Energy vs Determinants")
    plt.grid(True)

    save_path2 = json_path.with_name(json_path.stem + "_scatter.png")
    plt.savefig(save_path2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Scatter figure saved to {save_path2}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_trimci_energy.py result.json")
    else:
        json_file = sys.argv[1]
        plot_trimci_energy(json_file)
