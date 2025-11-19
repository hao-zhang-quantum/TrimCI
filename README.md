<p align="center">
  <img src="https://raw.githubusercontent.com/hao-zhang-quantum/TrimCI/44341c9d1a0cd070ed9c06ad4b1e437a54358514/banner.png"
       alt="TrimCI banner"
       width="800"/>
</p>


# TrimCI

**Trimmed Configuration Interaction (TrimCI)** is a **high-performance** framework for quantum many-body and quantum chemistry calculation.  
It constructs accurate ground states *directly from random Slater determinants* — without any guiding ansatz, Hartree–Fock reference, or prior human knowledge — through an iterative expansion–trimming cycle on the determinant graph.  

TrimCI demonstrates that accurate many-body ground states can *emerge from randomness*, achieving state-of-the-art accuracy and efficiency across molecular and lattice systems. It can **outperform human-designed ansatzes or human-provided knowledge** in hard problems, such as strongly correlated systems.

**Paper**
H. Zhang, M. Otten, “From Random Determinants to the Ground State,” arXiv:2511.14734 (2025).
https://arxiv.org/abs/2511.14734

---

## 🚀 Install

```
pip install trimci
```

Alternatively, you may build the package on your environment `python -m pip install .`.


## ⚡ Quick Example

1) A fast run in AUTO mode.

```
cd tutorial
tci --auto --goal speed -n 1000
```


2) An accurate run in AUTO mode.
```
cd tutorial
tci --auto --goal accuracy -n 1000
```


3) A custom run in FULL mode.
See `trimci_tutorial.ipynb` for more details.


4) More details are in the `paper` and `py/trimci/TrimCI_runner/trimci_driver.py`.



## ✨ Key Features
- **Emergent accuracy from randomness:** discovers the ground state without predefined ansatz or human bias.  
- **Expansion–trimming mechanism:** iteratively expands the determinant space via Hamiltonian couplings and trims away unimportant configurations.  
- **C++ backend, Python interface:** efficient C++ backend with OpenMP parallelization for core functions, while Python interface provides user-friendly access.
- **Massive efficiency gain:** achieves equivalent accuracy to selected-CI using \(10^2\)–\(10^5\times\) fewer determinants.  
- **Transferable module:** TrimCI wavefunctions can initialize or guide AFQMC, VMC, DMRG, tensor networks, and quantum algorithms (VQE, QPE).  
- **Explicit wavefunction:** produces a compact, analyzable coefficients and determinants dict enabling direct evaluation of observables and other measures.

---

## 🧩 Algorithm Overview

TrimCI operates on a graph whose nodes are Slater determinants and edges correspond to Hamiltonian couplings \(H_{ij}\).  
The algorithm alternates between two complementary stages:

1. **Expansion:**  
   Add neighboring determinants connected by large couplings \(|H_{ij}c_j|>\theta\).  
   This explores physically significant regions of the Hilbert space.

2. **Trimming:**  
   - *Local trimming:* random groups are diagonalized independently to remove negligible states.  
   - *Global trimming:* survivors are merged and re-diagonalized to select top-amplitude determinants.  

This two-level process refines the variational subspace nearly monotonically and rapidly converges toward the ground state.

---

## 🧠 Scientific Highlights

- **Molecular systems:**  
  Matches SHCI accuracy on Cr₂, [4Fe–4S], and the nitrogenase P-cluster while using \(10^2\)–\(10^5\times\) fewer determinants.  

- **Lattice systems:**  
  For the 8×8 Hubbard model, TrimCI reproduces >99 % of the AFQMC ground-state energy using only \(10^{-28}\) of the Hilbert space.  
  On 4×4 lattices, TrimCI achieves higher accuracy than AFQMC benchmarks.

- **Emergent structure:**  
  Starting from random determinants, TrimCI self-organizes a compact “core set” of dominant configurations.  
  The amplitude distribution follows a power law \(p(r) \propto r^{-(1+\alpha)}\) , revealing a *scale-free* organization and quantifiable **algorithmic entropy**.

---

## 🔗 Integration with Other Frameworks

TrimCI provides a compact and explicit coefficients and determinants dict that can:
- serve as a **trial or guiding wavefunction** for AFQMC and VMC,  
- initialize **DMRG** and tensor-network optimizations,  
- provide high-overlap initial states for **VQE** or **QPE** quantum algorithms,  
- enable cross-validation and hybrid workflows across classical and quantum domains.

---

## 📜 License
MIT License — see [`LICENSE`](LICENSE) for details.
