# molecule_setup.py

def setup_molecule(atom_str, basis, spin=0):
    """
    Build a molecule with PySCF and return (mol, mf, h1_mo, eri_mo).

    Requires optional dependency 'pyscf'. Install via:
    - pip install trimci[chem]
    or
    - pip install pyscf
    """
    try:
        from pyscf import gto, scf, ao2mo
    except ImportError as e:
        raise ImportError(
            "Optional dependency 'pyscf' is required for setup_molecule. "
            "Install with `pip install trimci[chem]` or `pip install pyscf`."
        ) from e

    mol = gto.M(atom=atom_str, basis=basis, spin=spin)
    mf = scf.RHF(mol)
    mf.verbose = 0  # suppress all SCF output
    mf.run()

    h1_mo = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff

    from pyscf.ao2mo import restore
    eri_mo_compressed = ao2mo.kernel(mol, mf.mo_coeff)
    n_orb = mf.mo_coeff.shape[1]
    eri_mo = restore(1, eri_mo_compressed, n_orb)

    return mol, mf, h1_mo, eri_mo
