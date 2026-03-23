"""
tbg — Twisted Bilayer Graphene tight-binding package
=====================================================

Constructs real-space tight-binding Hamiltonians for arbitrary-angle
(including incommensurate) twisted bilayer graphene (TBG) using the
Moon–Koshino Slater–Koster model.

Typical workflow::

    from tbg.lattice import generate_tbg_geometry
    from tbg.hamiltonian import build_hamiltonian
    from tbg.dos import compute_dos

    layers = generate_tbg_geometry(twist_angle_deg=1.08, n_cells=5)
    H = build_hamiltonian(layers)
    energies, dos = compute_dos(H)
"""

from tbg.lattice import generate_tbg_geometry, GrapheneLayer
from tbg.hamiltonian import build_hamiltonian
from tbg.dos import compute_dos, compute_ldos

__all__ = [
    "generate_tbg_geometry",
    "GrapheneLayer",
    "build_hamiltonian",
    "compute_dos",
    "compute_ldos",
]

__version__ = "0.1.0"
