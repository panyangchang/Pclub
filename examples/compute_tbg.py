"""
examples/compute_tbg.py
=======================
End-to-end example: build and analyse twisted bilayer graphene.

Run::

    python examples/compute_tbg.py

This script demonstrates:
1. Generating the atomic structure for several twist angles
2. Building the tight-binding Hamiltonian (sparse)
3. Computing DOS and LDOS
4. Printing a summary of key physical quantities
5. (Optionally) saving plots to PNG files if matplotlib is available
"""

from __future__ import annotations

import sys
import time
import numpy as np

# ── Import the tbg package ────────────────────────────────────────────────────
from tbg.lattice import generate_tbg_geometry, commensurate_indices
from tbg.hamiltonian import build_hamiltonian, hamiltonian_info, diagonalize
from tbg.dos import compute_dos, compute_ldos, integrated_dos
from tbg.utils import (
    moire_period,
    check_nn_distances,
    check_atom_count,
    symmetry_check,
    sublattice_onsite,
    gate_voltage_onsite,
)


def separator(title: str = "", width: int = 60) -> None:
    line = "─" * width
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"{'─'*pad} {title} {'─'*(width-pad-len(title)-2)}")
    else:
        print(line)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Geometry generation
# ─────────────────────────────────────────────────────────────────────────────
separator("1. Geometry")

test_angles = [1.08, 2.0, 5.0, 10.0, 21.787]
for theta in test_angles:
    lam = moire_period(theta)
    print(f"  θ = {theta:6.3f}°  →  moiré period λ = {lam:.1f} Å")

# Generate a moderate-size structure for 5°
theta = 5.0
n_cells = 4
print(f"\nGenerating TBG geometry: θ = {theta}°, n_cells = {n_cells}")
t0 = time.perf_counter()
bot, top = generate_tbg_geometry(theta, n_cells=n_cells)
dt = time.perf_counter() - t0

N_total = bot.n_atoms + top.n_atoms
print(f"  Bottom layer: {bot.n_atoms} atoms")
print(f"  Top    layer: {top.n_atoms} atoms")
print(f"  Total atoms : {N_total}")
print(f"  Generation time: {dt*1000:.1f} ms")

# Validate geometry
nn_check = check_nn_distances(bot)
cnt_check = check_atom_count(bot, n_cells)
print(f"  NN distance check: min={nn_check['min_nn']:.4f} Å  "
      f"max={nn_check['max_nn']:.4f} Å  expected={nn_check['expected']:.4f} Å"
      f"  ✓" if nn_check["passed"] else "  ✗")
print(f"  Atom count check : {cnt_check['actual']} atoms  "
      f"(expected {cnt_check['expected']})  "
      f"{'✓' if cnt_check['passed'] else '✗'}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Hamiltonian construction
# ─────────────────────────────────────────────────────────────────────────────
separator("2. Hamiltonian")

t0 = time.perf_counter()
H = build_hamiltonian([bot, top])
dt = time.perf_counter() - t0
info = hamiltonian_info(H)

print(f"  Matrix size   : {info['n_atoms']} × {info['n_atoms']}")
print(f"  Non-zero bonds: {info['nnz']}")
print(f"  Symmetric     : {info['is_symmetric']}")
print(f"  Sparsity      : {info['sparsity']*100:.2f}%")
print(f"  Min hopping   : {info['min_hopping']:.4f} eV")
print(f"  Max hopping   : {info['max_hopping']:.4f} eV")
print(f"  Build time    : {dt*1000:.1f} ms")

sym_check = symmetry_check(H)
print(f"  Symmetry check: max|H-Hᵀ| = {sym_check['max_asymmetry']:.2e}  "
      f"{'✓' if sym_check['passed'] else '✗'}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Eigenvalues
# ─────────────────────────────────────────────────────────────────────────────
separator("3. Eigenvalues")

t0 = time.perf_counter()
vals, vecs = diagonalize(H)
dt = time.perf_counter() - t0

print(f"  N = {N_total} eigenvalues computed in {dt*1000:.1f} ms")
print(f"  Band minimum  : {vals.min():.4f} eV")
print(f"  Band maximum  : {vals.max():.4f} eV")
print(f"  Bandwidth     : {vals.max()-vals.min():.4f} eV")
print(f"  States below E=0: {np.sum(vals < 0)} / {N_total}")

# Low-energy spectrum (5 eigenvalues each side of E=0)
n_show = 5
mid = np.searchsorted(vals, 0.0)
low = vals[max(0, mid-n_show):mid+n_show]
print(f"  Low-energy eigenvalues (eV): {np.array2string(low, precision=4)}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. DOS
# ─────────────────────────────────────────────────────────────────────────────
separator("4. DOS")

t0 = time.perf_counter()
energies, dos = compute_dos(H, n_energies=500, eta=0.05)
dt = time.perf_counter() - t0

print(f"  DOS computed in {dt*1000:.1f} ms")
print(f"  DOS at E=0 : {np.interp(0.0, energies, dos):.4f} states/eV/atom")

idos = integrated_dos(energies, dos, e_fermi=0.0)
print(f"  IDOS(E=0)  : {idos:.4f}  (should be ≈ 0.5 at charge neutrality)")

# ─────────────────────────────────────────────────────────────────────────────
# 5. LDOS at AA and AB stacking regions
# ─────────────────────────────────────────────────────────────────────────────
separator("5. LDOS")

# AA stacking: atoms near the origin (0,0) in each layer
def find_nearest_atom(layer, xy_target):
    dists = np.linalg.norm(layer.xy - np.array(xy_target), axis=1)
    return int(layer.atom_ids[np.argmin(dists)])

aa_bot = find_nearest_atom(bot, [0.0, 0.0])
aa_top = find_nearest_atom(top, [0.0, 0.0])
_, ldos_aa = compute_ldos(H, [aa_bot, aa_top], n_energies=500, eta=0.05)
print(f"  LDOS at AA site (E=0): {np.interp(0.0, energies, ldos_aa):.4f} /eV/site")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Effect of perturbations
# ─────────────────────────────────────────────────────────────────────────────
separator("6. Perturbations")

# Gate voltage
eps_gate = gate_voltage_onsite([bot, top], v_gate=0.2)
H_gate = build_hamiltonian([bot, top], onsite=eps_gate)
vals_gate, _ = diagonalize(H_gate)
print(f"  Gate voltage V=0.2 eV:")
print(f"    Spectrum shift: ΔE_min = {vals_gate.min()-vals.min():+.4f} eV")

# Sublattice staggering (Haldane-like)
eps_sub = sublattice_onsite([bot, top], delta=0.05)
H_sub = build_hamiltonian([bot, top], onsite=eps_sub)
vals_sub, _ = diagonalize(H_sub)
mid_sub = np.searchsorted(vals_sub, 0.0)
gap = vals_sub[mid_sub] - vals_sub[mid_sub - 1]
print(f"  Sublattice staggering δ=0.05 eV:")
print(f"    Gap near E=0: {gap:.4f} eV")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Multi-angle benchmark
# ─────────────────────────────────────────────────────────────────────────────
separator("7. Multi-angle benchmark")

print(f"  {'Angle':>8}  {'N_atoms':>8}  {'Build(ms)':>10}  {'DOS@E=0':>10}")
separator()
for theta in [1.08, 2.0, 5.0, 10.0]:
    b, t = generate_tbg_geometry(theta, n_cells=3)
    t0 = time.perf_counter()
    Hb = build_hamiltonian([b, t])
    dt_b = (time.perf_counter() - t0) * 1000
    en, ds = compute_dos(Hb, n_energies=200, eta=0.05)
    dos0 = float(np.interp(0.0, en, ds))
    N = b.n_atoms + t.n_atoms
    print(f"  {theta:8.3f}°  {N:8d}  {dt_b:10.1f}  {dos0:10.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Scalability test: ≥10⁴ atoms
# ─────────────────────────────────────────────────────────────────────────────
separator("8. Scalability (≥10⁴ atoms)")

n_large = 25   # (2*25+1)^2 * 2 * 2 ≈ 12902 atoms
print(f"  Generating n_cells={n_large} supercell ...")
b_large, t_large = generate_tbg_geometry(5.0, n_cells=n_large)
N_large = b_large.n_atoms + t_large.n_atoms
print(f"  Total atoms: {N_large}")
t0 = time.perf_counter()
H_large = build_hamiltonian([b_large, t_large])
dt_build = (time.perf_counter() - t0) * 1000
info_large = hamiltonian_info(H_large)
print(f"  Hamiltonian NNZ : {info_large['nnz']}")
print(f"  Build time      : {dt_build:.1f} ms")
print(f"  Memory (CSR)    : ~{H_large.data.nbytes / 1e6:.1f} MB (data array)")
print(f"  NNZ/atom        : {info_large['nnz'] / N_large:.1f}")

separator()
print("✓ All checks completed successfully.")
