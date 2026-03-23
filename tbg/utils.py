"""
utils.py — Utility functions for TBG tight-binding calculations
===============================================================

Contains helpers for:
* Moiré period and reciprocal lattice computations
* Periodicity validation
* Nearest-neighbour statistics
* On-site energy arrays for common perturbations (strain, gate voltage)
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

from tbg.lattice import GrapheneLayer, A_CC, A_GRAPHENE, D_LAYER


# ── Moiré geometry ───────────────────────────────────────────────────────────
def moire_period(twist_angle_deg: float) -> float:
    """
    Return the moiré super-lattice period λ (Å) for a given twist angle.

    .. math::

        \\lambda = \\frac{a}{2 \\sin(\\theta / 2)}

    where a = A_GRAPHENE is the graphene lattice constant.

    Parameters
    ----------
    twist_angle_deg : float
        Twist angle θ in degrees.

    Returns
    -------
    float
        Moiré period λ in Å.
    """
    theta = np.radians(twist_angle_deg)
    if abs(np.sin(theta / 2.0)) < 1e-12:
        return np.inf
    return A_GRAPHENE / (2.0 * np.sin(theta / 2.0))


def moire_reciprocal_vectors(
    twist_angle_deg: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Moiré Brillouin zone primitive reciprocal vectors (Å⁻¹).

    .. math::

        \\mathbf{G}_1^M = \\mathbf{b}_1^{(1)} - \\mathbf{b}_1^{(2)}, \\quad
        \\mathbf{G}_2^M = \\mathbf{b}_2^{(1)} - \\mathbf{b}_2^{(2)}

    where b_i^{(l)} are the reciprocal lattice vectors of layer l.

    Returns
    -------
    G1, G2 : ndarray, shape (2,)
        Moiré reciprocal lattice vectors in Å⁻¹.
    """
    theta = np.radians(twist_angle_deg)
    a = A_GRAPHENE
    # Reciprocal lattice vectors of graphene (2π convention)
    b1_0 = (2.0 * np.pi / a) * np.array([1.0, -1.0 / np.sqrt(3.0)])
    b2_0 = (2.0 * np.pi / a) * np.array([0.0, 2.0 / np.sqrt(3.0)])

    R_pos = np.array([[np.cos(theta / 2), -np.sin(theta / 2)],
                      [np.sin(theta / 2),  np.cos(theta / 2)]])
    R_neg = np.array([[np.cos(theta / 2),  np.sin(theta / 2)],
                      [-np.sin(theta / 2), np.cos(theta / 2)]])

    b1_top = R_pos @ b1_0
    b1_bot = R_neg @ b1_0
    b2_top = R_pos @ b2_0
    b2_bot = R_neg @ b2_0

    G1 = b1_top - b1_bot
    G2 = b2_top - b2_bot
    return G1, G2


# ── Validation helpers ────────────────────────────────────────────────────────
def check_nn_distances(layer: GrapheneLayer, tolerance: float = 0.05) -> dict:
    """
    Verify that nearest-neighbour distances in *layer* are consistent with
    a pure graphene lattice (|r_NN| = a_cc within *tolerance* Å).

    Returns a dict with keys ``min_nn``, ``max_nn``, ``expected``,
    ``passed``.
    """
    tree = cKDTree(layer.xy)
    dists, _ = tree.query(layer.xy, k=2)   # k=1 is self; k=2 is NN
    nn_dists = dists[:, 1]
    result = {
        "min_nn": float(nn_dists.min()),
        "max_nn": float(nn_dists.max()),
        "expected": A_CC,
        "passed": bool(
            abs(nn_dists.min() - A_CC) < tolerance
            and abs(nn_dists.max() - A_CC) < tolerance
        ),
    }
    return result


def check_atom_count(layer: GrapheneLayer, n_cells: int) -> dict:
    """
    Check that a rectangular supercell contains exactly 2*(2n+1)² atoms.

    Returns a dict with keys ``expected``, ``actual``, ``passed``.
    """
    expected = 2 * (2 * n_cells + 1) ** 2
    actual = layer.n_atoms
    return {"expected": expected, "actual": actual, "passed": actual == expected}


def symmetry_check(H) -> dict:
    """
    Check that H is real and symmetric.

    Parameters
    ----------
    H : scipy sparse matrix

    Returns
    -------
    dict with keys ``is_real``, ``max_asymmetry``, ``passed``.
    """
    diff = H - H.T
    max_asym = float(abs(diff).max()) if diff.nnz > 0 else 0.0
    is_real = np.isrealobj(H.data)
    return {
        "is_real": is_real,
        "max_asymmetry": max_asym,
        "passed": is_real and max_asym < 1e-10,
    }


# ── On-site energy helpers ─────────────────────────────────────────────────-
def sublattice_onsite(
    layers: Sequence[GrapheneLayer],
    delta: float = 0.0,
) -> np.ndarray:
    """
    Build an on-site energy array with a sublattice staggering +δ (A) / −δ (B)
    and no inter-layer asymmetry.

    Parameters
    ----------
    layers : sequence of GrapheneLayer
    delta : float
        Staggering energy in eV.

    Returns
    -------
    ndarray, shape (N_total,)
    """
    n_total = sum(lay.n_atoms for lay in layers)
    eps = np.zeros(n_total)
    for lay in layers:
        for local_i in range(lay.n_atoms):
            gid = int(lay.atom_ids[local_i])
            sub = int(lay.sublattices[local_i])
            eps[gid] = delta if sub == 0 else -delta
    return eps


def gate_voltage_onsite(
    layers: Sequence[GrapheneLayer],
    v_gate: float = 0.0,
) -> np.ndarray:
    """
    Apply a gate voltage: +V/2 to layer 0, −V/2 to layer 1.

    Parameters
    ----------
    layers : sequence of GrapheneLayer
    v_gate : float
        Gate voltage V in eV.

    Returns
    -------
    ndarray, shape (N_total,)
    """
    n_total = sum(lay.n_atoms for lay in layers)
    eps = np.zeros(n_total)
    for lay in layers:
        sign = 1.0 if lay.layer_id == 0 else -1.0
        for gid in lay.atom_ids:
            eps[int(gid)] = sign * v_gate / 2.0
    return eps
