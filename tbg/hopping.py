"""
hopping.py — Hopping parameters for twisted bilayer graphene
=============================================================

Implements the Moon–Koshino Slater–Koster model [PRB 87, 205404 (2013)],
which handles both intralayer and interlayer hopping through a unified
position-space formula.

Mathematical summary
--------------------
The transfer-integral between two pz orbitals at positions **r**_i and
**r**_j (with **r**_ij = **r**_j − **r**_i) is

.. math::

    t(\\mathbf{r}_{ij}) =
        n^2 \\, V_{pp\\sigma}(|\\mathbf{r}_{ij}|)
        + (1 - n^2) \\, V_{pp\\pi}(|\\mathbf{r}_{ij}|)

where

* :math:`n = r_{ij,z} / |\\mathbf{r}_{ij}|` is the direction cosine
  along z (the pz-orbital axis).

The radial functions are exponentially decaying Slater–Koster integrals:

.. math::

    V_{pp\\pi}(r) = V_{pp\\pi}^0 \\,
        \\exp\\!\\left(-\\frac{r - a_{cc}}{\\delta_0}\\right)

    V_{pp\\sigma}(r) = V_{pp\\sigma}^0 \\,
        \\exp\\!\\left(-\\frac{r - d_0}{\\delta_0}\\right)

Default parameters (Moon & Koshino 2013)
-----------------------------------------
+------------------+----------+--------------------------------------+
| Symbol           | Value    | Meaning                              |
+==================+==========+======================================+
| V_ppπ⁰           | −2.7 eV | nearest-neighbour intralayer hopping |
+------------------+----------+--------------------------------------+
| V_ppσ⁰           | +0.48 eV| interlayer hopping (AA stacking)     |
+------------------+----------+--------------------------------------+
| a_cc             | 1.42 Å  | C–C bond length                      |
+------------------+----------+--------------------------------------+
| d_0              | 3.35 Å  | interlayer distance                  |
+------------------+----------+--------------------------------------+
| δ_0              | 0.184 a | decay length (a = √3·a_cc)           |
+------------------+----------+--------------------------------------+

Cutoffs
-------
* Intralayer: r_cut = 3.0·a_cc  (includes NN and NNN)
* Interlayer: r_cut = 4.0·a_cc  (≈ 5.7 Å, keeps ≈ 20 interlayer pairs)

All values are in Å (lengths) and eV (energies).
"""

from __future__ import annotations

import numpy as np

from tbg.lattice import A_CC, A_GRAPHENE, D_LAYER

# ── Default tight-binding parameters ─────────────────────────────────────────
#: Intralayer π hopping (nearest-neighbour), eV
V_PP_PI: float = -2.7
#: Interlayer σ hopping (AA-stacking), eV
V_PP_SIGMA: float = 0.48
#: Decay length δ₀ = 0.184·a  (Å)
DELTA_0: float = 0.184 * A_GRAPHENE
#: Intralayer cutoff (NN + NNN), Å
R_CUT_INTRA: float = 3.0 * A_CC
#: Interlayer cutoff, Å
R_CUT_INTER: float = 4.5 * A_CC


def v_pp_pi(r: np.ndarray, v_pp_pi0: float = V_PP_PI) -> np.ndarray:
    """
    Radial part of the π Slater–Koster integral.

    .. math::
        V_{pp\\pi}(r) = V_{pp\\pi}^0 \\exp\\!\\left(-\\frac{r-a_{cc}}{\\delta_0}\\right)

    Parameters
    ----------
    r : array_like
        Inter-atomic distance(s) (Å).
    v_pp_pi0 : float
        Nearest-neighbour intralayer hopping (eV).  Default −2.7 eV.

    Returns
    -------
    ndarray
        V_ppπ(r) in eV.
    """
    r = np.asarray(r, dtype=float)
    return v_pp_pi0 * np.exp(-(r - A_CC) / DELTA_0)


def v_pp_sigma(r: np.ndarray, v_pp_sigma0: float = V_PP_SIGMA) -> np.ndarray:
    """
    Radial part of the σ Slater–Koster integral.

    .. math::
        V_{pp\\sigma}(r) = V_{pp\\sigma}^0 \\exp\\!\\left(-\\frac{r-d_0}{\\delta_0}\\right)

    Parameters
    ----------
    r : array_like
        Inter-atomic distance(s) (Å).
    v_pp_sigma0 : float
        Interlayer hopping at AA stacking (eV).  Default +0.48 eV.

    Returns
    -------
    ndarray
        V_ppσ(r) in eV.
    """
    r = np.asarray(r, dtype=float)
    return v_pp_sigma0 * np.exp(-(r - D_LAYER) / DELTA_0)


def hopping(
    r_ij: np.ndarray,
    *,
    v_pp_pi0: float = V_PP_PI,
    v_pp_sigma0: float = V_PP_SIGMA,
) -> float:
    """
    Slater–Koster transfer integral between two pz orbitals.

    .. math::
        t(\\mathbf{r}_{ij}) =
            n^2 V_{pp\\sigma}(|\\mathbf{r}_{ij}|)
            + (1-n^2) V_{pp\\pi}(|\\mathbf{r}_{ij}|)

    with :math:`n = r_{ij,z} / |\\mathbf{r}_{ij}|`.

    For **same-layer** bonds n = 0, so only the π term survives:
    t = V_ppπ(r_xy), which recovers the standard graphene tight-binding.

    For **inter-layer** bonds n ≈ d_0 / |r_ij|.

    Parameters
    ----------
    r_ij : array_like, shape (3,)
        Displacement vector **r**_j − **r**_i in Å.
    v_pp_pi0 : float
        Nearest-neighbour intralayer hopping (eV).
    v_pp_sigma0 : float
        Interlayer hopping at AA stacking (eV).

    Returns
    -------
    float
        Hopping amplitude t (eV).  Returns 0.0 if |r_ij| < 1e-6 (self-loop).
    """
    r_ij = np.asarray(r_ij, dtype=float)
    dist = float(np.linalg.norm(r_ij))
    if dist < 1e-6:
        return 0.0
    n = r_ij[2] / dist                          # direction cosine along z
    vpi = float(v_pp_pi(dist, v_pp_pi0))
    vsi = float(v_pp_sigma(dist, v_pp_sigma0))
    return n**2 * vsi + (1.0 - n**2) * vpi


def hopping_batch(
    r_ij_batch: np.ndarray,
    *,
    v_pp_pi0: float = V_PP_PI,
    v_pp_sigma0: float = V_PP_SIGMA,
) -> np.ndarray:
    """
    Vectorised Slater–Koster hopping for multiple bonds.

    Parameters
    ----------
    r_ij_batch : ndarray, shape (M, 3)
        Array of displacement vectors (Å).
    v_pp_pi0, v_pp_sigma0 : float
        Tight-binding parameters (eV).

    Returns
    -------
    ndarray, shape (M,)
        Hopping amplitudes (eV).
    """
    r_ij_batch = np.asarray(r_ij_batch, dtype=float)
    dists = np.linalg.norm(r_ij_batch, axis=1)
    safe = dists > 1e-6
    n2 = np.where(safe, (r_ij_batch[:, 2] / np.where(safe, dists, 1.0)) ** 2, 0.0)
    vpi = v_pp_pi(dists, v_pp_pi0)
    vsi = v_pp_sigma(dists, v_pp_sigma0)
    return n2 * vsi + (1.0 - n2) * vpi
