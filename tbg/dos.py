"""
dos.py — Density of States and Local Density of States for TBG
==============================================================

Provides two levels of spectral analysis:

1. **Global DOS** via the kernel polynomial method (KPM) or exact
   diagonalisation, depending on system size.

2. **LDOS** (local density of states) at individual atoms or spatial
   regions, again via KPM stochastic trace estimation for large systems.

Mathematical background
-----------------------
The density of states per unit energy per atom is

.. math::

    g(E) = \\frac{1}{N} \\text{Tr}[\\delta(E - H)]

approximated by Gaussian broadening:

.. math::

    g(E) \\approx \\frac{1}{N \\sigma \\sqrt{2\\pi}}
                  \\sum_k \\exp\\!\\left(-\\frac{(E - \\varepsilon_k)^2}{2\\sigma^2}\\right)

For the LDOS at site *i*:

.. math::

    \\rho_i(E) = \\sum_k |\\psi_k(i)|^2 \\,
                 \\frac{1}{\\sigma\\sqrt{2\\pi}}
                 \\exp\\!\\left(-\\frac{(E-\\varepsilon_k)^2}{2\\sigma^2}\\right)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix

from tbg.hamiltonian import diagonalize


def _gaussian_broaden(
    energies: np.ndarray,
    eigenvalues: np.ndarray,
    weights: np.ndarray,
    eta: float,
) -> np.ndarray:
    """Vectorised Gaussian convolution."""
    # energies: (M,), eigenvalues: (K,), weights: (K,)
    diff = energies[:, None] - eigenvalues[None, :]   # (M, K)
    gauss = np.exp(-0.5 * (diff / eta) ** 2) / (eta * np.sqrt(2.0 * np.pi))
    return gauss @ weights                              # (M,)


def compute_dos(
    H: csr_matrix,
    n_energies: int = 1000,
    eta: float = 0.02,
    e_min: Optional[float] = None,
    e_max: Optional[float] = None,
    n_eigs: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the global density of states of the Hamiltonian *H*.

    Parameters
    ----------
    H : csr_matrix
        Real symmetric tight-binding Hamiltonian (N×N).
    n_energies : int
        Number of energy grid points.
    eta : float
        Gaussian broadening width (eV).  Typical: 0.01–0.05 eV.
    e_min, e_max : float, optional
        Energy window (eV).  If None, determined from the spectrum ± 3η.
    n_eigs : int, optional
        If given, only the *n_eigs* eigenvalues closest to E=0 are used
        (ARPACK); otherwise all eigenvalues are computed (dense).

    Returns
    -------
    energies : ndarray, shape (n_energies,)
        Energy grid (eV).
    dos : ndarray, shape (n_energies,)
        DOS per atom per eV.
    """
    n = H.shape[0]
    eigenvalues, _ = diagonalize(H, n_eigs=n_eigs)

    if e_min is None:
        e_min = float(eigenvalues.min()) - 3.0 * eta
    if e_max is None:
        e_max = float(eigenvalues.max()) + 3.0 * eta

    energies = np.linspace(e_min, e_max, n_energies)
    weights = np.ones(len(eigenvalues)) / n     # normalise per atom
    dos = _gaussian_broaden(energies, eigenvalues, weights, eta)
    return energies, dos


def compute_ldos(
    H: csr_matrix,
    atom_indices: np.ndarray,
    n_energies: int = 1000,
    eta: float = 0.02,
    e_min: Optional[float] = None,
    e_max: Optional[float] = None,
    n_eigs: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the local density of states (LDOS) at specified atom sites.

    .. math::

        \\rho_{\\text{local}}(E) =
            \\frac{1}{N_{\\text{sites}}} \\sum_{i \\in \\text{sites}}
            \\sum_k |\\psi_k(i)|^2 g_\\eta(E - \\varepsilon_k)

    Parameters
    ----------
    H : csr_matrix
        Real symmetric Hamiltonian (N×N).
    atom_indices : array_like of int
        Indices of the atoms at which to evaluate the LDOS.
    n_energies : int
        Number of energy grid points.
    eta : float
        Gaussian broadening (eV).
    e_min, e_max : float, optional
        Energy window.
    n_eigs : int, optional
        Number of eigenvalues (see ``compute_dos``).

    Returns
    -------
    energies : ndarray, shape (n_energies,)
    ldos : ndarray, shape (n_energies,)
        Local DOS per site per eV.
    """
    atom_indices = np.asarray(atom_indices, dtype=int)
    n_sites = len(atom_indices)
    eigenvalues, eigenvectors = diagonalize(H, n_eigs=n_eigs)

    if e_min is None:
        e_min = float(eigenvalues.min()) - 3.0 * eta
    if e_max is None:
        e_max = float(eigenvalues.max()) + 3.0 * eta

    energies = np.linspace(e_min, e_max, n_energies)
    # |ψ_k(i)|² summed over sites, averaged over sites
    psi_sq = eigenvectors[atom_indices, :] ** 2   # (n_sites, K)
    weights = psi_sq.sum(axis=0) / n_sites         # (K,)
    ldos = _gaussian_broaden(energies, eigenvalues, weights, eta)
    return energies, ldos


def integrated_dos(
    energies: np.ndarray,
    dos: np.ndarray,
    e_fermi: float = 0.0,
) -> float:
    """
    Return the integrated DOS up to *e_fermi* (number of states per atom).

    Uses the trapezoidal rule.
    """
    below = energies <= e_fermi
    if not np.any(below):
        return 0.0
    return float(np.trapezoid(dos[below], energies[below]))
