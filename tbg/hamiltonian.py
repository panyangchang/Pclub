"""
hamiltonian.py — Real-space tight-binding Hamiltonian for TBG
=============================================================

Assembles the sparse Hermitian Hamiltonian matrix H for an arbitrary-angle
twisted bilayer graphene geometry.

Algorithm overview
------------------
1.  Stack all atom coordinates from both layers into a single (N, 3) array.
2.  Build a KD-tree from the in-plane coordinates to efficiently find
    *intralayer* neighbours (same z) and a separate KD-tree for
    *interlayer* pairs.
3.  For each bond within the respective cutoff radius, evaluate
    t(r_ij) via `hopping.hopping_batch` and record (row, col, value)
    triplets.
4.  Assemble a `scipy.sparse.csr_matrix` from the triplets.

Memory scalability
------------------
The number of non-zero entries scales as O(N) (sparse), so the approach
handles ≥ 10⁴ atoms.  For N = 10⁴ atoms the full matrix would require
~800 GB (dense float64), but the sparse representation needs only a few
MB for typical cutoff radii.

Matrix-element expression
-------------------------
For atoms i (position **r**_i) and j (position **r**_j) with
**r**_ij = **r**_j − **r**_i:

.. math::

    H_{ij} = t(\\mathbf{r}_{ij}) =
              n^2 V_{pp\\sigma}(|\\mathbf{r}_{ij}|)
              + (1 - n^2) V_{pp\\pi}(|\\mathbf{r}_{ij}|)

where n = (r_ij)_z / |r_ij|.  Diagonal entries H_{ii} = ε_i (on-site
energy, default 0 eV).

Cutoffs
-------
``r_cut_intra`` (default 3·a_cc ≈ 4.26 Å) includes NN and NNN intralayer
bonds.  ``r_cut_inter`` (default 4.5·a_cc ≈ 6.39 Å) captures all
interlayer bonds with amplitude > ~ 0.01 eV.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, lil_matrix

from tbg.lattice import GrapheneLayer, A_CC
from tbg.hopping import (
    hopping_batch,
    R_CUT_INTRA,
    R_CUT_INTER,
    V_PP_PI,
    V_PP_SIGMA,
)


def build_hamiltonian(
    layers: Sequence[GrapheneLayer],
    r_cut_intra: float = R_CUT_INTRA,
    r_cut_inter: float = R_CUT_INTER,
    onsite: Optional[np.ndarray] = None,
    v_pp_pi0: float = V_PP_PI,
    v_pp_sigma0: float = V_PP_SIGMA,
) -> csr_matrix:
    """
    Assemble the real-space tight-binding Hamiltonian as a sparse matrix.

    Parameters
    ----------
    layers : sequence of GrapheneLayer
        The two (or more) graphene layers.  Typically the tuple returned by
        ``generate_tbg_geometry``.
    r_cut_intra : float
        Intralayer hopping cutoff (Å).  Default: 3·a_cc (NN + NNN).
    r_cut_inter : float
        Interlayer hopping cutoff (Å).  Default: 4.5·a_cc.
    onsite : ndarray of float, shape (N_total,), optional
        On-site energies (eV) for each atom.  Defaults to all zeros.
        Can encode sublattice, strain, electrostatic gate, or impurity
        potentials.
    v_pp_pi0 : float
        π Slater–Koster amplitude (eV).  Default: −2.7.
    v_pp_sigma0 : float
        σ Slater–Koster amplitude (eV).  Default: +0.48.

    Returns
    -------
    H : scipy.sparse.csr_matrix, shape (N, N), dtype float64
        Real symmetric Hamiltonian.  All entries are real because on-site
        energies and hopping amplitudes are real in this model.

    Raises
    ------
    ValueError
        If ``layers`` is empty or ``onsite`` has the wrong length.

    Notes
    -----
    For very large systems (N > 10⁵) use the ``lil_matrix`` accumulator
    and convert to CSR at the end; or use the COO triplet approach (the
    default here) which is already memory-efficient.

    Pseudocode
    ----------
    ::

        # Step 1 — concatenate all positions
        pos = vstack([layer.positions for layer in layers])   # (N, 3)

        # Step 2 — separate layers by z-coordinate
        for each layer pair (same z = intralayer, different z = interlayer):
            build KD-tree of positions in layer k
            for each atom i in layer l:
                find neighbours j within r_cut
                compute r_ij = pos[j] - pos[i]
                t_ij = hopping_batch(r_ij)
                record (i, j, t_ij) triplets

        # Step 3 — assemble sparse matrix
        H = csr_matrix((values, (rows, cols)), shape=(N, N))
        H = (H + H.T) / 2    # enforce exact symmetry

        # Step 4 — add on-site energies
        H.setdiag(onsite)
    """
    if not layers:
        raise ValueError("layers must be non-empty.")

    # ── 1. Concatenate all atom positions ──────────────────────────────────
    pos_list = [layer.positions for layer in layers]
    pos = np.vstack(pos_list)          # (N_total, 3)
    n_total = pos.shape[0]

    if onsite is None:
        onsite = np.zeros(n_total, dtype=float)
    else:
        onsite = np.asarray(onsite, dtype=float)
        if onsite.shape != (n_total,):
            raise ValueError(
                f"onsite must have length {n_total}, got {onsite.shape}."
            )

    rows_list: list[np.ndarray] = []
    cols_list: list[np.ndarray] = []
    vals_list: list[np.ndarray] = []

    # ── 2. Group layers by z-height ────────────────────────────────────────
    unique_z = list({float(layer.z) for layer in layers})
    z_to_indices: dict[float, np.ndarray] = {}
    for z in unique_z:
        mask = np.abs(pos[:, 2] - z) < 0.1
        z_to_indices[z] = np.where(mask)[0]

    # ── 3. Intralayer hoppings ─────────────────────────────────────────────
    for z, idx in z_to_indices.items():
        if len(idx) == 0:
            continue
        xy = pos[idx, :2]
        tree = cKDTree(xy)
        # query_pairs returns pairs with distance in (0, r_cut_intra]
        pairs = tree.query_pairs(r_cut_intra, output_type="ndarray")
        if pairs.size == 0:
            continue
        # local indices → global indices
        i_global = idx[pairs[:, 0]]
        j_global = idx[pairs[:, 1]]
        r_ij = pos[j_global] - pos[i_global]        # (M, 3)  z-component = 0
        t_vals = hopping_batch(r_ij, v_pp_pi0=v_pp_pi0, v_pp_sigma0=v_pp_sigma0)
        # exclude negligible hoppings (|t| < 1e-6 eV)
        significant = np.abs(t_vals) > 1e-6
        rows_list.append(i_global[significant])
        cols_list.append(j_global[significant])
        vals_list.append(t_vals[significant])

    # ── 4. Interlayer hoppings ─────────────────────────────────────────────
    z_vals = list(z_to_indices.keys())
    for k, z1 in enumerate(z_vals):
        for z2 in z_vals[k + 1 :]:
            idx1 = z_to_indices[z1]
            idx2 = z_to_indices[z2]
            if len(idx1) == 0 or len(idx2) == 0:
                continue
            # Use 3-D KD-tree to find interlayer pairs within r_cut_inter
            tree2 = cKDTree(pos[idx2])
            results = tree2.query_ball_point(pos[idx1], r=r_cut_inter)
            i_locals, j_locals = [], []
            for local_i, neighbours in enumerate(results):
                if neighbours:
                    i_locals.extend([local_i] * len(neighbours))
                    j_locals.extend(neighbours)
            if not i_locals:
                continue
            i_locals_arr = np.array(i_locals)
            j_locals_arr = np.array(j_locals)
            i_global = idx1[i_locals_arr]
            j_global = idx2[j_locals_arr]
            r_ij = pos[j_global] - pos[i_global]
            t_vals = hopping_batch(r_ij, v_pp_pi0=v_pp_pi0, v_pp_sigma0=v_pp_sigma0)
            significant = np.abs(t_vals) > 1e-6
            rows_list.append(i_global[significant])
            cols_list.append(j_global[significant])
            vals_list.append(t_vals[significant])

    # ── 5. Assemble sparse matrix ──────────────────────────────────────────
    if rows_list:
        all_rows = np.concatenate(rows_list)
        all_cols = np.concatenate(cols_list)
        all_vals = np.concatenate(vals_list)
        # Upper-triangle entries; mirror for lower triangle
        all_rows_full = np.concatenate([all_rows, all_cols])
        all_cols_full = np.concatenate([all_cols, all_rows])
        all_vals_full = np.concatenate([all_vals, all_vals])
        H = csr_matrix(
            (all_vals_full, (all_rows_full, all_cols_full)),
            shape=(n_total, n_total),
            dtype=float,
        )
        # Sum duplicate entries and enforce exact symmetry
        H = (H + H.T) / 2.0
    else:
        H = csr_matrix((n_total, n_total), dtype=float)

    # ── 6. Add on-site energies ────────────────────────────────────────────
    H.setdiag(H.diagonal() + onsite)

    return H


def hamiltonian_info(H: csr_matrix) -> dict:
    """
    Return a dictionary of basic properties of the Hamiltonian matrix.

    Parameters
    ----------
    H : scipy.sparse.csr_matrix

    Returns
    -------
    dict with keys:
        * ``n_atoms``    — matrix dimension
        * ``nnz``        — number of non-zero off-diagonal entries
        * ``is_symmetric`` — whether max|H-H.T| < 1e-10
        * ``sparsity``   — fraction of zero entries
        * ``min_hopping``, ``max_hopping`` — off-diagonal extrema (eV)
    """
    n = H.shape[0]
    residual = float((H - H.T).data.max()) if (H - H.T).nnz else 0.0
    off = H.copy()
    off.setdiag(0)
    off.eliminate_zeros()
    nnz = off.nnz // 2        # count each bond once
    return {
        "n_atoms": n,
        "nnz": nnz,
        "is_symmetric": abs(residual) < 1e-10,
        "sparsity": 1.0 - H.nnz / (n * n),
        "min_hopping": float(off.data.min()) if off.nnz else 0.0,
        "max_hopping": float(off.data.max()) if off.nnz else 0.0,
    }


def diagonalize(
    H: csr_matrix,
    n_eigs: Optional[int] = None,
    sigma: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Diagonalise the Hamiltonian and return eigenvalues and eigenvectors.

    For small systems (N ≤ 2000) uses ``scipy.linalg.eigh`` (dense).
    For larger systems uses ``scipy.sparse.linalg.eigsh`` (ARPACK).

    Parameters
    ----------
    H : scipy.sparse.csr_matrix
        Real symmetric Hamiltonian.
    n_eigs : int, optional
        Number of eigenvalues to compute (ARPACK only).  If None, all
        eigenvalues are computed (dense solver).
    sigma : float
        Shift for ARPACK shift-invert mode.  Eigenvalues near *sigma*
        are found first.

    Returns
    -------
    eigenvalues : ndarray, shape (k,)
        Sorted eigenvalues in eV.
    eigenvectors : ndarray, shape (N, k)
        Corresponding eigenvectors.
    """
    import scipy.linalg
    import scipy.sparse.linalg

    n = H.shape[0]
    use_dense = (n_eigs is None) or (n <= 2000)
    if use_dense:
        H_dense = H.toarray()
        vals, vecs = scipy.linalg.eigh(H_dense)
    else:
        k = min(n_eigs, n - 2)
        vals, vecs = scipy.sparse.linalg.eigsh(H, k=k, sigma=sigma, which="LM")
        order = np.argsort(vals)
        vals, vecs = vals[order], vecs[:, order]
    return vals, vecs
