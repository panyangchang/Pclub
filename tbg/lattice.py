"""
lattice.py — Atomic geometry generation for twisted bilayer graphene
====================================================================

Provides functions to generate the two-layer graphene structure for
arbitrary twist angles, including incommensurate (non-magic) angles.

Atom-indexing strategy
----------------------
Every carbon atom receives a unique integer ``atom_id``:

* **Bottom layer (layer 0)**: ids 0 … N_bottom-1
* **Top    layer (layer 1)**: ids N_bottom … N_total-1

Within each layer, atoms are sorted by unit-cell indices (n1, n2) and
sublattice (A=0, B=1), giving a reproducible ordering that is
independent of twist angle.  The mapping

    atom_id  ↔  (layer_id, sublattice, n1, n2)

is stored in the ``GrapheneLayer.atom_table`` attribute so that defects,
strain, or edge modulations can be applied by simply look-up.

Physical constants (all lengths in Ångström, energies in eV)
-------------------------------------------------------------
a_cc  = 1.42 Å   C–C bond length
a     = √3·a_cc  graphene lattice constant
d_0   = 3.35 Å   interlayer separation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Optional
import numpy as np

# ── Physical constants ──────────────────────────────────────────────────────
A_CC: float = 1.42          # Å  C–C nearest-neighbour bond
A_GRAPHENE: float = A_CC * np.sqrt(3)  # Å  graphene lattice constant
D_LAYER: float = 3.35       # Å  default interlayer distance


# ── Helper: rotation matrix ─────────────────────────────────────────────────
def rotation_matrix_2d(theta: float) -> np.ndarray:
    """Return the 2×2 rotation matrix for angle *theta* (radians)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


# ── Lattice vectors and basis ────────────────────────────────────────────────
def graphene_lattice_vectors(angle: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return the two primitive lattice vectors of graphene, rotated by *angle*
    (radians).

    Convention::

        a1 = a (1, 0)
        a2 = a (1/2, √3/2)

    with a = A_GRAPHENE = √3·a_cc.
    """
    a = A_GRAPHENE
    a1_0 = a * np.array([1.0, 0.0])
    a2_0 = a * np.array([0.5, np.sqrt(3.0) / 2.0])
    R = rotation_matrix_2d(angle)
    return R @ a1_0, R @ a2_0


def graphene_basis_vectors(
    angle: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return the two-atom basis positions (sublattice A and B) of graphene,
    rotated by *angle* (radians).

    With the above convention::

        τ_A = (0, 0)
        τ_B = (a1 + a2) / 3       |τ_B| = a/√3 = a_cc ✓
    """
    a1, a2 = graphene_lattice_vectors(angle)
    tau_A = np.zeros(2)
    tau_B = (a1 + a2) / 3.0
    return tau_A, tau_B


# ── Core data structure ──────────────────────────────────────────────────────
@dataclass
class GrapheneLayer:
    """
    Container for all atomic data belonging to one graphene layer.

    Attributes
    ----------
    layer_id : int
        0 = bottom, 1 = top.
    angle : float
        Rotation angle of this layer (radians).
    z : float
        Height of the layer (Å).
    positions : ndarray, shape (N, 3)
        3-D Cartesian coordinates of all atoms (Å).
    sublattices : ndarray of int, shape (N,)
        Sublattice label: 0 = A, 1 = B.
    cell_indices : ndarray of int, shape (N, 2)
        Unit-cell indices (n1, n2) for each atom.
    atom_ids : ndarray of int, shape (N,)
        Globally unique integer IDs (offset by *global_offset*).
    atom_table : dict
        Reverse map (layer_id, sublattice, n1, n2) → atom_id.
    """

    layer_id: int
    angle: float
    z: float
    positions: np.ndarray
    sublattices: np.ndarray
    cell_indices: np.ndarray
    atom_ids: np.ndarray
    atom_table: dict = field(default_factory=dict)

    @property
    def n_atoms(self) -> int:
        """Number of atoms in this layer."""
        return len(self.positions)

    @property
    def xy(self) -> np.ndarray:
        """In-plane (x, y) coordinates, shape (N, 2)."""
        return self.positions[:, :2]

    @property
    def lattice_vectors(self) -> Tuple[np.ndarray, np.ndarray]:
        """Rotated lattice vectors of this layer."""
        return graphene_lattice_vectors(self.angle)

    def local_index(self, atom_id: int) -> int:
        """Convert global *atom_id* to local index within this layer."""
        return int(atom_id - self.atom_ids[0])


# ── Layer generator ──────────────────────────────────────────────────────────
def _generate_layer(
    angle: float,
    layer_id: int,
    n1_range: Tuple[int, int],
    n2_range: Tuple[int, int],
    z_height: float,
    global_offset: int,
    radius: Optional[float] = None,
) -> GrapheneLayer:
    """
    Generate all atoms for one graphene layer within a rectangular supercell
    (optionally trimmed to a circular disk of radius *radius*).

    Parameters
    ----------
    angle : float
        Layer rotation angle (radians).
    layer_id : int
        0 (bottom) or 1 (top).
    n1_range, n2_range : (int, int)
        Inclusive range of unit-cell indices along a1 / a2.
    z_height : float
        z-coordinate of the layer (Å).
    global_offset : int
        Starting global atom index (= 0 for bottom, N_bottom for top).
    radius : float, optional
        If given, only atoms with in-plane distance from origin ≤ *radius*
        are retained.  Useful for producing disk-shaped samples with a
        well-defined moiré period.

    Returns
    -------
    GrapheneLayer
    """
    a1, a2 = graphene_lattice_vectors(angle)
    tau_A, tau_B = graphene_basis_vectors(angle)

    positions = []
    sublattices = []
    cell_indices = []

    n1_min, n1_max = n1_range
    n2_min, n2_max = n2_range

    for n1 in range(n1_min, n1_max + 1):
        for n2 in range(n2_min, n2_max + 1):
            R = n1 * a1 + n2 * a2
            for sub, tau in enumerate([tau_A, tau_B]):
                pos_2d = R + tau
                if radius is not None and np.linalg.norm(pos_2d) > radius:
                    continue
                positions.append([pos_2d[0], pos_2d[1], z_height])
                sublattices.append(sub)
                cell_indices.append([n1, n2])

    n_atoms = len(positions)
    positions_arr = np.array(positions, dtype=float) if n_atoms else np.zeros((0, 3))
    sublattices_arr = np.array(sublattices, dtype=int)
    cell_indices_arr = np.array(cell_indices, dtype=int) if n_atoms else np.zeros((0, 2), dtype=int)
    atom_ids_arr = np.arange(n_atoms, dtype=int) + global_offset

    # Build reverse-lookup table
    atom_table: dict = {}
    for local_i in range(n_atoms):
        n1, n2 = cell_indices_arr[local_i]
        sub = sublattices_arr[local_i]
        atom_table[(layer_id, int(sub), int(n1), int(n2))] = int(atom_ids_arr[local_i])

    return GrapheneLayer(
        layer_id=layer_id,
        angle=angle,
        z=z_height,
        positions=positions_arr,
        sublattices=sublattices_arr,
        cell_indices=cell_indices_arr,
        atom_ids=atom_ids_arr,
        atom_table=atom_table,
    )


# ── Public API ───────────────────────────────────────────────────────────────
def generate_tbg_geometry(
    twist_angle_deg: float,
    n_cells: int = 5,
    interlayer_distance: float = D_LAYER,
    center_rotation: bool = True,
    disk_radius: Optional[float] = None,
) -> Tuple[GrapheneLayer, GrapheneLayer]:
    """
    Generate the atomic structure for twisted bilayer graphene at an
    arbitrary twist angle (including incommensurate angles).

    The strategy works for **any** angle because it constructs atoms in each
    layer independently (no requirement for a commensurate supercell) and
    relies purely on real-space coordinates for subsequent Hamiltonian
    assembly.

    Parameters
    ----------
    twist_angle_deg : float
        Twist angle in degrees.  Valid for 0 < θ ≤ 30°.
    n_cells : int
        Half-width of the supercell: generates a (2n+1)×(2n+1) unit-cell
        grid in each layer.  Larger values give more atoms and better
        moiré sampling; n=5 ≈ 242 atoms/layer.
    interlayer_distance : float
        Vertical separation between layers (Å).
    center_rotation : bool
        If True (default), rotate bottom by −θ/2 and top by +θ/2 so the
        twist is symmetric about the AA-stacking origin.
        If False, the bottom layer is unrotated (θ = 0) and the top layer
        is rotated by +θ.
    disk_radius : float, optional
        If given, only atoms within this in-plane radius (Å) are kept.
        Useful for producing circular samples that expose a well-defined
        moiré period.  If None, a square supercell is used.

    Returns
    -------
    bottom_layer : GrapheneLayer
        Layer 0 — globally indexed atoms 0 … N_bottom-1.
    top_layer : GrapheneLayer
        Layer 1 — globally indexed atoms N_bottom … N_total-1.

    Notes
    -----
    Atom-ID uniqueness
    ~~~~~~~~~~~~~~~~~~
    Each atom carries a globally unique ``atom_id`` integer.  Within a
    layer the ordering is deterministic: unit cells are iterated in
    (n1, n2) lexicographic order, and within each cell sublattice A
    precedes B.  This makes it straightforward to insert defects, local
    strain, or edge terminations by manipulating the relevant rows/columns
    of the Hamiltonian without breaking the indexing of other atoms.
    """
    theta = np.radians(twist_angle_deg)

    if center_rotation:
        angle_bottom = -theta / 2.0
        angle_top = theta / 2.0
    else:
        angle_bottom = 0.0
        angle_top = theta

    n = int(n_cells)
    bottom_layer = _generate_layer(
        angle=angle_bottom,
        layer_id=0,
        n1_range=(-n, n),
        n2_range=(-n, n),
        z_height=0.0,
        global_offset=0,
        radius=disk_radius,
    )

    top_layer = _generate_layer(
        angle=angle_top,
        layer_id=1,
        n1_range=(-n, n),
        n2_range=(-n, n),
        z_height=interlayer_distance,
        global_offset=bottom_layer.n_atoms,
        radius=disk_radius,
    )

    return bottom_layer, top_layer


def commensurate_indices(m: int, n: int) -> Tuple[int, int]:
    """
    Return the (m, n) Bravais–Coprime integer pair that labels a
    commensurate TBG supercell.

    The twist angle is given by [Lopes dos Santos et al., PRL 2007]::

        cos θ = (3m² + 3m + 1/2) / (3m² + 3m + 1)       [n = m+1]

    or more generally::

        cos θ = (n² + 4mn + m²) / (2(n² + mn + m²))

    Parameters
    ----------
    m, n : int
        Non-negative integers with gcd(m, n) = 1 and m ≠ n.

    Returns
    -------
    twist_angle_deg : float
        Commensurate twist angle in degrees.
    n_atoms_per_layer : int
        Number of atoms per layer in the minimal supercell.
    """
    m2_mn_n2 = m**2 + m * n + n**2
    cos_theta = (m**2 + 4 * m * n + n**2) / (2 * m2_mn_n2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta_deg = np.degrees(np.arccos(cos_theta))
    n_atoms = 4 * m2_mn_n2
    return theta_deg, n_atoms
