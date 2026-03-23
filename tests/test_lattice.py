"""
tests/test_lattice.py
=====================
Unit tests for tbg.lattice — atom generation and indexing.
"""

import numpy as np
import pytest

from tbg.lattice import (
    A_CC,
    A_GRAPHENE,
    D_LAYER,
    GrapheneLayer,
    commensurate_indices,
    generate_tbg_geometry,
    graphene_basis_vectors,
    graphene_lattice_vectors,
    rotation_matrix_2d,
)


# ── Rotation matrix ──────────────────────────────────────────────────────────
class TestRotationMatrix:
    def test_identity_at_zero(self):
        R = rotation_matrix_2d(0.0)
        np.testing.assert_allclose(R, np.eye(2), atol=1e-14)

    def test_ninety_degrees(self):
        R = rotation_matrix_2d(np.pi / 2)
        expected = np.array([[0, -1], [1, 0]], dtype=float)
        np.testing.assert_allclose(R, expected, atol=1e-14)

    def test_orthogonal(self):
        for theta in [0.1, 1.08, 5.0, 21.787]:
            R = rotation_matrix_2d(np.radians(theta))
            np.testing.assert_allclose(R @ R.T, np.eye(2), atol=1e-13)

    def test_det_one(self):
        for theta in [0.5, 3.0, 15.0]:
            R = rotation_matrix_2d(np.radians(theta))
            assert abs(np.linalg.det(R) - 1.0) < 1e-13


# ── Lattice and basis vectors ─────────────────────────────────────────────────
class TestLatticeVectors:
    def test_magnitude(self):
        a1, a2 = graphene_lattice_vectors(0.0)
        assert abs(np.linalg.norm(a1) - A_GRAPHENE) < 1e-12
        assert abs(np.linalg.norm(a2) - A_GRAPHENE) < 1e-12

    def test_angle_between_vectors(self):
        a1, a2 = graphene_lattice_vectors(0.0)
        cos_angle = np.dot(a1, a2) / (np.linalg.norm(a1) * np.linalg.norm(a2))
        assert abs(cos_angle - 0.5) < 1e-12   # 60° angle

    def test_basis_bond_length(self):
        tau_A, tau_B = graphene_basis_vectors(0.0)
        bond = np.linalg.norm(tau_B - tau_A)
        assert abs(bond - A_CC) < 1e-10

    def test_rotation_preserves_lengths(self):
        for angle_deg in [5.0, 15.0, 30.0]:
            angle = np.radians(angle_deg)
            a1, a2 = graphene_lattice_vectors(angle)
            assert abs(np.linalg.norm(a1) - A_GRAPHENE) < 1e-12
            assert abs(np.linalg.norm(a2) - A_GRAPHENE) < 1e-12


# ── Atom generation and indexing ──────────────────────────────────────────────
class TestGenerateTBGGeometry:
    def test_atom_count_rectangular(self):
        """Rectangular supercell must contain 2*(2n+1)^2 atoms per layer."""
        for n in [2, 3, 5]:
            bot, top = generate_tbg_geometry(5.0, n_cells=n)
            expected = 2 * (2 * n + 1) ** 2
            assert bot.n_atoms == expected, f"n={n}: bottom layer atom count wrong"
            assert top.n_atoms == expected, f"n={n}: top layer atom count wrong"

    def test_global_ids_unique(self):
        """All atom_ids must be globally unique across both layers."""
        bot, top = generate_tbg_geometry(2.0, n_cells=3)
        all_ids = np.concatenate([bot.atom_ids, top.atom_ids])
        assert len(np.unique(all_ids)) == len(all_ids)

    def test_layer_ids_correct(self):
        bot, top = generate_tbg_geometry(10.0, n_cells=2)
        assert bot.layer_id == 0
        assert top.layer_id == 1

    def test_z_heights(self):
        bot, top = generate_tbg_geometry(5.0, n_cells=2)
        assert np.allclose(bot.positions[:, 2], 0.0)
        assert np.allclose(top.positions[:, 2], D_LAYER)

    def test_custom_interlayer_distance(self):
        bot, top = generate_tbg_geometry(5.0, n_cells=2, interlayer_distance=3.5)
        assert np.allclose(top.positions[:, 2], 3.5)

    def test_atom_table_completeness(self):
        """atom_table must map every atom's (layer, sub, n1, n2) key."""
        bot, top = generate_tbg_geometry(3.0, n_cells=2)
        for lay in [bot, top]:
            for local_i in range(lay.n_atoms):
                n1, n2 = lay.cell_indices[local_i]
                sub = lay.sublattices[local_i]
                key = (lay.layer_id, int(sub), int(n1), int(n2))
                assert key in lay.atom_table
                assert lay.atom_table[key] == lay.atom_ids[local_i]

    def test_nn_distance_bottom_layer(self):
        """NN distance in the bottom layer must be a_cc."""
        from scipy.spatial import cKDTree
        bot, _ = generate_tbg_geometry(5.0, n_cells=3)
        tree = cKDTree(bot.xy)
        dists, _ = tree.query(bot.xy, k=2)
        nn = dists[:, 1]
        # Interior atoms only (edge atoms may have fewer neighbours)
        np.testing.assert_allclose(nn.min(), A_CC, atol=1e-6)

    def test_center_rotation_symmetry(self):
        """Center-rotation: layers rotated by ±θ/2 are mirror images."""
        theta = 5.0
        bot, top = generate_tbg_geometry(theta, n_cells=2, center_rotation=True)
        assert abs(bot.angle + top.angle) < 1e-12    # angles sum to zero

    def test_no_center_rotation(self):
        theta = 5.0
        bot, top = generate_tbg_geometry(theta, n_cells=2, center_rotation=False)
        assert abs(bot.angle) < 1e-12
        assert abs(top.angle - np.radians(theta)) < 1e-12

    def test_disk_radius_reduces_atoms(self):
        """Disk-trimmed sample must have fewer atoms than rectangular."""
        bot_rect, _ = generate_tbg_geometry(5.0, n_cells=4)
        bot_disk, _ = generate_tbg_geometry(5.0, n_cells=4, disk_radius=10.0)
        assert bot_disk.n_atoms < bot_rect.n_atoms

    @pytest.mark.parametrize("angle", [1.08, 1.47, 2.0, 5.0, 10.0, 21.787])
    def test_various_angles(self, angle):
        """All angles should produce valid geometries."""
        bot, top = generate_tbg_geometry(angle, n_cells=3)
        assert bot.n_atoms > 0
        assert top.n_atoms > 0
        assert bot.n_atoms == top.n_atoms

    def test_global_offset_continuity(self):
        """Top-layer IDs must start exactly where bottom-layer IDs end."""
        bot, top = generate_tbg_geometry(5.0, n_cells=3)
        assert top.atom_ids[0] == bot.n_atoms
        assert top.atom_ids[-1] == bot.n_atoms + top.n_atoms - 1


# ── Commensurate helper ────────────────────────────────────────────────────-
class TestCommensurateIndices:
    def test_magic_angle_approx(self):
        """(1,2) pair should give θ ≈ 21.79°."""
        theta, _ = commensurate_indices(1, 2)
        assert abs(theta - 21.787) < 0.01

    def test_atom_count_positive(self):
        for m, n in [(1, 2), (2, 3), (3, 4)]:
            _, n_atoms = commensurate_indices(m, n)
            assert n_atoms > 0

    def test_small_angle_gives_large_cell(self):
        """Smaller angles → more atoms in the moiré cell."""
        _, n32 = commensurate_indices(3, 4)
        _, n12 = commensurate_indices(1, 2)
        assert n32 > n12
