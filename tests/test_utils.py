"""
tests/test_utils.py
===================
Unit tests for tbg.utils — moiré geometry, validation helpers, on-site energies.
"""

import numpy as np
import pytest

from tbg.lattice import generate_tbg_geometry, A_GRAPHENE
from tbg.hamiltonian import build_hamiltonian
from tbg.utils import (
    moire_period,
    moire_reciprocal_vectors,
    check_nn_distances,
    check_atom_count,
    symmetry_check,
    sublattice_onsite,
    gate_voltage_onsite,
)


class TestMoirePeriod:
    def test_small_angle_large_period(self):
        """λ grows as θ → 0."""
        lam_1 = moire_period(1.0)
        lam_2 = moire_period(2.0)
        assert lam_1 > lam_2

    def test_zero_angle_infinite(self):
        lam = moire_period(0.0)
        assert lam == np.inf

    def test_formula(self):
        """λ = a / (2 sin(θ/2))."""
        theta = 5.0
        expected = A_GRAPHENE / (2.0 * np.sin(np.radians(theta) / 2.0))
        assert abs(moire_period(theta) - expected) < 1e-10

    def test_known_magic_angle(self):
        """At 1.08° the moiré period is ≈ 130 Å."""
        lam = moire_period(1.08)
        assert 120 < lam < 140


class TestMoireReciprocalVectors:
    def test_output_shapes(self):
        G1, G2 = moire_reciprocal_vectors(5.0)
        assert G1.shape == (2,)
        assert G2.shape == (2,)

    def test_magnitude_matches_period(self):
        """|G_M| ≈ 4π / (√3 λ) for the hexagonal moiré BZ."""
        theta = 5.0
        G1, G2 = moire_reciprocal_vectors(theta)
        lam = moire_period(theta)
        # For hexagonal lattice |G| = 4π/(√3 a_M) where a_M = λ
        expected = 4.0 * np.pi / (np.sqrt(3.0) * lam)
        assert abs(np.linalg.norm(G1) - expected) < 0.01


class TestCheckNNDistances:
    def test_perfect_lattice(self):
        bot, _ = generate_tbg_geometry(5.0, n_cells=3)
        result = check_nn_distances(bot)
        assert result["passed"]

    def test_expected_value(self):
        from tbg.lattice import A_CC
        bot, _ = generate_tbg_geometry(10.0, n_cells=3)
        result = check_nn_distances(bot)
        assert abs(result["expected"] - A_CC) < 1e-12


class TestCheckAtomCount:
    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_rectangular_supercell(self, n):
        bot, _ = generate_tbg_geometry(5.0, n_cells=n)
        result = check_atom_count(bot, n_cells=n)
        assert result["passed"], f"n={n}: {result}"


class TestSymmetryCheck:
    def test_passes_for_valid_H(self):
        bot, top = generate_tbg_geometry(5.0, n_cells=2)
        H = build_hamiltonian([bot, top])
        result = symmetry_check(H)
        assert result["passed"]
        assert result["is_real"]
        assert result["max_asymmetry"] < 1e-10


class TestOnsiteHelpers:
    def test_sublattice_onsite_shapes(self):
        bot, top = generate_tbg_geometry(5.0, n_cells=2)
        n_total = bot.n_atoms + top.n_atoms
        eps = sublattice_onsite([bot, top], delta=0.1)
        assert eps.shape == (n_total,)

    def test_sublattice_onsite_values(self):
        """A sites → +δ, B sites → −δ."""
        bot, top = generate_tbg_geometry(5.0, n_cells=2)
        eps = sublattice_onsite([bot, top], delta=0.3)
        for lay in [bot, top]:
            for local_i in range(lay.n_atoms):
                gid = int(lay.atom_ids[local_i])
                sub = int(lay.sublattices[local_i])
                expected = 0.3 if sub == 0 else -0.3
                assert abs(eps[gid] - expected) < 1e-12

    def test_gate_voltage_onsite_shapes(self):
        bot, top = generate_tbg_geometry(5.0, n_cells=2)
        n_total = bot.n_atoms + top.n_atoms
        eps = gate_voltage_onsite([bot, top], v_gate=0.2)
        assert eps.shape == (n_total,)

    def test_gate_voltage_onsite_values(self):
        """Bottom layer → +V/2, top layer → −V/2."""
        bot, top = generate_tbg_geometry(5.0, n_cells=2)
        eps = gate_voltage_onsite([bot, top], v_gate=0.4)
        for gid in bot.atom_ids:
            assert abs(eps[int(gid)] - 0.2) < 1e-12
        for gid in top.atom_ids:
            assert abs(eps[int(gid)] + 0.2) < 1e-12

    def test_onsite_affects_spectrum(self):
        """Gate voltage shifts eigenvalues relative to zero-gate case."""
        bot, top = generate_tbg_geometry(5.0, n_cells=2)
        H0 = build_hamiltonian([bot, top])
        eps_gate = gate_voltage_onsite([bot, top], v_gate=1.0)
        H_gate = build_hamiltonian([bot, top], onsite=eps_gate)
        from tbg.hamiltonian import diagonalize
        v0, _ = diagonalize(H0)
        vg, _ = diagonalize(H_gate)
        # Spectra should differ
        assert not np.allclose(v0, vg)
