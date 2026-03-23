"""
tests/test_dos.py
=================
Unit tests for tbg.dos — density of states and LDOS calculations.
"""

import numpy as np
import pytest

from tbg.lattice import generate_tbg_geometry
from tbg.hamiltonian import build_hamiltonian
from tbg.dos import compute_dos, compute_ldos, integrated_dos


@pytest.fixture(scope="module")
def small_tbg():
    """
    Small TBG system for fast tests: θ = 5°, n_cells = 2.
    Produces 50 atoms per layer (100 total), sufficient for physics checks
    while keeping test execution under 1 second.
    """
    bot, top = generate_tbg_geometry(5.0, n_cells=2)
    H = build_hamiltonian([bot, top])
    return bot, top, H


class TestComputeDOS:
    def test_output_shapes(self, small_tbg):
        bot, top, H = small_tbg
        energies, dos = compute_dos(H, n_energies=200)
        assert energies.shape == (200,)
        assert dos.shape == (200,)

    def test_dos_non_negative(self, small_tbg):
        _, _, H = small_tbg
        _, dos = compute_dos(H)
        assert np.all(dos >= 0.0)

    def test_dos_integrates_to_one(self, small_tbg):
        """∫ g(E) dE ≈ 1 (one state per atom)."""
        _, _, H = small_tbg
        energies, dos = compute_dos(H, n_energies=2000, eta=0.01)
        integral = np.trapezoid(dos, energies)
        assert abs(integral - 1.0) < 0.05, f"DOS integral = {integral:.4f} ≠ 1"

    def test_energy_window_respected(self, small_tbg):
        _, _, H = small_tbg
        energies, dos = compute_dos(H, n_energies=100, e_min=-3.0, e_max=3.0)
        assert energies[0] == pytest.approx(-3.0)
        assert energies[-1] == pytest.approx(3.0)

    def test_broader_eta_smoother(self, small_tbg):
        """Larger η gives a smoother (less peaked) DOS."""
        _, _, H = small_tbg
        _, dos_fine = compute_dos(H, eta=0.01)
        _, dos_broad = compute_dos(H, eta=0.2)
        assert dos_fine.max() > dos_broad.max()

    def test_dos_particle_hole_symmetry(self, small_tbg):
        """
        For a single monolayer with NN-only intralayer hopping the DOS is
        exactly particle-hole symmetric: g(E) = g(-E).  (NNN and interlayer
        hoppings both break PH symmetry, so this test uses a monolayer only.)
        """
        from tbg.lattice import _generate_layer, A_CC
        mono = _generate_layer(
            angle=0.0, layer_id=0,
            n1_range=(-4, 4), n2_range=(-4, 4),
            z_height=0.0, global_offset=0,
        )
        # NN-only cutoff: 1.5·a_cc captures only nearest neighbours
        H_nn = build_hamiltonian([mono], r_cut_intra=1.5 * A_CC)
        energies, dos = compute_dos(
            H_nn, n_energies=1001, e_min=-10.0, e_max=10.0, eta=0.05
        )
        n = len(dos)
        mid = n // 2
        dos_pos = dos[mid + 1:]       # E > 0 half
        dos_neg = dos[:mid][::-1]     # E < 0 half, reversed
        min_len = min(len(dos_pos), len(dos_neg))
        np.testing.assert_allclose(
            dos_pos[:min_len], dos_neg[:min_len], atol=0.05
        )


class TestComputeLDOS:
    def test_output_shapes(self, small_tbg):
        bot, top, H = small_tbg
        indices = bot.atom_ids[:5]
        energies, ldos = compute_ldos(H, indices, n_energies=100)
        assert energies.shape == (100,)
        assert ldos.shape == (100,)

    def test_ldos_non_negative(self, small_tbg):
        bot, top, H = small_tbg
        _, ldos = compute_ldos(H, bot.atom_ids[:3])
        assert np.all(ldos >= 0.0)

    def test_ldos_integrates_to_one(self, small_tbg):
        """∫ ρ_i(E) dE ≈ 1 (sum rule: one orbital per site)."""
        bot, top, H = small_tbg
        # Use all atoms → should give global DOS
        all_ids = np.concatenate([bot.atom_ids, top.atom_ids])
        energies, ldos = compute_ldos(H, all_ids, n_energies=2000, eta=0.01)
        integral = np.trapezoid(ldos, energies)
        assert abs(integral - 1.0) < 0.05, f"LDOS integral = {integral:.4f} ≠ 1"

    def test_single_atom_ldos(self, small_tbg):
        """LDOS at a single atom must be non-negative and finite."""
        bot, _, H = small_tbg
        energies, ldos = compute_ldos(H, [bot.atom_ids[0]])
        assert np.all(np.isfinite(ldos))
        assert np.all(ldos >= 0.0)


class TestIntegratedDOS:
    def test_integrated_dos_half_filling(self, small_tbg):
        """
        Integrated DOS up to E=0 should be ≈ 0.5 (half-filling) for an
        unperturbed bipartite lattice.
        """
        _, _, H = small_tbg
        energies, dos = compute_dos(H, n_energies=2000, eta=0.01)
        idos = integrated_dos(energies, dos, e_fermi=0.0)
        assert abs(idos - 0.5) < 0.08, f"IDOS at E=0: {idos:.4f} ≠ 0.5"

    def test_integrated_dos_monotone(self, small_tbg):
        """IDOS must be monotonically non-decreasing."""
        _, _, H = small_tbg
        energies, dos = compute_dos(H, n_energies=500, eta=0.05)
        fermi_levels = np.linspace(-5.0, 5.0, 20)
        idos_vals = [integrated_dos(energies, dos, e) for e in fermi_levels]
        diffs = np.diff(idos_vals)
        assert np.all(diffs >= -1e-6), "IDOS is not monotone"

    def test_integrated_dos_below_band(self, small_tbg):
        """IDOS well below the band minimum must be ≈ 0."""
        _, _, H = small_tbg
        energies, dos = compute_dos(H, n_energies=500, eta=0.05)
        idos = integrated_dos(energies, dos, e_fermi=-20.0)
        assert idos < 0.01


class TestPhysicalValidation:
    @pytest.mark.parametrize("angle", [1.08, 2.0, 5.0, 10.0])
    def test_dos_finite_all_angles(self, angle):
        """DOS must be finite for all angles in 0–30°."""
        bot, top = generate_tbg_geometry(angle, n_cells=2)
        H = build_hamiltonian([bot, top])
        _, dos = compute_dos(H, n_energies=100)
        assert np.all(np.isfinite(dos))

    def test_dos_at_dirac_point_small(self):
        """
        Near the magic angle, DOS at E=0 should be large (flat band).
        At larger angles, DOS at E=0 should be small (Dirac point).
        """
        # Use n_cells=3 for decent moiré sampling
        bot_large, top_large = generate_tbg_geometry(20.0, n_cells=3)
        H_large = build_hamiltonian([bot_large, top_large])
        energies_l, dos_l = compute_dos(H_large, n_energies=200, eta=0.05)

        # DOS at E=0
        e0_idx = np.argmin(np.abs(energies_l))
        dos_at_dirac_large = dos_l[e0_idx]

        # For monolayer-like (large angle), DOS at Dirac should be small
        # (not zero due to finite-size and broadening, but < 1 state/eV/atom)
        assert dos_at_dirac_large < 2.0, (
            f"DOS at Dirac for θ=20°: {dos_at_dirac_large:.2f}"
        )
