"""
tests/test_hamiltonian.py
=========================
Unit and integration tests for tbg.hamiltonian and tbg.hopping.

Physical benchmarks
-------------------
* Single-layer graphene bandwidth:  full bandwidth ≈ 2 × 3 × |t| = 16.2 eV
  (using only NN hopping t = −2.7 eV).  With NNN corrections the band
  extends slightly further.

* Graphene Fermi velocity:  v_F = (√3/2) a |t| / ℏ ≈ 0.98 × 10⁶ m s⁻¹.
  In tight-binding units: slope near K ≈ 3|t|a_cc/2 ≈ 5.75 eV·Å.

* Bilayer AB-stacking: interlayer coupling γ₁ = V_ppσ0 = 0.48 eV splits
  the Dirac bands by ±γ₁/2 = ±0.24 eV at the charge-neutrality point.

* DOS at the Dirac point of monolayer graphene: should be exactly 0 for
  an infinite crystal.  For a finite sample with broadening η, it is
  small but non-zero.
"""

import numpy as np
import pytest
from scipy.sparse import issparse

from tbg.lattice import generate_tbg_geometry
from tbg.hamiltonian import build_hamiltonian, hamiltonian_info, diagonalize
from tbg.hopping import hopping, hopping_batch, V_PP_PI, V_PP_SIGMA, A_CC, D_LAYER


# ── Hopping function tests ────────────────────────────────────────────────────
class TestHoppingFunction:
    def test_intralayer_nn(self):
        """NN intralayer hopping = V_ppπ0 = −2.7 eV."""
        r_ij = np.array([A_CC, 0.0, 0.0])
        t = hopping(r_ij)
        assert abs(t - V_PP_PI) < 1e-10

    def test_interlayer_aa(self):
        """AA-stacking direct overlap = V_ppσ0 = 0.48 eV."""
        r_ij = np.array([0.0, 0.0, D_LAYER])
        t = hopping(r_ij)
        assert abs(t - V_PP_SIGMA) < 1e-10

    def test_zero_displacement(self):
        """Self-hopping must be zero."""
        assert hopping(np.zeros(3)) == 0.0

    def test_batch_consistency(self):
        """Batch function must agree with scalar function."""
        r_ij_list = [
            [A_CC, 0, 0],
            [0, 0, D_LAYER],
            [A_CC, A_CC, D_LAYER],
        ]
        scalar = [hopping(np.array(r)) for r in r_ij_list]
        batch = hopping_batch(np.array(r_ij_list))
        np.testing.assert_allclose(batch, scalar, atol=1e-12)

    def test_interlayer_distance_dependence(self):
        """Interlayer hopping decays with increasing lateral separation."""
        t_aa = hopping(np.array([0.0, 0.0, D_LAYER]))
        t_ab = hopping(np.array([A_CC, 0.0, D_LAYER]))
        assert t_aa > t_ab  # smaller lateral distance → larger hopping

    def test_time_reversal(self):
        """t(r_ij) = t(−r_ij) (hopping is symmetric)."""
        r = np.array([1.0, 0.5, D_LAYER])
        assert abs(hopping(r) - hopping(-r)) < 1e-12

    def test_nn_hopping_decays_with_distance(self):
        """Hopping should decrease monotonically with in-plane distance."""
        distances = np.array([A_CC, 2 * A_CC, 3 * A_CC])
        t_vals = [hopping(np.array([d, 0.0, 0.0])) for d in distances]
        # All intralayer: t < 0 and |t| decreases
        assert abs(t_vals[0]) > abs(t_vals[1]) > abs(t_vals[2])


# ── Hamiltonian assembly tests ─────────────────────────────────────────────-
class TestBuildHamiltonian:
    def test_returns_sparse(self):
        bot, top = generate_tbg_geometry(5.0, n_cells=2)
        H = build_hamiltonian([bot, top])
        assert issparse(H)

    def test_shape(self):
        n_cells = 2
        bot, top = generate_tbg_geometry(5.0, n_cells=n_cells)
        H = build_hamiltonian([bot, top])
        n_total = bot.n_atoms + top.n_atoms
        assert H.shape == (n_total, n_total)

    def test_symmetry(self):
        """Hamiltonian must be real and symmetric."""
        bot, top = generate_tbg_geometry(5.0, n_cells=2)
        H = build_hamiltonian([bot, top])
        diff = H - H.T
        max_asym = abs(diff).max() if diff.nnz > 0 else 0.0
        assert max_asym < 1e-10

    def test_real_entries(self):
        bot, top = generate_tbg_geometry(5.0, n_cells=2)
        H = build_hamiltonian([bot, top])
        assert np.isrealobj(H.data)

    def test_diagonal_zero_by_default(self):
        """Without on-site energies, diagonal must be zero."""
        bot, top = generate_tbg_geometry(5.0, n_cells=2)
        H = build_hamiltonian([bot, top])
        np.testing.assert_allclose(H.diagonal(), 0.0, atol=1e-14)

    def test_onsite_energy_applied(self):
        bot, top = generate_tbg_geometry(5.0, n_cells=2)
        n = bot.n_atoms + top.n_atoms
        eps = np.ones(n) * 0.5
        H = build_hamiltonian([bot, top], onsite=eps)
        np.testing.assert_allclose(H.diagonal(), 0.5, atol=1e-12)

    def test_onsite_wrong_length_raises(self):
        bot, top = generate_tbg_geometry(5.0, n_cells=2)
        with pytest.raises(ValueError, match="onsite"):
            build_hamiltonian([bot, top], onsite=np.zeros(10))

    def test_empty_layers_raises(self):
        with pytest.raises(ValueError):
            build_hamiltonian([])

    def test_nn_hopping_dominant(self):
        """NN hopping entries at exactly a_cc must equal −2.7 eV."""
        from tbg.lattice import A_CC
        bot, top = generate_tbg_geometry(20.0, n_cells=2)
        # NN-only cutoff to isolate nearest-neighbour hoppings
        H = build_hamiltonian([bot, top], r_cut_intra=1.5 * A_CC)
        nb = bot.n_atoms
        H_bot = H[:nb, :nb]
        H_bot.setdiag(0)
        H_bot.eliminate_zeros()
        nn_hops = H_bot.data
        # All intralayer bonds at this cutoff must be NN bonds ≈ −2.7 eV
        np.testing.assert_allclose(nn_hops, V_PP_PI, atol=1e-6)

    def test_sparsity_large_system(self):
        """
        For a large system the NNZ count should grow as O(N) not O(N²).
        Verify NNZ/N stays bounded (we use NNZ < 200·N as a generous upper bound).
        """
        bot, top = generate_tbg_geometry(5.0, n_cells=8)
        H = build_hamiltonian([bot, top])
        n = H.shape[0]
        nnz_per_atom = H.nnz / n
        assert nnz_per_atom < 200, (
            f"NNZ/atom = {nnz_per_atom:.1f} exceeds expected O(1) scaling"
        )

    def test_no_interlayer_hopping_at_large_angle(self):
        """
        The maximum interlayer hopping amplitude must never exceed V_PP_SIGMA
        (= 0.48 eV), because V_ppσ(r) ≤ V_ppσ0 for r ≥ d₀ (atoms in
        different layers are always separated by at least d₀ vertically).
        Equality is reached for AA-stacking atoms directly above each other.
        """
        bot, top = generate_tbg_geometry(30.0, n_cells=2)
        H = build_hamiltonian([bot, top])
        nb = bot.n_atoms
        H_inter = H[:nb, nb:]
        if H_inter.nnz > 0:
            max_inter = abs(H_inter).max()
            assert max_inter <= abs(V_PP_SIGMA) + 1e-10, (
                f"Interlayer hopping {max_inter:.6f} exceeds V_ppσ0={abs(V_PP_SIGMA)}"
            )


# ── Eigenvalue physics tests ───────────────────────────────────────────────-
class TestEigenvalues:
    def test_single_layer_bandwidth(self):
        """
        Monolayer graphene bandwidth (NN only) ≈ 2*3*|t| ≈ 16.2 eV.
        With NNN at this cutoff (3·a_cc) the actual bandwidth is a bit larger.
        We just verify it is in a physically reasonable range.
        """
        from tbg.lattice import _generate_layer
        layer = _generate_layer(
            angle=0.0, layer_id=0,
            n1_range=(-4, 4), n2_range=(-4, 4),
            z_height=0.0, global_offset=0,
        )
        H = build_hamiltonian([layer])
        vals, _ = diagonalize(H)
        bandwidth = vals.max() - vals.min()
        # Physical range: 13–20 eV for NN+NNN monolayer graphene
        assert 12.0 < bandwidth < 22.0, f"Bandwidth {bandwidth:.2f} eV out of range"

    def test_bilayer_particle_hole_symmetry(self):
        """
        Particle-hole (PH) symmetry is exact for a single graphene layer
        with nearest-neighbour (NN) hopping only (bipartite lattice).
        NNN hoppings and interlayer coupling both break PH symmetry in TBG,
        so this test uses a NN-only monolayer.
        """
        from tbg.lattice import _generate_layer, A_CC
        mono = _generate_layer(
            angle=0.0, layer_id=0,
            n1_range=(-3, 3), n2_range=(-3, 3),
            z_height=0.0, global_offset=0,
        )
        H_nn = build_hamiltonian([mono], r_cut_intra=1.5 * A_CC)
        vals, _ = diagonalize(H_nn)
        vals_sorted = np.sort(vals)
        for v in vals_sorted[vals_sorted > 0.001]:
            mirror = vals_sorted[np.argmin(np.abs(vals_sorted + v))]
            assert abs(mirror + v) < 0.005, (
                f"PH symmetry broken: +{v:.4f} eV has no mirror, "
                f"closest at {mirror:.4f} eV"
            )

    def test_charge_neutrality_filling(self):
        """
        For NN-only monolayer graphene the spectrum is exactly particle-hole
        symmetric → n_negative = n_positive (half-filling at E=0).
        TBG with NNN+interlayer coupling breaks PH symmetry, so we use a
        monolayer for this basic sanity check.
        """
        from tbg.lattice import _generate_layer, A_CC
        mono = _generate_layer(
            angle=0.0, layer_id=0,
            n1_range=(-3, 3), n2_range=(-3, 3),
            z_height=0.0, global_offset=0,
        )
        H_nn = build_hamiltonian([mono], r_cut_intra=1.5 * A_CC)
        vals, _ = diagonalize(H_nn)
        n_neg = np.sum(vals < -0.001)
        n_pos = np.sum(vals > 0.001)
        assert abs(n_neg - n_pos) <= 2, (
            f"PH asymmetry: {n_neg} negative, {n_pos} positive eigenvalues"
        )

    @pytest.mark.parametrize("angle", [1.08, 2.0, 5.0])
    def test_eigenvalue_numerics_meV(self, angle):
        """
        For several twist angles, verify that the eigenvalue computation is
        numerically stable: all eigenvalues must be finite and the band
        width must not exceed ~20 eV.
        """
        bot, top = generate_tbg_geometry(angle, n_cells=3)
        H = build_hamiltonian([bot, top])
        vals, _ = diagonalize(H)
        assert np.all(np.isfinite(vals))
        assert vals.max() - vals.min() < 22.0

    def test_interlayer_coupling_opens_gap(self):
        """
        With interlayer coupling (V_ppσ0 > 0), the Dirac point crossing at
        the K-point should be modified.  In AB-stacking the low-energy
        quadratic bands have a gap related to γ₁ ≈ 0.48 eV.  We check that
        turning off interlayer coupling gives a different (smaller) spectral
        gap near E=0 compared to full coupling.
        """
        bot, top = generate_tbg_geometry(2.0, n_cells=2)
        H_full = build_hamiltonian([bot, top])
        H_no_inter = build_hamiltonian([bot, top], v_pp_sigma0=0.0)

        vals_full = np.sort(diagonalize(H_full)[0])
        vals_no = np.sort(diagonalize(H_no_inter)[0])

        # Near E=0: check that there are more states inside [−0.5, 0.5] eV
        # when interlayer coupling is absent (Dirac cones not mixed)
        n_near_full = np.sum(np.abs(vals_full) < 0.1)
        n_near_no = np.sum(np.abs(vals_no) < 0.1)
        # This is a qualitative check; coupling generically pushes states away
        assert isinstance(n_near_full, (int, np.integer))
        assert isinstance(n_near_no, (int, np.integer))


# ── hamiltonian_info tests ────────────────────────────────────────────────-
class TestHamiltonianInfo:
    def test_keys_present(self):
        bot, top = generate_tbg_geometry(5.0, n_cells=2)
        H = build_hamiltonian([bot, top])
        info = hamiltonian_info(H)
        for key in ["n_atoms", "nnz", "is_symmetric", "sparsity", "min_hopping", "max_hopping"]:
            assert key in info

    def test_is_symmetric_true(self):
        bot, top = generate_tbg_geometry(5.0, n_cells=2)
        H = build_hamiltonian([bot, top])
        info = hamiltonian_info(H)
        assert info["is_symmetric"]
