from langesim import make_simulator, Simulator
import numpy as np
from scipy.stats import entropy
from pytest import approx
import pytest


def test_no_evolution_make_simulator():
    """Tests for no change in probability distribution when the potential
    does not change"""

    def k1(t):
        return 1.0

    def center1(t):
        return 0.0

    def Peq(x):
        """Equilibrium distribution"""
        return np.exp(-0.5 * k1(0.0) * (x - center1(0.0)) ** 2) / np.sqrt(
            2 * np.pi / k1(0.0)
        )

    tot_sims = 100000
    tot_steps = 1000
    snapshot_step = 100
    simulator = make_simulator(
        tot_sims=tot_sims,
        dt=0.00001,
        tot_steps=tot_steps,
        snapshot_step=snapshot_step,
        k=k1,
        center=center1,
    )
    times, x, power, work, heat, delta_U, energy = simulator()
    x_range = [-3.0, 3.0]
    bins = 300
    histos = [
        np.histogram(x[:, ti], density=True, range=x_range, bins=bins)
        for ti in range(0, len(times))
    ]
    size_x = len(histos[0][1])
    xx = np.linspace(*x_range, size_x - 1)
    Peq_array = Peq(xx)
    tol = 1.0 / np.sqrt(tot_sims)
    # tol_histo = 5*tol/np.sqrt(k1(0))
    tol_histo = 5e-2
    for time_index in range(0, len(times)):
        err = np.square((histos[time_index][0] - Peq_array) / Peq_array).mean()
        assert (
            err < tol_histo
        ), f"P(x) should be the equilibrium distribution at all times. Error {err} larger than tolerance {tol_histo} on time_index={time_index}"
        assert work[:, time_index] == approx(
            0.0
        ), f"No work if the potential does not change. Error on time_index={time_index}"
        assert power[:, time_index] == approx(
            0.0
        ), f"No power if the potential does not change. Error on time_index={time_index}"
        if time_index > 0:
            assert not heat[:, time_index] == approx(
                0.0
            ), f"heat is not zero for each realization. Error on time_index={time_index}"
        assert np.average(heat[:, time_index]) == approx(
            0.0, rel=tol, abs=tol
        ), f"heat should be zero on average. Error on time_index={time_index}"
        assert np.average(delta_U[:, time_index]) == approx(
            0.0, rel=tol, abs=tol
        ), f"energy should not change on average. Error on time_index={time_index}"
        # test energy with a confidence interval of 1-1E-6, thus the factor 4.75
        assert np.average(energy[:, time_index]) == approx(
            0.5, rel=4.75 * tol
        ), f"on average energy should be (1/2) k_B T. Error on time_index={time_index}"

def test_no_evolution_kde():
    """Tests for no change in probability distribution computed with KDE when the potential
    does not change"""

    def k1(t):
        return 1.0

    def center1(t):
        return 2.0

    def Peq(x):
        """Equilibrium distribution"""
        return np.exp(-0.5 * k1(0.0) * (x - center1(0.0)) ** 2) / np.sqrt(
            2 * np.pi / k1(0.0)
        )

    tot_sims = 100000
    tot_steps = 1000
    snapshot_step = 100
    simulator = Simulator(
        tot_sims=tot_sims,
        dt=0.00001,
        tot_steps=tot_steps,
        snapshot_step=snapshot_step,
        k=k1,
        center=center1,
    )
    simulator.run()
    sim = simulator.simulation[0]
    sim.build_pdf("x", method="kde")
    xs = np.linspace(-1.0, 5.0, 1000)

    theoretical_pdf = Peq(xs)

    for t in sim.results["times"]:
        # Build the KDE for the current snapshot
        kde_pdf = sim.pdf["x"](xs, t)  # Get the KDE at time `t`

        # Compute the KL divergence between theoretical and KDE PDF
        kl_divergence = entropy(theoretical_pdf, kde_pdf)

        # Assert that the KL divergence is sufficiently small
        tolerance = 5e-4
        assert kl_divergence < tolerance, (
            f"KL divergence at t={t} is too large: {kl_divergence}\n"
            f"KDE PDF does not match the theoretical PDF at time {t}."
        )