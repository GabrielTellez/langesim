from langesim import Simulator
import numpy as np
import pytest


@pytest.fixture
def run_sim():
    """Runs a simulation and returns it"""
    ki = 0.01
    kf = 0.02
    tf = 0.1

    def k(t):
        return ki + (kf - ki) * t / tf

    tot_sims = 100_000
    simulator = Simulator(
        tot_sims=tot_sims, dt=0.0001, tot_steps=1_000, k=k, snapshot_step=100
    )
    simulator.run()
    return (tot_sims, ki, kf, tf, k, simulator.simulation[0])


def test_2d_law(run_sim):
    """Test the second law of thermodynamics:
    the irreversible work Wirr = W - Delta F
    should be positive
    """
    (tot_sims, ki, kf, tf, k, sim) = run_sim

    def DeltaF(t):
        return 0.5 * np.log(k(t) / k(0))

    times = sim.results["times"]
    work = sim.results["work"]
    wirr = np.average(work, axis=0) - DeltaF(times)
    assert np.all(wirr >= 0)


def test_Jarzynski_relation(run_sim):
    """Test the Jarzynski equality
    <exp(-work)>=exp(-Delta F)
    """
    (tot_sims, ki, kf, tf, k, sim) = run_sim

    def DeltaF(t):
        return 0.5 * np.log(k(t) / k(0))

    times = sim.results["times"]
    work = sim.results["work"]
    expW = np.average(np.exp(-work), axis=0)
    expWvar = np.var(np.exp(-work), axis=0)
    expF = np.exp(-DeltaF(times))
    tol = 5e-3
    for t, eW in enumerate(expW):
        assert eW == pytest.approx(
            expF[t], abs=tol
        ), f"Jarzynski equality not satisfied at time t={times[t]}"


def test_fuerte_Jarzynski_relation(run_sim):
    """Test the Jarzynski equality
    <exp(-work)>=exp(-Delta F)
    for a more variable potential.
    This used to fail before fixing the rounding bug in the times.
    """

    ki = 0.5
    kf = 1.0
    ko = 20.0
    tf = 10.0

    def k(t):
        """Protocolo por tramos rectos horizontal"""
        if t <= 0.0:
            return ki
        if t <= tf / 4.0:
            return ko / 2.0
        if t < tf / 2.0:
            return ko / 4.0
        if t < 3.0 * tf / 4.0:
            return ko / 10.0
        else:
            return kf

    kv = np.vectorize(k)

    simulator = Simulator(
        tot_sims=100_000,
        dt=0.0001,
        tot_steps=10_000,
        k=k,
        snapshot_step=1000,
    )

    def DeltaF(t):
        return 0.5 * np.log(kv(t) / k(0))

    simulator.run()
    sim = simulator.simulation[0]
    times = sim.results["times"]
    work = sim.results["work"]
    expW = np.average(np.exp(-work), axis=0)
    expF = np.exp(-DeltaF(times))
    tol = 5e-2
    for t, eW in enumerate(expW):
        assert eW == pytest.approx(
            expF[t], rel=tol
        ), f"Jarzynski equality not satisfied at time t={times[t]}"
