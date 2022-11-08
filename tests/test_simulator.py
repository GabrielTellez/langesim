from langesim import make_simulator
import numpy as np
from pytest import approx
import pytest


def test_make_simulator():
    """
    tests if the simulator with default parameters is created correctly and can be called.
    """

    simulator = make_simulator()
    assert callable(simulator)


def test_run_simulator():
    """
    tests the simulator runs and returns correct data shape.
    """

    def k1(t):
        return 1.0

    def center1(t):
        return 0.0

    tot_sims = 100
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
    shape = (tot_sims, int(tot_steps / snapshot_step) + 1)
    assert len(times) == int(tot_steps / snapshot_step) + 1
    assert np.shape(x) == shape
    assert np.shape(power) == shape
    assert np.shape(work) == shape
    assert np.shape(heat) == shape
    assert np.shape(delta_U) == shape
    assert np.shape(energy) == shape

    # Now test the run with different parameters from the default ones
    tot_sims2 = 200
    tot_steps2 = 1000
    snapshot_step2 = 230
    times, x, power, work, heat, delta_U, energy = simulator(
        tot_sims=tot_sims2, tot_steps=tot_steps2, snapshot_step=snapshot_step2
    )
    shape = (tot_sims2, int(tot_steps2 / snapshot_step2) + 1)
    assert len(times) == int(tot_steps2 / snapshot_step2) + 1
    assert np.shape(x) == shape
    assert np.shape(power) == shape
    assert np.shape(work) == shape
    assert np.shape(heat) == shape
    assert np.shape(delta_U) == shape
    assert np.shape(energy) == shape


def test_energy_conservation_fixed_potential():
    # test with fixed potential
    def k1(t):
        return 1.0

    def center1(t):
        return 0.0

    tot_sims = 10
    tot_steps = 10000
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
    for time_index in range(0, len(times)):
        assert delta_U[:, time_index] - (
            work[:, time_index] + heat[:, time_index]
        ) == approx(0.0)
        assert delta_U[:, time_index] - (
            energy[:, time_index] - energy[:, 0]
        ) == approx(0.0)


def test_energy_conservation_variable_potential():
    # test with moving center and k linear
    def k2(t):
        return 1.0 + 0.1 * t

    def center2(t):
        return 0.2 * t

    tot_sims = 10
    tot_steps = 10000
    snapshot_step = 100
    simulator = make_simulator(
        tot_sims=tot_sims,
        dt=0.00001,
        tot_steps=tot_steps,
        snapshot_step=snapshot_step,
        k=k2,
        center=center2,
    )
    times, x, power, work, heat, delta_U, energy = simulator()
    for time_index in range(0, len(times)):
        assert delta_U[:, time_index] - (
            work[:, time_index] + heat[:, time_index]
        ) == approx(0.0)
        assert delta_U[:, time_index] - (
            energy[:, time_index] - energy[:, 0]
        ) == approx(0.0)


def test_general_force():
    """Tests the simulator for a non harmonic force"""

    def f(x, t):
        return 1.0

    with pytest.raises(ValueError):
        # This will fail if the potential is not provided
        simulator = make_simulator(harmonic_potential=False, force=f)
        times, x, power, work, heat, delta_U, energy = simulator()


def test_general_potential():
    """Tests the simulator for a non harmonic potential"""

    def U(x, t):
        return -1.0 * x

    def init_cond():
        return 0.0

    simulator = make_simulator(
        harmonic_potential=False, potential=U, initial_distribution=init_cond
    )
    times, x, power, work, heat, delta_U, energy = simulator()


def test_nonconsistent_force_and_potential():
    """Tests if an error is raised when the potential U and the force f are
    non consistent: f not equal to -grad U
    """

    def f(x, t):
        return 1.0

    def U(x, t):
        return x**2

    def init_cond():
        return 0.0

    with pytest.raises(ValueError):
        # This should fail
        simulator = make_simulator(
            harmonic_potential=False,
            force=f,
            potential=U,
            initial_distribution=init_cond,
        )
        times, x, power, work, heat, delta_U, energy = simulator()
