from langesim import Simulation
import numpy as np
import pytest
import plotly.graph_objects as go


@pytest.fixture
def dummy_sim():
    """Builds a simulation with dummy paramaters and data"""

    def k(t):
        """t |--> 1.0"""
        return 1.0

    def center(t):
        """t |--> 0.0"""
        return 0.0

    tot_sims = 10
    dt = 0.01
    tot_steps = 100
    snapshot_step = 10
    noise_scaler = 1.0
    tot_snapshots = int(tot_steps / snapshot_step) + 1
    x = np.random.random_sample((tot_sims, tot_snapshots))
    work = np.random.random_sample((tot_sims, tot_snapshots))
    power = np.random.random_sample((tot_sims, tot_snapshots))
    heat = np.random.random_sample((tot_sims, tot_snapshots))
    delta_U = np.random.random_sample((tot_sims, tot_snapshots))
    energy = np.random.random_sample((tot_sims, tot_snapshots))
    times = np.arange(0, (1 + tot_steps) * dt, dt * snapshot_step)
    results = (times, x, power, work, heat, delta_U, energy)
    sim = Simulation(
        tot_sims=tot_sims,
        dt=dt,
        tot_steps=tot_steps,
        noise_scaler=noise_scaler,
        snapshot_step=snapshot_step,
        k=k,
        center=center,
        results=results,
        name="dummy simulation",
    )
    return (
        tot_sims,
        dt,
        tot_steps,
        noise_scaler,
        snapshot_step,
        k,
        center,
        results,
        sim,
    )


@pytest.mark.parametrize(
    "quantity", ["x", "power", "work", "heat", "delta_U", "energy"]
)
def test_plot_average(dummy_sim, quantity):
    """Test if a plot of an average is created

    Args:
        dummy_sim (tuple): simulation data and class
        quantity (string): quantity to plot the average
    """
    (
        tot_sims,
        dt,
        tot_steps,
        noise_scaler,
        snapshot_step,
        k,
        center,
        results,
        sim,
    ) = dummy_sim
    plot = sim.plot_average(quantity)
    assert isinstance(plot, go.Figure)


@pytest.mark.parametrize(
    "quantity", ["x", "power", "work", "heat", "delta_U", "energy"]
)
def test_plot_variance(dummy_sim, quantity):
    """Test if a plot of an variance is created

    Args:
        dummy_sim (tuple): simulation data and class
        quantity (string): quantity to plot the variance
    """
    (
        tot_sims,
        dt,
        tot_steps,
        noise_scaler,
        snapshot_step,
        k,
        center,
        results,
        sim,
    ) = dummy_sim
    plot = sim.plot_variance(quantity)
    assert isinstance(plot, go.Figure)


@pytest.mark.parametrize(
    "quantity", ["x", "power", "work", "heat", "delta_U", "energy"]
)
def test_plot_sim(dummy_sim, quantity):
    """Test if a plot of a quantity as function of time is created

    Args:
        dummy_sim (tuple): simulation data and class
        quantity (string): quantity to plot the variance
    """
    (
        tot_sims,
        dt,
        tot_steps,
        noise_scaler,
        snapshot_step,
        k,
        center,
        results,
        sim,
    ) = dummy_sim
    sim_list = [i for i in range(sim.tot_sims)]
    plot = sim.plot_sim(quantity, sim_list)
    assert isinstance(plot, go.Figure)


@pytest.mark.parametrize(
    "quantity", ["x", "power", "work", "heat", "delta_U", "energy"]
)
def test_animate_sim(dummy_sim, quantity):
    """Test if an animated plot of a quantity is created

    Args:
        dummy_sim (tuple): simulation data and class
        quantity (string): quantity to plot the variance
    """
    (
        tot_sims,
        dt,
        tot_steps,
        noise_scaler,
        snapshot_step,
        k,
        center,
        results,
        sim,
    ) = dummy_sim
    plot = sim.animate_pdf(quantity)
    assert isinstance(plot, go.Figure)
