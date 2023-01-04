from langesim import Simulation
import numpy as np
import pytest
import os
import plotly.graph_objects as go
import pickle


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


@pytest.fixture
def dummy_sim_const_histo():
    """Builds a dummy simulation with results yielding a constant
    histogram by segments
    """

    def k(t):
        """t |--> 1.0"""
        return 1.0

    def center(t):
        """t |--> 0.0"""
        return 0.0

    tot_sims = 10
    dt = 1.0
    tot_steps = 100
    snapshot_step = 101
    noise_scaler = 1.0
    # tot_snapshots = int(tot_steps / snapshot_step) + 1
    x = np.array([[0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 0.8]])
    x = x.transpose()
    work = x
    power = x
    heat = x
    delta_U = x
    energy = x
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


@pytest.fixture
def dummy_sim_gaussian():
    """Builds a dummy simulation with gaussian PDFs for results"""

    def k(t):
        """t |--> 1.0"""
        return 1.0

    def center(t):
        """t |--> 0.0"""
        return 0.0

    tot_sims = 1000000
    dt = 1.0
    tot_steps = 100
    snapshot_step = 50
    noise_scaler = 1.0
    # tot_snapshots = int(tot_steps/snapshot_step)+1
    x = np.random.normal(loc=[0.0, 1.0, 2.0], scale=[1.0, 3.0, 5.0], size=(tot_sims, 3))
    work = x
    power = x
    heat = x
    delta_U = x
    energy = x
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


def test_simulation_init(dummy_sim):
    """Tests correct creation of a simulation class and store of parameters"""
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
    assert sim.tot_sims == tot_sims
    assert sim.dt == dt
    assert sim.tot_steps == tot_steps
    assert sim.noise_scaler == noise_scaler
    assert sim.snapshot_step == snapshot_step
    assert sim.k == k
    assert sim.center == center


def test_simulation_init_store_results(dummy_sim):
    """Tests correct creation of a simulation class and store of the results"""
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
    (times, x, power, work, heat, delta_U, energy) = results
    labels = ["x", "power", "work", "heat", "delta_U", "energy"]
    assert sim.result_labels == labels
    assert np.array_equal(sim.results["times"], times)
    assert np.array_equal(sim.results["x"], x)
    assert np.array_equal(sim.results["power"], power)
    assert np.array_equal(sim.results["work"], work)
    assert np.array_equal(sim.results["heat"], heat)
    assert np.array_equal(sim.results["delta_U"], delta_U)
    assert np.array_equal(sim.results["energy"], energy)


def test_simulation_results_shape(dummy_sim):
    """Tests if the results have the correct shape

    Args:
        dummy_sim (tuple): dummy simulation (
      tot_sims, dt, tot_steps, noise_scaler, snapshot_step,
      k, center, results,
      Simulation
    )
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
    sim_res = sim.results
    tot_snapshots = int(tot_steps / snapshot_step) + 1
    shape = (tot_sims, tot_snapshots)
    assert len(sim_res["times"]) == tot_snapshots
    assert np.shape(sim_res["x"]) == shape
    assert np.shape(sim_res["power"]) == shape
    assert np.shape(sim_res["work"]) == shape
    assert np.shape(sim_res["heat"]) == shape
    assert np.shape(sim_res["delta_U"]) == shape
    assert np.shape(sim_res["energy"]) == shape


@pytest.mark.parametrize(
    "quantity", ["x", "power", "work", "heat", "delta_U", "energy"]
)
def test_build_histogram(dummy_sim, quantity):
    """Tests if a histogram is build with the correct shape"""
    np.warnings.filterwarnings("error", category=np.VisibleDeprecationWarning)
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
    bins = 300
    q_range = [-3.0, 3.0]
    sim.build_histogram(quantity, bins=bins, q_range=q_range)
    size = len(sim.results["times"])
    assert type(sim.histogram[quantity]) == list
    assert len(sim.histogram[quantity]) == size
    assert type(sim.histogram[quantity][0]) == tuple
    assert len(sim.histogram[quantity][0]) == 2


@pytest.mark.parametrize(
    "quantity", ["x", "power", "work", "heat", "delta_U", "energy"]
)
def test_build_pdf(dummy_sim, quantity):
    """test if the probability density functions are build correctly"""
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
    sim.build_pdf(quantity)
    assert callable(sim.pdf[quantity])


@pytest.mark.parametrize(
    "quantity", ["x", "power", "work", "heat", "delta_U", "energy"]
)
def test_pdf_call(dummy_sim, quantity):
    """Test a call to a probability density function"""
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
    sim.build_pdf(quantity)
    assert isinstance(sim.pdf[quantity](0.0, 0.0), float)


def test_pdf_call(dummy_sim_const_histo):
    """Test pdf correctly build for simple realization"""
    # with x =[[0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 0.8]])
    # the PDF with 4 bins should be
    # P(0) = 2 / norm
    # P(0.5) = 3 / norm
    # P(0.8) = P(1.0) = 5 / norm
    # with norm = 10*0.25
    bins = 4
    range_length = 1.0
    norm = 10 * range_length / bins

    def pdf_theo(x):
        if 0 <= x < 0.25:
            return 2.0 / norm
        if 0.25 <= x < 0.5:
            return 0.0
        if 0.5 <= x < 0.75:
            return 3 / norm
        if 0.75 <= x <= 1.0:
            return 5 / norm
        else:
            return 0.0

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
    ) = dummy_sim_const_histo
    sim.build_histogram("x", bins=bins, q_range=(0.0, 1.0))
    sim.build_pdf("x")
    for x in np.arange(0.0, 1.0, 0.1):
        assert sim.pdf["x"](x, 0) == pdf_theo(x)


def test_pdf_quantity_out_of_bounds(dummy_sim_const_histo):
    """Test pdf raise exception if quantity is out of bounds"""
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
    ) = dummy_sim_const_histo
    sim.build_histogram("x")
    sim.build_pdf("x")
    with pytest.raises(ValueError):
        # for x =[[0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 0.8]])
        # the PDF is not defined for x = 2, t=0
        sim.pdf["x"](2, 0)


def test_pdf_time_out_of_bounds(dummy_sim_const_histo):
    """Test pdf raise exception if quantity is out of bounds"""
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
    ) = dummy_sim_const_histo
    sim.build_histogram("x")
    sim.build_pdf("x")
    with pytest.raises(ValueError):
        # for x =[[0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 0.8]])
        # the PDF is not defined for t=1
        sim.pdf["x"](0, 1)


def test_pdf_gaussian_call(dummy_sim_gaussian):
    """Test pdf correctly build for simple realization"""

    def gaussian(x, x0, sigma):
        return np.exp(-((x - x0) ** 2) / (2 * sigma * sigma)) / np.sqrt(
            2 * np.pi * sigma**2
        )

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
    ) = dummy_sim_gaussian
    # x = np.random.normal(loc=[0.0, 1.0, 2.0], scale=[1.0, 3.0, 5.0], size=(tot_sims, 3))
    sim.build_histogram("x", bins=1000)
    sim.build_pdf("x")
    tol = 5e-2
    for x in np.arange(-1.0, 1.0, 0.1):
        assert sim.pdf["x"](x, 0) == pytest.approx(
            gaussian(x, 0.0, 1.0), rel=tol
        ), f"for x={x}, t=0"
    for x in np.arange(0.0, 3.0, 0.1):
        assert sim.pdf["x"](x, 51) == pytest.approx(
            gaussian(x, 1.0, 3.0), rel=tol
        ), f"for x={x}, t=51"


def test_averages(dummy_sim_const_histo):
    """Test average arrays correctly build for simple realization"""
    # with x =[[0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 0.8]])
    # the average of x is
    aver = (0.0 + 0.0 + 0.5 + 0.5 + 0.5 + 1.0 + 1.0 + 1.0 + 1.0 + 0.8) / 10.0
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
    ) = dummy_sim_const_histo
    sim.build_averages("x")
    assert sim.averages["x"][0] == aver


def test_average_call(dummy_sim_const_histo):
    """Test average function call for simple realization"""
    # with x =[[0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 0.8]])
    # the average of x is
    aver = (0.0 + 0.0 + 0.5 + 0.5 + 0.5 + 1.0 + 1.0 + 1.0 + 1.0 + 0.8) / 10.0
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
    ) = dummy_sim_const_histo
    sim.build_averages("x")
    assert sim.average_func["x"](0.0) == aver


def test_variance(dummy_sim_const_histo):
    """Test variance arrays correctly build for simple realization"""
    # with x =[[0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 0.8]])

    aver = (0.0 + 0.0 + 0.5 + 0.5 + 0.5 + 1.0 + 1.0 + 1.0 + 1.0 + 0.8) / 10.0
    aver2 = (
        0.0**2
        + 0.0**2
        + 0.5**2
        + 0.5**2
        + 0.5**2
        + 1.0**2
        + 1.0**2
        + 1.0**2
        + 1.0**2
        + 0.8**2
    ) / 10.0
    var = aver2 - aver**2

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
    ) = dummy_sim_const_histo
    sim.build_variances("x")
    assert sim.variances["x"][0] == var


def test_variance_call(dummy_sim_const_histo):
    """Test variance function call"""
    # with x =[[0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 0.8]])

    aver = (0.0 + 0.0 + 0.5 + 0.5 + 0.5 + 1.0 + 1.0 + 1.0 + 1.0 + 0.8) / 10.0
    aver2 = (
        0.0**2
        + 0.0**2
        + 0.5**2
        + 0.5**2
        + 0.5**2
        + 1.0**2
        + 1.0**2
        + 1.0**2
        + 1.0**2
        + 0.8**2
    ) / 10.0
    var = aver2 - aver**2

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
    ) = dummy_sim_const_histo
    sim.build_variances("x")
    assert sim.variance_func["x"](0) == var


def compare_sims_params(s1, s2):
    """Compare if two sims have equal parameters"""
    params = ["tot_sims", "dt", "tot_steps", "noise_scaler", "snapshot_step"]
    for k in params:
        assert s1.__dict__[k] == s2.__dict__[k]


def compare_sims_results(s1, s2):
    """Compare if two sims have equal results"""
    labels = ["times", "x", "power", "work", "heat", "delta_U", "energy"]
    for k in labels:
        assert np.array_equal(s1.results[k], s2.results[k])


def compare_sims_k_center(s1, s2):
    """Compare if two sims have the same functions for k(t) and
    center(t)"""
    t = np.arange(0, 10, 0.1)
    assert np.array_equal(s1.k(t), s2.k(t))
    assert np.array_equal(s1.center(t), s2.center(t))


def test_save_load_simulation(dummy_sim):
    """Tests that the class can be saved and reloaded"""
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
    filename = "dummy_sim_test_save.pickle"
    sim.save(filename)
    new_sim = Simulation.load(filename)
    compare_sims_params(sim, new_sim)
    compare_sims_results(sim, new_sim)
    compare_sims_k_center(sim, new_sim)
    os.remove(filename)


def test_load_wrong_simulation():
    """Tests that loading a wrong file for a simulation raises an error"""

    filename = "dummy_sim_test_save.pickle"
    wrong_data = 10
    with open(filename, "wb") as f:
        pickle.dump(wrong_data, f, pickle.DEFAULT_PROTOCOL)
    with pytest.raises(TypeError):
        # This should fail
        new_sim = Simulation.load(filename)
    os.remove(filename)


def test_name_simulation(dummy_sim):
    """Test if the simulation has a name"""
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
    assert sim.name == "dummy simulation"


def test_str_simulation(dummy_sim):
    """Test that the __str__ method shows the name of the simulation"""
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
    assert str(sim) == f'Simulation "{sim.name}"'


def test_analyse_simulation(dummy_sim):
    """Test the analysis of all results of simulation"""
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
    sim.analyse()
    for k in sim.result_labels:
        assert k in sim.histogram
        assert k in sim.pdf
        assert k in sim.averages
        assert k in sim.average_func
        assert k in sim.variances
        assert k in sim.variance_func
