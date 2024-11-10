"""Simulator and Simulation classes 
"""
from collections.abc import Callable
from typing import List, Tuple
import numpy as np
import plotly.graph_objects as go
import numba as nb
import cloudpickle
import pickle
from .graphic_utilities import animate_simulation, plot_quantity
from .sampler import make_sampler


def make_simulator(
    tot_sims=1000,
    dt=0.001,
    tot_steps=10000,
    noise_scaler=1.0,
    snapshot_step=100,
    k=None,
    center=None,
    harmonic_potential=True,
    force=None,
    potential=None,
    initial_distribution=None,
):
    """Makes a numba compiled njit langevin simulator of a brownian
    particle in an external time variable potential or force

    Args:
        tot_sims (int, optional): default total number of simulations. Defaults to 1000.
        dt (float, optional): default time step. Defaults to 0.001.
        tot_steps (int, optional): default number of steps of each simulation. Defaults to 10000.
        noise_scaler (float, optional): brownian noise variance k_B T. Defaults to 1.0.
        snapshot_step (int, optional): save a snapshot of simulation at
            each snapshot_step time. Defaults to 100.
        k (float function, optional): stiffness function k(t) of the potential. Defaults to k(t)=1.0.
        center (float function, optional): center function of the potential. Defaults to center(t)=0.0.
        harmonic_potential (boolean, optional): If True: the external potential
            is harmonic with stiffness k(t) and center(t).
            If False the external force is given by the force argument or
            the the external potential is given by potential argument
        force (float function(x,t), optional): the external force
        potential (float function(x,t), optional): the external potential
        initial_distribution (float function(), optional): initial
            condition for x(0). Default: sampled from Boltzmann factor at time 0: exp(-U(x,0))

    Returns:
        numba.Dispatcher: numba JIT compiled function that performs simulations
    """
    if harmonic_potential:
        if k == None:

            def k(t):
                """Default stiffness for the harmonic potential
                t |--> 1.0
                """
                return 1.0

        if center == None:

            def center(t):
                """Default center for the harmonic potential
                t |--> 0.0
                """
                return 0.0

        k = nb.njit(k)
        center = nb.njit(center)

        @nb.njit
        def f(x, t):
            """Force on the particle"""
            return -k(t) * (x - center(t))

        @nb.njit
        def U(x, t):
            """Harmonic potential energy"""
            return 0.5 * k(t) * (x - center(t)) * (x - center(t))

        if initial_distribution == None:

            @nb.njit
            def initial_distribution():
                """Samples initial position according to the equilibrium distribution

                Returns:
                    float: x random sample distributed according to exp(-k(0)*(x-center(0))**2/2)
                """
                return np.random.normal(center(0.0), scale=np.sqrt(1.0 / k(0.0)))

        else:
            initial_distribution = nb.njit(initial_distribution)

    else:
        # In general force mode
        # check that the force or the potential are defined
        if force == None and potential == None:
            raise ValueError(
                "In general force mode, the force or the potential have to be provided. Both cannot be None"
            )
        if force == None:
            # We need to compute the force from the potential
            U = nb.njit(potential)

            def force(x, t):
                dx = 1e-9
                return -(U(x + dx, t) - U(x, t)) / dx

        if potential == None:
            raise ValueError(
                "In general force mode, the potential have to be provided. It is too slow to compute it from the force."
            )
            # We need the potential from the force
            # this will never run:
            # f = nb.njit(force)

            # def potential(x, t):
            #     # Integral by basic numerical quadrature trapezoidal rule
            #     # This is way too slow
            #     dx = 1e-6
            #     x0 = 0.0
            #     xs = np.arange(x0 + dx, x - dx, dx)
            #     integral = 0.5 * (f(x0, t) + f(x, t))
            #     for xp in xs:
            #         integral += f(xp, t)
            #     integral = integral * dx
            #     return integral

        if initial_distribution == None:
            initial_distribution, _ = make_sampler(lambda x: np.exp(-potential(x, 0)))

        initial_distribution = nb.njit(initial_distribution)

        f = nb.njit(force)
        U = nb.njit(potential)
        # Test if potential and force are coherent
        dx = 1e-9
        xx = np.linspace(-10.0, 10.0, 100)
        tt = np.linspace(0.0, 10.0, 100)
        ff = [f(xx, t) for t in tt]
        minusgradU = [(U(xx, t) - U(xx + dx, t)) / dx for t in tt]
        if not np.allclose(ff, minusgradU):
            raise ValueError(
                "Force and potential are nonconsistent: Force is different from minus grad potential"
            )

    @nb.njit
    def one_simulation(
        dt=dt,
        tot_steps=tot_steps,
        xinit=0.0,
        noise_scaler=noise_scaler,
        snapshot_step=snapshot_step,
    ):
        """Function that performs one simulation

        Args:
            tot_sims (int, optional): default total number of simulations. Defaults to 1000.
            dt (float, optional): default time step. Defaults to 0.001.
            tot_steps (int, optional): default number of steps of each simulation. Defaults to 10000.
            noise_scaler (float, optional): brownian noise scale k_B T. Defaults to 1.0.
            snapshot_step (int, optional): save a snapshot of simulation at each snapshot_step time. Defaults to 100.


        Returns:
            x, power, work, heat, delta_U, energy (tuple): array of
              snapshots of the simulation
              x = position, power, work, heat, delta_U=energy difference between
               current state and initial state, energy

        """
        tot_snapshots = int(tot_steps / snapshot_step) + 1
        x = np.zeros(tot_snapshots, dtype=np.float64)
        work = np.zeros_like(x)
        power = np.zeros_like(x)
        heat = np.zeros_like(x)
        delta_U = np.zeros_like(x)
        energy = np.zeros_like(x)
        times_snapshots = np.zeros_like(x)
        xold = xinit
        x[0] = xinit
        energy[0] = U(x[0], 0)
        w = 0.0
        q = 0.0
        p = 0.0
        # step = 0
        snapshot_index = 0
        # Use a predefined array of times to avoid rounding errors from
        # computing it as step*dt
        times = np.arange(0, (1 + tot_steps) * dt, dt)
        for step, t in enumerate(times[:-1]):
            # while snapshot_index <= tot_snapshots:
            #   t = step * dt
            xnew = (
                xold
                + f(xold, t) * dt
                + np.random.normal() * np.sqrt(2.0 * dt * noise_scaler)
            )
            p = U(xnew, times[step + 1]) - U(xnew, t)
            w = w + p
            q = q + U(xnew, t) - U(xold, t)
            # step = step + 1
            if (step + 1) % snapshot_step == 0:
                # Take a snapshot of the simulation
                snapshot_index = snapshot_index + 1
                x[snapshot_index] = xnew
                power[snapshot_index] = p / dt
                work[snapshot_index] = w
                heat[snapshot_index] = q
                delta_U[snapshot_index] = U(xnew, times[step + 1]) - U(x[0], times[0])
                energy[snapshot_index] = U(xnew, times[step + 1])
                times_snapshots[snapshot_index] = times[step + 1]
                if snapshot_index >= tot_snapshots:
                    # No need to continue the simulation after the last snapshot
                    # have been taken
                    break
            xold = xnew

        return x, power, work, heat, delta_U, energy, times_snapshots

    @nb.jit(nopython=True, parallel=True)
    def many_sims_parallel(
        tot_sims=tot_sims,
        dt=dt,
        tot_steps=tot_steps,
        noise_scaler=noise_scaler,
        snapshot_step=snapshot_step,
    ):
        """Function that performs many simulations with a given initial
        condition at t=0

        Args:
          tot_sims (int, optional): default total number of simulations. Defaults to 1000.
          dt (float, optional): default time step. Defaults to 0.001.
          tot_steps (int, optional): default number of steps of each simulation. Defaults to 10000.
          noise_scaler (float, optional): brownian noise scale k_B T. Defaults to 1.0.
          snapshot_step (int, optional): save a snapshot of simulation at
          each snapshot_step time. Defaults to 100.

        Returns:
            times, x, power, work, heat, delta_U, energy:
              times: list of times of the snapshots
              x, power, work, heat, delta_U, energy = list of positions,
              power, ..., snapshots for each simulation
              ie. x[sim] = [x(0), x(snapshot_step*dt), x(2*snapshot_step*dt), .... ] for simulation number sim.
        """
        tot_snapshots = int(tot_steps / snapshot_step) + 1
        x = np.zeros((tot_sims, tot_snapshots))
        work = np.zeros_like(x)
        power = np.zeros_like(x)
        heat = np.zeros_like(x)
        delta_U = np.zeros_like(x)
        energy = np.zeros_like(x)
        # times = np.arange(0, (1 + tot_steps) * dt, dt * snapshot_step)
        times_snapshots = np.zeros_like(x)
        for sim_num in nb.prange(tot_sims):
            # initial position taken from a given initial_distribution
            xinit = initial_distribution()
            (
                x[sim_num],
                power[sim_num],
                work[sim_num],
                heat[sim_num],
                delta_U[sim_num],
                energy[sim_num],
                times_snapshots[sim_num],
            ) = one_simulation(
                dt=dt,
                tot_steps=tot_steps,
                xinit=xinit,
                noise_scaler=noise_scaler,
                snapshot_step=snapshot_step,
            )
        times = times_snapshots[0]  # all snapshot times should be the same.
        return times, x, power, work, heat, delta_U, energy

    return many_sims_parallel


################################################################################


class Simulation:
    """Stores simulation parameters and results.
    Analyses the results: builds PDF of the simulation results (position,
    work, etc..)

    Attributes:

        tot_sims (int): Total number of individual simulations.
        dt (float): time step
        tot_steps (int): total steps of the simulation.
        noise_scaler (float): brownian noise variance (k_B T). Defaults to 1.0.
        name (str): name of the simulation.
        k (Callable): function k(t) that gives the harmonic potential stifness as a function of time.
        center (Callable): function center(t) of the harmonic potential as a function of time
        harmonic_potential (bool): True if the potential is harmonic, False otherwise.
        force (Callable): function force(x,t) applied on the particle at position x at time t.
        potential (Callable): function potential(x,t) energy of the particle at position x at time t.
        results (dict): results of the simulations::

            results = {
                "times": times, # (ndarray): times where snapshot where taken.
                "x": x,         # (ndarray of shape (tot_sims, tot_snapshots)):
                                # x[sim][ts] is the position of the brownian particle in
                                # simulation number num and snapshot index ts
                "power": power, # (ndarray of shape (tot_sims, tot_snapshots)):
                                # power[sim][ts] is the power into the system
                                # at snapshot ts and simulation sim
                "work": work,   # (ndarray of shape (tot_sims, tot_snapshots)):
                                # work[sim][ts] is the work performed into the
                                # system in simulation sim up to snapshot ts
                "heat": heat,   # (ndarray of shape (tot_sims, tot_snapshots)):
                                # heat[sim][ts] into the system in simulation sim
                                # up to snapshot ts
                "delta_U": delta_U, # (ndarray of shape (tot_sims, tot_snapshots)):
                                    # delta_U[sim][ts] is the energy difference
                                    # between snapshot = 0 and current snapshot ts
                                    # in simulation sim
                "energy": energy,   # (ndarray of shape (tot_sims, tot_snapshots)):
                                    # energy[sim][ts] in simulation sim at snapshot ts
            }

        histogram (dict): histograms of quantities::

            histogram = {
                "x": list of histograms of position
                "power": list histogram of power
                "work": list of histogram of work
                "heat": list of histograms of heat
                "delta_U": list of histograms of energy difference between t=0 and time t
                "energy": list of histograms of energy
            }
            # Example: Simulation.histogram["x"][i] gives the histogram of "x" at time
            # snapshot number i.

        pdf (dict): probability distribution functions of x, power, work, heat,
            delta_U, energy::

            # Example: pdf["x"](x,t) gives the PDF of position x at time t.

        averages (dict): Averages of position, work, heat, power, delta_U and energy::

            # Example: averages["x"][i] is the position average at snapshot number i.

        average_func (dict): dictionary of function that give the average at a
            given time::

            # Example: average_func["x"](t)= position average at time t.

        variances (dict): variances of position, work, heat, power, delta_U and energy::

            # Example: variances["x"][i] is the position variance at snapshot number i.

        variance_func (dict): variances of position, power, heat, work, energy at time
            t::

            # t. Example: variance_func["x"](t)= position variance at time t.

    """

    result_labels = ["x", "power", "work", "heat", "delta_U", "energy"]

    def __init__(
        self,
        tot_sims,
        dt,
        tot_steps,
        noise_scaler,
        snapshot_step,
        k,
        center,
        results,
        name="",
        harmonic_potential=True,
        force=None,
        potential=None,
    ):
        """Initializes the Simulation class with parameters and raw results

        Args:
            tot_sims (int): total number of simulations.
            dt (float): time step.
            tot_steps (int): number of steps of each simulation.
            noise_scaler (float): brownian noise scale k_B T. Defaults to 1.0.
            snapshot_step (int): a snapshot of simulation has been saved each snapshot_step time.
            k (float function): stiffness of the potential
            center (float function): center of the potential
            results (tuple): results in the form (times, x, power, work, heat, delta_U, energy) where:

                times (ndarray):
                    ndarray of times where snapshot where taken.

                x (ndarray of shape (tot_sims, tot_snapshots)):
                    x[sim][ts] is the position of the brownian particle in simulation number num and snapshot ts

                power (ndarray of shape (tot_sims, tot_snapshots)):
                    power[sim][ts] is the power into the system at snapshot ts and simulation sim

                work (ndarray of shape (tot_sims, tot_snapshots)):
                    work[sim][ts] is the work performed into the system in simulation sim up to snapshot ts

                heat (ndarray of shape (tot_sims, tot_snapshots)):
                    heat[sim][ts] into the system in simulation sim up to snapshot ts

                delta_U (ndarray of shape (tot_sims, tot_snapshots)):
                    delta_U[sim][ts] is the energy difference between snapshot = 0 and current snapshot ts in
                    simulation sim

                energy (ndarray of shape (tot_sims, tot_snapshots)):
                    energy[sim][ts] in simulation sim at snapshot ts

            name (string, optional): name of the simulation
            harmonic_potential(boolean, optional): True if the potential is harmonic
            force (float function(x,t), optional): force when the potential
              is not harmonic
            potential (float function(x,t), optional): potential when the potential
              is not harmonic

        """
        self.tot_sims = tot_sims
        self.dt = dt
        self.tot_steps = tot_steps
        self.noise_scaler = noise_scaler
        self.snapshot_step = snapshot_step
        self.name = name
        self.k = k
        self.center = center
        self.harmonic_potential = harmonic_potential
        self.force = force
        self.potential = potential

        (times, x, power, work, heat, delta_U, energy) = results
        self.results = {
            "times": times,
            "x": x,
            "power": power,
            "work": work,
            "heat": heat,
            "delta_U": delta_U,
            "energy": energy,
        }
        self.histogram: dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}
        self.pdf: dict[str, Callable[[float, float], float]] = {}
        self.averages: dict[str, np.array] = {}
        self.average_func: dict[str, Callable[[float], float]] = {}
        self.variances: dict[str, np.array] = {}
        self.variance_func: dict[str, Callable[[float], float]] = {}

    def __str__(self):
        return f'Simulation "{self.name}"'

    def build_histogram(self, quantity, bins=300, q_range=None):
        """Builds the histogram of a quantity

        Args:
            quantity (string): quantity to build its histogram. Should be in ["x", "power", "work", "heat", "delta_U", "energy"]
            bins (int, optional): bins for the histogram. Defaults to 300.
            q_range (list, optional): range for the quantity. Defaults to None for automatic range. Not using automatic range can introduce bugs in the histograms if there are outliers.
        """
        if quantity not in self.result_labels:
            raise ValueError(f"quantity {quantity} must be in {self.result_labels}")
        self.histogram[quantity] = [
            np.histogram(
                self.results[quantity][:, ti],
                density=True,
                range=q_range,
                bins=bins,
            )
            for ti in range(0, len(self.results["times"]))
        ]

    def build_pdf(self, quantity, bins=300, q_range=None):
        """Builds the probability density function (PDF) for a quantity.
        The PDF is build and function is defined to access it in self.pdf(quantity)

        Args:
            quantity (string): quantity to build its pdf. Should be in ["x", "power", "work", "heat", "delta_U", "energy"]
            bins (int, optional): bins for the histogram. Defaults to 300.
            q_range (list, optional): range for the quantity. Defaults to None for automatic range. Not using automatic range can introduce bugs in the histograms if there are outliers.
        """
        if quantity not in self.result_labels:
            raise ValueError(f"quantity {quantity} must be in {self.result_labels}")
        # if quantity not in self.histogram.keys():
        # (Re)build the histogram as bins and q_range might be different
        # from previous evaluation
        self.build_histogram(quantity, bins, q_range)

        def pdf(x, t):
            # To do: Rewrite this to be numpy compatible? Nevermind: afterwards one can
            # use np.vectorize. Maybe use scipy interpolations.

            # time t to snapshot index ti
            bins_t = self.results["times"]
            if t < np.min(bins_t) or t > np.max(bins_t):
                raise ValueError(
                    f"In PDF of {quantity}: time={t} is out of bounds [{np.min(bins_t)}, {np.max(bins_t)}]"
                )
            ti = np.digitize(t, bins_t) - 1
            if ti < 0:
                ti = 0

            # self.histogram[quantity][ti, 0] # contains P(x)
            # self.histogram[quantity][ti, 1] # contains x
            # get the index corresponding to value x in the bins
            (hist, bins_x) = self.histogram[quantity][ti]
            if x < np.min(bins_x) or x > np.max(bins_x):
                raise ValueError(
                    f"{quantity}={x} is out of bounds [{np.min(bins_x)}, {np.max(bins_x)}]"
                )

            index_x = np.digitize(x, bins_x) - 1
            if index_x < 0:
                index_x = 0
            return hist[index_x]

        self.pdf[quantity] = pdf

        # Trying interp2d but this does not work because of different
        # ranges in x for different times t

        # t = self.results['times']
        # Using initial positions for all t does not work
        # x = self.histogram[quantity][0, 1][:-1]
        # This has the wrong shape
        # pdf_values = self.histogram[quantity][:, 0]
        # pdf = interp2d(x, t, pdf_values)

    def build_averages(self, quantity):
        """Computes the average of a quantity.
        The average at time t (with corresponding time_index of the snapshot)
        is stored in averages[quantity][time_index]
        A function giving the average as a function of time is created and
        stored in average_func(quantity)

        Args:
            quantity (string): quantity to build its averages. Should be in ["x", "power", "work", "heat", "delta_U", "energy"]

        Raises:
            ValueError: if quantity is not in ["x", "power", "work", "heat", "delta_U", "energy"]

        """
        if quantity not in self.result_labels:
            raise ValueError(f"quantity {quantity} must be in {self.result_labels}")
        self.averages[quantity] = np.average(self.results[quantity], axis=0)

        def av_fnct(t):
            # time t to snapshot index ti
            bins_t = self.results["times"]
            if t < np.min(bins_t) or t > np.max(bins_t):
                raise ValueError(
                    f"In average of {quantity}: time={t} is out of bounds [{np.min(bins_t)}, {np.max(bins_t)}]"
                )
            ti = np.digitize(t, bins_t) - 1
            if ti < 0:
                ti = 0
            return self.averages[quantity][ti]

        self.average_func[quantity] = av_fnct

    def build_variances(self, quantity):
        """Computes the variance of a quantity.
        The variance at time t (with corresponding time_index of the snapshot)
        is stored in variances[quantity][time_index]
        A function giving the variance as a function of time is created and
        stored in variance_func(quantity)

        Args:
            quantity (string): quantity to build its variances. Should be in ["x", "power", "work", "heat", "delta_U", "energy"]

        Raises:
            ValueError: if quantity is not in ["x", "power", "work", "heat", "delta_U", "energy"]

        """
        if quantity not in self.result_labels:
            raise ValueError(f"quantity {quantity} must be in {self.result_labels}")
        self.variances[quantity] = np.var(self.results[quantity], axis=0)

        def var_fnct(t):
            # time t to snapshot index ti
            bins_t = self.results["times"]
            if t < np.min(bins_t) or t > np.max(bins_t):
                raise ValueError(
                    f"In average of {quantity}: time={t} is out of bounds [{np.min(bins_t)}, {np.max(bins_t)}]"
                )
            ti = np.digitize(t, bins_t) - 1
            if ti < 0:
                ti = 0
            return self.variances[quantity][ti]

        self.variance_func[quantity] = var_fnct

    def animate_pdf(
        self,
        quantity,
        x_range=[-3.0, 3.0],
        y_range=[0, 1.5],
        bins=300,
        show_x_eq_distrib=None,
    ):
        """Shows an animation of the evolution of the PDF of a quantity

        Args:
            quantity (string): quantity to animate its PDF. Must be in ["x", "power", "work", "heat", "delta_U", "energy"]
            x_range (list, optional): range for the quantity in the PDF. Defaults to [-3.0, 3.0].
            y_range (list, optional): range for the PDF value. Defaults to [0, 1.5].
            bins (int, optional): bins for the histogram. Defaults to 300.
            show_x_eq_distrib (boolean, optional): if True the instantaneous
            equilibrium position distribution is shown. Defaults to None.

        Raises:
            ValueError: if quantity is not in ["x", "power", "work", "heat", "delta_U", "energy"]

        Returns:
            plotly.graph_objects.figure: animation of the PDF
        """
        if quantity not in self.result_labels:
            raise ValueError(f"quantity {quantity} must be in {self.result_labels}")
        if show_x_eq_distrib == None:
            show_x_eq_distrib = quantity == "x"
        return animate_simulation(
            self.results["times"],
            self.results[quantity],
            x_range=x_range,
            y_range=y_range,
            bins=bins,
            x_label=quantity,
            y_label=f"P({quantity},t)",
            show_x_eq_distrib=show_x_eq_distrib,
            k=self.k,
            center=self.center,
            harmonic_potential=self.harmonic_potential,
            potential=self.potential,
        )

    def save(self, filename):
        """Saves the simulation

        Args:
            filename (string): filename where the simulation is saved
        """
        with open(filename, "wb") as f:
            cloudpickle.dump(self, f, pickle.DEFAULT_PROTOCOL)

    @classmethod
    def load(cls, filename):
        """Loads a simulation from file

        Args:
            filename (string): filename of the simulation to load

        Returns:
            Simulation: the loaded simulation
        """
        with open(filename, "rb") as f:
            _sim = pickle.load(f)
        if isinstance(_sim, cls):
            return _sim
        else:
            raise TypeError(f'File "{filename}" does not contain a simulation.')

    def analyse(self):
        """Builds all histograms, PDF, averages and variances"""
        for k in self.result_labels:
            self.build_histogram(k)
            self.build_pdf(k)
            self.build_averages(k)
            self.build_variances(k)

    def plot_average(
        self, quantity, t_range=None, y_range=None, t_label="t", y_label=None
    ):
        """Plots <quantity> as a function of time

        Args:
            quantity (string): quantity to plot. Should be in ["x", "power", "work", "heat", "delta_U", "energy"]
            t_array (np.array): time axis array
            y_array (np.array): quantity to plot array
            t_range (list, optional): t range. Defaults to Autoscale.
            y_range (list, optional): y range. Defaults to Autoscale.
            t_label (str, optional): label for t axis. Defaults to 't'.
            y_label (str, optional): label for y axis. Defaults to ''.

        Raises:
            ValueError: if quantity is not in ["x", "power", "work", "heat", "delta_U", "energy"]

        Returns:
            plotly.graph_objects.figure: plot of the quantity
        """
        if quantity not in self.result_labels:
            raise ValueError(f"quantity {quantity} must be in {self.result_labels}")
        if quantity not in self.averages:
            self.build_averages(quantity)
        if y_label == None:
            y_label = f"<{quantity}>"
        # make figure
        fig = plot_quantity(
            self.results["times"],
            self.averages[quantity],
            t_range=t_range,
            y_range=y_range,
            t_label=t_label,
            y_label=y_label,
        )
        return fig

    def plot_variance(
        self, quantity, t_range=None, y_range=None, t_label="t", y_label=None
    ):
        """Plots the variance of quantity as a function of time

        Args:
            quantity (string): quantity to plot. Should be in ["x", "power", "work", "heat", "delta_U", "energy"]
            t_array (np.array): time axis array
            y_array (np.array): quantity to plot array
            t_range (list, optional): t range. Defaults to Autoscale.
            y_range (list, optional): y range. Defaults to Autoscale.
            t_label (str, optional): label for t axis. Defaults to 't'.
            y_label (str, optional): label for y axis. Defaults to ''.

        Raises:
            ValueError: if quantity is not in ["x", "power", "work", "heat", "delta_U", "energy"]

        Returns:
            plotly.graph_objects.figure: plot of the quantity
        """
        if quantity not in self.result_labels:
            raise ValueError(f"quantity {quantity} must be in {self.result_labels}")
        if quantity not in self.variances:
            self.build_variances(quantity)
        if y_label == None:
            y_label = f"Var({quantity})"
        # make figure
        fig = plot_quantity(
            self.results["times"],
            self.variances[quantity],
            t_range=t_range,
            y_range=y_range,
            t_label=t_label,
            y_label=y_label,
        )
        return fig

    def plot_sim(
        self,
        quantity: str,
        sim_list: List[int],
        sim_labels=None,
        t_range=None,
        y_range=None,
        t_label="t",
        y_label=None,
    ):
        """Plots quantity as a function of time for simulations listed in sim_list

        Args:
            quantity (str): quantity to plot. Should be in ["x", "power", "work", "heat", "delta_U", "energy"]
            sim_list (list of int): list of the simulation numbers to plot.
            sim_labels (list of str, optional): list of labels for each
              trace in the plot. Defaults to None.
            t_array (np.array): time axis array
            y_array (np.array): quantity to plot array
            t_range (list, optional): t range. Defaults to Autoscale.
            y_range (list, optional): y range. Defaults to Autoscale.
            t_label (str, optional): label for t axis. Defaults to 't'.
            y_label (str, optional): label for y axis. Defaults to ''.

        Raises:
            ValueError: if quantity is not in ["x", "power", "work", "heat", "delta_U", "energy"]

        Returns:
            plotly.graph_objects.figure: plot of the quantity
        """
        if quantity not in self.result_labels:
            raise ValueError(f"quantity {quantity} must be in {self.result_labels}")
        if y_label == None:
            y_label = quantity
        # Check if the list of simulation to plot is in range
        if any(num not in range(self.tot_sims) for num in sim_list):
            raise ValueError(
                f"Cannot plot a simulation {sim_list=} out of range {range(self.tot_sims)}"
            )
        if sim_labels == None:
            # Use simulation numbers as labels for each scatter
            sim_labels = [l for l in map(str, sim_list)]
        if len(sim_labels) != len(sim_list):
            raise ValueError(
                f"List of simulations {sim_list=} does not match list of labels {sim_labels=}"
            )
        # print(f"{sim_list=}, {sim_labels=}")
        t = self.results["times"]
        xs = self.results[quantity]
        # Initialize plot with the first trace
        fig = plot_quantity(
            t,
            xs[sim_list[0]],
            t_range=t_range,
            y_range=y_range,
            t_label=t_label,
            y_label=y_label,
            scatter_label=sim_labels[0],
        )
        # Add the rest of traces
        for (k, i) in enumerate(sim_list[1:], 1):
            # print(f"{i=}, {k=}, {sim_labels[k]=}")
            fig.add_trace(go.Scatter(x=t, y=xs[i], name=sim_labels[k]))
        return fig


##################################################################################


class Simulator:
    """Simulator class for Langevin dynamics of a harmonic oscillator with
    variable potential. Encapsulates the simulator, perform
    simulations, analyses them and store results
    of simulations.

    Attributes:

        tot_sims (int): default total number of simulations to run per batch.
        dt (float): default time step.
        tot_steps (int): default total steps of the simulation
        noise_scaler (float): default brownian noise variance (k_B T).
        k (Callable): function k(t) that gives the harmonic potential stifness as a function of time.
        center (Callable): function center(t) of the harmonic potential as a function of time
        harmonic_potential (bool): True if the potential is harmonic, False otherwise.
        force (Callable): function force(x,t) applied on the particle at position x at time t.
        potential (Callable): function potential(x,t) energy of the particle at position x at time t.
        initial_distribution (Callable): function without arguments that
            samples the initial distribution of the particles.
        simulator (numba.Dispatcher): numba JIT function that performs simulations.
        simulations_performed: number of batches of simulations that the Simulator has
            done. It is the number of times that Simulator.run() has been
            called.
        simulation (list): list of Simulation objects that contain the results
            of the simulations that have been runned.

    """

    def __init__(
        self,
        tot_sims=1000,
        dt=0.001,
        tot_steps=10000,
        noise_scaler=1.0,
        snapshot_step=100,
        k=None,
        center=None,
        harmonic_potential=True,
        force=None,
        potential=None,
        initial_distribution=None,
    ):
        """Initializes the Simulator

        Args:
            tot_sims (int, optional): total number of simulations. Defaults to 1000.
            dt (float, optional): time step. Defaults to 0.001.
            tot_steps (int, optional): total steps of each simulation. Defaults to 10000.
            noise_scaler (float, optional): brownian noise scale k_B T. Defaults to 1.0.
            snapshot_step (int, optional): save a snapshot of simulation at each snapshot_step time. Defaults to 100.
            k (float function, optional): stiffness function k(t) of the potential. Defaults to k(t)=1.0.
            center (float function, optional): center function of the potential. Defaults to center(t)=0.0.
            harmonic_potential (boolean, optional): If True: the external potential
              is harmonic with stiffness k(t) and center(t).
              If False the external force is given by the force argument or
              the the external potential is given by potential argument
            force (float function(x,t), optional): the external force
            potential (float function(x,t), optional): the external potential
            initial_distribution (float function(), optional): initial
              condition for x(0). Default: sampled from Boltzmann factor at time 0: exp(-U(x,0))
        """
        initial_distribution_not_compiled = initial_distribution
        if harmonic_potential:
            if k == None:

                def k(t):
                    """Default stiffness for the harmonic potential
                    t |--> 1.0
                    """
                    return 1.0

            if center == None:

                def center(t):
                    """Default center for the harmonic potential
                    t |--> 0.0
                    """
                    return 0.0

            def force(x, t):
                """Force on the particle"""
                return -k(t) * (x - center(t))

            def potential(x, t):
                """Harmonic potential energy"""
                return 0.5 * k(t) * (x - center(t)) * (x - center(t))

            if initial_distribution_not_compiled == None:

                def initial_distribution_not_compiled():
                    """Samples initial position according to the equilibrium distribution
                    Returns:
                        float: x random sample distributed according to exp(-k(0)*(x-center(0)**2/2))
                    """
                    return np.random.normal(center(0.0), scale=np.sqrt(1.0 / k(0.0)))

        # store the default parameters for simulations
        self.tot_sims = tot_sims
        self.dt = dt
        self.tot_steps = tot_steps
        self.noise_scaler = noise_scaler
        self.snapshot_step = snapshot_step
        self.k = k
        self.center = center
        self.harmonic_potential = harmonic_potential
        self.force = force
        self.potential = potential
        self.initial_distribution = initial_distribution_not_compiled
        self.simulator = make_simulator(
            tot_sims=tot_sims,
            dt=dt,
            tot_steps=tot_steps,
            noise_scaler=noise_scaler,
            snapshot_step=snapshot_step,
            k=k,
            center=center,
            harmonic_potential=harmonic_potential,
            force=force,
            potential=potential,
            initial_distribution=initial_distribution,
        )
        self.simulations_performed = 0
        # list of Simulations classes to store results of simulations
        self.simulation: List[Simulation] = []

    def run(
        self,
        tot_sims=None,
        dt=None,
        tot_steps=None,
        noise_scaler=None,
        snapshot_step=None,
        name="",
    ):
        """Runs a batch of tot_sim simulations and appends the results to
        Simulator.simulation list.

        Args:
          tot_sims (int, optional): total number of simulations.
          dt (float, optional): time step.
          tot_steps (int, optional): total steps of each simulation.
          noise_scaler (float, optional): brownian noise scale k_B T.
          snapshot_step (int, optional): save a snapshot of simulation at
            each snapshot_step time.
          name (str, optional): name of the simulation
        """
        if tot_sims == None:
            tot_sims = self.tot_sims
        if dt == None:
            dt = self.dt
        if tot_steps == None:
            tot_steps = self.tot_steps
        if noise_scaler == None:
            noise_scaler = self.noise_scaler
        if snapshot_step == None:
            snapshot_step = self.snapshot_step

        results = self.simulator(tot_sims, dt, tot_steps, noise_scaler, snapshot_step)
        sim = Simulation(
            tot_sims,
            dt,
            tot_steps,
            noise_scaler,
            snapshot_step,
            self.k,
            self.center,
            results,
            name,
            harmonic_potential=self.harmonic_potential,
            force=self.force,
            potential=self.potential,
        )
        self.simulation.append(sim)
        self.simulations_performed += 1

    def analyse(self, sim_num=None):
        """Performs the analysis of simulation number sim_num

        Args:
            sim_num (int, optional): simulation number. Defaults to last simulation performed.
        """
        if sim_num == None:
            sim_num = self.simulations_performed - 1
        if sim_num < 0 or sim_num > self.simulations_performed - 1:
            raise ValueError(f"Simulation number {sim_num} does not exists")

        self.simulation[sim_num].analyse()
