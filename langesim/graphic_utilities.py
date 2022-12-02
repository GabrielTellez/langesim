"""Graphics utilities to plot the simulations results.
"""
import numpy as np
import plotly.graph_objects as go
from scipy.integrate import quad


def animate_simulation(
    times,
    xst,
    x_range=[-3.0, 6.0],
    y_range=[0, 1.5],
    bins=300,
    x_label="x",
    y_label="P(x,t)",
    show_x_eq_distrib=True,
    k=None,
    center=None,
    harmonic_potential=True,
    potential=None,
):
    """Plot and animates a simulation data results

    Args:
        times (list of float): list of times where snapshots where taken
        xst (list of list of float): list of snapshots of many
          simulations. Should have shape (tot_sims, tot_snapshots)
        x_range (list, optional): range of the data to plot. Defaults to [-3.0, 6.0].
        y_range (list, optional): range of the histogram of xst. Defaults to [0, 1.5].
        bins (int, optional): bins to compute histogram of xst. Defaults to 300.
        x_label (str, optional): label for xst in the plot. Defaults to 'x'.
        y_label (str, optional): label for the probability density of xst. Defaults to 'P(x,t)'.
        show_x_eq_distrib (bool, optional): show the equilibrium
          distribution corresponding to a harmonic oscilator with center(t)
          and stiffness k(t). Defaults to True.
        k (float function, optional): stiffness function of the potential. Defaults to k(t)=1.0.
        center (float function, optional): center function of the potential. Defaults to center(t)=0.0.
        harmonic_potential: True if working with a harmonic potential
        potential: potential energy to use when harmonic_potential=False
    Returns:
        plotly.graph_objects.figure: animation of the simulation data
    """
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

    if not harmonic_potential and potential == None and show_x_eq_distrib:
        raise ValueError(
            "Cannot show the equilibrium distribution if the potential is not provided in general force mode."
        )
    num_points = 1000
    xx = np.linspace(*x_range, num_points)
    # range should be automatic (default) for histograms or bug with outliers
    histos = [
        np.histogram(xst[:, ti], density=True, bins=bins)
        for ti in range(0, len(times))
    ]
    if harmonic_potential:
        b = [
            np.exp(-0.5 * k(t) * (xx - center(t)) ** 2) / np.sqrt(2 * np.pi / k(t))
            for t in times
        ]
    else:
        # Un-normalized Boltzmann factor
        def expU(x, t):
            return np.exp(-potential(x, t))
        # Normalize the PDF
        def expUnorm(x, t):
            Z = quad(expU, -np.inf, np.inf, args = (t,))[0]
            return expU(x,t) / Z

        b = [expUnorm(xx, t) for t in times]

    # make figure
    fig_dict = {"data": [], "layout": {}, "frames": []}
    fig_dict["layout"] = go.Layout(
        xaxis=dict(range=x_range, autorange=False),
        yaxis=dict(range=y_range, autorange=False),
        xaxis_title=x_label,
        yaxis_title=y_label,
    )
    fig_dict["layout"]["updatemenus"] = [
        {
            "type": "buttons",
            "buttons": [
                {
                    "args": [
                        None,
                        {
                            "frame": {"duration": 500, "redraw": False},
                            "fromcurrent": True,
                            "transition": {
                                "duration": 300,
                                "easing": "quadratic-in-out",
                            },
                        },
                    ],
                    "label": "Play",
                    "method": "animate",
                },
                {
                    "args": [
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        },
                    ],
                    "label": "Pause",
                    "method": "animate",
                },
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top",
        }
    ]
    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 14},
            "prefix": "t = ",
            "visible": True,
            "xanchor": "right",
        },
        "transition": {"duration": 300, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": [],
    }
    fig_dict["data"] = [go.Bar(x=histos[0][1], y=histos[0][0], name=y_label)]
    if show_x_eq_distrib:
        fig_dict["data"].append(go.Scatter(x=xx, y=b[0], name=f"Eq. distr."))
    # make frames
    for time_index in range(0, len(times)):
        frame_data = [go.Bar(x=histos[time_index][1], y=histos[time_index][0])]
        if show_x_eq_distrib:
            frame_data.append(go.Scatter(x=xx, y=b[time_index]))
        frame = go.Frame(data=frame_data, name=time_index, traces=[0, 1])
        fig_dict["frames"].append(frame)
        slider_step = {
            "args": [
                [time_index],
                {
                    "frame": {"duration": 300, "redraw": False},
                    "mode": "immediate",
                    "transition": {"duration": 300},
                },
            ],
            "label": round(times[time_index], 3),
            "method": "animate",
        }
        sliders_dict["steps"].append(slider_step)
    fig_dict["layout"]["sliders"] = [sliders_dict]
    fig = go.Figure(fig_dict)
    fig.update_layout(bargap=0)
    return fig


################################################################################


def plot_quantity(
    t_array,
    y_array,
    t_range=None,
    y_range=None,
    t_label="t",
    y_label="",
    scatter_label=None,
):
    """Plots y_array as function of t_array

    Args:
        t_array (np.array): time axis array of
        y_array (np.array): quantity to plot array
        t_range (list, optional): t range. Defaults to Autoscale.
        y_range (list, optional): y range. Defaults to Autoscale.
        t_label (str, optional): label for t axis. Defaults to 't'.
        y_label (str, optional): label for y axis. Defaults to ''.

    Returns:
        plotly.graph_objects.figure: the plot of the quantity
    """
    if scatter_label == None:
        scatter_label = y_label

    fig_dict = {"data": [], "layout": {}}

    if t_range == None:
        xaxis_dict = dict(autorange=True)
    else:
        xaxis_dict = dict(range=t_range, autorange=False)
    if y_range == None:
        yaxis_dict = dict(autorange=True)
    else:
        yaxis_dict = dict(range=y_range, autorange=False)
    fig_dict["layout"] = go.Layout(
        xaxis=xaxis_dict, yaxis=yaxis_dict, xaxis_title=t_label, yaxis_title=y_label
    )
    fig_dict["data"].append(go.Scatter(x=t_array, y=y_array, name=scatter_label))
    fig = go.Figure(fig_dict)
    return fig


################################################################################
