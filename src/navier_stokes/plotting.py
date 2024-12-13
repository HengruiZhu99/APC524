import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

# Handy type alias
FArray = NDArray[np.float64]


def plot_results(
    x: FArray,
    u_initial: FArray,
    u_comparison: tuple[FArray, ...],
    labels: tuple[str, ...],
    title: str = "",
):
    """
    Displays a plot of different solutions to 1D Navier stokes equation
    in comparison with the initial condition.

    Args:
        x (FArray) = position
        u_initial (FArray) = fluid velocity at initial condition
        u_comparison (tuple[FArray, ...]) = different solutions for comparison
        labels (tuple[str,...]) = labels for items in u_comparison
        title (str) = title for the plot
    """

    plt.figure(figsize=(10, 5), dpi=200)
    plt.plot(x, u_initial, label="Initial condition")

    # plot each solution with corresponding label
    for u_plot, label in zip(u_comparison, labels):
        plt.plot(x, u_plot, label=label, linestyle="--")

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("u")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_kinetic_energy(time_values: FArray, ke_values: tuple[float, ...]):
    """
    Displays a plot of kinetic energy over time

    Args:
        time_values (FArray) = time steps
        ke_values (tuple[float, ...]) = kinetic energy at given time steps
    """
    plt.figure(figsize=(10, 5))
    plt.plot(time_values, ke_values, marker="o", linestyle="-", label="Kinetic Energy")
    plt.title("Evolution of Average Kinetic Energy vs Time")
    plt.xlabel("Time Step")
    plt.ylabel("Average Kinetic Energy")
    plt.legend()
    plt.grid(True)
    plt.show()
