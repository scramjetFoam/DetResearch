import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from scripts.final_manuscript.plot_settings import set_palette, set_style


def load_data(data_path: str):
    with pd.HDFStore(
        data_path,
        "r",
    ) as store:
        plot_data = pd.DataFrame()
        for _, row in store.data_fixed_uncert.iterrows():
            plot_data = pd.concat(
                [
                    plot_data,
                    pd.DataFrame(
                        {
                            "fuel": ["CH4"] * 2,
                            "oxidizer": ["N2O"] * 2,
                            "diluent": [row["diluent"]] * 2,
                            "phi_nom": [row["phi_nom"]] * 2,
                            "dil_mf_nom": [row["dil_mf_nom"]] * 2,
                            "cell_size": [row["cell_size_measured"], row["cell_size_westbrook"]],
                            "source": ["measurement", "simulation"],
                            "uncertainty": [row["u_cell_size_measured"], np.NaN],
                        }
                    ),
                ]
            )
        return plot_data


def plot(data: pd.DataFrame, show_title: bool, save_plot: bool):
    grid = sns.relplot(
        x="dil_mf_nom",
        y="cell_size",
        row="source",
        col="phi_nom",
        hue="diluent",
        data=data,
        kind="scatter",
        zorder=2,
    )
    grid.fig.set_size_inches((11, 8.5))
    if show_title:
        grid.fig.suptitle("Diluent Comparison", weight="bold")

    measurements = data[data["source"] == "measurement"]
    plot_args = dict(
        x="dil_mf_nom",
        y="cell_size",
        yerr="uncertainty",
        ls="None",
        linewidth=0.5,
        zorder=1,
        capsize=4,
        alpha=0.4,
    )
    grid.axes[0][0].errorbar(
        **plot_args, data=measurements[(measurements["phi_nom"] == 0.4) & (measurements["diluent"] == "CO2")]
    )
    grid.axes[0][0].errorbar(
        **plot_args, data=measurements[(measurements["phi_nom"] == 0.4) & (measurements["diluent"] == "N2")]
    )
    grid.axes[0][1].errorbar(
        **plot_args, data=measurements[(measurements["phi_nom"] == 0.7) & (measurements["diluent"] == "CO2")]
    )
    grid.axes[0][1].errorbar(
        **plot_args, data=measurements[(measurements["phi_nom"] == 0.7) & (measurements["diluent"] == "N2")]
    )
    grid.axes[0][2].errorbar(
        **plot_args, data=measurements[(measurements["phi_nom"] == 1.0) & (measurements["diluent"] == "CO2")]
    )
    grid.axes[0][2].errorbar(
        **plot_args, data=measurements[(measurements["phi_nom"] == 1.0) & (measurements["diluent"] == "N2")]
    )

    for ax in grid.axes.flatten():
        title = ax.get_title()
        ax.set_title(
            title.replace("source = ", "")
            .replace("measurement ", "Measured")
            .replace("simulation ", "Simulated")
            .replace("| ", "\n")
            .replace("phi_nom", r"$\phi_{nom}$")
        )
        if ax.get_xlabel():
            ax.set_xlabel("Nominal Diluent Mole Fraction")
        if ax.get_ylabel():
            ax.set_ylabel("Cell size (mm)")

    grid.tight_layout()

    if save_plot:
        grid.fig.savefig("plots/simulated_vs_measured.pdf")


def main():
    show_plot = False
    save_plot = True
    show_title = True

    set_palette()
    set_style()

    script_dir = os.path.dirname(__file__)
    data_path = os.path.join(
        os.path.dirname(script_dir), "simulation_measurement_comparison", "simulated_and_measured_2024_03_02.h5"
    )

    data = load_data(data_path=data_path)
    plot(data=data, show_title=show_title, save_plot=save_plot)

    if show_plot:
        plt.show()


if __name__ == "__main__":
    main()
