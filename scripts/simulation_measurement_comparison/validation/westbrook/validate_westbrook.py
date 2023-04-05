import concurrent.futures
import dataclasses
import datetime
import sys
import time
from concurrent.futures import ProcessPoolExecutor
import os
from typing import Tuple

import traceback
import warnings

import cantera as ct
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from sdtoolbox.postshock import CJspeed

from funcs.simulation import cell_size as cs

sns.set_style("darkgrid")

DATA_DIR = os.path.abspath(os.path.dirname(__file__))
MECH = "gri30.cti"


@dataclasses.dataclass(frozen=True)
class Condition:
    fuel: str
    oxidizer: str
    phi: float


CONDITION_MAP = {
    "2H2_O2": Condition(
        fuel="H2",
        oxidizer="O2",
        phi=1,
    ),
    "C2H2_O2": Condition(
        fuel="H2",
        oxidizer="O2",
        phi=5/2,
    ),
    "C2H2_2_5O2": Condition(
        fuel="C2H2",
        oxidizer="O2",
        phi=1,
    ),
    "CH4_2O2": Condition(
        fuel="CH4",
        oxidizer="O2",
        phi=1,
    )
}


def load_txt(file_name: str) -> pd.DataFrame:
    # saved with p0 in atm and cell size in cm
    df = pd.read_csv(os.path.join(DATA_DIR, file_name), skiprows=1, names=["pressure", "cell_size"])
    df["cell_size"] *= 100  # cm -> mm
    df["pressure"] *= 101325  # atm -> Pa
    mixture = file_name.rstrip(".txt")
    df["mixture"] = mixture
    condition = CONDITION_MAP[mixture]
    df["fuel"] = condition.fuel
    df["oxidizer"] = condition.oxidizer
    df["phi"] = condition.phi

    return df


def load_all_data() -> pd.DataFrame:
    df = pd.DataFrame()
    for file in os.listdir(DATA_DIR):
        if file.endswith(".txt"):
            df = pd.concat((df, load_txt(file_name=file)))

    return df


def simulate_single_condition(idx_and_row: Tuple[int, pd.Series], error_log: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        idx = idx_and_row[0]
        row = idx_and_row[1].copy()
        try:
            phi = row["phi"]
            fuel = row["fuel"]
            oxidizer = row["oxidizer"]
            p_0 = row["pressure"]
            t_0 = 300  # todo: verify this please
            gas = ct.Solution(MECH)
            gas.set_equivalence_ratio(phi, fuel, oxidizer)
            cj_speed = CJspeed(p_0, t_0, gas.mole_fraction_dict(), MECH)

            # guess at simulation parameters using measured cell sizes
            # convert cell size to meters, then guess double the westbrook estimate for induction length
            ind_len_estimate = row["cell_size"] / 1000 / (29 / 2)
            cv_end_time = ind_len_estimate / cj_speed
            min_n_steps = 20

            _simulated = cs.calculate(
                mechanism=MECH,
                initial_temp=t_0,
                initial_press=p_0,
                fuel=fuel,
                oxidizer=oxidizer,
                equivalence=phi,
                diluent=None,
                diluent_mol_frac=0,
                cj_speed=cj_speed,
                max_step_znd=ind_len_estimate / min_n_steps,
                cv_end_time=cv_end_time,
                max_step_cv=cv_end_time / min_n_steps,
            )
            row["cell_size_gavrikov"] = _simulated.cell_size.gavrikov * 1000  # m -> mm
            row["cell_size_ng"] = _simulated.cell_size.ng * 1000  # m -> mm
            row["cell_size_westbrook"] = _simulated.cell_size.westbrook * 1000  # m -> mm
            row["gavrikov_criteria_met"] = _simulated.gavrikov_criteria_met
        except Exception as e:
            with open(error_log, "a") as f:
                f.write(
                    f"Caught exception: {e}\n\n"
                    f"Fuel {fuel}\n"
                    f"Oxidizer: {oxidizer}\n"
                    f"Initial pressure: {p_0} Pa\n\n"
                    f"{traceback.format_exc()}"
                    "================\n\n"
                )
                f.flush()

    return idx, row


def simulate_measured_conditions(df_measured: pd.DataFrame) -> pd.DataFrame:
    n_meas = len(df_measured)
    start = datetime.datetime.now().isoformat()
    error_log = f"error_log_{start}"

    with ProcessPoolExecutor() as executor:
        with tqdm(total=n_meas, unit="calc", file=sys.stdout, colour="green", desc="Running") as counter:
            futures = {
                executor.submit(simulate_single_condition, idx_and_row=idx_and_row, error_log=error_log)
                for idx_and_row in df_measured.iterrows()
            }
            results = []

            for done in concurrent.futures.as_completed(futures):
                results.append(done.result())
                counter.update()

            counter.set_description_str("Done")

    df_out = pd.DataFrame()
    for result in sorted(results, key=lambda r: r[0]):
        df_out = pd.concat((df_out, result[1].to_frame().T))

    return df_out


if __name__ == "__main__":
    data = load_all_data()

    print(f"\nloaded {len(data)} data points from literature\n")

    simulated = simulate_measured_conditions(data)
    simulated["pressure"] = pd.to_numeric(simulated["pressure"])
    simulated["cell_size"] = pd.to_numeric(simulated["cell_size"])
    simulated["phi"] = pd.to_numeric(simulated["phi"])
    simulated["cell_size_gavrikov"] = pd.to_numeric(simulated["cell_size_gavrikov"])
    simulated["cell_size_ng"] = pd.to_numeric(simulated["cell_size_ng"])
    simulated["cell_size_westbrook"] = pd.to_numeric(simulated["cell_size_westbrook"])

    with pd.HDFStore(os.path.join(DATA_DIR, "westbrook_validation.h5"), "w") as store:
        with warnings.catch_warnings():
            # pandas doesn't do types well here, and frankly I'm sick of hearing about it
            warnings.simplefilter("ignore")
            store["data"] = simulated

    plot_data = pd.DataFrame()
    for (_, row) in simulated.iterrows():
        plot_data = pd.concat([
            plot_data,
            pd.DataFrame({
                "mixture": [row["mixture"]] * 2,
                "fuel": [row["fuel"]] * 2,
                "oxidizer": [row["oxidizer"]] * 2,
                "phi": [row["phi"]] * 2,
                "pressure": [row["pressure"]] * 2,
                "cell_size": [row["cell_size"], row["cell_size_westbrook"]],
                "source": ["literature", "simulation"]
            })
        ])

    grid = sns.relplot(x="pressure", y="cell_size", style="source", row="mixture", data=plot_data, kind="scatter")
    grid.set(xscale="log", yscale="log")
    for ax in grid.axes[0]:
        ax.invert_xaxis()
    plt.show()
