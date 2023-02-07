import dataclasses
import multiprocessing as mp
import os
import warnings

import cantera as ct
import seaborn as sns
import pandas as pd
import tqdm
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


def simulate_single_condition(idx_and_row: pd.Series):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        row = idx_and_row[1].copy()
        phi = row["phi"]
        fuel = row["fuel"]
        oxidizer = row["oxidizer"]
        p_0 = row["pressure"]
        t_0 = 300  # todo: verify this please
        gas = ct.Solution(MECH)
        gas.set_equivalence_ratio(phi, fuel, oxidizer)
        cj_speed = CJspeed(p_0, t_0, gas.mole_fraction_dict(), MECH)
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
        )
        row["cell_size_gavrikov"] = _simulated.cell_size.gavrikov * 1000  # m -> mm
        row["cell_size_ng"] = _simulated.cell_size.ng * 1000  # m -> mm
        row["cell_size_westbrook"] = _simulated.cell_size.westbrook * 1000  # m -> mm
        row["gavrikov_criteria_met"] = _simulated.gavrikov_criteria_met

    return row


def simulate_measured_conditions(df_measured: pd.DataFrame):  # todo: error handling!
    with mp.Pool(maxtasksperchild=1) as p:
        # noinspection PyTypeChecker
        df_result = pd.DataFrame(
            tqdm.tqdm(p.imap(simulate_single_condition, df_measured.iterrows()), total=len(df_measured))
        )

    return df_result.sort_index()


if __name__ == "__main__":
    data = load_all_data()

    print(f"loaded {len(data)} data points from literature")

    simulated = simulate_measured_conditions(data)
    with pd.HDFStore(os.path.join(DATA_DIR, "westbrook_validation.h5"), "w") as store:
        store["data"] = simulated

    grid = sns.relplot(x="pressure", y="cell_size", style="mixture", data=simulated, kind="scatter")
    grid.set(xscale="log", yscale="log")
    for ax in grid.axes[0]:
        ax.invert_xaxis()

    grid = sns.relplot(x="pressure", y="cell_size_westbrook", style="mixture", data=simulated, kind="scatter")
    grid.set(xscale="log", yscale="log")
    for ax in grid.axes[0]:
        ax.invert_xaxis()
    plt.show()
