import datetime
import multiprocessing as mp
import os
import warnings

import cantera as ct
import numpy as np
import pandas as pd
import tqdm
from sdtoolbox.postshock import CJspeed
from uncertainties import unumpy as unp

from funcs.simulation import cell_size as cs
from funcs.simulation import thermo

FUEL = "CH4"
OXIDIZER = "N2O"
MECH = "gri30.cti"


def main():
    df_measured = read_in_measured_data()
    df_result = simulate_measured_conditions(df_measured)
    today = datetime.date.today().isoformat().replace("-", "_")
    data_file = f"simulated_and_measured_{today}.h5"
    with pd.HDFStore(os.path.join(os.path.dirname(__file__), data_file), "w") as store:
        store["data"] = df_result


def get_column_mean_with_uncertainty(df: pd.DataFrame, column: str):
    mean = np.mean(unp.uarray(df[column], df[f"u_{column}"]))

    # noinspection PyUnresolvedReferences
    return mean.nominal_value, mean.std_dev


def read_in_measured_data():
    data_loc = os.path.join(os.path.join(os.path.dirname(__file__), "measurements.h5"))
    with pd.HDFStore(data_loc, "r") as store:
        df = store.data

    df = df[(df["fuel"] == FUEL) & (df["oxidizer"] == OXIDIZER)]
    df_out = pd.DataFrame()
    for idx, ((diluent, phi_nom, dil_mf_nom), df_grouped) in enumerate(
        df.groupby(["diluent", "phi_nom", "dil_mf_nom"])
    ):
        p_0, u_p_0 = get_column_mean_with_uncertainty(df_grouped, "p_0")
        t_0, u_t_0 = get_column_mean_with_uncertainty(df_grouped, "t_0")
        phi, u_phi = get_column_mean_with_uncertainty(df_grouped, "phi")
        dil_mf, u_dil_mf = get_column_mean_with_uncertainty(df_grouped, "dil_mf")
        wave_speed, u_wave_speed = get_column_mean_with_uncertainty(df_grouped, "wave_speed")
        cell_size, u_cell_size = get_column_mean_with_uncertainty(df_grouped, "cell_size")
        this_row = pd.DataFrame(
            data={
                "diluent": diluent,
                "phi_nom": phi_nom,
                "dil_mf_nom": dil_mf_nom,
                "p_0": p_0,
                "u_p_0": u_p_0,
                "t_0": t_0,
                "phi": phi,
                "u_phi": u_phi,
                "dil_mf": dil_mf,
                "u_dil_mf": u_dil_mf,
                "wave_speed": wave_speed,
                "u_wave_speed": u_wave_speed,
                "cell_size_measured": cell_size,
                "u_cell_size_measured": u_cell_size,
            },
            index=[idx],
        )
        df_out = pd.concat((df_out, this_row), axis=0)

    return df_out


def simulate_measured_conditions(df_measured: pd.DataFrame):
    with mp.Pool() as p:
        # noinspection PyTypeChecker
        df_result = pd.DataFrame(
            tqdm.tqdm(p.imap(simulate_single_condition, df_measured.iterrows()), total=len(df_measured))
        )

    return df_result.sort_index()


def simulate_single_condition(idx_and_row: pd.Series):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        row = idx_and_row[1].copy()
        phi = row["phi"]
        dil = row["diluent"]
        dil_mf = row["dil_mf"]
        p_0 = row["p_0"]
        t_0 = row["t_0"]
        gas = ct.Solution(MECH)
        gas.set_equivalence_ratio(
            phi,
            FUEL,
            OXIDIZER
        )
        q = thermo.diluted_species_dict(
            gas.mole_fraction_dict(),
            dil,
            dil_mf
        )
        cj_speed = CJspeed(p_0, t_0, q, MECH)
        simulated = cs.calculate(
            mechanism=MECH,
            initial_temp=t_0,
            initial_press=p_0,
            fuel=FUEL,
            oxidizer=OXIDIZER,
            equivalence=phi,
            diluent=dil,
            diluent_mol_frac=dil_mf,
            cj_speed=cj_speed,
            # cv_end_time=12e-6,
            # max_step_cv=1e-6,
            # max_tries_cv=1,
            # max_step_znd=1e-4,
            # max_tries_znd=5,
            # znd_end_time=5e-4,
        )
        row["cell_size_gavrikov"] = simulated.cell_size.gavrikov * 1000  # m -> mm
        row["cell_size_ng"] = simulated.cell_size.ng * 1000  # m -> mm
        row["cell_size_westbrook"] = simulated.cell_size.westbrook * 1000  # m -> mm
        row["gavrikov_criteria_met"] = simulated.gavrikov_criteria_met

        row["znd_step"] = simulated.znd_step
        row["znd_end_time"] = simulated.znd_end_time
        row["znd_tries"] = simulated.znd_tries
        row["znd_max_temp_time"] = simulated.znd_max_temp_time

    return row


if __name__ == "__main__":
    main()
