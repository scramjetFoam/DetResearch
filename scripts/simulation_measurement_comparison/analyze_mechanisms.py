import dataclasses
from multiprocessing import Lock, Pool
import os.path
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, List, Any, Dict

import cantera as ct
import numpy as np
import pandas as pd
import tqdm
from sdtoolbox.postshock import CJspeed

from funcs.simulation.sensitivity.istarmap import istarmap  # noqa: F401
from funcs.simulation.cell_size import CellSize
from funcs.simulation import thermo


DATA_FILE = os.path.join(os.path.abspath(os.path.dirname(__file__)), "mechanism_comparison_results.h5")


class DataColumn(Enum):
    mechanism = "mech"
    fuel = "fuel"
    oxidizer = "oxidizer"
    diluent = "diluent"
    diluent_mole_fraction = "dil_mf"
    cj_speed = "cj_speed"
    cj_time = "cj_time"
    gavrikov = "cs_gavrikov"
    ng = "cs_ng"
    westbrook = "cs_westbrook"
    cell_size_time = "cs_time"

    @classmethod
    def all(cls) -> List[str]:
        return [item.value for _, item in cls.__members__.items()]


@dataclasses.dataclass
class DataUpdate:
    column: DataColumn
    value: Any


class Mechanism(Enum):
    GRI3 = "gri30.cti"
    GRI3HighT = "gri30_highT.cti"
    SanDiego = "sandiego20161214.cti"
    JetSurf = "JetSurf2.cti"
    Blanquart = "Blanquart2018.cti"
    Aramco = "aramco2.cti"
    FFCM = "ffcm1.cti"

    @classmethod
    def validate_all(cls):
        print("Validating mechanisms")
        ctis = [mech.value for _, mech in cls.__members__.items()]
        with Pool() as p:
            results = p.map(cls._validate, ctis)
        p.join()

        good_mechanisms = []
        bad_mechanisms = []
        for (cti, success) in results:
            if success:
                good_mechanisms.append(cti)
            else:
                bad_mechanisms.append(cti)

        if good_mechanisms:
            msg = f"\u001b[32mFOUND: {', '.join(sorted(good_mechanisms))}\u001b[0m"
            print(msg)
        if bad_mechanisms:
            msg = f"\u001b[31mNOT FOUND: {', '.join(bad_mechanisms)}\u001b[0m"
            print(msg)

    @staticmethod
    def _validate(cti: str) -> Tuple[str, bool]:
        success = False
        try:
            ct.Solution(cti)
            success = True
        except ct._cantera.CanteraError:
            pass

        return cti, success


class Diluent(Enum):
    NONE = None
    CO2 = "CO2"
    N2 = "N2"

    def as_string(self):
        if self is Diluent.NONE:
            return "None"
        else:
            return self.value


@dataclasses.dataclass(frozen=True)
class InitialConditions:
    p0: int = 101325  # Pa
    t0: int = 300  # K
    phi: int = 1
    fuel: str = "CH4"
    oxidizer: str = "N2O"

    @classmethod
    def undiluted_gas(cls, mechanism: Mechanism) -> ct.Solution:
        gas = ct.Solution(mechanism.value)
        gas.TP = cls.t0, cls.p0
        gas.set_equivalence_ratio(cls.phi, cls.fuel, cls.oxidizer)

        return gas

    @classmethod
    def diluted_gas(cls, mechanism: Mechanism, diluent: Diluent, diluent_mol_frac: float) -> ct.Solution:
        gas = cls.undiluted_gas(mechanism)
        if diluent.value is None:
            return gas
        else:
            new_mole_fractions = thermo.diluted_species_dict(
                spec=gas.mole_fraction_dict(),
                diluent=diluent.as_string(),
                diluent_mol_frac=diluent_mol_frac,
            )
            gas.X = new_mole_fractions


def get_all_cj_speeds():  # todo
    pass


def _new_data_row(mech: Mechanism, diluent: Diluent, diluent_mol_frac: float, updates_dict: Dict[str, str]):
    data = pd.DataFrame({
        DataColumn.mechanism.value: mech.value,
        DataColumn.fuel.value: InitialConditions.fuel,
        DataColumn.oxidizer.value: InitialConditions.oxidizer,
        DataColumn.diluent.value: diluent.as_string(),
        DataColumn.diluent_mole_fraction.value: diluent_mol_frac,
        **updates_dict
    }, index=[0])

    return data


def update_results(mech: Mechanism, diluent: Diluent, diluent_mol_frac: float, updates: List[DataUpdate], lock: Lock):
    with lock:
        with pd.HDFStore(DATA_FILE, "a") as store:
            updates_dict = {update.column.value: update.value for update in updates}
            if "/data" in store.keys():
                for update in updates:
                    mask = (
                        (store["data"][DataColumn.mechanism.value] == mech.value)
                        & (store["data"][DataColumn.diluent.value] == diluent.value)
                        & np.isclose(store["data"][DataColumn.diluent_mole_fraction.value], diluent_mol_frac)
                    )
                    if mask.any():
                        store["data"].loc[mask, update.column.value] = update.value
                    else:
                        store.append("data", _new_data_row(mech, diluent, diluent_mol_frac, updates_dict))
                        break
            else:
                df = pd.DataFrame([], columns=DataColumn.all())
                df = pd.concat(
                    (df, _new_data_row(mech, diluent, diluent_mol_frac, updates_dict)),
                    axis=0,
                    ignore_index=True,
                )
                store.put("data", df, format="table")


def get_cj_speed(mech: Mechanism, diluent: Diluent, diluent_mol_frac: float, lock: Lock, force_calc: bool = False,) -> float:
    if force_calc:
        cj = _calculate_cj_speed(mech, diluent, diluent_mol_frac, lock)
    else:
        cj = _try_loading_cj_speed(mech, diluent, diluent_mol_frac, lock)
        if cj is None:
            cj = _calculate_cj_speed(mech, diluent, diluent_mol_frac, lock)

    return cj


def _calculate_cj_speed(mech: Mechanism, diluent: Diluent, diluent_mol_frac: float, lock: Lock) -> float:
    if diluent is not Diluent.NONE and diluent_mol_frac > 0:
        gas = InitialConditions.diluted_gas(mech, diluent, diluent_mol_frac)
    else:
        gas = InitialConditions.undiluted_gas(mech)

    t0 = time.time()
    cj = CJspeed(
        P1=InitialConditions.p0,
        T1=InitialConditions.t0,
        q=gas.mole_fraction_dict(),
        mech=mech.value
    )
    t1 = time.time()
    updates = [
        DataUpdate(DataColumn.cj_speed, cj),
        DataUpdate(DataColumn.cj_time, t1-t0)
    ]
    update_results(mech, diluent, diluent_mol_frac, updates, lock)

    return cj


def _try_loading_cj_speed(mech: Mechanism, diluent: Diluent, diluent_mol_frac: float, lock: Lock) -> Optional[float]:
    cj = None
    with lock:
        with pd.HDFStore(DATA_FILE, "a") as store:
            if "/data" in store.keys():
                maybe_data = store["data"][
                    (store["data"][DataColumn.mechanism.value] == mech.value)
                    & (store["data"][DataColumn.diluent.value] == diluent.as_string())
                    & np.isclose(store["data"][DataColumn.diluent_mole_fraction.value], diluent_mol_frac)
                ]
                if len(maybe_data):
                    cj = maybe_data[DataColumn.cj_speed.value].values[0]

    return cj


def simulate_cell_size(mech: Mechanism, diluent: Diluent, diluent_mol_frac: float):  # todo
    pass


if __name__ == "__main__":
    # Mechanism.validate_all()
    _lock = Lock()
    print(get_cj_speed(Mechanism.GRI3, Diluent.NONE, 0.0, _lock))
