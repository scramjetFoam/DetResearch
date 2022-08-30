import dataclasses
import multiprocessing as mp
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import cantera as ct
import pandas as pd
import tqdm
from sdtoolbox.postshock import CJspeed

from funcs.simulation.sensitivity.istarmap import istarmap  # noqa: F401
from funcs.simulation.cell_size import CellSize
from funcs.simulation import thermo


class Mechanism(Enum):
    GRI3 = "gri30.cti"
    GRI3HighT = "gri30_highT.cti"
    SanDiego = "sandiego20161214.cti"
    JetSurf = "JetSurf2.cti"
    Blanquart = "Blanquart2018.cti"
    Aramco = "aramco2.cti"
    FFCM = "ffcm1.cti"

    @staticmethod
    def validate_all():
        for name, mech in Mechanism.__members__.items():
            try:
                ct.Solution(mech.value)
            except ct._cantera.CanteraError:
                print()
                print(name)
                print()
                raise


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

    def undiluted_gas(self, mechanism: str) -> ct.Solution:
        gas = ct.Solution(mechanism)
        gas.TP = self.t0, self.p0
        gas.set_equivalence_ratio(self.phi, self.fuel, self.oxidizer)

        return gas

    def diluted_gas(self, mechanism: str, diluent: Diluent, diluent_mol_frac: float) -> ct.Solution:
        gas = self.undiluted_gas(mechanism)
        if diluent.value is None:
            return gas
        else:
            new_mole_fractions = thermo.diluted_species_dict(
                spec=gas.mole_fraction_dict(),
                diluent=diluent.as_string(),
                diluent_mol_frac=diluent_mol_frac,
            )
            gas.X = new_mole_fractions


