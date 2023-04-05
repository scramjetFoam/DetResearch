"""
Detonation cell size estimation functions, based on the work of Shepherd et al:
http://shepherd.caltech.edu/EDL/PublicResources/sdt/nb/sdt_intro.slides.html

A description of the theory, numerical methods, and applications are described
in the following report:

    Numerical Solution Methods for Shock and Detonation Jump Conditions, S.
    Browne, J. Ziegler, and J. E. Shepherd, GALCIT Report FM2006.006 - R3,
    California Institute of Technology Revised September, 2018.

This script uses SDToolbox, which can be found at
http://shepherd.caltech.edu/EDL/PublicResources/sdt/
"""
import dataclasses
from typing import Optional

import cantera as ct
import numpy as np
import sdtoolbox

from .thermo import diluted_species_dict


@dataclasses.dataclass(frozen=True)
class ZndResult:
    induction_length: float
    max_thermicity: float
    velocity: float


@dataclasses.dataclass(frozen=True)
class CvResult:
    induction_time: float
    limit_species_mole_fraction: Optional[np.ndarray]
    temperature: Optional[np.ndarray]
    time: Optional[np.ndarray]


def wrapped_cvsolve(
    gas,
    limit_species_idx: int,
    max_tries=10,
    t_end=1e-6,
    max_step=1e-5,
    induction_time_only: bool = False,
):
    """
    Look jack, I don't have time for your `breaking` malarkey

    Try cvsolve a handful of times with increasing max time lengths to see if
    maybe it's just being mean.

    Parameters
    ----------
    gas : ct.Solution
        gas object to work on
    limit_species_idx : Index of limit species in Solution object
    max_tries : int
        how motivated are you
    t_end : float
        initial end time, which is doubled each iteration
    max_step : float
        maximum cvsolve step time
    induction_time_only : Everything else is None

    Returns
    -------
    idk whatever was supposed to come out in the first place, it's late
    """
    tries = 0
    # if you don't reset the Solution's properties it'll go crazy during the
    # iteration process. Iterations are supposed to be independent, so that's...
    # you know, pretty bad
    init_tpx = gas.TPX
    out = None
    while tries <= max_tries:
        gas.TPX = init_tpx
        tries += 1
        if tries < max_tries:
            # this exception is broad on purpose.
            # noinspection PyBroadException
            try:
                out = sdtoolbox.cv.cvsolve(gas, t_end=t_end, max_step=max_step)
                break
            except:  # noqa: E722
                t_end *= 2
                max_step *= 2
        else:
            # let it break if it's gonna break after max tries
            out = sdtoolbox.cv.cvsolve(gas, t_end=t_end, max_step=max_step)

    if induction_time_only:
        return CvResult(
            induction_time=out["ind_time"],
            limit_species_mole_fraction=None,
            temperature=None,
            time=None,
        )
    else:
        return CvResult(
            induction_time=out["ind_time"],
            limit_species_mole_fraction=out["speciesX"][limit_species_idx],
            temperature=out["T"],
            time=out["time"],
        )


def wrapped_zndsolve(
    gas,
    base_gas,
    cj_speed,
    t_end,
    max_step,
    max_tries=5
):
    tries = 0
    init_tpx = gas.TPX
    init_tpx_base = base_gas.TPX
    out = None
    while tries <= max_tries:
        # retry the simulation if the initial time step doesn't work
        # drop time step by one order of magnitude every failure
        gas.TPX = init_tpx
        base_gas.TPX = init_tpx_base
        tries += 1
        if tries < max_tries:
            try:
                out = sdtoolbox.znd.zndsolve(
                    gas,
                    base_gas,
                    cj_speed,
                    advanced_output=True,
                    t_end=t_end,
                    max_step=max_step
                )
                break
            except (ct.CanteraError, ValueError):
                max_step /= 10.
        else:
            # let it break if it's gonna break after max tries
            out = sdtoolbox.znd.zndsolve(
                gas,
                base_gas,
                cj_speed,
                advanced_output=True,
                t_end=t_end,
                max_step=max_step
            )
    return ZndResult(
        induction_length=out["ind_len_ZND"],
        max_thermicity=out["thermicity"].max(),
        velocity=out["U"][0],
    )


@dataclasses.dataclass(frozen=True)
class ModelResults:
    gavrikov: float
    ng: float
    westbrook: float

    def values(self) -> np.array:
        return np.array((self.gavrikov, self.ng, self.westbrook))


@dataclasses.dataclass(frozen=True)
class CellSizeResults:
    cell_size: ModelResults
    induction_length: ModelResults
    gavrikov_criteria_met: bool
    reaction_equation: str
    k_i: float


def calculate(
    mechanism: str,
    initial_temp: float,
    initial_press: float,
    fuel: str,
    oxidizer: str,
    equivalence: float,
    diluent: Optional[str],
    diluent_mol_frac: float,
    cj_speed: float,
    perturbed_reaction: Optional[float] = None,
    perturbation_fraction: float = 1e-2,
    max_tries_znd: int = 10,
    max_step_znd: float = 1e-4,
    max_tries_cv: int = 15,
    cv_end_time: float = 1e-6,
    max_step_cv: float = 5e-7,
) -> CellSizeResults:
    base_gas = _build_gas_object(
        mechanism=mechanism,
        equivalence=equivalence,
        fuel=fuel,
        oxidizer=oxidizer,
        diluent=diluent,
        diluent_mol_frac=diluent_mol_frac,
        initial_temp=initial_temp,
        initial_press=initial_press,
        perturbed_reaction=perturbed_reaction,
        perturbation_fraction=perturbation_fraction,
    )
    q = base_gas.X

    # FIND EQUILIBRIUM POST SHOCK STATE FOR GIVEN SPEED
    gas = sdtoolbox.postshock.PostShock_eq(
        U1=cj_speed,
        P1=initial_press,
        T1=initial_temp,
        q=q,
        mech=mechanism,
        perturbed_rxn_no=perturbed_reaction,
        perturbation_fraction=perturbation_fraction,
    )

    cj_speed_density_corrected = cj_speed * base_gas.density / gas.density

    # FIND FROZEN POST SHOCK STATE FOR GIVEN SPEED
    gas = sdtoolbox.postshock.PostShock_fr(
        U1=cj_speed,
        P1=initial_press,
        T1=initial_temp,
        q=q,
        mech=mechanism,
        perturbed_rxn_no=perturbed_reaction,
        perturbation_fraction=perturbation_fraction,
    )

    # SOLVE ZND DETONATION ODES
    znd_result = wrapped_zndsolve(
        gas=gas,
        base_gas=base_gas,
        cj_speed=cj_speed,
        t_end=2e-3,
        max_step=max_step_znd,
        max_tries=max_tries_znd,
    )

    # Find CV parameters including effective activation energy
    gas.TPX = initial_temp, initial_press, q
    gas = sdtoolbox.postshock.PostShock_fr(
        U1=cj_speed,
        P1=initial_press,
        T1=initial_temp,
        q=q,
        mech=mechanism,
        perturbed_rxn_no=perturbed_reaction,
        perturbation_fraction=perturbation_fraction,
    )
    temp_vn, press_vn = gas.TP
    temp_a = temp_vn * 1.02
    gas.TPX = temp_a, press_vn, q

    #  Gather limiting species and index -- fuel for lean mixtures, oxygen for rich mixtures
    if equivalence <= 1:
        limit_species = fuel
    else:
        limit_species = 'O2'
    limit_species_idx = gas.species_index(limit_species)

    # cv_out_0 = sdtoolbox.cv.cvsolve(gas)
    cv_out_0 = wrapped_cvsolve(
        gas,
        limit_species_idx,
        max_tries_cv,
        cv_end_time,
        max_step_cv,
        induction_time_only=False,
    )

    temp_b = temp_vn * 0.98
    gas.TPX = temp_b, press_vn, q
    # cv_out_1 = sdtoolbox.cv.cvsolve(gas, t_end=10e-6)
    cv_out_1 = wrapped_cvsolve(
        gas,
        limit_species_idx,
        max_tries_cv,
        cv_end_time,
        max_step_cv,
        induction_time_only=True,
    )

    # Approximate effective activation energy for CV explosion
    tau_a = cv_out_0.induction_time
    tau_b = cv_out_1.induction_time
    if tau_a == 0 or tau_b == 0:
        activation_energy = 0
    else:
        activation_energy = 1 / temp_vn * (np.log(tau_a / tau_b) / ((1 / temp_a) - (1 / temp_b)))

    # Tps should be post-shock temperature at 1.3 Dcj per Gavrikov
    temp_post_shock_gavrikov = sdtoolbox.postshock.PostShock_fr(
        U1=cj_speed * 1.3,
        P1=initial_press,
        T1=initial_temp,
        q=q,
        mech=mechanism,
        perturbed_rxn_no=perturbed_reaction,
        perturbation_fraction=perturbation_fraction,
    ).T
    gavrikov_criteria = {
        "Ea/RTps": activation_energy / ct.gas_constant / temp_post_shock_gavrikov,
        "Tvn/T0": temp_vn / initial_temp
    }
    # see pg 32 of https://doi.org/10.1016/S0010-2180(99)00076-0
    gavrikov_criteria_met = all((
        gavrikov_criteria["Ea/RTps"] <= 16,
        gavrikov_criteria["Ea/RTps"] >= 3,
        gavrikov_criteria["Tvn/T0"] <= 8,
        gavrikov_criteria["Tvn/T0"] >= 1.5,
    ))

    #  Find Gavrikov induction time based on 50% limiting species
    #  consumption, fuel for lean mixtures, oxygen for rich mixtures
    gas.TPX = temp_vn, press_vn, q
    try:
        mf_initial = gas.mole_fraction_dict()[limit_species]
    except KeyError:
        mf_initial = 0.
    gas.equilibrate('UV')
    mf_final = gas.mole_fraction_dict()[limit_species]
    temp_final = gas.T
    mf_gav = 0.5*(mf_initial - mf_final) + mf_final
    t_gav = np.nanmax(cv_out_0.time[cv_out_0.limit_species_mole_fraction > mf_gav], initial=0)

    #  Westbrook time based on 50% temperature rise
    temp_west = 0.5*(temp_final - temp_vn) + temp_vn
    t_west = np.nanmax(cv_out_0.time[cv_out_0.temperature < temp_west], initial=0)

    # Ng et al definition of max thermicity width
    # Equation 2
    chi_ng = activation_energy * znd_result.induction_length / (cj_speed_density_corrected / znd_result.max_thermicity)

    induction_length = ModelResults(
        westbrook=t_west*znd_result.velocity,
        gavrikov=t_gav*znd_result.velocity,
        ng=znd_result.induction_length,
    )

    # calculate and return cell size results
    cell_size = ModelResults(
        westbrook=_cell_size_westbrook(induction_length=induction_length.westbrook),
        gavrikov=_cell_size_gavrikov(
            temp_0=initial_temp,
            temp_vn=temp_vn,
            activation_energy=activation_energy,
            induction_length=induction_length.gavrikov,
        ),
        ng=_cell_size_ng(chi_ng=chi_ng, induction_length=induction_length.ng),
    )

    if perturbed_reaction is None:
        reaction_equation = None
        k_i = None
    else:
        reaction_equation = base_gas.reaction_equation(perturbed_reaction)
        k_i = base_gas.forward_rate_constants[perturbed_reaction]
    return CellSizeResults(
        cell_size=cell_size,
        induction_length=induction_length,
        gavrikov_criteria_met=gavrikov_criteria_met,
        reaction_equation=reaction_equation,
        k_i=k_i,
    )


def _build_gas_object(
    mechanism: str,
    equivalence: float,
    fuel: str,
    oxidizer: str,
    diluent: Optional[str],
    diluent_mol_frac: float,
    initial_temp: float,
    initial_press: float,
    perturbed_reaction: Optional[int],
    perturbation_fraction: float
) -> ct.Solution:
    gas = ct.Solution(mechanism)

    gas.set_equivalence_ratio(equivalence, fuel, oxidizer)
    if diluent is not None and diluent_mol_frac > 0:
        gas.X = diluted_species_dict(gas.mole_fraction_dict(), diluent, diluent_mol_frac)

    gas.TP = initial_temp, initial_press
    if perturbed_reaction is not None:
        gas.set_multiplier(1 + perturbation_fraction, perturbed_reaction)
    return gas


def _cell_size_ng(chi_ng: float, induction_length: float) -> float:
    """
    Calculates cell size using the correlation given by Ng, H. D., Ju, Y.,
    & Lee, J. H. S. (2007). Assessment of detonation hazards in
    high-pressure hydrogen storage from chemical sensitivity analysis.
    International Journal of Hydrogen Energy, 32(1), 93–99.
    https://doi.org/10.1016/j.ijhydene.2006.03.012 using equations (1) and
    (2) along with the coefficients given in Table 1.

    Parameters
    ----------

    Returns
    -------
    cell_size : float
        Estimated cell size (m)
    """
    # Coefficients from Table 1
    a_0 = 30.465860763763
    a = np.array([89.55438805808153, -130.792822369483, 42.02450507117405])
    b = np.array([-0.02929128383850, 1.026325073064710e-5, -1.031921244571857e-9])

    # Equation 1
    chi_pow = np.power(chi_ng, range(1, len(a)+1))
    cell_size = (a_0 + (a / chi_pow + b * chi_pow).sum()) * induction_length

    return cell_size


def _cell_size_gavrikov(temp_0: float, temp_vn: float, activation_energy: float, induction_length: float) -> float:
    """
    Calculates cell size using the correlation given by Gavrikov, A.I.,
    Efimenko, A.A., & Dorofeev, S.B. (2000). A model for detonation cell
    size prediction from chemical kinetics. Combustion and Flame, 120(1–2),
    19–33. https://doi.org/10.1016/S0010-2180(99)00076-0 using equation (5)
    along with coefficients given in Table 1

    Parameters
    ----------

    Returns
    -------
    cell_size : float
        Estimated cell size (m)
    """
    # Coefficients from Table 1
    a = -0.007843787493
    b = 0.1777662961
    c = 0.02371845901
    d = 1.477047968
    e = 0.1545112957
    f = 0.01547021569
    g = -1.446582357
    h = 8.730494354
    i = 4.599907939
    j = 7.443410379
    k = 0.4058325462
    m = 1.453392165

    # Equation 5
    gav_y = temp_vn / temp_0
    cell_size = np.power(
        10,
        gav_y * (a * gav_y - b)
        + activation_energy * (c * activation_energy - d + (e - f * gav_y) * gav_y)
        + g * np.log(gav_y)
        + h * np.log(activation_energy)
        + gav_y * (i / activation_energy - k * gav_y / np.power(activation_energy, m))
        - j
    ) * induction_length

    return cell_size


def _cell_size_westbrook(induction_length: float) -> float:
    """
    Calculates cell size using the correlation given by Westbrook, C. K., &
    Urtiew, P. A. (1982). Chemical kinetic prediction of critical parameters
    in gaseous detonations. Symposium (International) on Combustion, 19(1),
    615–623. https://doi.org/10.1016/S0082-0784(82)80236-1

    Parameters
    ----------

    Returns
    -------
    cell_size : float
        Estimated cell size (m)
    """
    return 29 * induction_length
