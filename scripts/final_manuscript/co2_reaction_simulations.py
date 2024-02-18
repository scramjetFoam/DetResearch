import concurrent.futures
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import cantera as ct
from tqdm import tqdm
from rich.traceback import install

import sdtoolbox as sdt
import simulation.thermo
from funcs.simulation.cell_size import calculate_westbrook_only, build_gas_object


install(show_locals=True)


def get_important_reaction_indices(gas: ct.Solution) -> list[int]:
    rxn_indices = []
    for i, rxn, in enumerate(gas.reactions()):
        reactants_and_products = [*rxn.reactants.keys(), *rxn.products.keys()]
        if "CO2" in reactants_and_products and "OH" in reactants_and_products:
            rxn_indices.append(i)
    return rxn_indices


def get_important_species_indices(gas: ct.Solution) -> list[int]:
    species_indices = []
    for spec in ["CO2", "OH", "H"]:
        try:
            species_indices.append(gas.species_index(spec))
        except ValueError:
            warnings.warn(f"Species not found: {spec}")
    return species_indices


def simulate(
    mech: str,
    fuel: str,
    oxidizer: str,
    diluent: str,
    t0: float,
    p0: float,
    phi: float,
    dil_mf: float,
) -> None:
    base_gas = build_gas_object(
        mechanism=mech,
        equivalence=phi,
        fuel=fuel,
        oxidizer=oxidizer,
        diluent=diluent,
        diluent_mol_frac=dil_mf,
        initial_temp=t0,
        initial_press=p0,
    )
    q = base_gas.mole_fraction_dict()
    cj_speed = sdt.postshock.CJspeed(p0, t0, q, mech)
    with warnings.catch_warnings():
        # don't fuck up my tqdm counter >:(
        warnings.simplefilter("ignore")
        calculate_westbrook_only(
            mechanism=mech,
            initial_temp=t0,
            initial_press=p0,
            fuel=fuel,
            oxidizer=oxidizer,
            equivalence=phi,
            diluent=diluent,
            diluent_mol_frac=dil_mf,
            cj_speed=cj_speed,
            rxn_indices=get_important_reaction_indices(gas=base_gas),
            spec_indices=get_important_species_indices(gas=base_gas),
            db_path="/home/mick/DetResearch/scripts/final_manuscript/co2_reaction_study_less_znd.sqlite",
            batch_threshold=100_000,
            znd_end_time=3e-5,
        )


def main():
    mech = "gri30_highT.xml"
    fuel = "CH4"
    oxidizer = "N2O"
    t0 = 300
    p0 = 101325
    phi = 1.0

    co2_dil_mfs = [0.1, 0.2]
    n2_dil_mfs = [simulation.thermo.match_adiabatic_temp(
        mech=mech,
        fuel=fuel,
        oxidizer=oxidizer,
        phi=phi,
        dil_original="CO2",
        dil_original_mol_frac=mf_co2,
        dil_new="N2",
        init_temp=t0,
        init_press=p0,
    ) for mf_co2 in co2_dil_mfs] + co2_dil_mfs  # Tad matched + mole fraction matched

    n_conditions = len(co2_dil_mfs) + len(n2_dil_mfs)
    with tqdm(total=n_conditions, unit="simulation", file=sys.stdout, colour="green", desc="Running") as counter:
        # for dil_mf in co2_dil_mfs:
        #     simulate(
        #         mech=mech,
        #         fuel=fuel,
        #         oxidizer=oxidizer,
        #         diluent="CO2",
        #         t0=t0,
        #         p0=p0,
        #         phi=phi,
        #         dil_mf=dil_mf,
        #     )
        # for dil_mf in n2_dil_mfs:
        #     simulate(
        #         mech=mech,
        #         fuel=fuel,
        #         oxidizer=oxidizer,
        #         diluent="N2",
        #         t0=t0,
        #         p0=p0,
        #         phi=phi,
        #         dil_mf=dil_mf,
        #     )

        with ProcessPoolExecutor() as executor:
            futures = set()
            for dil_mf in co2_dil_mfs:
                # noinspection PyTypeChecker
                f = executor.submit(
                    simulate,
                    mech=mech,
                    fuel=fuel,
                    oxidizer=oxidizer,
                    diluent="CO2",
                    t0=t0,
                    p0=p0,
                    phi=phi,
                    dil_mf=dil_mf,
                )
                futures.add(f)
            for dil_mf in n2_dil_mfs:
                # noinspection PyTypeChecker
                f = executor.submit(
                    simulate,
                    mech=mech,
                    fuel=fuel,
                    oxidizer=oxidizer,
                    diluent="N2",
                    t0=t0,
                    p0=p0,
                    phi=phi,
                    dil_mf=dil_mf,
                )
                futures.add(f)
            results = []

            for done in concurrent.futures.as_completed(futures):
                results.append(done.result())
                counter.update()

            counter.set_description_str("Done")


if __name__ == "__main__":
    main()
