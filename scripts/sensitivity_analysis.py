import multiprocessing as mp

import cantera as ct
from tqdm import tqdm

import funcs.simulation.sensitivity.detonation.database as db
# noinspection PyUnresolvedReferences
from funcs.simulation.sensitivity import istarmap
from funcs.simulation.sensitivity.detonation import analysis
from funcs.simulation import thermo

if __name__ == '__main__':
    import warnings
    warnings.simplefilter('ignore')

    # Inputs
    perturbation_fraction = 1e-2
    max_step_znd = 1e-4  # default 1e-4
    db_path = "sensitivity_3.sqlite"
    mechanism = "gri30_highT.cti"
    initial_temp = 300
    initial_press = 101325
    equivalence = 1
    fuel = 'H2'
    oxidizer = 'O2'
    diluent_to_match = 'CO2'
    diluent = "N2"
    diluent_mol_frac_to_match = 0.05
    always_overwrite_existing = False

    # Preparations
    if diluent_to_match == diluent:
        diluent_mol_frac = diluent_mol_frac_to_match
    else:
        diluent_mol_frac = thermo.match_adiabatic_temp(
            mechanism,
            fuel,
            oxidizer,
            equivalence,
            diluent_to_match,
            diluent_mol_frac_to_match,
            diluent,
            initial_temp,
            initial_press
        )
    print("Initializing study... ", end="", flush=True)
    test_conditions = analysis.initialize_study(
        db_path=db_path,
        test_conditions=db.TestConditions(
            mechanism=mechanism,
            initial_temp=initial_temp,
            initial_press=initial_press,
            fuel=fuel,
            oxidizer=oxidizer,
            equivalence=equivalence,
            diluent=diluent,
            diluent_mol_frac=diluent_mol_frac,
        )
    )
    print("Done!")
    reactions = ct.Reaction().listFromFile(mechanism)

    # Calculations
    _lock = mp.Lock()
    n_rxns = len(reactions)
    with mp.Pool(initializer=analysis.init, initargs=(_lock,)) as p:
        # noinspection PyUnresolvedReferences
        for _ in tqdm(
            p.istarmap(
                analysis.perform_study,
                [
                    [
                        test_conditions,
                        perturbation_fraction,
                        perturbed_rxn_no,
                        db_path,
                        max_step_znd,
                        always_overwrite_existing,
                    ] for perturbed_rxn_no in range(n_rxns)
                ]
            ),
            total=n_rxns
        ):
            pass
    print("woo!")
