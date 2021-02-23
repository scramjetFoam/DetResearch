import multiprocessing as mp

import cantera as ct
from tqdm import tqdm

import funcs.simulation.sensitivity.database as db
# noinspection PyUnresolvedReferences
from funcs.simulation.sensitivity import istarmap
from funcs.simulation.sensitivity.analysis import perform_study, init


# from itertools import permutations
# import numpy as np
# import cantera as ct


# fuels = {'CH4', 'C3H8'}
# oxidizers = {'N2O'}
# equivs = {0.4, 0.7, 1.0}
# init_pressures = {101325.}
# init_temps = {300}
# dilution_numbers = np.array([1, 5.5, 10], dtype=float)
# dilution = {
#     'None': [0],
#     'CO2': dilution_numbers * 1e-2,
#     'NO': dilution_numbers * 1e-4
# }

def study_with_progress_bar(
        progress_bar,
        mechanism,
        initial_temp,
        initial_press,
        equivalence,
        fuel,
        oxidizer,
        diluent,
        diluent_mol_frac,
        inert,
        perturbation_fraction,
        rxn_no,
        lock
):
    perform_study(
        mechanism,
        initial_temp,
        initial_press,
        equivalence,
        fuel,
        oxidizer,
        diluent,
        diluent_mol_frac,
        inert,
        perturbation_fraction,
        rxn_no,
        lock
    )
    progress_bar.update(1)


if __name__ == '__main__':
    import warnings
    warnings.simplefilter('ignore')
    # inert = 'AR'
    _inert = None
    # _mechanism = 'Mevel2017.cti'
    # _mechanism = 'aramco2.cti'
    _inert_species = [_inert]
    _mechanism = "gri30_highT.cti"
    _initial_temp = 300
    _initial_press = 101325
    _equivalence = 1
    _fuel = 'CH4'
    _oxidizer = 'N2O'
    # diluent = 'AR'
    _diluent = 'CO2'
    _diluent_mol_frac = 0.1
    _perturbation_fraction = 1e-3

    t = db.Table(
        'sensitivity.sqlite',
        'data'
    )
    exist_check = t.fetch_test_rows(
        mechanism=_mechanism,
        initial_temp=_initial_temp,
        initial_press=_initial_press,
        fuel=_fuel,
        oxidizer=_oxidizer,
        equivalence=_equivalence,
        diluent=_diluent,
        diluent_mol_frac=_diluent_mol_frac,
        inert=_inert
    )['rxn_table_id']
    # if len(exist_check) > 0:
    #     t.delete_test(exist_check[0])

    reactions = []
    # noinspection PyCallByClass,PyArgumentList
    for rxn in ct.Reaction.listFromFile(_mechanism):
        if not any([
            s in list(rxn.reactants) + list(rxn.products)
            for s in _inert_species
        ]):
            reactions.append(rxn)

    # pbar = tqdm(total=len(reactions))

    _lock = mp.Lock()
    p = mp.Pool(initializer=init, initargs=(_lock,))
    for _ in tqdm(
        p.istarmap(
            perform_study,
            [
                [
                    # pbar,
                    _mechanism,
                    _initial_temp,
                    _initial_press,
                    _equivalence,
                    _fuel,
                    _oxidizer,
                    _diluent,
                    _diluent_mol_frac,
                    _inert,
                    _perturbation_fraction,
                    i,
                    # _lock
                ] for i in range(len(reactions))
            ]
        ), total=len(reactions)
    ):
        pass
    p.close()
    # for i in range(len(reactions)):
    #     perform_study(
    #         _mechanism,
    #         _initial_temp,
    #         _initial_press,
    #         _equivalence,
    #         _fuel,
    #         _oxidizer,
    #         _diluent,
    #         _diluent_mol_frac,
    #         _inert,
    #         _perturbation_fraction,
    #         i,
    #         _lock
    #     )
    #     pbar.update(1)
    print("woo!")
