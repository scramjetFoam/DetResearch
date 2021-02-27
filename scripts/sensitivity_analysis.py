import multiprocessing as mp

import cantera as ct
from tqdm import tqdm

import funcs.simulation.sensitivity.database as db
# noinspection PyUnresolvedReferences
from funcs.simulation.sensitivity import istarmap
from funcs.simulation.sensitivity.analysis import perform_study, init

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
    _diluent_mol_frac = 0.2
    _perturbation_fraction = 1e-2
    db_name = "sensitivity_2.sqlite"

    t = db.Table(
        db_name,
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

    reactions = []
    # noinspection PyCallByClass,PyArgumentList
    for rxn in ct.Reaction.listFromFile(_mechanism):
        if not any([
            s in list(rxn.reactants) + list(rxn.products)
            for s in _inert_species
        ]):
            reactions.append(rxn)

    # PARALLEL -- remove _lock to args in cell_size.CellSize
    _lock = mp.Lock()
    n_rxns = len(reactions)
    with mp.Pool(initializer=init, initargs=(_lock,)) as p:
        for _ in tqdm(
            p.istarmap(
                perform_study,
                [
                    [
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
                        db_name,
                    ] for i in range(n_rxns)
                ]
            ),
            total=n_rxns,
            colour="499c54"
        ):
            pass

    # SERIAL -- add _lock to args in cell_size.CellSize
    # pbar = tqdm(total=n_rxns)
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
