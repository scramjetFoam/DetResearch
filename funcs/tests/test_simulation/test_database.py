import os
from tempfile import NamedTemporaryFile

import cantera as ct
import deepdiff
import funcs.simulation.sensitivity.detonation.database as db

TEST_DIR = os.path.abspath(os.path.dirname(__file__))


def test_database():
    db_file = NamedTemporaryFile()
    db.DataBase(path=db_file.name)


def test_base_reaction_table():
    db_file = NamedTemporaryFile()
    test_db = db.DataBase(path=db_file.name)
    table = test_db.base_rxn_table
    
    mechanism = "gri30.cti"
    assert not table.has_mechanism(mechanism=mechanism)

    table.store_all_reactions(gas=ct.Solution(mechanism), mechanism=mechanism)
    table.cur.execute(
        f"select count(*) n_rxns from {table.name} where mechanism = :mech",
        dict(mech=mechanism),
    )
    result = table.cur.fetchone()
    assert result["n_rxns"] == 325
    assert table.has_mechanism(mechanism=mechanism)


def test_test_conditions_table():
    db_file = NamedTemporaryFile()
    test_db = db.DataBase(path=db_file.name)
    table = test_db.test_conditions_table

    expected_row = dict(
        mechanism="gri30.cti",
        initial_temp=300.0,
        initial_press=101325.0,
        fuel="H2",
        oxidizer="O2",
        equivalence=1.0,
        diluent="CO2",
        diluent_mol_frac=0.2,
        cj_speed=1234.5,
        ind_len_west=1e-1,
        ind_len_gav=1e-2,
        ind_len_ng=1e-3,
        cell_size_west=2e-1,
        cell_size_gav=2e-2,
        cell_size_ng=2e-3,
    )
    assert not table.test_exists(test_id=1)

    test_id = table.insert_new_row(
        mechanism=expected_row["mechanism"],
        initial_temp=expected_row["initial_temp"],
        initial_press=expected_row["initial_press"],
        fuel=expected_row["fuel"],
        oxidizer=expected_row["oxidizer"],
        equivalence=expected_row["equivalence"],
        diluent=expected_row["diluent"],
        diluent_mol_frac=expected_row["diluent_mol_frac"],
    )
    assert table.test_exists(test_id=test_id)

    table.add_results(
        test_id=test_id,
        cj_speed=expected_row["cj_speed"],
        ind_len_west=expected_row["ind_len_west"],
        ind_len_gav=expected_row["ind_len_gav"],
        ind_len_ng=expected_row["ind_len_ng"],
        cell_size_west=expected_row["cell_size_west"],
        cell_size_gav=expected_row["cell_size_gav"],
        cell_size_ng=expected_row["cell_size_ng"],
    )
    expected_row["test_id"] = test_id
    row = table.fetch_row_by_id(test_id=test_id)
    assert deepdiff.DeepDiff(expected_row, row, exclude_regex_paths=r"date_stored") == {}
    row = table.fetch_rows_by_id(test_ids=[test_id])
    assert deepdiff.DeepDiff(
        {k: [v] for k, v in expected_row.items()},
        row,
        exclude_regex_paths=r"date_stored",
    ) == {}


# todo: tests for PerturbedResultsTable
