import os
from tempfile import NamedTemporaryFile
from unittest import TestCase

import cantera as ct
import deepdiff
import funcs.simulation.sensitivity.detonation.database as db

TEST_DIR = os.path.abspath(os.path.dirname(__file__))

# todo: delete databases at the end of each test


class TestDataBase(TestCase):
    def setUp(self):
        self.db_file = NamedTemporaryFile()

    def test_create(self):
        test = db.DataBase(path=self.db_file.name)
        # todo: more tests


class TestBaseReactionTable(TestCase):
    def setUp(self):
        self.db_file = NamedTemporaryFile()
        self.db = db.DataBase(path=self.db_file.name)
        self.table = self.db.base_rxn_table

    def test_store_and_check(self):
        """
        Base reaction table stores and checks mechanisms properly
        """
        mechanism = "gri30.cti"

        assert not self.table.has_mechanism(mechanism=mechanism)

        self.table.store_all_reactions(gas=ct.Solution(mechanism), mechanism=mechanism)
        self.table.cur.execute(
            f"select count(*) n_rxns from {self.table.name} where mechanism = :mech",
            dict(mech=mechanism),
        )
        result = self.table.cur.fetchone()
        self.assertEqual(result["n_rxns"], 325)

        assert self.table.has_mechanism(mechanism=mechanism)


class TestTestConditionsTable(TestCase):
    def setUp(self):
        self.db_file = NamedTemporaryFile()
        self.db = db.DataBase(path=self.db_file.name)
        self.table = self.db.test_conditions_table

    def test_store_and_check(self):
        """
        Test conditions table stores and checks test data properly
        """
        mechanism = "gri30.cti"
        initial_temp = 300.0
        initial_press = 101325.0
        fuel = "H2"
        oxidizer = "O2"
        equivalence = 1.0
        diluent = "CO2"
        diluent_mol_frac = 0.2
        cj_speed = 1234.5
        ind_len_west = 1e-1
        ind_len_gav = 1e-2
        ind_len_ng = 1e-3
        cell_size_west = 2e-1
        cell_size_gav = 2e-2
        cell_size_ng = 2e-3

        assert not self.table.test_exists(test_id=0)

        test_id = self.table.insert_new_row(
            mechanism=mechanism,
            initial_temp=initial_temp,
            initial_press=initial_press,
            fuel=fuel,
            oxidizer=oxidizer,
            equivalence=equivalence,
            diluent=diluent,
            diluent_mol_frac=diluent_mol_frac,
        )

        assert self.table.test_exists(test_id=test_id)

        self.table.add_results(
            test_id=test_id,
            cj_speed=cj_speed,
            ind_len_west=ind_len_west,
            ind_len_gav=ind_len_gav,
            ind_len_ng=ind_len_ng,
            cell_size_west=cell_size_west,
            cell_size_gav=cell_size_gav,
            cell_size_ng=cell_size_ng,
        )

        expected_row = dict(
            test_id=test_id,
            mechanism=mechanism,
            initial_temp=initial_temp,
            initial_press=initial_press,
            fuel=fuel,
            oxidizer=oxidizer,
            equivalence=equivalence,
            diluent=diluent,
            diluent_mol_frac=diluent_mol_frac,
            cj_speed=cj_speed,
            ind_len_west=ind_len_west,
            ind_len_gav=ind_len_gav,
            ind_len_ng=ind_len_ng,
            cell_size_west=cell_size_west,
            cell_size_gav=cell_size_gav,
            cell_size_ng=cell_size_ng,
        )
        row = self.table.fetch_row_by_id(test_id=test_id)
        assert deepdiff.DeepDiff(expected_row, row, exclude_regex_paths=r"date_stored") == {}
        row = self.table.fetch_rows_by_id(test_ids=[test_id])
        assert deepdiff.DeepDiff(
            {k: [v] for k, v in expected_row.items()},
            row,
            exclude_regex_paths=r"date_stored",
        ) == {}


# todo: tests for PerturbedResultsTable
