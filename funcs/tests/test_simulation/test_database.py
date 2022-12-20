import os
from tempfile import NamedTemporaryFile
from unittest import TestCase

import cantera as ct
import funcs.simulation.sensitivity.detonation.database as db

TEST_DIR = os.path.abspath(os.path.dirname(__file__))

# todo: delete databases at the end of each test


def bind(instance, func, as_name=None):  # pragma: no cover
    """
    https://stackoverflow.com/questions/1015307/python-bind-an-unbound-method

    Bind the function *func* to *instance*, with either provided name *as_name*
    or the existing name of *func*. The provided *func* should accept the
    instance as the first argument, i.e. "self".
    """
    if as_name is None:
        as_name = func.__name__
    bound_method = func.__get__(instance, instance.__class__)
    setattr(instance, as_name, bound_method)
    return bound_method


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
