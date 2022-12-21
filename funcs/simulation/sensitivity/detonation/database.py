# -*- coding: utf-8 -*-
"""
PURPOSE:
    Database management for data storage

CREATED BY:
    Mick Carter
    Oregon State University
    CIRE and Propulsion Lab
    cartemic@oregonstate.edu
"""
import sqlite3
import warnings
from functools import lru_cache
from typing import Any, Dict, List, Optional

import cantera as ct


class DataBase:  # todo: implement these changes throughout the code base
    """
    A class for database-level operations
    """

    def __init__(self, path: str):
        self._path = path
        self.con = sqlite3.connect(path)
        self.con.row_factory = sqlite3.Row
        self.test_conditions_table = TestConditionsTable(con=self.con)
        self.base_rxn_table = BaseReactionTable(con=self.con)
        self.perturbed_results_table = PerturbedResultsTable(
            con=self.con,
            base_rxn_table=self.base_rxn_table,
            test_conditions_table=self.test_conditions_table,
        )

    def __del__(self):
        self.con.commit()
        self.con.close()

    def new_test(
        self,
        mechanism,
        initial_temp,
        initial_press,
        fuel,
        oxidizer,
        equivalence,
        diluent,
        diluent_mol_frac,
    ):
        if not self.base_rxn_table.has_mechanism(mechanism=mechanism):
            self.base_rxn_table.store_all_reactions(gas=ct.Solution(mechanism), mechanism=mechanism)
        test_id = self.test_conditions_table.insert_new_row(
            mechanism=mechanism,
            initial_temp=initial_temp,
            initial_press=initial_press,
            fuel=fuel,
            oxidizer=oxidizer,
            equivalence=equivalence,
            diluent=diluent,
            diluent_mol_frac=diluent_mol_frac,
        )

        return test_id


def build_query_str(inputs: Dict[str, Any], table_name: str):
    """
    Builds a SQL query string. Any inputs which are None will be left wild.

    Parameters
    ----------
    inputs : dict
        Dictionary of keyword arguments to build a query string around.
        This has been left as flexible as possible so that this method can
        build query strings for any of the table types.
    table_name : str
        Table which is being queried

    Returns
    -------
    cmd_str : str
        SQL command to search for the desired inputs
    """
    where = " WHERE " if inputs else ""
    sql_varnames = [f"{item} = :{item}" for item in inputs.keys()]
    cmd_str = f"SELECT * FROM {table_name} {where} {' AND '.join(sql_varnames)};"

    return cmd_str


def rows_to_dict(cur: sqlite3.Cursor):
    data = {}
    for row in (dict(r) for r in cur.fetchall()):
        for col, value in row.items():
            data.setdefault(col, []).append(value)

    return data


def row_to_dict(cur: sqlite3.Cursor):
    return dict(cur.fetchone())


class BaseReactionTable:
    name = "rxn_base"

    def __init__(self, con: sqlite3.Connection):
        self.cur = con.cursor()
        if not self.cur.fetchall():
            self._create()

    def _create(self):
        """
        Creates a table of base (unperturbed) reactions and their rate constants
        in the current database
        """
        self.cur.execute(
            f"""
            CREATE TABLE {self.name} (
                mechanism TEXT,
                rxn_no INTEGER,
                rxn TEXT,
                k_i REAL,
                PRIMARY KEY (mechanism, rxn_no)
            );
            """
        )

    @property
    @lru_cache(maxsize=None)
    def columns(self) -> List[str]:
        """
        A list of all column names in the current table.
        """
        self.cur.execute(f"PRAGMA table_info({self.name});")

        return [item[1] for item in self.cur.fetchall()]

    def store_all_reactions(self, gas: ct.Solution, mechanism: str):
        self.cur.execute(
            f"DELETE FROM {self.name} WHERE mechanism = :mechanism",
            {"mechanism": mechanism},
        )
        for rxn_no, rxn, k_i in zip(range(gas.n_reactions), gas.reaction_equations(), gas.forward_rate_constants):
            self.cur.execute(
                f"""
                INSERT INTO {self.name} VALUES (
                    :mechanism,
                    :rxn_no,
                    :rxn,
                    :k_i
                );
                """,
                {
                    "mechanism": mechanism,
                    "rxn_no": rxn_no,
                    "rxn": rxn,
                    "k_i": k_i,
                },
            )
            self.cur.connection.commit()

    def has_mechanism(self, mechanism: str) -> bool:
        self.cur.execute(
            f"SELECT COUNT(*) > 0 has_some FROM {self.name} WHERE mechanism = :mechanism",
            {"mechanism": mechanism},
        )

        return bool(self.cur.fetchone()["has_some"])


class TestConditionsTable:
    name = "test_conditions"

    def __init__(self, con: sqlite3.Connection):
        self.cur = con.cursor()
        if not self.cur.fetchall():
            self._create()

    def _create(self):
        """
        Creates a table of test conditions and results in the current database
        """
        self.cur.execute(
            f"""
            CREATE TABLE {self.name} (
                test_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                date_stored TEXT,
                mechanism TEXT,
                initial_temp REAL,
                initial_press REAL,
                fuel TEXT,
                oxidizer TEXT,
                equivalence REAL,
                diluent TEXT,
                diluent_mol_frac REAL,
                cj_speed REAL,
                ind_len_west REAL,
                ind_len_gav REAL,
                ind_len_ng REAL,
                cell_size_west REAL,
                cell_size_gav REAL,
                cell_size_ng REAL
            );
            """
        )

    @property
    @lru_cache(maxsize=None)
    def columns(self) -> List[str]:
        """
        A list of all column names in the current table.
        """
        self.cur.execute(f"PRAGMA table_info({self.name});")

        return [item[1] for item in self.cur.fetchall()]

    def test_exists(self, test_id: Optional[int]) -> bool:
        """
        Checks the current table for a specific row of data

        Parameters
        ----------
        test_id
            Test ID corresponding to updated row
        """
        self.cur.execute(f"SELECT * from {self.name} WHERE test_id = :test_id", {"test_id": test_id})

        return len(self.cur.fetchall()) > 0

    def fetch_rows_by_id(self, test_ids: List[int]):
        cleaned_ids = []
        for test_id in test_ids:
            if not isinstance(test_id, int):
                raise ValueError("Test IDs must be integers.")
            cleaned_ids.append(str(test_id))
        self.cur.execute(f"SELECT * FROM {self.name} where test_id in ({','.join(cleaned_ids)})")

        return rows_to_dict(self.cur)

    def fetch_row_by_id(self, test_id: int):
        if not isinstance(test_id, int):
            raise ValueError("Test IDs must be integers.")
        self.cur.execute(f"SELECT * FROM {self.name} where test_id = {test_id}")

        return row_to_dict(self.cur)

    def insert_new_row(
        self,
        mechanism,
        initial_temp,
        initial_press,
        fuel,
        oxidizer,
        equivalence,
        diluent,
        diluent_mol_frac,
    ) -> int:
        """
        Stores a row of test data in the current table.

        Parameters
        ----------
        mechanism : str
            Mechanism used for the current row's computation
        initial_temp : float
            Initial temperature for the current row, in Kelvin
        initial_press : float
            Initial pressure for the current row, in Pascals
        equivalence : float
            Equivalence ratio for the current row
        fuel : str
            Fuel used in the current row
        oxidizer : str
            Oxidizer used in the current row
        diluent : str
            Diluent used in the current row
        diluent_mol_frac : float
            Mole fraction of diluent used in the current row
            protect existing entries

        Returns
        -------
        test_id : int
            Test ID
        """
        self.cur.execute(
            f"""
            INSERT INTO {self.name} VALUES (
                Null,
                datetime('now', 'localtime'),
                :mechanism,
                :initial_temp,
                :initial_press,
                :fuel,
                :oxidizer,
                :equivalence,
                :diluent,
                :diluent_mol_frac,
                Null,
                Null,
                Null,
                Null,
                Null,
                Null,
                Null
            );
            """,
            {
                "mechanism": mechanism,
                "initial_temp": initial_temp,
                "initial_press": initial_press,
                "fuel": fuel,
                "oxidizer": oxidizer,
                "equivalence": equivalence,
                "diluent": diluent,
                "diluent_mol_frac": diluent_mol_frac,
            },
        )
        self.cur.connection.commit()

        return self.cur.lastrowid

    def add_results(
        self,
        test_id,
        cj_speed,
        ind_len_west,
        ind_len_gav,
        ind_len_ng,
        cell_size_west,
        cell_size_gav,
        cell_size_ng,
    ):
        """
        Updates the stored test results for `test_id`

        Parameters
        ----------
        test_id:
            Test ID corresponding to updated row
        cj_speed : float
            CJ speed to update
        ind_len_west : float
            Induction length (Westbrook)
        ind_len_gav : float
            Induction length (Gavrikov)
        ind_len_ng : float
            Induction length (Ng)
        cell_size_west : float
            Cell size (Westbrook)
        cell_size_gav : float
            Cell size (Gavrikov)
        cell_size_ng : float
            Cell size (Ng)
        """
        self.cur.execute(
            f"""
            UPDATE {self.name} SET
                date_stored = datetime('now', 'localtime'),
                cj_speed = :cj_speed,
                ind_len_west = :ind_len_west,
                ind_len_gav = :ind_len_gav,
                ind_len_ng = :ind_len_ng,
                cell_size_west = :cell_size_west,
                cell_size_gav = :cell_size_gav,
                cell_size_ng = :cell_size_ng
            WHERE
                test_id = :test_id
            """,
            {
                "cj_speed": cj_speed,
                "ind_len_west": ind_len_west,
                "ind_len_gav": ind_len_gav,
                "ind_len_ng": ind_len_ng,
                "cell_size_west": cell_size_west,
                "cell_size_gav": cell_size_gav,
                "cell_size_ng": cell_size_ng,
                "test_id": test_id,
            },
        )
        self.cur.connection.commit()


class PerturbedResultsTable:
    # todo: update all tests
    # todo: update everything that uses this stuff

    name = "perturbed_results"

    def __init__(
        self,
        con: sqlite3.Connection,
        base_rxn_table: BaseReactionTable,
        test_conditions_table: TestConditionsTable,
        testing=False,
    ):
        self.cur = con.cursor()
        self._testing = testing
        self._base_rxn_table = base_rxn_table
        self._test_conditions_table = test_conditions_table
        if not self.cur.fetchall():
            self._create()

    @property
    @lru_cache(maxsize=None)
    def columns(self):
        """
        A list of all column names in the current table.
        """
        self.cur.execute(f"PRAGMA table_info({self.name});")

        return [item[1] for item in self.cur.fetchall()]

    def test_exists(self, rxn_no):  # todo: we also need mechanism -- use composite PK
        """
        Checks the current table for a specific row of data

        Parameters

        Returns
        -------
        row_found : bool
            True if a row with the given information was found in the current
            table, False if not
        """
        self.cur.execute(
            f"""
            SELECT * from {self.name} WHERE rxn_no = :rxn_no
            """,
            {"rxn_no": rxn_no},
        )

        return len(self.cur.fetchall()) > 0

    def _create(self):
        """
        Creates a table of perturbed reaction results in the current database
        """
        self.cur.execute(
            f"""
            CREATE TABLE {self.name} (
                test_id INTEGER,
                rxn_no INTEGER,
                stored_date TEXT,
                rxn TEXT,
                k_i_pert REAL,
                ind_len_west REAL,
                ind_len_gav REAL,
                ind_len_ng REAL,
                cell_size_west REAL,
                cell_size_gav REAL,
                cell_size_ng REAL,
                sens_ind_len_west REAL,
                sens_ind_len_gav REAL,
                sens_ind_len_ng REAL,
                sens_cell_size_west REAL,
                sens_cell_size_gav REAL,
                sens_cell_size_ng REAL,
                PRIMARY KEY (test_id, rxn_no)
            );
            """
        )

    def update_row(
        self,
        test_id,
        rxn_no,
        rxn,
        k_i,
        ind_len_west,
        ind_len_gav,
        ind_len_ng,
        cell_size_west,
        cell_size_gav,
        cell_size_ng,
        sens_ind_len_west,
        sens_ind_len_gav,
        sens_ind_len_ng,
        sens_cell_size_west,
        sens_cell_size_gav,
        sens_cell_size_ng,
    ):
        """
        Updates the CJ velocity and forward reaction rate (k_i) for a set of
        conditions.

        Parameters
        ----------
        ind_len_west : float
            Induction length (Westbrook)
        ind_len_gav : float
            Induction length (Gavrikov)
        ind_len_ng : float
            Induction length (Ng)
        cell_size_west : float
            Cell size (Westbrook)
        cell_size_gav : float
            Cell size (Gavrikov)
        cell_size_ng : float
            Cell size (Ng)
        """
        self.cur.execute(
            f"""
            UPDATE {self.name} SET
                k_i = :k_i,
                ind_len_west = :ind_len_west,
                ind_len_gav = :ind_len_gav,
                ind_len_ng = :ind_len_ng,
                cell_size_west = :cell_size_west,
                cell_size_gav = :cell_size_gav,
                cell_size_ng = :cell_size_ng,
                sens_ind_len_west = :sens_ind_len_west,
                sens_ind_len_gav = :sens_ind_len_gav,
                sens_ind_len_ng = :sens_ind_len_ng,
                sens_cell_size_west = :sens_cell_size_west,
                sens_cell_size_gav = :sens_cell_size_gav,
                sens_cell_size_ng = :sens_cell_size_ng
            WHERE
                (test_id, rxn_no) = (:rxn_no, :test_id)
            """,
            {
                "test_id": test_id,
                "rxn_no": rxn_no,
                "rxn": rxn,
                "k_i": k_i,
                "ind_len_west": ind_len_west,
                "ind_len_gav": ind_len_gav,
                "ind_len_ng": ind_len_ng,
                "cell_size_west": cell_size_west,
                "cell_size_gav": cell_size_gav,
                "cell_size_ng": cell_size_ng,
                "sens_ind_len_west": sens_ind_len_west,
                "sens_ind_len_gav": sens_ind_len_gav,
                "sens_ind_len_ng": sens_ind_len_ng,
                "sens_cell_size_west": sens_cell_size_west,
                "sens_cell_size_gav": sens_cell_size_gav,
                "sens_cell_size_ng": sens_cell_size_ng,
            },
        )
        self.cur.connection.commit()

    def store_row(
        self,
        test_id,
        rxn_no,
        rxn,
        k_i,
        ind_len_west,
        ind_len_gav,
        ind_len_ng,
        cell_size_west,
        cell_size_gav,
        cell_size_ng,
        sens_ind_len_west,
        sens_ind_len_gav,
        sens_ind_len_ng,
        sens_cell_size_west,
        sens_cell_size_gav,
        sens_cell_size_ng,
        overwrite_existing=False,
    ):
        """
        Stores a row of data in the current table.

        If a row with this information already exists in the current table,
        overwrite_existing decides whether to overwrite the existing data or
        disregard the current data.

        Parameters
        ----------
        test_id : int
            Test ID
        rxn_no : int
            Reaction number of the perturbed reaction in the mechanism's
            reaction list
        rxn : str
            Equation for the perturbed reaction
        k_i : float
            Forward reaction rate constant for the perturbed reaction
        ind_len_west : float
            Induction length (Westbrook)
        ind_len_gav : float
            Induction length (Gavrikov)
        ind_len_ng : float
            Induction length (Ng)
        cell_size_west : float
            Cell size (Westbrook)
        cell_size_gav : float
            Cell size (Gavrikov)
        cell_size_ng : float
            Cell size (Ng)
        sens_ind_len_west : float
            Induction length sensitivity (Westbrook)
        sens_ind_len_gav : float
            Induction length sensitivity (Gavrikov)
        sens_ind_len_ng : float
            Induction length sensitivity (Ng)
        sens_cell_size_west : float
            Cell size (Westbrook)
        sens_cell_size_gav : float
            Cell size sensitivity (Gavrikov)
        sens_cell_size_ng : float
            Cell size sensitivity (Ng)
        overwrite_existing : bool
            True to overwrite an existing entry if it exists, False to
            protect existing entries

        Returns
        -------
        test_id : int
            Test ID
        """
        if self.test_exists(rxn_no):
            # a row with the current information was found
            if overwrite_existing:
                self.update_row(
                    test_id=test_id,
                    rxn_no=rxn_no,
                    rxn=rxn,
                    k_i=k_i,
                    ind_len_west=ind_len_west,
                    ind_len_gav=ind_len_gav,
                    ind_len_ng=ind_len_ng,
                    cell_size_west=cell_size_west,
                    cell_size_gav=cell_size_gav,
                    cell_size_ng=cell_size_ng,
                    sens_ind_len_west=sens_ind_len_west,
                    sens_ind_len_gav=sens_ind_len_gav,
                    sens_ind_len_ng=sens_ind_len_ng,
                    sens_cell_size_west=sens_cell_size_west,
                    sens_cell_size_gav=sens_cell_size_gav,
                    sens_cell_size_ng=sens_cell_size_ng,
                )
            else:
                # warn the user that the current input was ignored
                warnings.warn("Cannot overwrite row unless overwrite_existing=True", Warning)

        else:
            # no rows with the current information were found
            self.insert_row(
                test_id=test_id,
                rxn_no=rxn_no,
                rxn=rxn,
                k_i=k_i,
                ind_len_west=ind_len_west,
                ind_len_gav=ind_len_gav,
                ind_len_ng=ind_len_ng,
                cell_size_west=cell_size_west,
                cell_size_gav=cell_size_gav,
                cell_size_ng=cell_size_ng,
                sens_ind_len_west=sens_ind_len_west,
                sens_ind_len_gav=sens_ind_len_gav,
                sens_ind_len_ng=sens_ind_len_ng,
                sens_cell_size_west=sens_cell_size_west,
                sens_cell_size_gav=sens_cell_size_gav,
                sens_cell_size_ng=sens_cell_size_ng,
            )

    def insert_row(
        self,
        test_id: int,
        rxn_no,
        rxn,
        k_i,
        ind_len_west,
        ind_len_gav,
        ind_len_ng,
        cell_size_west,
        cell_size_gav,
        cell_size_ng,
        sens_ind_len_west,
        sens_ind_len_gav,
        sens_ind_len_ng,
        sens_cell_size_west,
        sens_cell_size_gav,
        sens_cell_size_ng,
    ):
        self.cur.execute(
            f"""
            INSERT INTO {self.name} VALUES (
                :test_id,
                :rxn_no,
                stored_date: datetime('now', 'localtime'),
                :rxn,
                :k_i,
                :ind_len_west,
                :ind_len_gav,
                :ind_len_ng,
                :cell_size_west,
                :cell_size_gav,
                :cell_size_ng,
                :sens_ind_len_west,
                :sens_ind_len_gav,
                :sens_ind_len_ng,
                :sens_cell_size_west,
                :sens_cell_size_gav,
                :sens_cell_size_ng
            );
            """,
            {
                "test_id": test_id,
                "rxn_no": rxn_no,
                "rxn": rxn,
                "k_i": k_i,
                "ind_len_west": ind_len_west,
                "ind_len_gav": ind_len_gav,
                "ind_len_ng": ind_len_ng,
                "cell_size_west": cell_size_west,
                "cell_size_gav": cell_size_gav,
                "cell_size_ng": cell_size_ng,
                "sens_ind_len_west": sens_ind_len_west,
                "sens_ind_len_gav": sens_ind_len_gav,
                "sens_ind_len_ng": sens_ind_len_ng,
                "sens_cell_size_west": sens_cell_size_west,
                "sens_cell_size_gav": sens_cell_size_gav,
                "sens_cell_size_ng": sens_cell_size_ng,
            },
        )
        self.cur.connection.commit()

    def fetch_rows(
        self,
        test_id=None,
        rxn_no=None,
        rxn=None,
        k_i=None,
        ind_len_west=None,
        ind_len_gav=None,
        ind_len_ng=None,
        cell_size_west=None,
        cell_size_gav=None,
        cell_size_ng=None,
        sens_ind_len_west=None,
        sens_ind_len_gav=None,
        sens_ind_len_ng=None,
        sens_cell_size_west=None,
        sens_cell_size_gav=None,
        sens_cell_size_ng=None,
    ):
        """
        Fetches all rows from the current database with the desired inputs.
        Any inputs which are None will be left wild.

        Parameters
        ----------

        Returns
        -------
        data : dict
            Dictionary containing the rows of the current table which match
            the input criteria. Keys are column names, and values are lists.
        """
        inputs = {
            "test_id": test_id,
            "rxn_no": rxn_no,
            "rxn": rxn,
            "k_i": k_i,
            "ind_len_west": ind_len_west,
            "ind_len_gav": ind_len_gav,
            "ind_len_ng": ind_len_ng,
            "cell_size_west": cell_size_west,
            "cell_size_gav": cell_size_gav,
            "cell_size_ng": cell_size_ng,
            "sens_ind_len_west": sens_ind_len_west,
            "sens_ind_len_gav": sens_ind_len_gav,
            "sens_ind_len_ng": sens_ind_len_ng,
            "sens_cell_size_west": sens_cell_size_west,
            "sens_cell_size_gav": sens_cell_size_gav,
            "sens_cell_size_ng": sens_cell_size_ng,
        }
        cmd_str = build_query_str(inputs, self.name)
        self.cur.execute(cmd_str, inputs)
        info = self.cur.fetchall()
        labels = self.columns
        data = {lbl: [] for lbl in labels}
        for row in info:
            for lbl, d in zip(labels, row):
                data[lbl].append(d)

        return data
