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
import inspect
import sqlite3
import warnings
from typing import Any, Dict, List, Optional

import cantera as ct


def _formatwarnmsg_impl(msg):  # pragma: no cover
    # happier warning format :)
    s = "%s: %s\n" % (msg.category.__name__, msg.message)
    return s


warnings._formatwarnmsg_impl = _formatwarnmsg_impl
warnings.simplefilter("always")


class DataBase:
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
        cj_speed,
        ind_len_west,
        ind_len_gav,
        ind_len_ng,
        cell_size_west,
        cell_size_gav,
        cell_size_ng,
    ):
        if not self.base_rxn_table.has_mechanism(mechanism=mechanism):
            self.base_rxn_table.store_all_reactions(
                gas=ct.Solution(mechanism), mechanism=mechanism
            )
        test_id = self.test_conditions_table.insert_new_row(
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

        return test_id

    def list_tables(self):
        """
        Finds every table in a given database.

        Returns
        -------
        table_list : list
            All tables within the requested database
        """
        with sqlite3.connect(self._path) as con:
            cur = con.cursor()
            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' and name != 'sqlite_sequence';"
            )

        tables = cur.fetchall()
        con.close()
        table_list = [item[0] for item in tables]
        return table_list


def get_table_with_name(cur: sqlite3.Cursor, name: str):
    name = clean_table_name(name)
    cur.execute(f"select name from sqlite_master where type='table' and name='{name}';")


def clean_table_name(name: str):
    """
    Cleans a table name string to keep me from doing anything too stupid.
    Alphanumeric values and underscores are allowed; anything else will
    throw a NameError.

    Parameters
    ----------
    name : str

    Returns
    -------
    str
    """
    if any([not (char.isalnum() or char == "_") for char in name]):
        raise NameError(
            "Table name must be entirely alphanumeric. Underscores are allowed."
        )
    else:
        return name.lower()


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
    table_name = clean_table_name(table_name)
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
        get_table_with_name(self.cur, self.name)
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
        for rxn_no, rxn, k_i in zip(
            range(gas.n_reactions), gas.reaction_equations(), gas.forward_rate_constants
        ):
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
        get_table_with_name(self.cur, self.name)
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
    def columns(self) -> List[str]:
        """
        A list of all column names in the current table.
        """
        self.cur.execute(f"PRAGMA table_info({self.name});")

        return [item[1] for item in self.cur.fetchall()]

    # noinspection PyUnusedLocal
    def test_exists(
        self,
        mechanism=None,
        initial_temp=None,
        initial_press=None,
        fuel=None,
        oxidizer=None,
        equivalence=None,
        diluent=None,
        diluent_mol_frac=None,
    ):
        """
        Checks the current table for a specific row of data

        Parameters
        ----------
        mechanism : str
            Mechanism used for the desired row's computation
        initial_temp : float
            Initial temperature for the desired row, in Kelvin
        initial_press : float
            Initial pressure for the desired row, in Pascals
        equivalence : float
            Equivalence ratio for the desired row
        fuel : str
            Fuel used in the desired row
        oxidizer : str
            Oxidizer used in the desired row
        diluent : str
            Diluent used in the desired row
        diluent_mol_frac : float
            Mole fraction of diluent used in the desired row

        Returns
        -------
        row_found : bool
        """
        inputs = {
            key: value
            for key, value in inspect.getargvalues(inspect.currentframe())[3].items()
            if key in self.columns
        }
        query_str = build_query_str(inputs, self.name)
        self.cur.execute(
            query_str,
            {key: value for key, value in inputs.items() if value is not None},
        )

        return len(self.cur.fetchall()) > 0

    def fetch_rows_by_value(
        self,
        mechanism=None,
        initial_temp=None,
        initial_press=None,
        fuel=None,
        oxidizer=None,
        equivalence=None,
        diluent=None,
        diluent_mol_frac=None,
    ):
        """
        Fetches all rows from the current database with the desired inputs.
        Any inputs which are None will be left wild.

        Parameters
        ----------
        mechanism : str
            Mechanism to search for
        initial_temp : float
            Initial temperature to search for, in Kelvin
        initial_press : float
            Initial pressure to search for, in Pascals
        fuel : str
            Fuel to search for
        oxidizer : str
            Oxidizer to search for
        equivalence : float
            Equivalence ratio to search for
        diluent : str
            Diluent to search for
        diluent_mol_frac : float
            Mole fraction of diluent to search for

        Returns
        -------
        data : dict
            Dictionary containing the rows of the current table which match
            the input criteria. Keys are column names, and values are lists.
        """
        inputs = {
            "mechanism": mechanism,
            "initial_temp": initial_temp,
            "initial_press": initial_press,
            "equivalence": equivalence,
            "fuel": fuel,
            "oxidizer": oxidizer,
            "diluent": diluent,
            "diluent_mol_frac": diluent_mol_frac,
        }
        cmd_str = build_query_str(inputs=inputs, table_name=self.name)
        self.cur.execute(cmd_str, inputs)

        return rows_to_dict(self.cur)

    def fetch_rows_by_id(self, test_ids: List[int]):
        cleaned_ids = []
        for test_id in test_ids:
            if not isinstance(test_id, int):
                raise ValueError("Test IDs must be integers.")
            cleaned_ids.append(str(test_id))
        self.cur.execute(
            f"SELECT * FROM {self.name} where test_id in ({','.join(cleaned_ids)})"
        )

        return rows_to_dict(self.cur)

    def fetch_row_by_id(self, test_id: int):
        if not isinstance(test_id, int):
            raise ValueError("Test IDs must be integers.")
        self.cur.execute(f"SELECT * FROM {self.name} where test_id = {test_id}")

        return row_to_dict(self.cur)

    def update_existing_row(
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
        Updates the CJ velocity and forward reaction rate (k_i) for a set of
        conditions.

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
        cj_speed,
        ind_len_west,
        ind_len_gav,
        ind_len_ng,
        cell_size_west,
        cell_size_gav,
        cell_size_ng,
    ) -> int:
        """
        Stores a row of data in the current table.

        If a row with this information already exists in the current table,
        overwrite_existing decides whether to overwrite the existing data or
        disregard the current data.

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
        cj_speed : float
            Current CJ speed
        diluent : str
            Diluent used in the current row
        diluent_mol_frac : float
            Mole fraction of diluent used in the current row
            protect existing entries
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

        Returns
        -------
        test_id : int
            Test ID
        """
        # todo: don't create duplicate rows
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
                :cj_speed,
                :ind_len_west,
                :ind_len_gav,
                :ind_len_ng,
                :cell_size_west,
                :cell_size_gav,
                :cell_size_ng
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
                "cj_speed": cj_speed,
                "ind_len_west": ind_len_west,
                "ind_len_gav": ind_len_gav,
                "ind_len_ng": ind_len_ng,
                "cell_size_west": cell_size_west,
                "cell_size_gav": cell_size_gav,
                "cell_size_ng": cell_size_ng,
            },
        )
        self.cur.connection.commit()

        return self.cur.lastrowid

    def store_row(
        self,
        test_id: Optional[
            int
        ],  # todo: use this instead of overwrite_existing -- None makes new row, ID edits
        mechanism,
        initial_temp,
        initial_press,
        fuel,
        oxidizer,
        equivalence,
        diluent,
        diluent_mol_frac,
        cj_speed,
        ind_len_west,
        ind_len_gav,
        ind_len_ng,
        cell_size_west,
        cell_size_gav,
        cell_size_ng,
    ):
        """
        Stores a row of data in the current table.

        If a row with this information already exists in the current table,
        overwrite_existing decides whether to overwrite the existing data or
        disregard the current data.

        Parameters
        ----------
        test_id : Optional[int]
            Row ID
            Attempted writes to an existing row will be ignored unless test_id=None is passed
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
        cj_speed : float
            Current CJ speed
        diluent : str
            Diluent used in the current row
        diluent_mol_frac : float
            Mole fraction of diluent used in the current row
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

        Returns
        -------
        test_id : Optional[int]
            Row ID
        """
        if self.test_exists(
            mechanism=mechanism,
            initial_temp=initial_temp,
            initial_press=initial_press,
            equivalence=equivalence,
            fuel=fuel,
            oxidizer=oxidizer,
            diluent=diluent,
            diluent_mol_frac=diluent_mol_frac,
        ):
            # a row with the current information was found
            if test_id is None:
                [test_id] = self.fetch_rows_by_value(
                    mechanism=mechanism,
                    initial_temp=initial_temp,
                    initial_press=initial_press,
                    equivalence=equivalence,
                    fuel=fuel,
                    oxidizer=oxidizer,
                    diluent=diluent,
                    diluent_mol_frac=diluent_mol_frac,
                )["test_id"]
                self.update_existing_row(
                    test_id=test_id,
                    cj_speed=cj_speed,
                    ind_len_west=ind_len_west,
                    ind_len_gav=ind_len_gav,
                    ind_len_ng=ind_len_ng,
                    cell_size_west=cell_size_west,
                    cell_size_gav=cell_size_gav,
                    cell_size_ng=cell_size_ng,
                )

                return test_id
            else:
                # warn the user that the current input was ignored
                warnings.warn(
                    "Cannot overwrite row unless test_id=None is passed", Warning
                )

                return None

        else:
            # no rows with the current information were found
            return self.insert_new_row(
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


class PerturbedResultsTable:
    # todo: sort out test ID (ex reaction table ID)
    # todo: add to DataBase
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
        get_table_with_name(self.cur, self.name)
        if not self.cur.fetchall():
            self._create()

    def columns(self):
        """
        Returns
        -------
        table_info : list
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
                warnings.warn(
                    "Cannot overwrite row unless overwrite_existing=True", Warning
                )

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
        labels = self.columns()
        data = {lbl: [] for lbl in labels}
        for row in info:
            for lbl, d in zip(labels, row):
                data[lbl].append(d)

        return data
