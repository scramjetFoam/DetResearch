import dataclasses
import sqlite3
from enum import Enum
from typing import Optional

from cantera import Species
from retry import retry


class DatabaseError(Exception):
    pass


class TableName(Enum):
    Conditions: str = "conditions"
    Reactions: str = "reactions"
    Species: str = "species"


class SimulationType(Enum):
    Znd: str = "znd"
    Cv: str = "cv"


class SqliteDataBase:
    def __init__(self, path: str):
        self._path = path
        self.con = sqlite3.connect(path)
        self.con.row_factory = sqlite3.Row

    def __del__(self):
        self.con.commit()
        self.con.close()


class SqliteTable:
    def __init__(self, db: SqliteDataBase, table_name: str):
        self.db = db
        self.cur = self.db.con.cursor()
        self.name = table_name

    def table_exists(self) -> bool:
        self.cur.execute("select name from sqlite_master where type='table' and name=:name", {"name": self.name})
        return self.cur.fetchone() is not None


@dataclasses.dataclass
class Conditions:
    sim_type: str
    mech: str
    initial_temp: float
    initial_press: float
    fuel: str
    oxidizer: str
    equivalence: float
    diluent: Optional[str]
    dil_mf: float


class ConditionTable(SqliteTable):
    def __init__(self, db: SqliteDataBase):
        super().__init__(db=db, table_name=TableName.Conditions.value)
        if not self.table_exists():
            self.__create()

    @retry(sqlite3.OperationalError, tries=5)
    def __create(self):
        self.cur.execute(
            f"""
            CREATE TABLE {self.name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                sim_type TEXT NOT NULL,
                mech TEXT NOT NULL,
                initial_temp REAL NOT NULL,
                initial_press REAL NOT NULL,
                fuel TEXT NOT NULL,
                oxidizer TEXT NOT NULL,
                equivalence REAL NOT NULL,
                diluent TEXT,
                dil_mf REAL NOT NULL
            );
            """
        )

    @retry(sqlite3.OperationalError, tries=5)
    def insert(self, test_conditions: Conditions) -> int:
        """
        Stores a row of test data in the current table.
        """

        self.cur.execute(
            f"""
            INSERT INTO {self.name} VALUES (
                Null,
                :sim_type,
                :mech,
                :initial_temp,
                :initial_press,
                :fuel,
                :oxidizer,
                :equivalence,
                :diluent,
                :dil_mf
            );
            """,
            dataclasses.asdict(test_conditions),
        )
        self.cur.connection.commit()
        return self.cur.lastrowid


@dataclasses.dataclass
class ReactionData:
    condition_id: int
    time: float
    reaction: str
    fwd_rate_constant: float
    fwd_rate_of_progress: float


class ReactionTable(SqliteTable):
    def __init__(self, db: SqliteDataBase):
        super().__init__(db=db, table_name=TableName.Reactions.value)
        if not self.table_exists():
            self.__create()

    @retry(sqlite3.OperationalError, tries=5)
    def __create(self):
        self.cur.execute(
            f"""
            CREATE TABLE {self.name} (
                condition_id INTEGER NOT NULL,
                time REAL NOT NULL,
                reaction TEXT NOT NULL,
                fwd_rate_constant REAL NOT NULL,
                fwd_rate_of_progress REAL NOT NULL,
                FOREIGN KEY(condition_id) REFERENCES {TableName.Conditions.value}(id) ON UPDATE CASCADE
            );
            """
        )

    @retry(sqlite3.OperationalError, tries=5)
    def insert(self, data: ReactionData, commit: bool = True):
        self.cur.execute(
            f"""
            INSERT INTO {self.name} VALUES (
                :condition_id,
                :time,
                :reaction,
                :fwd_rate_constant,
                :fwd_rate_of_progress
            );
            """,
            {
                "condition_id": data.condition_id,
                "time": data.time,
                "reaction": data.reaction,
                "fwd_rate_constant": data.fwd_rate_constant,
                "fwd_rate_of_progress": data.fwd_rate_of_progress,
            },
        )
        if commit:
            self.cur.connection.commit()


@dataclasses.dataclass
class SpeciesData:
    condition_id: int
    time: float
    species: Species
    mole_frac: float
    concentration: float
    creation_rate: float


class SpeciesTable(SqliteTable):
    def __init__(self, db: SqliteDataBase):
        super().__init__(db=db, table_name=TableName.Species.value)
        if not self.table_exists():
            self.__create()

    @retry(sqlite3.OperationalError, tries=5)
    def __create(self):
        self.cur.execute(
            f"""
            CREATE TABLE {self.name} (
                condition_id INTEGER NOT NULL,
                time REAL NOT NULL,
                species TEXT NOT NULL,
                mole_frac REAL NOT NULL,
                concentration REAL NOT NULL,
                creation_rate REAL NOT NULL,
                FOREIGN KEY(condition_id) REFERENCES {TableName.Conditions.value}(id) ON UPDATE CASCADE
            );
            """
        )

    @retry(sqlite3.OperationalError, tries=5)
    def insert(self, data: SpeciesData, commit: bool = True):
        self.cur.execute(
            f"""
            INSERT INTO {self.name} VALUES (
                :condition_id,
                :time,
                :species,
                :mole_frac,
                :concentration,
                :creation_rate
            );
            """,
            {
                "condition_id": data.condition_id,
                "time": data.time,
                "species": data.species.name,
                "mole_frac": data.mole_frac,
                "concentration": data.concentration,
                "creation_rate": data.creation_rate,
            }
        )
        if commit:
            self.cur.connection.commit()


class SimulationDatabase:
    db: SqliteDataBase
    conditions: ConditionTable
    conditions_id: int
    reactions: ReactionTable
    species: SpeciesTable

    def __init__(self, db: SqliteDataBase, conditions: Conditions):
        self.db = db
        self.conditions = ConditionTable(db)
        self.conditions_id = self.conditions.insert(conditions)
        self.reactions = ReactionTable(db)
        self.species = SpeciesTable(db)
