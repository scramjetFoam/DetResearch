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
    BulkProperties: str = "bulk_properties"


class SimulationType(Enum):
    Znd: str = "znd"
    Cv: str = "cv"


class SqliteDataBase:
    def __init__(self, path: str, timeout: float = 600):
        self.path = path
        self.timeout = timeout
        self.con = self.connect()
        self.con.row_factory = sqlite3.Row

    def __del__(self):
        try:
            self.con.commit()
        except sqlite3.ProgrammingError:
            self.con = self.connect()
        self.con.close()

    def connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=self.timeout)


class SqliteTable:
    def __init__(self, db: SqliteDataBase, table_name: str, clear_existing_data: bool = False):
        self.db = db
        self.cur = self.db.con.cursor()
        self.name = table_name
        if self.table_exists() and clear_existing_data:
            self.__clear()

    def table_exists(self) -> bool:
        self.cur.execute("select name from sqlite_master where type='table' and name=:name", {"name": self.name})
        return self.cur.fetchone() is not None

    @retry(sqlite3.OperationalError, tries=10, backoff=2, max_delay=2)
    def __clear(self):
        self.cur.execute(
            f"""
            DROP TABLE IF EXISTS {self.name};
            """
        )


@dataclasses.dataclass
class Conditions:
    sim_type: str
    mech: str
    match: Optional[str]
    dil_condition: str
    initial_temp: float
    initial_press: float
    fuel: str
    oxidizer: str
    equivalence: float
    diluent: Optional[str]
    dil_mf: float


class ConditionTable(SqliteTable):
    def __init__(self, db: SqliteDataBase, clear_existing_data: bool = False):
        super().__init__(db=db, table_name=TableName.Conditions.value, clear_existing_data=clear_existing_data)
        if not self.table_exists():
            self.__create()

    @retry(sqlite3.OperationalError, tries=10, backoff=2, max_delay=2)
    def __create(self):
        self.cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                start DATETIME NOT NULL,
                end DATETIME,
                sim_type TEXT NOT NULL,
                mech TEXT NOT NULL,
                match TEXT,
                dil_condition TXT NOT NULL,
                initial_temp REAL NOT NULL,
                initial_press REAL NOT NULL,
                fuel TEXT NOT NULL,
                oxidizer TEXT NOT NULL,
                equivalence REAL NOT NULL,
                diluent TEXT,
                dil_mf REAL NOT NULL,
                temp_vn REAL,
                temp_west REAL,
                t_ind REAL,
                u_znd REAL,
                u_cj REAL,
                cell_size REAL
            );
            """
        )

    @retry(sqlite3.OperationalError, tries=10, backoff=2, max_delay=2)
    def insert(self, test_conditions: Conditions) -> int:
        """
        Stores a row of test data in the current table.
        """

        self.cur.execute(
            f"""
            INSERT INTO {self.name} VALUES (
                Null,
                CURRENT_TIMESTAMP,
                Null,
                :sim_type,
                :mech,
                :match,
                :dil_condition,
                :initial_temp,
                :initial_press,
                :fuel,
                :oxidizer,
                :equivalence,
                :diluent,
                :dil_mf,
                Null,
                Null,
                Null,
                Null,
                Null,
                Null
            );
            """,
            dataclasses.asdict(test_conditions),
        )
        self.cur.connection.commit()
        return self.cur.lastrowid


@dataclasses.dataclass
class BulkPropertiesData:
    condition_id: int
    run_no: int
    time: float
    temperature: float
    pressure: float
    velocity: Optional[float] = None


class BulkPropertiesTable(SqliteTable):
    def __init__(self, db: SqliteDataBase, clear_existing_data: bool = False):
        super().__init__(db=db, table_name=TableName.BulkProperties.value, clear_existing_data=clear_existing_data)
        if not self.table_exists():
            self.__create()

    @retry(sqlite3.OperationalError, tries=10, backoff=2, max_delay=2)
    def __create(self):
        self.cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.name} (
                condition_id INTEGER NOT NULL,
                run_no INTEGER NOT NULL,
                time REAL NOT NULL,
                temperature REAL NOT NULL,
                pressure REAL NOT NULL,
                velocity REAL,
                PRIMARY KEY (condition_id, run_no, time),
                FOREIGN KEY(condition_id) REFERENCES {TableName.Conditions.value}(id)
                ON UPDATE CASCADE ON DELETE CASCADE
            );
            """
        )

    @retry(sqlite3.OperationalError, tries=10, backoff=2, max_delay=2)
    def insert_or_update(self, data: BulkPropertiesData, commit: bool = True):
        self.cur.execute(
            f"""
            INSERT INTO {self.name} VALUES (
                :condition_id,
                :run_no,
                :time,
                :temperature,
                :pressure,
                :velocity
            )
            ON CONFLICT(condition_id, run_no, time) DO UPDATE SET
                temperature=excluded.temperature,
                pressure=excluded.pressure,
                velocity=excluded.velocity;
            """,
            {
                "condition_id": data.condition_id,
                "run_no": data.run_no,
                "time": data.time,
                "temperature": data.temperature,
                "pressure": data.pressure,
                "velocity": data.velocity,
            },
        )
        if commit:
            self.cur.connection.commit()


@dataclasses.dataclass
class ReactionData:
    condition_id: int
    run_no: int
    time: float
    reaction: str
    fwd_rate_constant: float
    fwd_rate_of_progress: float


class ReactionTable(SqliteTable):
    def __init__(self, db: SqliteDataBase, clear_existing_data: bool = False):
        super().__init__(db=db, table_name=TableName.Reactions.value, clear_existing_data=clear_existing_data)
        if not self.table_exists():
            self.__create()

    @retry(sqlite3.OperationalError, tries=10, backoff=2, max_delay=2)
    def __create(self):
        self.cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.name} (
                condition_id INTEGER NOT NULL,
                run_no INTEGER NOT NULL,
                time REAL NOT NULL,
                reaction TEXT NOT NULL,
                fwd_rate_constant REAL NOT NULL,
                fwd_rate_of_progress REAL NOT NULL,
                PRIMARY KEY (condition_id, run_no, time, reaction),
                FOREIGN KEY(condition_id) REFERENCES {TableName.Conditions.value}(id)
                ON UPDATE CASCADE ON DELETE CASCADE
            );
            """
        )

    @retry(sqlite3.OperationalError, tries=10, backoff=2, max_delay=2)
    def insert_or_update(self, data: ReactionData, commit: bool = True):
        self.cur.execute(
            f"""
            INSERT INTO {self.name} VALUES (
                :condition_id,
                :run_no,
                :time,
                :reaction,
                :fwd_rate_constant,
                :fwd_rate_of_progress
            )
            ON CONFLICT(condition_id, run_no, time, reaction) DO UPDATE SET
                fwd_rate_constant=excluded.fwd_rate_constant,
                fwd_rate_of_progress=excluded.fwd_rate_of_progress;
            """,
            {
                "condition_id": data.condition_id,
                "run_no": data.run_no,
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
    run_no: int
    time: float
    species: Species
    mole_frac: float
    concentration: float
    creation_rate: float


class SpeciesTable(SqliteTable):
    def __init__(self, db: SqliteDataBase, clear_existing_data: bool = False):
        super().__init__(db=db, table_name=TableName.Species.value, clear_existing_data=clear_existing_data)
        if not self.table_exists():
            self.__create()

    @retry(sqlite3.OperationalError, tries=10, backoff=2, max_delay=2)
    def __create(self):
        self.cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.name} (
                condition_id INTEGER NOT NULL,
                run_no INTEGER NOT NULL,
                time REAL NOT NULL,
                species TEXT NOT NULL,
                mole_frac REAL NOT NULL,
                concentration REAL NOT NULL,
                creation_rate REAL NOT NULL,
                PRIMARY KEY (condition_id, run_no, time, species),
                FOREIGN KEY(condition_id) REFERENCES {TableName.Conditions.value}(id)
                ON UPDATE CASCADE ON DELETE CASCADE
            );
            """
        )

    @retry(sqlite3.OperationalError, tries=10, backoff=2, max_delay=2)
    def insert_or_update(self, data: SpeciesData, commit: bool = True):
        self.cur.execute(
            f"""
            INSERT INTO {self.name} VALUES (
                :condition_id,
                :run_no,
                :time,
                :species,
                :mole_frac,
                :concentration,
                :creation_rate
            )
            ON CONFLICT(condition_id, run_no, time, species) DO UPDATE SET
                mole_frac=excluded.mole_frac,
                concentration=excluded.concentration,
                creation_rate=excluded.creation_rate;
            """,
            {
                "condition_id": data.condition_id,
                "run_no": data.run_no,
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
        self.bulk_properties = BulkPropertiesTable(self.db)

    def reconnect(self):
        self.db = SqliteDataBase(path=self.db.path)
        self.conditions = ConditionTable(self.db)
        self.reactions = ReactionTable(self.db)
        self.species = SpeciesTable(self.db)
        self.bulk_properties = BulkPropertiesTable(self.db)


def clear_simulation_database(path: str):
    """
    Convenience function to make resets easier between simulation re-runs
    """
    db = SqliteDataBase(path=path)
    _ = ConditionTable(db, clear_existing_data=True)
    _ = ReactionTable(db, clear_existing_data=True)
    _ = SpeciesTable(db, clear_existing_data=True)
    _ = BulkPropertiesTable(db, clear_existing_data=True)
