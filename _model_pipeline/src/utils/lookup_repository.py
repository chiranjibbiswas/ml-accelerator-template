""""Repository for managing lookup tables in a Synapse SQL database using SQLAlchemy and pandas."""

import os
import struct
import urllib
import pandas as pd
from sqlalchemy import create_engine, Table, text
from sqlalchemy.engine import Engine

from azure.identity import ManagedIdentityCredential

# ---------------------------
# Token acquisition
# ---------------------------
def get_engine(synapse_server, synapse_db, synapse_user, synapse_password):
    connection_string = (
        f"Driver={{ODBC Driver 18 for SQL Server}};"
        f"Server=tcp:{synapse_server},1433;"
        f"Database={synapse_db};"
        f"Uid={synapse_user};"
        f"Pwd={synapse_password};"
        "Encrypt=yes;"
        "TrustServerCertificate=no;"
        "Connection Timeout=30;"
    )
    odbc_str = urllib.parse.quote_plus(connection_string)
    engine = create_engine(
        f"mssql+pyodbc:///?odbc_connect={odbc_str}",
        fast_executemany=True,
    )
    return engine


class LookupTableRepository:
    """Generic repository for inserting and loading lookup tables."""

    def __init__(self, engine: Engine, table: Table):
        self.engine = engine
        self.table = table

    def insert_dataframe(self, df: pd.DataFrame):
        """Insert a pandas DataFrame into the table."""
        with self.engine.begin() as conn:
            records = df.to_dict(orient="records")
            conn.execute(self.table.insert(), records)
        print(f"Inserted {len(df)} rows into {self.table.name}.")

    def load_to_dataframe(self, filters: dict = None) -> pd.DataFrame:
        """
        Load all records (or filtered) from the table into a DataFrame.
        filters: dict of column_name -> value to filter rows
        """
        stmt = self.table.select()
        if filters:
            for col, val in filters.items():
                stmt = stmt.where(self.table.c[col] == val)

        with self.engine.connect() as conn:
            df = pd.read_sql(stmt, conn)
        print(f"Loaded {len(df)} rows from {self.table.name}.")
        return df
    
    def query_to_dataframe(self, sql: str) -> pd.DataFrame:
        """Execute a raw SQL query and load results into a DataFrame."""
        with self.engine.connect() as conn:
            df = pd.read_sql(text(sql), conn)
        return df