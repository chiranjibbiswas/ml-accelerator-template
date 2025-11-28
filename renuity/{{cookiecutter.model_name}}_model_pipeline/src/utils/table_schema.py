""""
This module defines the schema for various lookup tables used in the application using SQLAlchemy.
"""

from sqlalchemy import Table, Column, Integer, MetaData, String, DECIMAL

# ---------------------------
# Define table (Core)
# ---------------------------

def get_tables(schema:str):
    metadata = MetaData(schema=schema)

    btd_raw_to_prioritization_score = Table(
        "btd_raw_to_prioritization_score",
        metadata,
        Column("lead_prioritization_score", Integer, nullable=False),
        Column("min_btd_raw_score", DECIMAL(16,15)),
        Column("max_btd_raw_score", DECIMAL(16,15)),
        Column("btd_predicted_rate", DECIMAL(5,1)),
        Column("model_version", Integer, nullable=False)
    )

    btd_subsource_creditrating_prioritization = Table(
        "btd_subsource_creditrating_prioritization",
        metadata,
        Column("sub_source", String(256), nullable=False),
        Column("experian_credit_rating", String(256), nullable=False),
        Column("btd_predicted_rate", DECIMAL(5,1), nullable=False),
        Column("lead_prioritization_score", Integer, nullable=False),
        Column("model_version", Integer, nullable=False)
    )
    return btd_raw_to_prioritization_score, btd_subsource_creditrating_prioritization
