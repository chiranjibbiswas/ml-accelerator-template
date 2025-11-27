import pandas as pd
import numpy as np
from typing import Tuple
from datetime import date
import os

from sqlalchemy import text
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl.chart import LineChart, Reference

from utils import (
    KEY_VAULT_NAME,
    TARGET_VARIABLE,

)
from utils import get_logger, get_engine, features, Config
from utils.parse_json_to_csv import list_blobs_within_date_range, process_blobs_to_csv
logger = get_logger(__name__)


# ------------------------
# Data Handling
# ------------------------
def create_validation_dataset(raw_container_client, root_folder: str, blob_path: str, start_date: date, end_date: date, local_dir: str) -> pd.DataFrame:
    """Create and upload validation dataset (last 2 months)."""
    try:
        validation_file_path = f"{local_dir}/raw_validation_dataset.csv"
        validation_blob_path = f"{blob_path}/raw_validation_dataset.csv"
        blob_client = raw_container_client.get_blob_client(validation_blob_path)

        blob_info_list = list_blobs_within_date_range(raw_container_client, root_folder, start_date, end_date)
        raw_validation_data = process_blobs_to_csv(raw_container_client, blob_info_list)

        raw_validation_data.columns = [col.lower() for col in raw_validation_data.columns]
        raw_validation_data.to_csv(validation_file_path, index=False)

        with open(validation_file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)

        return raw_validation_data
    except Exception as e:
        logger.error(f"Error creating validation dataset: {e}")
        raise


def create_dataset(raw_container_client: str, local_dir: str, raw_validation_blob_path:str, args) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create validation datasets merged with labels."""
    try:
        val_cols = ['leadid'] + features + [TARGET_VARIABLE]
        # raw csv data path
        root_folder = "raw_data"
        
        # processed raw csv data path
        synapse_config = Config(os.getenv("KEY_VAULT_NAME"))
        engine = get_engine(synapse_server=synapse_config.synapse_server, #NOSONAR
                            synapse_db=synapse_config.synapse_db,
                            synapse_user=synapse_config.synapse_user,
                            synapse_password=synapse_config.synapse_password)

        raw_validation_data = create_validation_dataset(raw_container_client, root_folder, raw_validation_blob_path, args.validation_start_date, args.validation_end_date, local_dir)
        raw_validation_data = raw_validation_data[(raw_validation_data.lead_issued==1) & (raw_validation_data.division=='MAC')]

        # with engine.connect() as conn: #NOSONAR
        #     query = text(f"""
        #         SELECT leadid, btd_flag
        #         FROM {synapse_config.synapse_db}.dbo.lead_activities 
        #         WHERE entrydate BETWEEN '{start_date}' AND '{cur_date}'
        #     """)
        #     leads = pd.read_sql(query, conn)

        # training_data = raw_training_data[cols].merge(leads[["leadid", "btd_flag"]], on="leadid", how="inner").drop("leadid", axis=1) #NOSONAR
        # validation_data = raw_validation_data[cols].merge(leads[["leadid", "btd_flag"]], on="leadid", how="inner")

        validation_data = raw_validation_data[val_cols]

        logger.info(f"Validation dataset shape: {validation_data.shape}")
        # return only features + btd_flag
        return validation_data
    except Exception as e:
        logger.error(f"Error creating datasets: {e}")
        raise


def upload_to_azure_blob(upload_file, blob_path, container_client):
    try:
        file_name =upload_file.split('/')[-1]
        blob_client = container_client.get_blob_client(blob_path)
        with open(upload_file, "rb") as data:  # open in binary mode
            blob_client.upload_blob(data, overwrite=True)
        logger.info(f"File:{file_name} uploaded to blob:{blob_path}")
    except Exception as e:
        logger.error(f"Error uploading a file:{e}")
        raise e


def btd_score_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Compute bucketed BTD scores."""
    try:
        # --- Prep CTE ---
        prep = df[["btd_1_score", "sold", "btd_flag"]].copy()

        # --- Bucketed CTE ---
        # Equivalent of NTILE(100) OVER (ORDER BY btd_1_score DESC)
        prep = prep.sort_values("btd_1_score", ascending=False).reset_index(drop=True)
        prep["bucket_num"] = pd.qcut(
            prep.index + 1, 20, labels=False
        ) + 1  # NTILE is 1-based

        bucketed = prep[["bucket_num", "btd_1_score", "sold", "btd_flag"]]

        # --- Buckets CTE ---
        buckets = (
            bucketed.groupby("bucket_num", as_index=False)
            .agg(
                min_probability=("btd_1_score", "min"),
                max_probability=("btd_1_score", "max"),
                sold_count=("sold", lambda x: (x == 1).sum()),
                btd_sold_count=("btd_flag", lambda x: (x == 1).sum()),
            )
        )

        # Compute BTDRate
        buckets["BTDRate"] = (
            (buckets["btd_sold_count"] / buckets["sold_count"].replace(0, np.nan)) * 100
        ).round(1)

        # --- Final Select ---
        buckets = buckets.sort_values("bucket_num")

        # lead_prioritization_score
        buckets["lead_prioritization_score"] = buckets["bucket_num"]

        # min_btd_raw_score
        buckets["min_btd_raw_score"] = np.where(
            buckets["bucket_num"] == 20, 0.0, buckets["min_probability"]
        )

        # max_btd_raw_score (use lagged min_probability)
        buckets["max_btd_raw_score"] = buckets["min_probability"].shift(1)
        buckets.loc[buckets["bucket_num"] == 1, "max_btd_raw_score"] = 0.99999999999999999

        # btd_predicted_rate
        buckets["btd_predicted_rate"] = buckets["BTDRate"]

        # Final output
        result = buckets[
            ["lead_prioritization_score", "min_btd_raw_score", "max_btd_raw_score", "btd_predicted_rate"]
        ].sort_values("lead_prioritization_score", ascending=False).reset_index(drop=True)

        return result
    except Exception as e:
        logger.error(f"Error inserting into btd_score_buckets: {e}")
        raise


def generate_report(metrics, final_df:pd.DataFrame, report_name:str):
    try:
        # Create Workbook
        wb = Workbook()
        ws = wb.active
        ws.title = "BTD Analysis"
        metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['values']).rename_axis('metrics')
        metrics_df.reset_index(drop=False, inplace=True)
        # Insert DataFrame
        for r in dataframe_to_rows(final_df, index=False, header=True):
            ws.append(r)

        # Add as Excel table
        tab = Table(displayName="BTDTable", ref=f"A1:D{len(final_df)+1}")
        style = TableStyleInfo(name="TableStyleMedium9", showFirstColumn=False,
                            showLastColumn=False, showRowStripes=True, showColumnStripes=False)
        tab.tableStyleInfo = style
        ws.add_table(tab)

        # Write second table at F1 (side by side)---
        start_col = 7  # column F
        start_row = 1

        for ridx, r in enumerate(dataframe_to_rows(metrics_df, index=False, header=True), start=start_row):
            for cidx, value in enumerate(r, start=start_col):
                ws.cell(row=ridx, column=cidx, value=value)

        # Add as table
        tab2 = Table(displayName="ModelMetrics", ref=f"G1:H{len(metrics_df)+1}")
        style2 = TableStyleInfo(name="TableStyleMedium2", showRowStripes=True)
        tab2.tableStyleInfo = style2
        ws.add_table(tab2)

        # Create Line Chart
        chart = LineChart()
        chart.style = 13
        chart.y_axis.title = "BTD predicted rate (%)"
        chart.x_axis.title = "Lead prioritization score"
        chart.width = 20     # make it wider
        chart.height = 10     # make it taller

        # Data range (BTDRateSold column)
        data = Reference(ws, min_col=4, min_row=1, max_row=len(final_df)+1)  # Column D
        # Categories (bucket_num column)
        cats = Reference(ws, min_col=1, min_row=2, max_row=len(final_df)+1)  # Column A, skip header

        chart.add_data(data, titles_from_data=True)
        chart.set_categories(cats)

        ws.add_chart(chart, "J7")
        
        # Save Workbook
        wb.save(report_name)
        logger.info(f"Excel report generated: {report_name}")
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise e