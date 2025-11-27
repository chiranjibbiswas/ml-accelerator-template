""" 
Utilities to parse JSON files from Azure Blob Storage into CSV format.
Handles both month-level and day-level JSON files, combining them into a single DataFrame.

"""

from datetime import datetime, date
import json
import pandas as pd
from typing import List, Tuple

from .setup_logger import get_logger
logger = get_logger(__name__)


def month_range(start_date:date, end_date:date):
    """Yield (year, month) tuples between two dates inclusive."""
    current = date(start_date.year, start_date.month, 1)
    while current <= end_date:
        yield current.year, current.month
        # next month
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)


def list_blobs_within_date_range(container_client, root_folder:str, start_date:date, end_date:date) -> List[Tuple]:
    """List blob files between start and end date, handling both month- and day-level files."""
    blob_list = []
    for year, month in month_range(start_date, end_date):
        prefix = f"{root_folder}/{year}/{month:02d}/"
        logger.info(f"Scanning prefix: {prefix}")
        for blob in container_client.list_blobs(name_starts_with=prefix):
            if not blob.name.endswith(".json"):
                continue

            parts = blob.name.split("/")
            try:
                # day-level file: root/year/month/day/file.json
                if len(parts) >= 5:
                    year, month, day = map(int, parts[1:4])
                    blob_date = datetime(year, month, day).date()
                    if start_date <= blob_date <= end_date:
                        blob_list.append((blob.name, "day"))
                # month-level file: root/year/month/file.json
                elif len(parts) >= 4:
                    year, month = map(int, parts[1:3])
                    blob_date = date(year, month, 1)
                    # include whole month if any overlap
                    start_month = date(start_date.year, start_date.month, 1)
                    end_month = date(end_date.year, end_date.month, 1)
                    if start_month <= blob_date <= end_month:
                        blob_list.append((blob.name, "month"))

            except Exception:
                continue

    logger.info(f"Found {len(blob_list)} files in range")
    return blob_list


def process_blobs_to_csv(container_client, blob_info_list:List[Tuple]) -> pd.DataFrame:
    """Download blobs and parse into DataFrame, supporting both month- and day-level JSON."""
    rows = []
    for blob_name, granularity in blob_info_list:
        blob_client = container_client.get_blob_client(blob_name)
        data = blob_client.download_blob().readall().decode("utf-8")

        try:
            json_data = json.loads(data)
            if granularity == "month":
                # month-level file contains list of rows
                if isinstance(json_data, list):
                    rows.extend(json_data)
                else:
                    logger.warning(f"Month file {blob_name} not a list, skipping.")
            elif granularity == "day":
                # day-level file contains single row
                if isinstance(json_data, dict):
                    rows.append(json_data)
                else:
                    logger.warning(f"Day file {blob_name} not dict, skipping.")
        except Exception as e:
            logger.error(f"Error parsing {blob_name}: {e}")

    if rows:
        df = pd.DataFrame(rows)
        return df
    else:
        logger.error("⚠️ No data found in given range.")
        raise RuntimeError("No data found in given range")
