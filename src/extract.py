"""
TERRA-FIRE Data Engineering Pipeline
Module: src/extract.py
Description: Data Extraction and Ingestion Layer for Planetary & Historical Datasets.
"""

import os
import json
import logging
from typing import Tuple, Dict, Any, Optional
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("terra_fire.extract")

# Default paths to external storage or local raw directory
DEFAULT_DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "TERRAFIRE-db"))
LOCAL_RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))


def resolve_dataset_path(filename: str) -> str:
    """Resolves dataset location between TERRAFIRE-db and local data/raw directory."""
    db_path = os.path.join(DEFAULT_DB_DIR, filename)
    local_path = os.path.join(LOCAL_RAW_DIR, filename)
    if os.path.exists(db_path):
        return db_path
    elif os.path.exists(local_path):
        return local_path
    else:
        raise FileNotFoundError(f"Dataset '{filename}' not found in '{DEFAULT_DB_DIR}' or '{LOCAL_RAW_DIR}'.")


def extract_era5_hourly(filepath: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Extracts ERA5-Land hourly surface reanalysis data.
    Handles the 3-line metadata header gracefully, extracting geospatial coordinates and timezone.
    """
    if filepath is None:
        filepath = resolve_dataset_path("ERA5_Land_Hourly_SierraFria_2023_2024.csv")
    
    logger.info(f"Extracting ERA5-Land from: {filepath}")
    
    # Extract metadata header from first 2 lines
    meta_df = pd.read_csv(filepath, nrows=1)
    metadata = {
        "latitude": float(meta_df["latitude"].iloc[0]),
        "longitude": float(meta_df["longitude"].iloc[0]),
        "elevation_m": float(meta_df["elevation"].iloc[0]),
        "utc_offset_seconds": int(meta_df["utc_offset_seconds"].iloc[0]),
        "timezone": str(meta_df["timezone"].iloc[0]),
        "timezone_abbreviation": str(meta_df["timezone_abbreviation"].iloc[0])
    }
    
    # Read observations starting at line 4 (skiprows=3)
    df = pd.read_csv(filepath, skiprows=3)
    logger.info(f"ERA5-Land extracted successfully. Records: {len(df)}, Attributes: {df.shape[1]}")
    return df, metadata


def extract_nasa_firms(year: int = 2023, filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Extracts NASA FIRMS VIIRS 375m active fire detections for Mexico.
    """
    filename = f"NASA_FIRMS_VIIRS_SNPP_{year}_Mexico.csv"
    if filepath is None:
        filepath = resolve_dataset_path(filename)
        
    logger.info(f"Extracting NASA FIRMS VIIRS ({year}) from: {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"NASA FIRMS ({year}) extracted. Total detections: {len(df)}")
    return df


def extract_conafor_fires(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Extracts CONAFOR Historical Forest Fire Registry (2012–2025).
    """
    if filepath is None:
        filepath = resolve_dataset_path("CONAFOR_Incendios_2012_2025.xlsx")
        
    logger.info(f"Extracting CONAFOR forest fires registry from: {filepath}")
    df = pd.read_excel(filepath)
    logger.info(f"CONAFOR registry extracted. Total records: {len(df)}, Attributes: {df.shape[1]}")
    return df


def extract_sentinel2_stac(filepath: Optional[str] = None) -> Dict[str, Any]:
    """
    Extracts Copernicus Sentinel-2 MSI STAC metadata JSON item.
    """
    if filepath is None:
        filepath = resolve_dataset_path("Sentinel2_L2A_STAC_Item_SierraFria_20240411.json")
        
    logger.info(f"Extracting Sentinel-2 STAC metadata from: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        stac_data = json.load(f)
    return stac_data


def profile_dataframe(name: str, df: pd.DataFrame) -> Dict[str, Any]:
    """Generates a structured profiling diagnostic dictionary for a dataset."""
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0].to_dict()
    
    profile = {
        "dataset_name": name,
        "total_records": len(df),
        "total_variables": df.shape[1],
        "duplicate_rows": int(df.duplicated().sum()),
        "columns_with_nulls": null_cols,
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
    }
    return profile
