"""
TERRA-FIRE Data Engineering Pipeline
Module: src/pipeline.py
Description: End-to-End ETL Pipeline Orchestrator.
"""

import os
import sys
import logging
from typing import Dict, Any

from src.extract import (
    extract_era5_hourly,
    extract_nasa_firms,
    extract_conafor_fires,
    extract_sentinel2_stac
)
from src.clean import (
    clean_era5_data,
    clean_firms_data,
    clean_conafor_data
)
from src.integrate import (
    integrate_spatiotemporal_matrix,
    export_analysis_ready_parquet
)
from src.validate import run_data_validation_suite

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("terra_fire.pipeline")


def run_full_pipeline(
    processed_parquet_path: str = "data/processed/terra_fire_spatiotemporal_matrix_v1.parquet",
    sample_csv_path: str = "data/sample/terra_fire_matrix_sample.csv"
) -> Dict[str, Any]:
    """
    Executes the complete reproducible ETL pipeline for project TERRA-FIRE:
    1. Extraction & Ingestion
    2. Data Quality Cleaning & Standardization
    3. Biophysical Transformation & Downscaling
    4. Spatiotemporal Fusion & Target Formulation
    5. Apache Parquet Export
    6. Formal Quality Assertion Suite
    """
    logger.info("=" * 70)
    logger.info("STARTING TERRA-FIRE DATA ENGINEERING LAB ETL PIPELINE")
    logger.info("=" * 70)
    
    # ---------------------------------------------------------
    # Phase 1: Extraction
    # ---------------------------------------------------------
    logger.info(">>> Phase 1: Extracting raw planetary & administrative datasets...")
    df_era5_raw, era5_meta = extract_era5_hourly()
    
    df_firms23 = extract_nasa_firms(2023)
    df_firms24 = extract_nasa_firms(2024)
    import pandas as pd
    df_firms_raw = pd.concat([df_firms23, df_firms24], ignore_index=True)
    
    df_conafor_raw = extract_conafor_fires()
    stac_metadata = extract_sentinel2_stac()
    
    # ---------------------------------------------------------
    # Phase 2: Cleaning & Standardization
    # ---------------------------------------------------------
    logger.info(">>> Phase 2: Cleaning and standardizing observations...")
    df_era5_clean, log_era5 = clean_era5_data(df_era5_raw)
    
    bbox_ags = {"min_lat": 21.5, "max_lat": 22.6, "min_lon": -103.1, "max_lon": -101.8}
    df_firms_clean, log_firms = clean_firms_data(df_firms_raw, bbox=bbox_ags)
    
    df_conafor_clean, log_conafor = clean_conafor_data(df_conafor_raw, target_state="Aguascalientes")
    
    # Save clean intermediate samples for testing
    os.makedirs("data/sample", exist_ok=True)
    df_era5_clean.head(1000).to_csv("data/sample/era5_hourly_sample.csv", index=False)
    df_firms_clean.to_csv("data/sample/firms_viirs_sierra_fria_sample.csv", index=False)
    df_conafor_clean.to_csv("data/sample/conafor_fires_aguascalientes_sample.csv", index=False)
    
    # ---------------------------------------------------------
    # Phase 3 & 4: Transformation, Integration & Target Labeling
    # ---------------------------------------------------------
    logger.info(">>> Phase 3 & 4: Spatiotemporal integration and Parquet generation...")
    matrix_df, int_record = integrate_spatiotemporal_matrix(
        df_era5_clean=df_era5_clean,
        df_firms_clean=df_firms_clean,
        df_conafor_clean=df_conafor_clean,
        matching_radius_deg=0.08
    )
    
    # ---------------------------------------------------------
    # Phase 5: Analytical Export
    # ---------------------------------------------------------
    logger.info(">>> Phase 5: Exporting Analysis-Ready Parquet...")
    export_metrics = export_analysis_ready_parquet(
        df=matrix_df,
        output_parquet_path=processed_parquet_path,
        output_sample_csv_path=sample_csv_path
    )
    
    # ---------------------------------------------------------
    # Phase 6: Formal Validation Suite
    # ---------------------------------------------------------
    logger.info(">>> Phase 6: Executing data quality assertion checks...")
    all_passed, checklist = run_data_validation_suite(processed_parquet_path)
    
    readiness_decision = "READY FOR ANALYSIS" if all_passed else "CONDITIONALLY READY"
    
    logger.info("=" * 70)
    logger.info(f"PIPELINE COMPLETED SUCCESSFULLY. DECISION: [{readiness_decision}]")
    logger.info("=" * 70)
    
    return {
        "status": "SUCCESS",
        "readiness_decision": readiness_decision,
        "export_metrics": export_metrics,
        "integration_record": int_record,
        "validation_passed": all_passed,
        "validation_checklist": checklist
    }


if __name__ == "__main__":
    result = run_full_pipeline()
    print("\nSummary Result:")
    for k, v in result.items():
        if k != "validation_checklist":
            print(f"  {k}: {v}")
