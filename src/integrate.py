"""
TERRA-FIRE Data Engineering Pipeline
Module: src/integrate.py
Description: Spatiotemporal Fusion, Ground-Truth Target Labeling, and Parquet Feature Store Generation.
"""

import os
import logging
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import pandas as pd

from src.transform import (
    calculate_lapse_rate_temperature,
    transform_climate_daily_aggregates
)

logger = logging.getLogger("terra_fire.integrate")


def build_regional_monitoring_grid() -> pd.DataFrame:
    """
    Constructs the canonical 25-cell spatial benchmark grid for the Aguascalientes & Sierra Fría
    regional wildfire monitoring network (5x5 grid lattice from 21.65°N to 22.35°N, -102.75°W to -102.05°W).
    Each voxel represents a calibrated topographic & biophysical zone across varying vegetation strata.
    """
    grid_cells = [
        # Lat 21.65 (South corridor: Sierra del Laurel / Calvillo / South Valley)
        {"voxel_id": "AGS_V01", "lat": 21.65, "lon": -102.75, "elevation_m": 2420.0, "slope_deg": 22.0, "aspect_rad": 1.45, "dist_roads_m": 850.0, "dist_agri_m": 1900.0, "fuel_type": "Bosque de Encino"},
        {"voxel_id": "AGS_V02", "lat": 21.65, "lon": -102.57, "elevation_m": 2150.0, "slope_deg": 14.5, "aspect_rad": 3.12, "dist_roads_m": 350.0, "dist_agri_m": 1200.0, "fuel_type": "Matorral Submontano"},
        {"voxel_id": "AGS_V03", "lat": 21.65, "lon": -102.40, "elevation_m": 1920.0, "slope_deg": 8.0,  "aspect_rad": 0.85, "dist_roads_m": 180.0, "dist_agri_m": 450.0,  "fuel_type": "Pastizal Inducido"},
        {"voxel_id": "AGS_V04", "lat": 21.65, "lon": -102.22, "elevation_m": 1860.0, "slope_deg": 5.5,  "aspect_rad": 4.20, "dist_roads_m": 120.0, "dist_agri_m": 310.0,  "fuel_type": "Pastizal Inducido"},
        {"voxel_id": "AGS_V05", "lat": 21.65, "lon": -102.05, "elevation_m": 2040.0, "slope_deg": 11.0, "aspect_rad": 0.35, "dist_roads_m": 410.0, "dist_agri_m": 950.0,  "fuel_type": "Matorral Xerofilo"},

        # Lat 21.82 (Central-South corridor: Calvillo Valley / Jesus Maria / Aguascalientes East)
        {"voxel_id": "AGS_V06", "lat": 21.82, "lon": -102.75, "elevation_m": 2580.0, "slope_deg": 26.5, "aspect_rad": 2.10, "dist_roads_m": 1100.0, "dist_agri_m": 2400.0, "fuel_type": "Pino-Encino"},
        {"voxel_id": "AGS_V07", "lat": 21.82, "lon": -102.57, "elevation_m": 2240.0, "slope_deg": 16.0, "aspect_rad": 3.85, "dist_roads_m": 480.0, "dist_agri_m": 1500.0, "fuel_type": "Bosque de Encino"},
        {"voxel_id": "AGS_V08", "lat": 21.82, "lon": -102.40, "elevation_m": 1980.0, "slope_deg": 9.5,  "aspect_rad": 5.15, "dist_roads_m": 210.0, "dist_agri_m": 600.0,  "fuel_type": "Matorral Submontano"},
        {"voxel_id": "AGS_V09", "lat": 21.82, "lon": -102.22, "elevation_m": 1880.0, "slope_deg": 6.0,  "aspect_rad": 1.95, "dist_roads_m": 90.0,  "dist_agri_m": 280.0,  "fuel_type": "Pastizal Inducido"},
        {"voxel_id": "AGS_V10", "lat": 21.82, "lon": -102.05, "elevation_m": 2120.0, "slope_deg": 13.5, "aspect_rad": 2.75, "dist_roads_m": 520.0, "dist_agri_m": 1100.0, "fuel_type": "Matorral Xerofilo"},

        # Lat 22.00 (Central corridor: Western Escarpment / Central Valley / Tepezala)
        {"voxel_id": "AGS_V11", "lat": 22.00, "lon": -102.75, "elevation_m": 2690.0, "slope_deg": 29.0, "aspect_rad": 0.95, "dist_roads_m": 1400.0, "dist_agri_m": 3100.0, "fuel_type": "Bosque de Pino"},
        {"voxel_id": "AGS_V12", "lat": 22.00, "lon": -102.57, "elevation_m": 2350.0, "slope_deg": 18.0, "aspect_rad": 3.60, "dist_roads_m": 620.0, "dist_agri_m": 1750.0, "fuel_type": "Pino-Encino"},
        {"voxel_id": "AGS_V13", "lat": 22.00, "lon": -102.40, "elevation_m": 2010.0, "slope_deg": 10.0, "aspect_rad": 0.15, "dist_roads_m": 240.0, "dist_agri_m": 720.0,  "fuel_type": "Matorral Submontano"},
        {"voxel_id": "AGS_V14", "lat": 22.00, "lon": -102.22, "elevation_m": 1910.0, "slope_deg": 6.5,  "aspect_rad": 4.80, "dist_roads_m": 110.0, "dist_agri_m": 350.0,  "fuel_type": "Pastizal Inducido"},
        {"voxel_id": "AGS_V15", "lat": 22.00, "lon": -102.05, "elevation_m": 2280.0, "slope_deg": 17.5, "aspect_rad": 2.45, "dist_roads_m": 680.0, "dist_agri_m": 1400.0, "fuel_type": "Matorral Xerofilo"},

        # Lat 22.18 (Core Sierra Fria & San Jose de Gracia corridor)
        {"voxel_id": "AGS_V16", "lat": 22.18, "lon": -102.75, "elevation_m": 2850.0, "slope_deg": 33.5, "aspect_rad": 3.40, "dist_roads_m": 2100.0, "dist_agri_m": 4800.0, "fuel_type": "Bosque de Pino"},
        {"voxel_id": "AGS_V17", "lat": 22.18, "lon": -102.57, "elevation_m": 2595.0, "slope_deg": 21.0, "aspect_rad": 1.10, "dist_roads_m": 750.0, "dist_agri_m": 2200.0, "fuel_type": "Pino-Encino"},
        {"voxel_id": "AGS_V18", "lat": 22.18, "lon": -102.40, "elevation_m": 2140.0, "slope_deg": 12.5, "aspect_rad": 4.10, "dist_roads_m": 380.0, "dist_agri_m": 980.0,  "fuel_type": "Bosque de Encino"},
        {"voxel_id": "AGS_V19", "lat": 22.18, "lon": -102.22, "elevation_m": 1950.0, "slope_deg": 7.0,  "aspect_rad": 0.55, "dist_roads_m": 150.0, "dist_agri_m": 420.0,  "fuel_type": "Pastizal Inducido"},
        {"voxel_id": "AGS_V20", "lat": 22.18, "lon": -102.05, "elevation_m": 2340.0, "slope_deg": 19.0, "aspect_rad": 2.85, "dist_roads_m": 820.0, "dist_agri_m": 1650.0, "fuel_type": "Matorral Xerofilo"},

        # Lat 22.35 (North corridor: High Peaks & Zacatecas Border)
        {"voxel_id": "AGS_V21", "lat": 22.35, "lon": -102.75, "elevation_m": 2920.0, "slope_deg": 35.0, "aspect_rad": 1.70, "dist_roads_m": 2600.0, "dist_agri_m": 5600.0, "fuel_type": "Bosque de Pino"},
        {"voxel_id": "AGS_V22", "lat": 22.35, "lon": -102.57, "elevation_m": 2680.0, "slope_deg": 24.5, "aspect_rad": 3.75, "dist_roads_m": 1200.0, "dist_agri_m": 2900.0, "fuel_type": "Pino-Encino"},
        {"voxel_id": "AGS_V23", "lat": 22.35, "lon": -102.40, "elevation_m": 2220.0, "slope_deg": 14.0, "aspect_rad": 0.25, "dist_roads_m": 450.0, "dist_agri_m": 1150.0, "fuel_type": "Bosque de Encino"},
        {"voxel_id": "AGS_V24", "lat": 22.35, "lon": -102.22, "elevation_m": 1990.0, "slope_deg": 8.0,  "aspect_rad": 4.95, "dist_roads_m": 180.0, "dist_agri_m": 510.0,  "fuel_type": "Matorral Submontano"},
        {"voxel_id": "AGS_V25", "lat": 22.35, "lon": -102.05, "elevation_m": 2410.0, "slope_deg": 20.5, "aspect_rad": 2.30, "dist_roads_m": 940.0, "dist_agri_m": 1850.0, "fuel_type": "Matorral Xerofilo"}
    ]
    return pd.DataFrame(grid_cells)


def integrate_spatiotemporal_matrix(
    df_era5_clean: pd.DataFrame,
    df_firms_clean: pd.DataFrame,
    df_conafor_clean: pd.DataFrame,
    matching_radius_deg: float = 0.08
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fuses climate reanalysis, spatial topography, biophysical vegetation dynamics,
    and ground-truth fire events into an integrated Analysis-Ready Parquet Matrix.
    """
    logger.info("Starting spatiotemporal integration...")
    
    # 1. Transform climate into daily time-series
    daily_climate = transform_climate_daily_aggregates(df_era5_clean)
    dates = daily_climate["date"].tolist()
    
    # 2. Build spatial grid
    grid_df = build_regional_monitoring_grid()
    num_voxels = len(grid_df)
    num_days = len(dates)
    total_rows = num_voxels * num_days
    
    logger.info(f"Generating Cartesian space-time product: {num_voxels} voxels x {num_days} days = {total_rows} observations.")
    
    # Repeat grid across all dates
    expanded_rows = []
    for _, voxel in grid_df.iterrows():
        temp_v = pd.DataFrame([voxel.to_dict()] * num_days)
        temp_v["date"] = dates
        expanded_rows.append(temp_v)
        
    matrix_df = pd.concat(expanded_rows, ignore_index=True)
    
    # 3. Merge daily climate attributes by date
    matrix_df = matrix_df.merge(daily_climate, on="date", how="left")
    
    # 4. Topographic downscaling: lapse-rate adjusted temperature
    matrix_df["temp_2m_max_c"] = calculate_lapse_rate_temperature(
        temp_c=matrix_df["temp_2m_max_c"],
        dem_elevation_m=matrix_df["elevation_m"],
        reference_elevation_m=2595.0,
        gamma=-0.0065
    )
    
    # 5. Biophysical Vegetation Dynamics (NDMI, SAVI, NBR)
    # Seasonal phenology modeling: dry season (March–May) vs monsoon greening (July–September)
    day_of_year = matrix_df["date"].dt.dayofyear
    seasonal_cycle = np.sin((day_of_year - 120) * (2 * np.pi / 365))
    
    # Base moisture correlated with fuel type and season
    fuel_moist_factor = matrix_df["elevation_m"] / 3000.0
    matrix_df["ndmi"] = np.clip(0.15 + (0.25 * seasonal_cycle) + (0.10 * fuel_moist_factor) - (0.05 * matrix_df["vpd_max_kpa"]), -1.0, 1.0)
    matrix_df["savi"] = np.clip(0.30 + (0.20 * seasonal_cycle) + (0.08 * fuel_moist_factor), -1.0, 1.0)
    matrix_df["nbr"]  = np.clip(0.25 + (0.22 * seasonal_cycle) - (0.04 * matrix_df["days_since_rain"] / 30.0), -1.0, 1.0)
    
    # 6. Target Label Formulation: fire_ignition_48h
    # Ground-truth: match FIRMS detections and CONAFOR ignitions within radius and 48-72h anticipatory window (1 to 3 days prior)
    logger.info("Formulating 48-72h predictive ignition target labels...")
    
    firms_events = df_firms_clean[df_firms_clean["confidence_category"].isin(["nominal", "high"])].copy()
    firms_events["event_date"] = pd.to_datetime(pd.to_datetime(firms_events["timestamp_utc"]).dt.date)
    
    conafor_events = df_conafor_clean.copy()
    conafor_events["event_date"] = pd.to_datetime(pd.to_datetime(conafor_events["date_start"]).dt.date)
    
    fire_targets = np.zeros(len(matrix_df), dtype=np.int32)
    
    # Match FIRMS thermal detections
    for _, fire in firms_events.iterrows():
        f_date = fire["event_date"]
        f_lat = fire["latitude"]
        f_lon = fire["longitude"]
        
        spatial_match = (
            np.sqrt((matrix_df["lat"] - f_lat)**2 + (matrix_df["lon"] - f_lon)**2) <= matching_radius_deg
        )
        temporal_match = (
            (matrix_df["date"] >= (f_date - pd.Timedelta(days=3))) &
            (matrix_df["date"] <= (f_date - pd.Timedelta(days=1)))
        )
        match_idx = matrix_df[spatial_match & temporal_match].index
        if len(match_idx) > 0:
            fire_targets[match_idx] = 1
            
    # Match CONAFOR validated ignitions
    for _, fire in conafor_events.iterrows():
        f_date = fire["event_date"]
        f_lat = fire["latitude"]
        f_lon = fire["longitude"]
        
        spatial_match = (
            np.sqrt((matrix_df["lat"] - f_lat)**2 + (matrix_df["lon"] - f_lon)**2) <= matching_radius_deg
        )
        temporal_match = (
            (matrix_df["date"] >= (f_date - pd.Timedelta(days=3))) &
            (matrix_df["date"] <= (f_date - pd.Timedelta(days=1)))
        )
        match_idx = matrix_df[spatial_match & temporal_match].index
        if len(match_idx) > 0:
            fire_targets[match_idx] = 1
            
    matrix_df["fire_ignition_48h"] = fire_targets
    positive_count = int(matrix_df["fire_ignition_48h"].sum())
    logger.info(f"Target formulation completed. Positive ignition windows (48-72h): {positive_count} ({positive_count/len(matrix_df)*100:.2f}%)")
    
    # 7. Format Timestamp and Column Order
    matrix_df["timestamp"] = matrix_df["date"]
    
    final_cols = [
        "voxel_id", "timestamp", "fire_ignition_48h",
        "ndmi", "savi", "nbr",
        "vpd_max_kpa", "vpd_72h_max_kpa",
        "temp_2m_max_c", "rh_2m_min_pct",
        "wind_u10_mean", "wind_v10_mean", "wind_speed_max_kmh",
        "soil_moist_mean", "days_since_rain", "precip_72h_sum_mm",
        "elevation_m", "slope_deg", "aspect_rad",
        "dist_roads_m", "dist_agri_m", "fuel_type"
    ]
    matrix_df = matrix_df[final_cols].copy()
    
    # Cast optimized dtypes
    matrix_df["voxel_id"] = matrix_df["voxel_id"].astype("string")
    matrix_df["fire_ignition_48h"] = matrix_df["fire_ignition_48h"].astype("int32")
    matrix_df["days_since_rain"] = matrix_df["days_since_rain"].astype("int32")
    matrix_df["fuel_type"] = matrix_df["fuel_type"].astype("category")
    
    float_cols = [c for c in matrix_df.columns if c not in ["voxel_id", "timestamp", "fire_ignition_48h", "days_since_rain", "fuel_type"]]
    for fc in float_cols:
        matrix_df[fc] = matrix_df[fc].astype("float32")
        
    integration_record = {
        "total_records": len(matrix_df),
        "total_variables": len(final_cols),
        "positive_ignition_windows": positive_count,
        "negative_windows": len(matrix_df) - positive_count,
        "date_range": [str(matrix_df["timestamp"].min().date()), str(matrix_df["timestamp"].max().date())],
        "voxels_count": num_voxels
    }
    
    return matrix_df, integration_record


def export_analysis_ready_parquet(
    df: pd.DataFrame,
    output_parquet_path: str,
    output_sample_csv_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Exports the integrated DataFrame into production Apache Parquet format
    with Snappy compression, generating storage comparison metrics.
    """
    os.makedirs(os.path.dirname(output_parquet_path), exist_ok=True)
    df.to_parquet(output_parquet_path, engine="pyarrow", compression="snappy", index=False)
    parquet_size_bytes = os.path.getsize(output_parquet_path)
    parquet_size_mb = round(parquet_size_bytes / (1024 * 1024), 2)
    
    csv_size_mb = None
    if output_sample_csv_path is not None:
        os.makedirs(os.path.dirname(output_sample_csv_path), exist_ok=True)
        # Export representative sample
        df.head(2000).to_csv(output_sample_csv_path, index=False)
        csv_size_mb = round(os.path.getsize(output_sample_csv_path) / (1024 * 1024), 2)
        
    metrics = {
        "parquet_path": output_parquet_path,
        "parquet_size_mb": parquet_size_mb,
        "sample_csv_path": output_sample_csv_path,
        "sample_csv_size_mb": csv_size_mb,
        "records": len(df),
        "columns": df.shape[1]
    }
    logger.info(f"Parquet exported: {output_parquet_path} ({parquet_size_mb} MB)")
    return metrics
