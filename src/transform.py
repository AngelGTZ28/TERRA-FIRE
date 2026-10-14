"""
TERRA-FIRE Data Engineering Pipeline
Module: src/transform.py
Description: Feature Transformation, Biophysical Modeling, and Climate Physics Engine.
"""

import logging
from typing import Tuple, Dict, Any, List
import numpy as np
import pandas as pd

logger = logging.getLogger("terra_fire.transform")


def calculate_tetens_vpd(temp_c: pd.Series, dew_point_c: pd.Series) -> pd.Series:
    """
    Computes Vapor Pressure Deficit (VPD in kPa) using the empirical Tetens formulation.
    VPD = e_s(T) - e_a(T_dew)
    e_s(T) = 0.61078 * exp((17.27 * T) / (T + 237.3)) [kPa]
    e_a(T_dew) = 0.61078 * exp((17.27 * T_dew) / (T_dew + 237.3)) [kPa]
    """
    es = 0.61078 * np.exp((17.27 * temp_c) / (temp_c + 237.3))
    ea = 0.61078 * np.exp((17.27 * dew_point_c) / (dew_point_c + 237.3))
    vpd = np.maximum(0.0, es - ea)
    return pd.Series(vpd, index=temp_c.index, name="vpd_kpa")


def decompose_wind_vectors(wind_speed_kmh: pd.Series, wind_dir_deg: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Converts horizontal wind speed from km/h to m/s and decomposes it into
    zonal (u10, west-to-east) and meridional (v10, south-to-north) vector components.
    u10 = -ws * sin(theta_rad)
    v10 = -ws * cos(theta_rad)
    """
    ws_ms = wind_speed_kmh / 3.6
    rad = np.deg2rad(wind_dir_deg)
    u10 = -ws_ms * np.sin(rad)
    v10 = -ws_ms * np.cos(rad)
    return (
        pd.Series(u10, index=wind_speed_kmh.index, name="wind_u10"),
        pd.Series(v10, index=wind_speed_kmh.index, name="wind_v10")
    )


def calculate_lapse_rate_temperature(
    temp_c: pd.Series,
    dem_elevation_m: pd.Series,
    reference_elevation_m: float = 2595.0,
    gamma: float = -0.0065
) -> pd.Series:
    """
    Applies Environmental Lapse-Rate Downscaling to adjust ambient air temperature
    across varying topographic terrain elevations.
    T_downscaled = T_ref + gamma * (Z_dem - Z_ref)
    gamma: standard environmental lapse rate (-0.0065 °C/m)
    """
    elev_diff = dem_elevation_m - reference_elevation_m
    t_downscaled = temp_c + (gamma * elev_diff)
    return pd.Series(t_downscaled, index=temp_c.index, name="temp_2m_c_downscaled")


def calculate_days_since_rain(daily_precip_mm: pd.Series, threshold_mm: float = 1.0) -> pd.Series:
    """
    Computes consecutive cumulative days without significant precipitation (< threshold_mm).
    Serves as an empirical proxy for fine fuel moisture depletion.
    """
    days_counter = []
    current_dry_days = 0
    for p in daily_precip_mm:
        if p < threshold_mm:
            current_dry_days += 1
        else:
            current_dry_days = 0
        days_counter.append(current_dry_days)
    return pd.Series(days_counter, index=daily_precip_mm.index, name="days_since_rain")


def transform_climate_daily_aggregates(df_hourly: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms continuous hourly ERA5-Land series into daily multi-scale predictive drivers:
    - Daily max temperature (°C)
    - Daily min relative humidity (%)
    - Daily max VPD (kPa)
    - Daily mean wind vector (u10, v10)
    - Daily mean soil moisture (m3/m3)
    - Cumulative daily precipitation (mm)
    - Days since last significant rain (days)
    - 72-hour antecedent rolling maximum VPD (kPa)
    """
    logger.info("Transforming hourly climate into daily predictive aggregates...")
    
    # Compute hourly physical transforms
    df = df_hourly.copy()
    df["vpd_kpa"] = calculate_tetens_vpd(df["temp_2m_c"], df["dew_point_2m_c"])
    u10, v10 = decompose_wind_vectors(df["wind_speed_10m_kmh"], df["wind_dir_10m_deg"])
    df["wind_u10"] = u10
    df["wind_v10"] = v10
    
    # Group by calendar date (local)
    df["date"] = df["timestamp_local"].dt.date
    daily = df.groupby("date").agg(
        temp_2m_max_c=("temp_2m_c", "max"),
        temp_2m_min_c=("temp_2m_c", "min"),
        temp_2m_mean_c=("temp_2m_c", "mean"),
        rh_2m_min_pct=("rh_2m_pct", "min"),
        rh_2m_mean_pct=("rh_2m_pct", "mean"),
        vpd_max_kpa=("vpd_kpa", "max"),
        vpd_mean_kpa=("vpd_kpa", "mean"),
        wind_speed_max_kmh=("wind_speed_10m_kmh", "max"),
        wind_u10_mean=("wind_u10", "mean"),
        wind_v10_mean=("wind_v10", "mean"),
        soil_moist_mean=("soil_moist_0_to_7cm_m3m3", "mean"),
        precip_sum_mm=("precip_mm", "sum")
    ).reset_index()
    
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)
    
    # 72-hour (3-day) rolling maximum VPD and precipitation
    daily["vpd_72h_max_kpa"] = daily["vpd_max_kpa"].rolling(window=3, min_periods=1).max()
    daily["precip_72h_sum_mm"] = daily["precip_sum_mm"].rolling(window=3, min_periods=1).sum()
    daily["days_since_rain"] = calculate_days_since_rain(daily["precip_sum_mm"], threshold_mm=1.0)
    
    logger.info(f"Daily climate aggregates produced. Days: {len(daily)}")
    return daily
