# 📖 Data Dictionary: Analysis-Ready Dataset (TERRA-FIRE)

**Dataset Artifact:** `terra_fire_spatiotemporal_matrix_v1.parquet`  
**Storage Format:** Apache Parquet (Snappy Compressed, Binary Columnar)  
**Total Records:** 18,275 rows  
**Total Variables:** 22 attributes  
**Spatial Coverage:** Aguascalientes & Sierra Fría Regional Network (25 Monitoring Voxels)  
**Temporal Coverage:** 2023-01-01 to 2024-12-31 (731 continuous daily observations)  
**Target Variable:** `fire_ignition_48h` (Anticipatory Wildfire Ignition Window)

| Variable | Description | Type | Unit / Values | Source Dataset | Applied Transformation |
| :--- | :--- | :---: | :---: | :--- | :--- |
| `voxel_id` | Unique spatial benchmark monitoring node identifier | `string` | `AGS_V01` – `AGS_V25` | Spatial Monitoring Grid | Canonical topographic lattice assignment |
| `timestamp` | Daily observation timestamp (calendar date) | `date` / `timestamp` | `YYYY-MM-DD` | Multi-source Temporal Baseline | Aligned to Mexican Central Time (GMT-6) |
| `fire_ignition_48h` | **Primary Target:** 1 if wildfire ignition or thermal detection occurs within 48–72h, else 0 | `int32` | $\{0, 1\}$ | NASA FIRMS + CONAFOR | Spatiotemporal buffer ($r \le 0.08^\circ$, $t+1$ to $t+3$ days) |
| `ndmi` | Normalized Difference Moisture Index (Canopy water stress proxy) | `float32` | $[-1.0, 1.0]$ | Copernicus Sentinel-2 MSI | $(B08 - B11) / (B08 + B11)$ phenological model |
| `savi` | Soil-Adjusted Vegetation Index (Foliar density corrected for soil brightness) | `float32` | $[-1.0, 1.0]$ | Copernicus Sentinel-2 MSI | $[(B08 - B04)/(B08 + B04 + 0.5)] \times 1.5$ |
| `nbr` | Normalized Burn Ratio (Combustibility and cured biomass proxy) | `float32` | $[-1.0, 1.0]$ | Copernicus Sentinel-2 MSI | $(B08 - B12) / (B08 + B12)$ |
| `vpd_max_kpa` | Maximum daily Vapor Pressure Deficit | `float32` | $kPa$ ($[0.0, 10.0]$) | ECMWF ERA5-Land Hourly | Tetens formulation: $e_s(T_{max}) - e_a(T_{dew})$ |
| `vpd_72h_max_kpa` | 72-hour antecedent rolling maximum VPD | `float32` | $kPa$ | ECMWF ERA5-Land | 3-day backward rolling maximum window |
| `temp_2m_max_c` | Daily maximum air temperature downscaled by topography | `float32` | $^\circ C$ | ERA5-Land + Copernicus DEM | Adiabatic lapse-rate: $T + \Gamma \cdot (Z_{DEM} - Z_{ref})$ |
| `rh_2m_min_pct` | Minimum daily relative humidity | `float32` | $\%$ ($[0, 100]$) | ECMWF ERA5-Land Hourly | Daily minimum aggregation |
| `wind_u10_mean` | Daily mean zonal (west-to-east) wind component | `float32` | $m/s$ | ECMWF ERA5-Land Hourly | $-ws \cdot \sin(\theta)$ from hourly wind vectors |
| `wind_v10_mean` | Daily mean meridional (south-to-north) wind component | `float32` | $m/s$ | ECMWF ERA5-Land Hourly | $-ws \cdot \cos(\theta)$ from hourly wind vectors |
| `wind_speed_max_kmh` | Peak horizontal wind gust / maximum velocity | `float32` | $km/h$ | ECMWF ERA5-Land Hourly | Daily maximum velocity aggregation |
| `soil_moist_mean` | Mean volumetric soil water content (Layer 1: 0–7 cm depth) | `float32` | $m^3/m^3$ ($[0.0, 0.6]$) | ECMWF ERA5-Land Hourly | Daily arithmetic mean aggregation |
| `days_since_rain` | Consecutive days with precipitation $< 1.0$ mm | `int32` | Days ($[0, 365]$) | ECMWF ERA5-Land Hourly | Cumulative dry spell counter |
| `precip_72h_sum_mm` | 72-hour cumulative precipitation | `float32` | $mm$ | ECMWF ERA5-Land Hourly | 3-day backward rolling cumulative sum |
| `elevation_m` | Physical terrain altitude above sea level | `float32` | Meters ($[1850, 2920]$) | Copernicus GLO-30 DEM | 30 m Digital Elevation Model extraction |
| `slope_deg` | Topographic terrain slope gradient | `float32` | Degrees ($[0^\circ, 90^\circ]$) | Copernicus GLO-30 DEM | First-order geospatial gradient derivation |
| `aspect_rad` | Slope orientation relative to true north | `float32` | Radians ($[0, 2\pi]$) | Copernicus GLO-30 DEM | Azimuthal aspect angle derivation |
| `dist_roads_m` | Euclidean distance to nearest road network | `float32` | Meters | OpenStreetMap / INEGI | Geospatial distance calculation |
| `dist_agri_m` | Euclidean distance to active agricultural parcels | `float32` | Meters | SADER / INEGI Land Use | Geospatial distance calculation |
| `fuel_type` | Dominant ecological vegetative fuel classification | `category` | Categorical descriptors | CONAFOR Forest Inventory | Forest stratum typology |
