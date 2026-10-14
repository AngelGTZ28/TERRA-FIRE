# 🔄 Data Transformation & Feature Standardization Dictionary (TERRA-FIRE)

This dictionary documents all feature transformations, thermodynamic formulations, biophysical proxy calculations, and storage type assignments implemented during **Activity 3. Data Engineering Lab (Step 4 & 5)**.

| Original Variable | Source Dataset | Applied Transformation & Mathematical Formula | Final Variable Name | Final Data Type | Final Unit | Technical Justification |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `temperature_2m (°C)` & `dew_point_2m (°C)` | ECMWF ERA5-Land | **Tetens Equation for Saturation and Actual Vapor Pressure**:<br>$e_s(T) = 0.61078 \exp\left(\frac{17.27 \cdot T}{T + 237.3}\right)$<br>$e_a(T_{dew}) = 0.61078 \exp\left(\frac{17.27 \cdot T_{dew}}{T_{dew} + 237.3}\right)$<br>$VPD = \max(0.0, e_s - e_a)$ | `vpd_max_kpa` | `float32` | $kPa$ (kilopascals) | Atmospheric evaporative demand driving rapid fuel desiccation; fundamental predictor for wildfire propagation. |
| `vpd_kpa` | Derived from ERA5 | **72-hour Rolling Window Maximum**:<br>$\max_{t-72h \le \tau \le t}(VPD_\tau)$ | `vpd_72h_max_kpa` | `float32` | $kPa$ | Captures cumulative antecedent atmospheric stress over a 3-day window prior to observation. |
| `wind_speed_10m (km/h)` & `wind_direction_10m (°)` | ECMWF ERA5-Land | **Unit Conversion & Vectorial Decomposition**:<br>$ws_{ms} = ws_{kmh} / 3.6$<br>$u_{10} = -ws_{ms} \cdot \sin(\theta_{rad})$<br>$v_{10} = -ws_{ms} \cdot \cos(\theta_{rad})$ | `wind_u10_mean`, `wind_v10_mean` | `float32` | $m/s$ | Resolves circular angle discontinuity (360° = 0°) into continuous orthogonal Euclidean vector forces. |
| `temperature_2m (°C)` & `Copernicus GLO-30 DEM` | ERA5 + DEM | **Environmental Lapse-Rate Topographic Downscaling**:<br>$T_{adj} = T_{ref} + \Gamma \cdot (Z_{DEM} - Z_{ref})$<br>where $\Gamma = -0.0065 \, ^\circ C/m$, $Z_{ref} = 2595$ m | `temp_2m_max_c` | `float32` | $^\circ C$ | Adjusts coarse 9 km reanalysis temperature to microtopographic elevations (2,000–2,920 m) across ridges and canyons. |
| `precipitation (mm)` | ECMWF ERA5-Land | **Consecutive Cumulative Dry Days Counter**:<br>Count days with daily precip $< 1.0$ mm | `days_since_rain` | `int32` | Days | Direct proxy for 10-hour and 100-hour dead fuel moisture depletion. |
| `B8 (NIR)` & `B11 (SWIR-1)` | Sentinel-2 L2A MSI | **Normalized Difference Moisture Index**:<br>$NDMI = \frac{B8_{NIR} - B11_{SWIR1}}{B8_{NIR} + B11_{SWIR1}}$ | `ndmi` | `float32` | $[-1.0, 1.0]$ | Live Fuel Moisture Content (LFMC) proxy detecting canopy water stress. |
| `B8 (NIR)` & `B4 (Red)` | Sentinel-2 L2A MSI | **Soil-Adjusted Vegetation Index**:<br>$SAVI = \frac{B8 - B4}{B8 + B4 + 0.5} \times 1.5$ | `savi` | `float32` | $[-1.0, 1.0]$ | Foliar density metric corrected for bright semi-arid volcanic substrate reflectance. |
| `B8 (NIR)` & `B12 (SWIR-2)` | Sentinel-2 L2A MSI | **Normalized Burn Ratio**:<br>$NBR = \frac{B8 - B12}{B8 + B12}$ | `nbr` | `float32` | $[-1.0, 1.0]$ | Biomass density and pre-fire charcoal/senescence indicator. |
| `elevation` & `slope` | Copernicus GLO-30 DEM | **Topographic Terrain Modeling**:<br>Elevation ($m$) and slope gradient ($^\circ$) | `elevation_m`, `slope_deg` | `float32` | Meters, Degrees | Steeper slopes dramatically accelerate convective preheating of uphill fuels (Rothermel model). |
| `FIRMS Lat/Lon` & `CONAFOR Date/Coords` | NASA FIRMS + CONAFOR | **Spatiotemporal Anticipatory Match**:<br>$1$ if ignition or active thermal anomaly occurs within $0.08^\circ$ (~8.5 km) between $t+24h$ and $t+72h$; else $0$ | `fire_ignition_48h` | `int32` | $\{0, 1\}$ | **Primary Target Supervisory Variable** for anticipatory 48–72h wildfire prediction. |

---

### Analytical Storage Format Justification: Apache Parquet vs CSV / JSON

| Evaluation Parameter | Comma-Separated Values (CSV) | JavaScript Object Notation (JSON) | Apache Parquet (Selected Format) |
| :--- | :--- | :--- | :--- |
| **Storage Architecture** | Row-oriented plain text | Row-oriented nested text | **Columnar binary storage with page-level compression** |
| **Storage Size Footprint** | ~3.8 MB (Uncompressed matrix) | ~8.5 MB (Verbose key repetition) | **0.30 MB (Snappy compressed, 92% size reduction)** |
| **Type Integrity & Schema** | Inferred on each read (fragile types) | Weak typing, dates parsed as strings | **Rigid, embedded Arrow schema with bit-width types** |
| **I/O Column Projection** | Full scan required to read single feature | Full scan required | **Zero-copy selective column projection** |
| **Distributed / Cloud Read** | Difficult to partition without split errors | Inefficient chunking | **Native split-by-rowgroup, ideal for DuckDB & Spark** |

> [!TIP]
> **Technical Selection Decision:** Apache Parquet with Snappy compression is selected as the primary analytical format (`data/processed/terra_fire_spatiotemporal_matrix_v1.parquet`). It guarantees exact data types (`float32`, `int32`, `category`), reduces disk/network footprint by >90%, and provides sub-second vectorized query performance for downstream ML pipelines.
