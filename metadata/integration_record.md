# 🔗 Data Integration Record & Diagram (TERRA-FIRE)

This document formalizes the multi-source dataset fusion, granularity alignment, and spatiotemporal integration performed during **Activity 3. Data Engineering Lab (Step 5 & 6)**.

---

### Data Integration Record Table

| Source Dataset | Integration Key | Integration Method | Records Before Integration | Records After Integration | Identified Granularity & Integration Issues | Applied Mitigation Rule |
| :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| **ECMWF ERA5-Land Reanalysis** | `date` (`YYYY-MM-DD`) | Temporal rolling aggregation + Left Join to Cartesian grid | 17,544 hourly observations | 18,275 space-time observations (731 days × 25 voxels) | **Temporal Granularity Mismatch**: ERA5 is continuous hourly time-series; target matrix requires daily risk horizons. | Transformed hourly records into daily aggregates (daily max temp, min RH, max VPD, cumulative rain, 72h rolling metrics) prior to merging. |
| **Topographic Monitoring Grid (DEM / Landsat baseline)** | `voxel_id` (`AGS_V01` – `AGS_V25`) | Cartesian Space-Time Expansion | 25 spatial nodes | 18,275 space-time observations | **Continuous Field vs Discrete Nodes**: Elevation rasters are continuous floating grids. | Sampled 25 representative benchmark voxels covering key forest and shrub strata of the Aguascalientes / Sierra Fría corridor. |
| **NASA FIRMS VIIRS Active Fire Detections** | Spatial `(latitude, longitude)` + Temporal `timestamp_utc` | Spatiotemporal Radius & Window Intersection | 2,211 regional detections (2023–2024) | 876 positive anticipatory ignition windows flagged | **Event-Driven Asynchrony**: Satellite orbits detect active thermal anomalies irregularly at 01:30 and 13:30 local pass times. | Formulated Euclidean spatial distance buffer ($r \le 0.08^\circ \approx 8.5$ km) and 48–72h forward-looking predictive window ($t+1$ to $t+3$ days). |
| **CONAFOR Historical Forest Fire Registry** | Spatial `(latitude, longitude)` + Temporal `date_start` | Spatiotemporal Radius & Window Intersection | 282 validated state fire records | Included in positive ground-truth target flags | **Administrative Logging Latency**: Official fire reports often record bureaucratic detection/arrival hours rather than ignition initiation. | Fused jointly with NASA FIRMS active fire detections to calibrate ground-truth ignition veracity. |

---

### Spatiotemporal Integration Architecture Diagram

```mermaid
flowchart TD
    subgraph S1["Raw Climate Stream"]
        ERA5["ECMWF ERA5-Land Hourly<br/>(17,544 rows, 9 cols)"] --> CLN1["Clean & Localize GMT-6<br/>(+ UTC timestamp)"]
        CLN1 --> TRF1["Daily Microclimatic Aggregation<br/>(Max Temp, Min RH, Tetens VPD, 72h Rolling)"]
    end

    subgraph S2["Topographic Lattice"]
        GRID["25 Regional Monitoring Voxels<br/>(AGS_V01 to AGS_V25)<br/>Elevation, Slope, Fuel Type, Distances"]
    end

    subgraph S3["Ground-Truth Incident Streams"]
        FIRMS["NASA FIRMS VIIRS<br/>(2,211 detections)"]
        CONAFOR["CONAFOR Fire Registry<br/>(282 records)"]
        FIRMS & CONAFOR --> SP_BUF["Spatiotemporal Matcher<br/>Radius <= 0.08° (~8.5 km)<br/>Window: 48h to 72h ahead"]
    end

    TRF1 & GRID --> CART["Cartesian Product Expansion<br/>(25 voxels x 731 days = 18,275 rows)"]
    CART --> DOWN["Lapse-Rate Downscaling &<br/>Biophysical Index Modeling (NDMI, SAVI, NBR)"]
    DOWN & SP_BUF --> TARGET["Binary Supervisory Label Assignment<br/>(fire_ignition_48h: 876 positive / 17,399 negative)"]
    TARGET --> PARQ["Apache Parquet Feature Store<br/>(terra_fire_spatiotemporal_matrix_v1.parquet)<br/>0.30 MB, Snappy Encoded"]
```
