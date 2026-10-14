# 📊 Data Storage Decision Matrix (TERRA-FIRE)

This matrix formalizes the data storage classification across the project datasets according to their volume, sensitivity, access license, and version control feasibility in compliance with the **Activity 3. Data Engineering Lab** requirements.

| Dataset Name | Primary Source / Provider | Approximate Size | Fits in Memory? | Classification | Storage Location | Technical Justification |
| :--- | :--- | :---: | :---: | :--- | :--- | :--- |
| **CONAFOR Incendios Forestales (2012–2025)** | CONAFOR / SEMARNAT | 18.6 MB (101,003 records, 37 cols) | Yes | **Sample Only** (in Git) / **External Storage** (Full) | `TERRAFIRE-db/` (Full) & `data/sample/` (Regional) | The national dataset exceeds ideal repository limits. A verified regional sample for Aguascalientes / Sierra Fría (282 clean records) is committed to Git for full reproducibility, while raw national tables remain external. |
| **NASA FIRMS VIIRS Active Fire Detections (2023–2024)** | NASA LANCE / FIRMS | 52.4 MB (657,791 detections) | Yes | **Sample Only** (in Git) / **External Storage** (Full) | `TERRAFIRE-db/` (Full) & `data/sample/` (Regional) | High-cadence national satellite observations generate >650k rows. Storing the full raw CSV in GitHub causes repository bloat. The regional testbed sample (2,211 detections) is versioned in `data/sample/`. |
| **ECMWF ERA5-Land Hourly Surface Reanalysis** | ECMWF / Copernicus C3S | 0.93 MB (17,544 rows, 9 cols) | Yes | **Repository Suitable** / **Sample Only** | `data/sample/era5_hourly_sample.csv` | Microclimatic hourly reanalysis for the Sierra Fría reference station fits easily in local memory and repository limits (<1 MB). |
| **Copernicus Sentinel-2 MSI STAC Metadata** | ESA / Microsoft Planetary Computer | 0.03 MB (JSON Scene Item) | Yes | **Repository Suitable** | `data/sample/sentinel2_stac_sample.json` | Lightweight STAC catalog metadata item indexing multispectral Cloud-Optimized GeoTIFFs (B04, B08, B11, B12). Perfectly suitable for repository storage. |
| **Copernicus DEM (GLO-30) Elevation Raster** | ESA / Airbus Defence | 35.4 MB (per 1°x1° COG tile) | Yes | **External Storage Required** | Cloud Object Store / `TERRAFIRE-db/` | Heavy 32-bit floating point elevation raster. Excluded from Git tracking via `.gitignore`; accessible via AWS Open Data S3 or local DB directory. |
| **Analysis-Ready Parquet Feature Store (`matrix_v1.parquet`)** | TERRA-FIRE Engine (Processed) | ~1.5–5 MB (Columnar, Snappy) | Yes | **Repository Suitable** (Pilot) / **External Storage** (Full Matrix) | `data/processed/terra_fire_spatiotemporal_matrix_v1.parquet` | The optimized columnar format with Snappy compression reduces in-memory footprint by >75% compared to raw CSVs, enabling direct versioning of the analytical pilot. |

---

### Data Processing Scale Assessment

| Dataset | Total Records | Fits in Working Memory? (RAM) | Processing Strategy | Justification |
| :--- | :---: | :---: | :--- | :--- |
| **CONAFOR Historic** | 101,003 | Yes (~35 MB RAM) | **In-Memory Pandas + Vectorized Cleaning** | Dataset fits entirely into RAM. No distributed cluster (Spark) is justified; vectorized string and numeric transformations execute in <1.5 seconds. |
| **NASA FIRMS (2023–2024)** | 657,791 | Yes (~110 MB RAM) | **In-Memory Pandas / Bounding Box Indexing** | Conventional memory processing handles ~650k rows efficiently. Spatial bounding box indexing reduces working volume for regional extraction. |
| **ECMWF ERA5-Land** | 17,544 | Yes (~5 MB RAM) | **In-Memory Pandas + Rolling Windows** | Hourly time-series for 2 continuous years processes instantaneously using Pandas datetime index and rolling window operators. |
| **Integrated Feature Matrix** | ~17,500 voxels | Yes (<15 MB RAM) | **Standard Scaled Pipeline -> Parquet Output** | Conforms to **Scenario A / Scaled Pipeline**: conventional memory processing is optimal, avoiding unnecessary distributed compute overhead. |
