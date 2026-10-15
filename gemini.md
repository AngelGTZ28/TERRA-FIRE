# 🧠 TERRA-FIRE: Master Project Memory & Context (`gemini.md`)
> **AI Assistant Reference Manual & Technical Knowledge Base**  
> **Last Updated:** October 14, 2026 | **Project Status:** TRL 2 Validated ➔ Ready for PoC (TRL 3)

---

## 📌 1. Project Identity & Governance
* **Project Name:** **TERRA-FIRE** (Predictive Wildfire Intelligence & Offline Tactical Cartography Engine)
* **Mission:** Transition Mexico's forest fire management from reactive suppression to anticipatory, micro-territorial (10–20 m) mitigation **48 to 72 hours prior to ignition**.
* **Target Users:** 
  1. *Rural ejidatarios & farmers:* Safe agricultural burn advisories (NOM-015-SEMARNAT/AGRICULTURA-2023).
  2. *Community fire brigades:* 100% offline tactical risk cartography and escape route navigation in remote mountain corridors.
  3. *CONAFOR & Civil Protection dispatchers:* Anticipatory territorial resource pre-allocation.
* **Institution:** Universidad Tecnológica El Retoño (UTR) — Aguascalientes, México.
* **Academic Program:** Information Technologies and Digital Innovation Engineering (LITD9A).
* **Course:** Data Science & Integrative Project III (Sep–Dec 2026).
* **Advising Professor:** Dr. Sergio Valadez Godínez.
* **Engineering Team:** 
  - Angel Emilio Gutierrez Lozano (ID: 4341)
  - Emiliano Betanzos Valtierra (ID: 4795)

---

## 🏛️ 2. Architectural Ecosystem & Workspace Directories
The project is decoupled into three peer workspaces located in `/home/angel/Workspace/TERRA-FIRE-PROJECT/`:

```text
TERRA-FIRE-PROJECT/
├── TERRA-FIRE/             # 🛠️ GitHub Code Repository (ETL Pipeline, ML, PWA)
├── TERRA-FIRE-DATABASES/   # 📦 External Raw Datasets Storage (>72 MB)
└── TerraFire_vault/        # 📚 Obsidian Knowledge Vault & Official Academic Reports
```

### The 4 Decoupled Computing Layers:
1. **Layer 1: Project Repository (GitHub):** Source code (`src/`), executable notebooks (`notebooks/`), metadata dictionaries (`metadata/`), `.gitignore`, `requirements.txt`, small representative samples (`data/sample/`), and compressed Parquet artifacts (`data/processed/`).
2. **Layer 2: Development Environment (GitHub Codespaces / VS Code):** Linux/Windows workstation environment running Python 3.12 and pinned dependencies.
3. **Layer 3: Data Storage:** Heavy raw tables (>72 MB) stored externally in `TERRA-FIRE-DATABASES/` (excluded from Git).
4. **Layer 4: Data Processing:** Vectorized in-memory compute infrastructure (Python + Pandas/PyArrow) using a **Scaled Data Pipeline** architecture (processing in <3.5 s without unnecessary cluster overhead).

---

## 🛰️ 3. Primary Datasets & Ingestion Peculiarities

| Dataset | Provider | Records & Size | Storage Path | Ingestion Peculiarities & Mitigations |
| :--- | :--- | :---: | :--- | :--- |
| **CONAFOR Historical Fires (2012–2025)** | CONAFOR / SEMARNAT | 101,003 rows<br>(18.6 MB `.xlsx`) | `TERRA-FIRE-DATABASES/CONAFOR_Incendios_2012_2025.xlsx` | • Headers contain corrupted non-ASCII accents (`'Ao'`, `'Herbceo'`).<br>• 48 records have positive longitudes (inverted by $\times -1.0$).<br>• 946 null vegetation rows imputed as `'No Especificado'`.<br>• Strata hectares nulls imputed as `0.0` ha. |
| **NASA FIRMS VIIRS S-NPP (2023–2024)** | NASA LANCE / FIRMS | 657,791 rows<br>(52.4 MB `.csv`) | `TERRA-FIRE-DATABASES/NASA_FIRMS_VIIRS_SNPP_*.csv` | • `acq_time` is integer (e.g. `820`); zero-padded to `08:20:00` and concatenated with `acq_date` to form UTC timestamp.<br>• Filtered to Mexico bounding box $[14.0, 33.5]^\circ\text{N}, [-118.5, -86.0]^\circ\text{W}$. |
| **ECMWF ERA5-Land Hourly** | Copernicus C3S | 17,544 rows<br>(0.93 MB `.csv`) | `TERRA-FIRE-DATABASES/ERA5_Land_Hourly_SierraFria_2023_2024.csv` | • **Crucial:** Lines 1–2 contain station metadata; line 3 is blank; table header is at line 4. Must be loaded using `skiprows=3`.<br>• Local time is GMT-6; UTC offset (+6h) added for satellite alignment. |
| **Copernicus DEM (GLO-30)** | ESA / Airbus | 1°x1° Tile<br>(35.4 MB `.tif`) | `TERRA-FIRE-DATABASES/Copernicus_DSM_*.dem` | • 32-bit float elevation values relative to EGM2008.<br>• Used to compute slope ($0^\circ–90^\circ$), aspect, and lapse-rate thermal downscaling. |
| **Copernicus Sentinel-2 MSI L2A** | ESA / Microsoft PC | Scene Item<br>(0.03 MB `.json`) | `TERRA-FIRE-DATABASES/Sentinel2_L2A_STAC_Item_*.json` | • Multispectral surface reflectance (B04 Red, B08 NIR, B11 SWIR-1, B12 SWIR-2).<br>• Derives Live Fuel Moisture Content (LFMC) via NDMI, SAVI, and NBR. |

---

## ⚗️ 4. Mathematical Formulations & Feature Engineering

### 1. Tetens Equation for Vapor Pressure Deficit ($VPD$)
Evaporative atmospheric suction desiccation force:
$$e_s(T) = 0.61078 \exp\left(\frac{17.27 \cdot T}{T + 237.3}\right) \quad [kPa]$$
$$e_a(T_{dew}) = 0.61078 \exp\left(\frac{17.27 \cdot T_{dew}}{T_{dew} + 237.3}\right) \quad [kPa]$$
$$VPD = \max(0.0, e_s(T) - e_a(T_{dew})) \quad [kPa]$$

### 2. Environmental Lapse-Rate Topographic Downscaling
Correcting coarse 9 km reanalysis temperature to microtopographic elevations:
$$T_{downscaled} = T_{ref} + \Gamma \cdot (Z_{DEM} - Z_{ref})$$
Where $\Gamma = -0.0065 \, ^\circ C/m$, $Z_{ref} = 2595$ m (Sierra Fría station elevation).

### 3. Horizontal Wind Vector Decomposition
Resolves 360° circular discontinuity into continuous Euclidean forces:
$$u_{10} = -(ws_{kmh} / 3.6) \cdot \sin(\theta_{rad}) \quad (\text{zonal, west-to-east})$$
$$v_{10} = -(ws_{kmh} / 3.6) \cdot \cos(\theta_{rad}) \quad (\text{meridional, south-to-north})$$

### 4. Biophysical Indices (Sentinel-2 MSI)
- **Normalized Difference Moisture Index (NDMI):** $NDMI = \frac{B08 - B11}{B08 + B11}$ (Canopy water stress proxy).
- **Soil-Adjusted Vegetation Index (SAVI):** $SAVI = \frac{B08 - B04}{B08 + B04 + 0.5} \times 1.5$ (Foliar density adjusted for bright soil).
- **Normalized Burn Ratio (NBR):** $NBR = \frac{B08 - B12}{B08 + B12}$ (Biomass combustibility proxy).

---

## 📊 5. Analysis-Ready Dataset (`matrix_v1.parquet`) Specification

* **Path:** `data/processed/terra_fire_spatiotemporal_matrix_v1.parquet`
* **Format:** Apache Parquet with Snappy compression (0.30 MB, 92% compression vs CSV).
* **Dimensions:** 18,275 rows $\times$ 22 columns (25 regional monitoring nodes $\times$ 731 continuous days across 2023–2024).
* **Target Label (`fire_ignition_48h`):** Binary $\{0, 1\}$. Evaluated via spatial buffer ($r \le 0.08^\circ \approx 8.5$ km) and 48–72h forward-looking predictive window ($t+1$ to $t+3$ days).
  - Positive ignition windows: **876** (4.79% class prevalence).
  - Negative baseline windows: **17,399** (95.21%).

### Summary Data Dictionary:
`voxel_id` (string), `timestamp` (date), `fire_ignition_48h` (int32), `ndmi` (float32), `savi` (float32), `nbr` (float32), `vpd_max_kpa` (float32), `vpd_72h_max_kpa` (float32), `temp_2m_max_c` (float32), `rh_2m_min_pct` (float32), `wind_u10_mean` (float32), `wind_v10_mean` (float32), `wind_speed_max_kmh` (float32), `soil_moist_mean` (float32), `days_since_rain` (int32), `precip_72h_sum_mm` (float32), `elevation_m` (float32), `slope_deg` (float32), `aspect_rad` (float32), `dist_roads_m` (float32), `dist_agri_m` (float32), `fuel_type` (category).

---

## 💻 6. Quick Execution & Command Reference

### Python Environment Setup
```bash
# Navigate to repo
cd /home/angel/Workspace/TERRA-FIRE-PROJECT/TERRA-FIRE

# Install dependencies
pip install -r requirements.txt
```

### Run Full ETL Pipeline
```powershell
python -m src.pipeline
```

### Launch Interactive Notebook
```powershell
jupyter lab notebooks/data_engineering_lab.ipynb
```

### Run Automated Quality Assertion Suite
```powershell
python -c "import src.validate as v; passed, checklist = v.run_data_validation_suite('data/processed/terra_fire_spatiotemporal_matrix_v1.parquet'); print('All tests passed:', passed)"
```

### Git Branch & Versioning Standards
- Main branch: `main`
- Functional branches: `data-cleaning`, `data-integration`, `validation`
- Commit conventions: Semantic prefixes (`feat:`, `feat(clean):`, `feat(integrate):`, `test(validation):`, `docs:`) with descriptive titles.

---

## 🗺️ 7. Project Roadmap & Upcoming Steps
1. **Activity 3 (Completed):** Data Engineering Lab — ETL pipeline, Parquet Feature Store, and 7-gate validation suite (**READY FOR ANALYSIS**).
2. **Activity 4 (Next):** Exploratory Data Analysis (EDA) — Distribution analysis, multi-collinearity checks, spatial autocorrelation (Moran's I), and seasonal trend decomposition.
3. **Activity 5:** Predictive Machine Learning — 5-Fold Blocked Spatial Cross-Validation with XGBoost & LightGBM; PR-AUC and Brier Score optimization.
4. **Activity 6:** Explainable AI (XAI) — TreeSHAP sub-second edge explainability calculation.
5. **Activity 7:** Offline-First PWA Implementation — MapLibre GL vector tiles, Tippecanoe `.pbf`, and SQLite/WASM client caching.
