# 🔥 TERRA-FIRE: Predictive Wildfire Intelligence Engine
## 🧪 Activity 3 & 4: Data Engineering & Data Intelligence Lab

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Storage-Parquet](https://img.shields.io/badge/storage-Apache%20Parquet-orange.svg)](https://parquet.apache.org/)
[![Validation-Pass](https://img.shields.io/badge/validation-100%25%20Passed-brightgreen.svg)]()
[![Analysis-Ready](https://img.shields.io/badge/data--intelligence-READY%20FOR%20ANALYSIS-success.svg)]()
[![License-Open](https://img.shields.io/badge/data-Open%20Access-green.svg)]()

> **Institution:** Universidad Tecnológica El Retoño (UTR)  
> **Course:** Data Science (LITD9A) — Sep–Dec 2026  
> **Professor:** Dr. Sergio Valadez Godínez  
> **Authors:** Angel Emilio Gutierrez Lozano (ID: 4341) & Emiliano Betanzos Valtierra (ID: 4795)  
> **Repository:** [https://github.com/AngelGTZ28/TERRA-FIRE](https://github.com/AngelGTZ28/TERRA-FIRE)

---

## 📌 1. Project Purpose & Scope
**TERRA-FIRE** is a predictive wildfire intelligence and offline tactical cartography platform designed to transition Mexican wildfire management from reactive suppression to anticipatory, micro-territorial mitigation **48 to 72 hours prior to ignition**.

This repository houses:
1. **Activity 3 (Data Engineering Lab):** Complete, reproducible ETL pipeline that ingests, cleans, harmonizes, and validates multi-source Earth Observation, meteorological reanalysis, and wildfire records into an optimized **Apache Parquet Feature Store** (`data/processed/terra_fire_spatiotemporal_matrix_v1.parquet`).
2. **Activity 4 (Data Intelligence Lab):** Comprehensive exploratory data analysis (EDA), descriptive statistics (central tendency and dispersion), multivariate hypothesis testing (Welch's $t$-tests), and publication-grade visual analytics (12 analytical dashboards) establishing empirical evidence for predictive modeling.

---

## 🛰️ 2. Primary Datasets & Sources of Origin

| Dataset Name | Source / Provider | Temporal Coverage | Native Format | Origin URL / Discovery |
| :--- | :--- | :--- | :---: | :--- |
| **CONAFOR Forest Fires** | Comisión Nacional Forestal (SEMARNAT) | 2012–2025 (101,003 records) | `.xlsx` | [IDEFOR Open Data](https://idefor.cnf.gob.mx/documents/3249/download) |
| **NASA FIRMS Active Fires** | NASA LANCE / FIRMS (VIIRS 375m S-NPP) | 2023–2024 (657,791 detections) | `.csv` | [NASA Earthdata FIRMS](https://firms.modaps.eosdis.nasa.gov/) |
| **ECMWF ERA5-Land** | Copernicus Climate Change Service (C3S) | 2023–2024 (17,544 hourly steps) | `.csv` | [Copernicus Climate Data Store](https://doi.org/10.24381/cds.e2161bac) |
| **Copernicus DEM (GLO-30)** | European Space Agency (ESA) / Airbus | Static 30m Global Baseline | `.tif` (COG) | [Copernicus Data Space](https://doi.org/10.5270/ESA-c5d3d65) |
| **Sentinel-2 L2A MSI** | ESA / European Commission | 5-day revisit cadence | `.json` / COG | [Microsoft Planetary Computer STAC](https://planetarycomputer.microsoft.com/) |

---

## 🗄️ 3. Architecture & Four-Layer Computing Model

In strict accordance with the academic guidelines, four computing layers are decoupled:
1. **Layer 1: Project Repository (GitHub):** Source code (`src/`), executable notebooks (`notebooks/`), metadata dictionaries (`metadata/`), visual dashboards (`reports/figures/`), representative sample datasets (`data/sample/`), and compressed Parquet artifacts (`data/processed/`).
2. **Layer 2: Development Environment (GitHub Codespaces / VS Code):** Linux/Windows workstation environment running Python 3.10+ and pip virtual environments.
3. **Layer 3: Data Storage:** Heavy raw files (>72 MB combined) reside in external storage (`TERRAFIRE-db` / institutional cloud storage) and are ignored by `.gitignore`. Small samples (<5 MB) and the compressed Parquet matrix (<1 MB) are maintained within the repository.
4. **Layer 4: Data Processing:** In-memory vectorized Python + Pandas/PyArrow following a **Scaled Data Pipeline** approach (processing in <3.5 s without unnecessary cluster overhead).

---

## 📁 4. Repository Structure

```text
TERRA-FIRE/
├── data/
│   ├── sample/                     # Representative subsets for rapid testing & git versioning
│   │   ├── conafor_fires_aguascalientes_sample.csv
│   │   ├── era5_hourly_sample.csv
│   │   ├── firms_viirs_sierra_fria_sample.csv
│   │   ├── sentinel2_stac_sample.json
│   │   └── terra_fire_matrix_sample.csv
│   ├── raw/                        # Original raw datasets (unmodified; excluded from Git)
│   │   ├── README.md               # Acquisition provenance, source URLs & checksums
│   │   └── .gitkeep
│   ├── intermediate/               # Temporary cache generated during ETL (excluded from Git)
│   │   └── .gitkeep
│   └── processed/                  # Final Analysis-Ready Dataset
│       └── terra_fire_spatiotemporal_matrix_v1.parquet (0.30 MB, Snappy Columnar)
├── metadata/                       # Audit registries, cleaning logs, and validation matrices
│   ├── cleaning_log.md             # Empirical cleaning rules and affected record counts
│   ├── data_dictionary.md          # 22-variable data dictionary, schema, and units
│   ├── integration_record.md       # Spatiotemporal granularity reconciliation
│   ├── storage_decision_matrix.md  # 4-tier storage classification and scale assessment
│   ├── transformation_dictionary.md# Biophysical and thermodynamic formulas
│   └── validation_checklist.md     # 7-point quality assurance checklist (100% Pass)
├── notebooks/                      # Executable analytical notebooks
│   ├── data_engineering_lab.ipynb  # Interactive E2E engineering and ETL notebook
│   └── data_intelligence_lab.ipynb # Complete descriptive statistics and visual analytics
├── reports/                        # Publication-grade analytical dashboards
│   └── figures/                    # 12 high-resolution (300 DPI) visual analytics figures
│       ├── fig01_target_distribution.png
│       ├── fig02_vpd_distributions.png
│       ├── fig03_temp_humidity_distributions.png
│       ├── fig04_biophysical_indices.png
│       ├── fig05_topography_anthropogenic.png
│       ├── fig06_fuel_types.png
│       ├── fig07_comparative_boxplots.png
│       ├── fig08_anthropogenic_proximity.png
│       ├── fig09_fuel_type_comparative.png
│       ├── fig10_seasonality_timeline.png
│       ├── fig11_thermodynamic_phase_space.png
│       └── fig12_topographic_niche.png
├── src/                            # Modular Python packages
│   ├── __init__.py
│   ├── clean.py                    # Spatial bounds clipping and categorical encoding
│   ├── data_intelligence_analysis.py # Standalone statistical engine & dashboard generator
│   ├── extract.py                  # Ingestion layer and profiling diagnostics
│   ├── integrate.py                # Spatiotemporal fusion and target label generator
│   ├── pipeline.py                 # Master orchestrator script (python -m src.pipeline)
│   ├── transform.py                # Thermodynamic formulas and lapse-rate downscaling
│   └── validate.py                 # Automated quality assertion suite (Pass/Fail)
├── .gitignore                      # Protection against large binaries (>50MB) and caches
├── gemini.md                       # Master project memory, system architecture, and constraints
├── README.md                       # Comprehensive project documentation
└── requirements.txt                # Pinned production dependencies
```

---

## ⚙️ 5. Dependencies & Environment Setup

### Installation
```bash
# 1. Clone repository
git clone https://github.com/AngelGTZ28/TERRA-FIRE.git
cd TERRA-FIRE

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 3. Install pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 6. Execution Instructions

### A. Run Data Engineering (ETL Pipeline)
```bash
python -m src.pipeline
```

### B. Run Data Intelligence Analysis (Statistics & Visual Dashboards)
```bash
python src/data_intelligence_analysis.py
```
*Outputs all 12 publication charts into `reports/figures/` and prints comprehensive central tendency, dispersion, and comparative matrices to console in <3.0 seconds.*

### C. Launch Interactive Jupyter Notebooks
```bash
# Data Engineering Pipeline Notebook
jupyter lab notebooks/data_engineering_lab.ipynb

# Data Intelligence Lab Notebook
jupyter lab notebooks/data_intelligence_lab.ipynb
```

---

## 📊 7. Dataset Profile (`matrix_v1.parquet`)

- **Total Records:** 18,275 rows (25 monitoring voxels $\times$ 731 continuous days across 2023–2024).
- **Total Variables:** 22 attributes (18 quantitative, 2 qualitative, 1 temporal, 1 target).
- **Target Variable (`fire_ignition_48h`):** Binary classification indicating wildfire emergence within 48–72h ($r \le 8.5\text{ km}$).
  - Non-ignition baseline: **17,399 records (95.21%)**
  - Positive anticipatory ignition: **876 records (4.79%)**
- **Missing Values:** Exactly 0 across all 402,050 data points (100% complete).
- **Memory Footprint:** 1.63 MB uncompressed in RAM, 0.30 MB serialized in Apache Parquet (92% compression).

---

## 🧠 8. Main Data Intelligence Findings

1. **Anthropogenic Proximity Triggers Wildfires:** Ignitions occur at a mean distance of **651.6 m** (median **420.0 m**) from agricultural parcels, contrasted with **1,650.7 m** (median **1,200.0 m**) for baseline conditions ($p < 10^{-30}$), proving that traditional agricultural burns (*roza y quema*) along agrarian borders are the primary wildfire vector.
2. **Extreme Combustible Selectivity:** **63.58%** of all detected ignitions (557 of 876 positive events) concentrate in *Pastizal Inducido*, with an additional **17.92%** in *Matorral Xerófilo*, while dense temperate forests contribute $<5\%$ combined.
3. **Thermodynamic Desiccation Precursors:** Prior to ignition, $VPD_{max}$ surges by **+16.38%** (mean $2.21\text{ kPa}$, median $2.09\text{ kPa}$) and relative humidity plunges to a critical median of **18.0%**, establishing atmospheric suction as an anticipatory precursor detectable 48–72h ahead.
4. **Foothill Valley Clustering:** Wildfires concentrate in low-elevation valleys (mean $2,004.9\text{ msnm}$, mean slope $9.29^\circ$) where human accessibility and agricultural plots intersect fine fuels.
5. **Readiness Classification:** **READY FOR ADVANCED ANALYSIS** (TRL 3 proof-of-concept ready for 5-Fold Spatially Blocked Cross-Validation with XGBoost/LightGBM).

---

## 🗺️ 9. Project Roadmap & Upcoming Steps
1. **Activity 3 (Completed):** Data Engineering Lab — ETL pipeline, Parquet Feature Store, and 7-gate validation suite (**READY FOR ANALYSIS**).
2. **Activity 4 (Completed):** Data Intelligence Lab — Statistical profiling, dispersion analysis, bivariate group contrasts, and visual dashboards (**READY FOR ADVANCED ANALYSIS**).
3. **Activity 5 (Next):** Predictive Machine Learning — 5-Fold Blocked Spatial Cross-Validation with XGBoost & LightGBM; PR-AUC and Brier Score optimization.
4. **Activity 6:** Explainable AI (XAI) — TreeSHAP sub-second edge explainability calculation for NOM-015 agricultural burn permits.
5. **Activity 7:** Offline-First PWA Implementation — MapLibre GL vector tiles (`.pbf`, $<12\text{ MB}$) for community brigades in cellular dead zones.

---

## 👥 Academic Credits
- **Angel Emilio Gutierrez Lozano** (ID: 4341)
- **Emiliano Betanzos Valtierra** (ID: 4795)
- **Docente Asesor:** Dr. Sergio Valadez Godínez
- **Carrera:** Ingeniería en Tecnologías de la Información e Innovación Digital (LITD9A)
- **Universidad Tecnológica El Retoño** — 2026
