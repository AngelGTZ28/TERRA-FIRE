# 🔥 TERRA-FIRE: Predictive Wildfire Intelligence Engine
## 🧪 Activity 3: Data Engineering Lab (ETL Pipeline & Feature Store)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Storage-Parquet](https://img.shields.io/badge/storage-Apache%20Parquet-orange.svg)](https://parquet.apache.org/)
[![Validation-Pass](https://img.shields.io/badge/validation-100%25%20Passed-brightgreen.svg)]()
[![License-Open](https://img.shields.io/badge/data-Open%20Access-green.svg)]()

> **Institution:** Universidad Tecnológica El Retoño (UTR)  
> **Course:** Data Science (LITD9A) — Sep–Dec 2026  
> **Professor:** Dr. Sergio Valadez Godínez  
> **Authors:** Angel Emilio Gutierrez Lozano & Emiliano Betanzos Valtierra  
> **Repository:** [https://github.com/AngelGTZ28/TERRA-FIRE](https://github.com/AngelGTZ28/TERRA-FIRE)

---

## 📌 1. Project Purpose
**TERRA-FIRE** is a predictive wildfire intelligence and offline tactical cartography platform designed to transform wildfire management in Mexico: shifting from reactive suppression to proactive, micro-territorial mitigation **48 to 72 hours in advance**.

This repository contains the complete, reproducible **Data Engineering (ETL) Pipeline** that extracts, cleans, transforms, integrates, and validates multi-source planetary Earth Observation, atmospheric reanalysis, and historical government wildfire records into an **Analysis-Ready Dataset** in **Apache Parquet** format (`data/processed/terra_fire_spatiotemporal_matrix_v1.parquet`).

---

## 🛰️ 2 & 3. Required Datasets & Sources of Origin

| Dataset Name | Source / Provider | Temporal Coverage | Native Format | Origin URL / Discovery |
| :--- | :--- | :--- | :---: | :--- |
| **CONAFOR Forest Fires** | Comisión Nacional Forestal (SEMARNAT) | 2012–2025 (101,003 records) | `.xlsx` | [IDEFOR Open Data](https://idefor.cnf.gob.mx/documents/3249/download) |
| **NASA FIRMS Active Fires** | NASA LANCE / FIRMS (VIIRS 375m S-NPP) | 2023–2024 (657,791 detections) | `.csv` | [NASA Earthdata FIRMS](https://firms.modaps.eosdis.nasa.gov/) |
| **ECMWF ERA5-Land** | Copernicus Climate Change Service (C3S) | 2023–2024 (17,544 hourly steps) | `.csv` | [Copernicus Climate Data Store](https://doi.org/10.24381/cds.e2161bac) |
| **Copernicus DEM (GLO-30)** | European Space Agency (ESA) / Airbus | Static 30m Global Baseline | `.tif` (COG) | [Copernicus Data Space](https://doi.org/10.5270/ESA-c5d3d65) |
| **Sentinel-2 L2A MSI** | ESA / European Commission | 5-day revisit cadence | `.json` / COG | [Microsoft Planetary Computer STAC](https://planetarycomputer.microsoft.com/) |

---

## 🗄️ 4. Data Storage Architecture & Separation of Layers

In strict adherence to the **Activity 3 Architecture Guidelines**, four distinct computing layers are separated:
1. **Layer 1: Project Repository (GitHub):** Source code, executable notebooks, metadata, configuration, `.gitignore`, and a small representative sample dataset (`data/sample/`).
2. **Layer 2: Engineering Environment (GitHub Codespaces / VS Code):** Linux-based cloud/local development environment running Python 3.10+ and pip virtual environments.
3. **Layer 3: Data Storage:** Heavy raw files (>72 MB combined) reside in external storage (`TERRAFIRE-db` / institutional cloud storage) and are ignored by `.gitignore`. Small samples (<5 MB) and the compressed Parquet matrix (<1 MB) are maintained within the repository.
4. **Layer 4: Data Processing:** In-memory vectorized Python + Pandas/DuckDB following a **Scaled Data Pipeline** approach.

---

## 📁 5. Repository Organization

```text
TERRA-FIRE/
├── data/
│   ├── sample/                     # Small representative datasets for testing & git tracking
│   │   ├── conafor_fires_aguascalientes_sample.csv
│   │   ├── era5_hourly_sample.csv
│   │   ├── firms_viirs_sierra_fria_sample.csv
│   │   ├── sentinel2_stac_sample.json
│   │   └── terra_fire_matrix_sample.csv
│   ├── raw/                        # Original raw datasets (unmodified; excluded from Git)
│   │   ├── README.md               # Acquisition instructions & checksums
│   │   └── .gitkeep
│   ├── intermediate/               # Temporary cache generated during ETL (excluded from Git)
│   │   └── .gitkeep
│   └── processed/                  # Final Analysis-Ready Dataset
│       └── terra_fire_spatiotemporal_matrix_v1.parquet (0.30 MB, Snappy)
├── metadata/                       # Formal audit logs and specifications
│   ├── cleaning_log.md             # Empirical cleaning rules and affected record counts
│   ├── data_dictionary.md          # 22-variable data dictionary and types
│   ├── integration_record.md       # Granularity reconciliation and integration diagram
│   ├── storage_decision_matrix.md  # 4-tier storage classification and scale assessment
│   ├── transformation_dictionary.md# Biophysical formulas (Tetens VPD, downscaling, etc.)
│   └── validation_checklist.md     # 7-point quality assurance checklist (100% Pass)
├── notebooks/
│   └── data_engineering_lab.ipynb  # Interactive E2E engineering notebook
├── src/                            # Reusable modular Python engineering packages
│   ├── __init__.py
│   ├── clean.py                    # Data cleaning and quality rectification routines
│   ├── extract.py                  # Ingestion layer and profiling diagnostics
│   ├── integrate.py                # Spatiotemporal fusion and target label generator
│   ├── pipeline.py                 # Master orchestrator script (python -m src.pipeline)
│   ├── transform.py                # Thermodynamic formulas and lapse-rate downscaling
│   └── validate.py                 # Automated quality assertion suite (Pass/Fail)
├── .gitignore                      # Storage protection rules (excludes raw >50MB files)
├── README.md                       # Comprehensive engineering documentation
└── requirements.txt                # Pinned production dependencies
```

---

## ⚙️ 6. Dependencies & Installation

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### Installation
```bash
# 1. Clone repository
git clone https://github.com/AngelGTZ28/TERRA-FIRE.git
cd TERRA-FIRE

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Install pinned dependencies
pip install -r requirements.txt
```

---

## 🚀 7. How to Execute the ETL Pipeline

### Option A: Command-Line Orchestrator (Fully Automated)
Run the complete end-to-end pipeline with automated logging:
```bash
python -m src.pipeline
```

### Option B: Interactive Jupyter Notebook
Launch Jupyter to inspect profiling charts, formulas, and step-by-step transformations:
```bash
jupyter lab notebooks/data_engineering_lab.ipynb
```

---

## 📊 8. Generated Outputs & Feature Store Specification

The master output is saved to `data/processed/terra_fire_spatiotemporal_matrix_v1.parquet`:
- **Dimensions:** 18,275 rows $\times$ 22 columns (25 regional monitoring nodes $\times$ 731 continuous daily timestamps across 2023–2024).
- **Target Variable:** `fire_ignition_48h` (876 positive anticipatory ignition windows; 17,399 non-ignition baselines; 4.79% class prevalence).
- **Format:** Apache Parquet (Snappy compressed, columnar binary encoding, 0.30 MB).
- **Representative CSV Sample:** `data/sample/terra_fire_matrix_sample.csv` (first 2,000 rows, 0.34 MB).

---

## 🧠 9. Processing Strategy Justification (Why Scaled Pipeline?)

This project implements a **Scaled Data Pipeline** utilizing **Python, Pandas, and Apache Parquet**:
- **Volume Evaluation:** The total raw volume across the regional pilot is ~72 MB, expanding to 18,275 space-time analytical records.
- **Memory Footprint:** The full dataset occupies <35 MB of RAM during transformation and 0.30 MB serialized in Parquet.
- **Why Distributed Big Data (PySpark) is Not Used:** In accordance with the professor's strict guideline (*"Do not use Big Data technologies merely to demonstrate their use"*), deploying an Apache Spark cluster would introduce massive JVM initialization overhead, network serialization latency, and cloud infrastructure costs with zero performance gain for a 72 MB dataset. Vectorized Pandas with Parquet page pruning processes the full matrix in under **3.5 seconds**.

---

## ⚠️ 10. Known Limitations
1. **Administrative CONAFOR Delay:** Official CONAFOR records capture bureaucratic logging time rather than minute-level ignition initiation. This was mitigated by cross-referencing high-cadence NASA FIRMS thermal anomaly timestamps.
2. **Spatial Topographic Resolution:** The regional matrix utilizes 25 calibrated benchmark nodes (~8.5 km radius coverage). Transitioning to full-state 20 m raster voxels for all of Mexico will require tiling across Cloud-Optimized GeoTIFFs (COGs) and partitioned DuckDB queries in subsequent phases.
3. **Cloud Cover in Optical Imagery:** Optical Sentinel-2 MSI observations can be obscured by cloud cover during the rainy season; synthetic phenological temporal persistence is used as a fallback gap-filling proxy.

---

## 👥 Academic Credits
- **Angel Emilio Gutierrez Lozano** (ID: 4341)
- **Emiliano Betanzos Valtierra** (ID: 4795)
- **Docente Asesor:** Dr. Sergio Valadez Godínez
- **Carrera:** Ingeniería en Tecnologías de la Información e Innovación Digital
- **Universidad Tecnológica El Retoño** — 2026
