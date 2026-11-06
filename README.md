# 🔥 TERRA-FIRE: Predictive Wildfire Intelligence Engine
## 🧪 Activities 3, 4, 5 & 6: Data Engineering, Data Intelligence, Variable Impact & Pattern Discovery

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Storage-Parquet](https://img.shields.io/badge/storage-Apache%20Parquet-orange.svg)](https://parquet.apache.org/)
[![Validation-Pass](https://img.shields.io/badge/validation-100%25%20Passed-brightgreen.svg)]()
[![Pattern-Discovery](https://img.shields.io/badge/pattern--discovery-READY%20FOR%20SEGMENTATION-success.svg)]()
[![License-Open](https://img.shields.io/badge/data-Open%20Access-green.svg)]()

> **Institution:** Universidad Tecnológica El Retoño (UTR)  
> **Course:** Data Science (`LITD9A`) — Sep–Dec 2026  
> **Professor:** Dr. Sergio Valadez Godínez  
> **Authors:** Angel Emilio Gutierrez Lozano (ID: 4341) & Emiliano Betanzos Valtierra (ID: 4795)  
> **Repository:** [https://github.com/AngelGTZ28/TERRA-FIRE](https://github.com/AngelGTZ28/TERRA-FIRE)

---

## 📌 1. Project Purpose & Scope
**TERRA-FIRE** is a predictive wildfire intelligence and offline tactical cartography platform designed to transition Mexican wildfire management from reactive suppression to anticipatory, micro-territorial mitigation **48 to 72 hours prior to ignition**.

This repository houses:
1. **Activity 3 (Data Engineering Lab):** Complete, reproducible ETL pipeline that ingests, cleans, harmonizes, and validates multi-source Earth Observation, meteorological reanalysis, and wildfire records into an optimized **Apache Parquet Feature Store** (`data/processed/terra_fire_spatiotemporal_matrix_v1.parquet`).
2. **Activity 4 (Data Intelligence Lab):** Comprehensive exploratory data analysis (EDA), descriptive statistics (central tendency and dispersion), multivariate hypothesis testing (Welch's $t$-tests), and publication-grade visual analytics (`fig01`–`fig12`).
3. **Activity 5 (Variable Impact Analysis):** Parametric (Pearson $r$) and non-parametric (Spearman $\rho$) correlation analysis, Ordinary Least Squares (OLS) simple regression modeling across continuous physical systems ($N = 18,275$) and quantile-binned empirical ignition probabilities ($N = 20$), 95% confidence interval estimation, Variance Inflation Factor (VIF) multicollinearity diagnostics, spatial confounding resolution, and multi-criteria Variable Impact Ranking (`fig13`–`fig20`).
4. **Activity 6 (Pattern Discovery Challenge):** Multivariate Z-score standardization (`StandardScaler`) and Principal Component Analysis (`PCA(n_components=3)`) across the 12-variable Core Spatiotemporal Matrix (**75.0% dimensionality reduction compressing 12 variables into strictly 3 orthogonal components `['PC1', 'PC2', 'PC3']` while retaining 87.87% of total variance with $\text{VIF} = 1.00$**), 2D/3D PCA biplots, seasonal hysteresis loop discovery, Hotelling's $T^2$ & Q-residual outlier diagnostics, and identification of the **Quadrant IV Pyro-Anthropogenic Convergence Zone** capturing **68.61% of all wildfires** (`fig21`–`fig27`).

---

## 🛰️ 2. Dataset Used & Primary Sources of Origin

- **Analysis-Ready Dataset:** `data/processed/terra_fire_spatiotemporal_matrix_v1.parquet` (0.30 MB, Snappy Columnar)
- **Dimensions:** 18,275 daily observations $\times$ 22 attributes (25 regional monitoring nodes $\times$ 731 days across 2023–2024).
- **Target Variable (`fire_ignition_48h`):** Binary indicator $\{0, 1\}$ with **876 positive ignition windows (4.79%)** and **17,399 baseline windows (95.21%)**.

| Dataset Name | Source / Provider | Temporal Coverage | Native Format | Origin URL / Discovery |
| :--- | :--- | :--- | :---: | :--- |
| **CONAFOR Forest Fires** | Comisión Nacional Forestal (SEMARNAT) | 2012–2025 (101,003 records) | `.xlsx` | [IDEFOR Open Data](https://idefor.cnf.gob.mx/documents/3249/download) |
| **NASA FIRMS Active Fires** | NASA LANCE / FIRMS (VIIRS 375m S-NPP) | 2023–2024 (657,791 detections) | `.csv` | [NASA Earthdata FIRMS](https://firms.modaps.eosdis.nasa.gov/) |
| **ECMWF ERA5-Land** | Copernicus Climate Change Service (C3S) | 2023–2024 (17,544 hourly steps) | `.csv` | [Copernicus Climate Data Store](https://doi.org/10.24381/cds.e2161bac) |
| **Copernicus DEM (GLO-30)** | European Space Agency (ESA) / Airbus | Static 30m Global Baseline | `.tif` (COG) | [Copernicus Data Space](https://doi.org/10.5270/ESA-c5d3d65) |
| **Sentinel-2 L2A MSI** | ESA / European Commission | 5-day revisit cadence | `.json` / COG | [Microsoft Planetary Computer STAC](https://planetarycomputer.microsoft.com/) |

---

## 🔬 3. Variables Analyzed & Statistical Techniques (Activities 5 & 6)

### Variables Analyzed
- **Target Variable (1):** `fire_ignition_48h` (Binary 48–72h wildfire emergence; strictly excluded from PCA to prevent Target Leakage).
- **High-Impact Explanatory Variables (7 Quantitative + 1 Categorical):** `dist_agri_m` (Agricultural parcel distance), `dist_roads_m` (Road network distance), `temp_2m_max_c` (Adiabatically downscaled max temperature), `rh_2m_min_pct` (Min relative humidity), `soil_moist_mean` (Upper 0–7 cm volumetric soil moisture), `vpd_max_kpa` (Daily peak Vapor Pressure Deficit), `ndmi` (Sentinel-2 Canopy Moisture Index), and `fuel_type` (Ecological fuel stratum).
- **Medium-Impact Covariates & Confounders (5 in Core PCA):** `slope_deg`, `elevation_m`, `vpd_72h_max_kpa`, `precip_72h_sum_mm`, `days_since_rain` (plus `savi` pruned due to collinearity with `ndmi`).
- **Low-Impact / Redundant Variables Pruned (5):** `nbr` ($\text{VIF} > 100$ collinear with `ndmi`), `wind_v10_mean` (statistically non-significant noise, $p = 0.621$), `wind_speed_max_kmh`, `wind_u10_mean`, and `aspect_rad`.

### Statistical & Multivariate Techniques Implemented
1. **Parametric & Non-Parametric Correlation Analysis:** Pearson linear correlation ($r$), Point-Biserial correlation ($r_{pb}$), and Spearman rank-order correlation ($\rho$) with two-tailed asymptotic significance tests.
2. **Simple Linear Regression Modeling (`statsmodels.api.OLS`):** Continuous physical system regressions ($N = 18,275$) and vigintile empirical probability regressions ($N = 20\text{ quantiles}$).
3. **Principal Component Analysis & Dimensionality Reduction (`StandardScaler` + `PCA(n_components=3)`):** Exploratory eigenvalue spectrum verification (`pca = PCA()`) followed by explicit truncation (`pca_reduced = PCA(n_components=3)`) under the Kaiser Criterion ($\lambda_k \ge 1.0$), generating `X_pca_reduced` ($18,275 \times 3$), Hotelling's $T^2$ Mahalanobis distance, and Q-residuals (SPE).

---

## 📁 4. Repository Structure

```text
TERRA-FIRE/
├── data/
│   ├── sample/                          # Representative subsets for rapid testing & git versioning
│   ├── raw/                             # Original raw datasets (unmodified; excluded from Git)
│   ├── intermediate/                    # Temporary cache generated during ETL (excluded from Git)
│   └── processed/                       # Final Analysis-Ready Dataset
│       └── terra_fire_spatiotemporal_matrix_v1.parquet (0.30 MB, Snappy Columnar)
├── metadata/                            # Audit registries, cleaning logs, and validation matrices
├── notebooks/                           # Executable analytical notebooks
│   ├── data_engineering_lab.ipynb       # Activity 3: Interactive E2E engineering and ETL notebook
│   ├── data_intelligence_lab.ipynb      # Activity 4: Descriptive statistics and visual analytics
│   ├── variable_impact_analysis.ipynb   # Activity 5: Correlation, regression & impact assessment
│   └── pattern_discovery_challenge.ipynb# Activity 6: Multivariate PCA (12->3), biplots & pattern discovery
├── reports/                             # Publication-grade analytical dashboards
│   └── figures/                         # 27 high-resolution (300 DPI) visual analytics figures
│       ├── fig01_target_distribution.png ... fig12_topographic_niche.png (Activity 4)
│       ├── fig13_correlation_heatmaps.png ... fig20_variable_impact_ranking.png (Activity 5)
│       ├── fig21_pca_explained_variance.png
│       ├── fig22_pca_component_loadings.png
│       ├── fig23_pca_2d_biplot_ignition.png
│       ├── fig24_pca_ecological_regimes.png
│       ├── fig25_pca_3d_projection.png
│       ├── fig26_pca_latent_profiles.png
│       └── fig27_pca_outliers_hotelling.png
├── src/                                 # Modular Python packages
│   ├── extract.py, clean.py, transform.py, integrate.py, validate.py, pipeline.py
│   ├── data_intelligence_analysis.py    # Activity 4 statistical engine & dashboard generator
│   ├── variable_impact_analysis.py      # Activity 5 correlation, OLS regression & VIF engine
│   └── pattern_discovery_challenge.py   # Activity 6 multivariate PCA (12->3) & pattern discovery engine
├── Work/                                # Official Deliverable Markdown Reports & Figures
│   ├── Activity 5. Variable Impact Analysis.md
│   ├── Activity 6. Pattern Discovery Challenge.md
│   ├── variable_impact_analysis.ipynb
│   ├── pattern_discovery_challenge.ipynb
│   ├── variable_impact_analysis.py
│   ├── pattern_discovery_challenge.py
│   └── figures/
├── .gitignore
├── gemini.md                            # Master project memory, system architecture, and context
├── README.md                            # Comprehensive project documentation
└── requirements.txt                     # Pinned production dependencies
```

---

## ⚙️ 5. Dependencies & Environment Setup

### Core Dependencies (`requirements.txt`)
- `pandas>=2.2.0`, `numpy>=1.26.0`, `pyarrow>=15.0.0`, `fastparquet>=2024.2.0`
- `scipy>=1.12.0`, `statsmodels>=0.14.0`, `scikit-learn>=1.4.0`
- `matplotlib>=3.8.0`, `seaborn>=0.13.0`, `plotly>=5.20.0`, `jupyterlab>=4.1.0`

### Installation & Execution
```bash
# 1. Clone repository & activate environment
git clone https://github.com/AngelGTZ28/TERRA-FIRE.git
cd TERRA-FIRE
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Run Activity 3 (ETL Pipeline)
python -m src.pipeline

# 3. Run Activity 4 (Data Intelligence Lab)
python src/data_intelligence_analysis.py

# 4. Run Activity 5 (Variable Impact Analysis)
python src/variable_impact_analysis.py

# 5. Run Activity 6 (Pattern Discovery Challenge)
python src/pattern_discovery_challenge.py

# 6. Launch Interactive Jupyter Notebooks
jupyter lab notebooks/pattern_discovery_challenge.ipynb
```

---

## 🧠 6. Main Findings (Activities 4, 5 & 6)

1. **Anthropogenic Proximity is the Dominant Rank Predictor of Ignition:** `dist_agri_m` exhibits the strongest rank correlation with `fire_ignition_48h` ($\rho = -0.2155, p = 6.36 \times 10^{-191}$, Cohen's $d = -0.993$), followed by `dist_roads_m` ($\rho = -0.2063$), proving escaped agricultural burns (*roza y quema* under NOM-015) are the primary wildfire trigger.
2. **High Explanatory Power in Quantile Ignition Probability Regressions:** Across 20 equal-frequency quantiles ($N = 20$), simple OLS regressions prove that `temp_2m_max_c` ($R^2 = 0.8930$), `rh_2m_min_pct` ($R^2 = 0.8232$), `vpd_max_kpa` ($R^2 = 0.7816$), and `soil_moist_mean` ($R^2 = 0.7803$) linearly explain **78% to 89% of the variance in 48–72h wildfire ignition probability**.
3. **High-Efficiency PCA Dimensionality Reduction (Activity 6):** Applying `PCA(n_components=3)` to the 12 standardized High/Medium-Impact spatiotemporal variables compresses the feature space into the reduced matrix `X_pca_reduced` ($18,275 \times 3$, columns `['PC1', 'PC2', 'PC3']`), achieving a **75.0% reduction in dimensions while retaining 87.87% of total variance** ($\text{VIF} = 1.00$):
   - **$\text{PC1}$ ($40.76\%, \lambda_1 = 4.8918$):** *Atmospheric-Edaphic Desiccation & Drought Stress Continuum* ($d = +0.760$).
   - **$\text{PC2}$ ($33.41\%, \lambda_2 = 4.0088$):** *Orographic Altitude vs. Anthropogenic Agricultural Valley Interface* ($d = -0.738$).
   - **$\text{PC3}$ ($13.70\%, \lambda_3 = 1.6440$):** *Vegetative Canopy Phenology & Warm Monsoon Hydration Regime*.
4. **Discovery of the Quadrant IV Pyro-Anthropogenic Convergence Zone ($28.1\times$ Risk Amplification):** In the reduced $(\text{PC1}, \text{PC2})$ plane, **Quadrant IV ($\text{PC1} > 0 \land \text{PC2} \le 0$)** contains $28.50\%$ of observations but concentrates **68.61% of all regional wildfires ($601 / 876$)** at an **11.54% ignition rate** ($28.1\times$ higher than Quadrant II's $0.41\%$), reaching **23.34%** in the **Extreme Apex ($\text{PC1} > 2.5 \land \text{PC2} < -1.0$)**.
5. **Readiness Classification:** **READY FOR SEGMENTATION**.

---

## 🗺️ 7. Project Roadmap & Upcoming Steps
1. **Activity 3 (Completed):** Data Engineering Lab — ETL pipeline, Parquet Feature Store, and 7-gate validation suite (**READY FOR ANALYSIS**).
2. **Activity 4 (Completed):** Data Intelligence Lab — Statistical profiling, dispersion analysis, bivariate group contrasts, and visual dashboards (**READY FOR ADVANCED ANALYSIS**).
3. **Activity 5 (Completed):** Variable Impact Analysis — Correlation matrices, OLS regression modeling, VIF multicollinearity diagnostics, confounding analysis, and Variable Impact Ranking (**READY FOR PATTERN DISCOVERY**).
4. **Activity 6 (Completed):** Pattern Discovery Challenge — Multivariate 12-to-3 PCA dimensionality reduction (`X_pca_reduced` $18,275 \times 3$), 2D/3D biplots, seasonal hysteresis loop, and Hotelling's $T^2$ outlier diagnostics (`fig21`–`fig27`, **READY FOR SEGMENTATION**).
5. **Next Stage (Intelligent Segmentation Lab):** Unsupervised clustering ($K$-Means, Gaussian Mixture Models, DBSCAN) exclusively in the orthogonal 3D PCA subspace (`['PC1', 'PC2', 'PC3']`) to segment micro-territorial pyro-climatic risk regimes.
6. **Machine Learning Challenge & Model Performance Validation:** 5-Fold Spatially Blocked Cross-Validation (`GroupKFold` by `voxel_id`) with XGBoost & LightGBM; PR-AUC, Brier Score, and TreeSHAP explainability.

---

## 👥 Academic Credits
- **Angel Emilio Gutierrez Lozano** (ID: 4341)
- **Emiliano Betanzos Valtierra** (ID: 4795)
- **Docente Asesor:** Dr. Sergio Valadez Godínez
- **Carrera:** Ingeniería en Tecnologías de la Información e Innovación Digital (`LITD9A`)
- **Universidad Tecnológica El Retoño** — 2026
