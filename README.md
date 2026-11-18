# 🔥 TERRA-FIRE: Predictive Wildfire Intelligence Engine
## 🧪 Activities 3, 4, 5, 6, 7 & 8: Data Engineering, Data Intelligence, Variable Impact, Pattern Discovery, Machine Learning Challenge & Intelligent Segmentation Lab

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Storage-Parquet](https://img.shields.io/badge/storage-Apache%20Parquet-orange.svg)](https://parquet.apache.org/)
[![Validation-Pass](https://img.shields.io/badge/validation-100%25%20Passed-brightgreen.svg)]()
[![ML-Readiness](https://img.shields.io/badge/machine--learning-READY%20FOR%20MODEL%20VALIDATION-success.svg)]()
[![Segmentation-Readiness](https://img.shields.io/badge/segmentation-READY%20FOR%20ADVANCED%20ANALYTICS-brightgreen.svg)]()
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
5. **Activity 7 (Machine Learning Challenge):** Leakage-free 80/20 stratified supervised binary classification (`fire_ignition_48h`) benchmarking **Naive Bayes (`GaussianNB`)**, **Support Vector Machine (`SVC` RBF, `class_weight='balanced'`)**, and a **basic feedforward Artificial Neural Network (`MLPClassifier` $64 \times 32$ ReLU)** across both the **3D Orthogonal PCA Subspace (`PC1, PC2, PC3`, $\text{VIF} = 1.00$)** and the **12D Standardized Physical Feature Space**, with serialized models in `models/*.pkl` and visual analytics dashboards `fig28`–`fig33`.
6. **Activity 8 (Intelligent Segmentation Lab):** Unsupervised $K$-Means clustering (`KMeans(n_clusters=4, init='k-means++', n_init=20, random_state=42)`) operating strictly without target labels in the **3D Orthogonal PCA Subspace (`['PC1', 'PC2', 'PC3']`, $87.87\%$ variance retained, $\text{VIF} = 1.00$)**, evaluating candidate $K \in \{2..8\}$ via Elbow WCSS and Silhouette Score ($\bar{s} = 0.3039$, $+21.7\%$ over 12D raw space), discovering 4 operational pyro-climatic regimes, and exporting `models/kmeans_segmentation_model.pkl` and dashboards `fig34`–`fig37`.

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

## 🔬 3. Variables Analyzed & Statistical / Machine Learning Techniques (Activities 5, 6, 7 & 8)

### Variables Analyzed
- **Target Variable (1):** `fire_ignition_48h` (Binary 48–72h wildfire emergence; strictly excluded from PCA and unsupervised $K$-Means clustering to prevent Target Leakage, and evaluated *post-hoc* for segment validation).
- **High-Impact Explanatory Variables (7 Quantitative + 1 Categorical):** `dist_agri_m` (Agricultural parcel distance), `dist_roads_m` (Road network distance), `temp_2m_max_c` (Adiabatically downscaled max temperature), `rh_2m_min_pct` (Min relative humidity), `soil_moist_mean` (Upper 0–7 cm volumetric soil moisture), `vpd_max_kpa` (Daily peak Vapor Pressure Deficit), `ndmi` (Sentinel-2 Canopy Moisture Index), and `fuel_type` (Ecological fuel stratum).
- **Medium-Impact Covariates & Confounders (5 in Core PCA):** `slope_deg`, `elevation_m`, `vpd_72h_max_kpa`, `precip_72h_sum_mm`, `days_since_rain` (plus `savi` pruned due to collinearity with `ndmi`).
- **Low-Impact / Redundant Variables Pruned (5):** `nbr` ($\text{VIF} > 100$ collinear with `ndmi`), `wind_v10_mean` (statistically non-significant noise, $p = 0.621$), `wind_speed_max_kmh`, `wind_u10_mean`, and `aspect_rad`.

### Statistical, Supervised & Unsupervised Techniques Implemented
1. **Parametric & Non-Parametric Correlation Analysis:** Pearson linear correlation ($r$), Point-Biserial correlation ($r_{pb}$), and Spearman rank-order correlation ($\rho$) with two-tailed asymptotic significance tests.
2. **Simple Linear Regression Modeling (`statsmodels.api.OLS`):** Continuous physical system regressions ($N = 18,275$) and vigintile empirical probability regressions ($N = 20\text{ quantiles}$).
3. **Principal Component Analysis & Dimensionality Reduction (`StandardScaler` + `PCA(n_components=3)`):** Exploratory eigenvalue spectrum verification (`pca = PCA()`) followed by explicit truncation (`pca_reduced = PCA(n_components=3)`) under the Kaiser Criterion ($\lambda_k \ge 1.0$), generating `X_pca_reduced` ($18,275 \times 3$), Hotelling's $T^2$ Mahalanobis distance, and Q-residuals (SPE).
4. **Supervised Binary Classification (`GaussianNB`, `SVC` RBF, `MLPClassifier` $64 \times 32$):** Leakage-free 80/20 stratified training, 5-Fold Stratified Cross-Validation, Platt probability calibration, out-of-fold threshold optimization ($\tau^*$ and $\tau_{\text{sens}}$), and 10-repeat permutation importance.
5. **Unsupervised Intelligent Segmentation (`KMeans` with `k-means++`):** Candidate evaluation across $K \in \{2..8\}$ in 3D orthogonal PCA space vs. 12D standardized space using Within-Cluster Sum of Squares (WCSS / Inertia), Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index, 2D/3D centroid cartography, and post-hoc contingency $\chi^2$ / ANOVA validation.

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
├── models/                              # Serialized Supervised & Unsupervised ML Models
│   ├── naive_bayes_model.pkl
│   ├── svm_rbf_model.pkl
│   ├── mlp_ann_model.pkl
│   ├── activity7_metrics_summary.json
│   ├── kmeans_segmentation_model.pkl
│   └── activity8_segmentation_summary.json
├── metadata/                            # Audit registries, cleaning logs, and validation matrices
├── notebooks/                           # Executable analytical notebooks
│   ├── data_engineering_lab.ipynb       # Activity 3: Interactive E2E engineering and ETL notebook
│   ├── data_intelligence_lab.ipynb      # Activity 4: Descriptive statistics and visual analytics
│   ├── variable_impact_analysis.ipynb   # Activity 5: Correlation, regression & impact assessment
│   ├── pattern_discovery_challenge.ipynb# Activity 6: Multivariate PCA (12->3), biplots & pattern discovery
│   ├── machine_learning_challenge.ipynb # Activity 7: Supervised ML (Naive Bayes, SVM, MLP) & Evaluation
│   └── intelligent_segmentation_lab.ipynb # Activity 8: Unsupervised K-Means Segmentation & Validation
├── reports/                             # Publication-grade analytical reports & dashboards
│   ├── Activity 5. Variable Impact Analysis.md
│   ├── Activity 6. Pattern Discovery Challenge.md
│   ├── Activity 7. Machine Learning Challenge.md
│   ├── Activity 8. Intelligent Segmentation Lab.md
│   └── figures/                         # 37 high-resolution (300 DPI) visual analytics figures
│       ├── fig01_target_distribution.png ... fig12_topographic_niche.png (Activity 4)
│       ├── fig13_correlation_heatmaps.png ... fig20_variable_impact_ranking.png (Activity 5)
│       ├── fig21_pca_explained_variance.png ... fig27_pca_outliers_hotelling.png (Activity 6)
│       ├── fig28_ml_dataset_partition_and_feature_strategy.png ... fig33_ml_model_comparison_and_importance.png (Activity 7)
│       └── fig34_kmeans_elbow_and_silhouette.png ... fig37_cluster_distributions_and_domain_characterization.png (Activity 8)
├── src/                                 # Modular Python packages
│   ├── extract.py, clean.py, transform.py, integrate.py, validate.py, pipeline.py
│   ├── data_intelligence_analysis.py    # Activity 4 statistical engine & dashboard generator
│   ├── variable_impact_analysis.py      # Activity 5 correlation, OLS regression & VIF engine
│   ├── pattern_discovery_challenge.py   # Activity 6 multivariate PCA (12->3) & pattern discovery engine
│   ├── machine_learning_challenge.py    # Activity 7 supervised classification, CV & visual analytics engine
│   └── intelligent_segmentation_lab.py  # Activity 8 unsupervised K-Means segmentation & centroid engine
├── Work/                                # Official Deliverable Markdown Reports & Figures
│   ├── Activity 5. Variable Impact Analysis.md ... Activity 8. Intelligent Segmentation Lab.md
│   ├── variable_impact_analysis.ipynb ... intelligent_segmentation_lab.ipynb
│   ├── variable_impact_analysis.py ... intelligent_segmentation_lab.py
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
- `scipy>=1.12.0`, `statsmodels>=0.14.0`, `scikit-learn>=1.4.0`, `joblib>=1.3.0`
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

# 6. Run Activity 7 (Machine Learning Challenge)
python src/machine_learning_challenge.py

# 7. Run Activity 8 (Intelligent Segmentation Lab)
python src/intelligent_segmentation_lab.py

# 8. Launch Interactive Jupyter Notebooks
jupyter lab notebooks/intelligent_segmentation_lab.ipynb
```

---

## 🧠 6. Main Findings (Activities 4, 5, 6, 7 & 8)

1. **Anthropogenic Proximity is the Dominant Rank Predictor of Ignition:** `dist_agri_m` exhibits the strongest rank correlation with `fire_ignition_48h` ($\rho = -0.2155, p = 6.36 \times 10^{-191}$, Cohen's $d = -0.993$), followed by `dist_roads_m` ($\rho = -0.2063$), proving escaped agricultural burns (*roza y quema* under NOM-015) are the primary wildfire trigger.
2. **High Explanatory Power in Quantile Ignition Probability Regressions:** Across 20 equal-frequency quantiles ($N = 20$), simple OLS regressions prove that `temp_2m_max_c` ($R^2 = 0.8930$), `rh_2m_min_pct` ($R^2 = 0.8232$), `vpd_max_kpa` ($R^2 = 0.7816$), and `soil_moist_mean` ($R^2 = 0.7803$) linearly explain **78% to 89% of the variance in 48–72h wildfire ignition probability**.
3. **High-Efficiency PCA Dimensionality Reduction (Activity 6):** Applying `PCA(n_components=3)` to the 12 standardized High/Medium-Impact spatiotemporal variables compresses the feature space into the reduced matrix `X_pca_reduced` ($18,275 \times 3$, columns `['PC1', 'PC2', 'PC3']`), achieving a **75.0% reduction in dimensions while retaining 87.87% of total variance** ($\text{VIF} = 1.00$):
   - **$\text{PC1}$ ($40.76\%, \lambda_1 = 4.8918$):** *Atmospheric-Edaphic Desiccation & Drought Stress Continuum* ($d = +0.760$).
   - **$\text{PC2}$ ($33.41\%, \lambda_2 = 4.0088$):** *Orographic Altitude vs. Anthropogenic Agricultural Valley Interface* ($d = -0.738$).
   - **$\text{PC3}$ ($13.70\%, \lambda_3 = 1.6440$):** *Vegetative Canopy Phenology & Warm Monsoon Hydration Regime*.
4. **Discovery of the Quadrant IV Pyro-Anthropogenic Convergence Zone ($28.1\times$ Risk Amplification):** In the reduced $(\text{PC1}, \text{PC2})$ plane, **Quadrant IV ($\text{PC1} > 0 \land \text{PC2} \le 0$)** contains $28.50\%$ of observations but concentrates **68.61% of all regional wildfires ($601 / 876$)** at an **11.54% ignition rate** ($28.1\times$ higher than Quadrant II's $0.41\%$), reaching **23.34%** in the **Extreme Apex ($\text{PC1} > 2.5 \land \text{PC2} < -1.0$)**.
5. **Supervised Machine Learning Benchmark & Model Selection (Activity 7):**
   - Orthogonal PCA ($p=3, \text{VIF}=1.00$) rescues `GaussianNB` from 12D multicollinearity ($\text{VIF}=80.05$), cutting Brier calibration error by **80.5% ($0.2212 \to 0.0430$)** and boosting $\text{PR-AUC}$ by **+25.2% ($0.1269 \to 0.1589$)**.
   - **Selected Best Model (`MLPClassifier` $64 \times 32$ ReLU):** Achieves **Test $\text{ROC-AUC} = 0.8205$, $\text{PR-AUC} = 0.1715$, $\text{Accuracy} = 89.22\%$** in 3D PCA space ($1.39\text{ ms}$ inference), and **Test $\text{ROC-AUC} = 0.9049$ ($5\text{-Fold CV } 0.8946 \pm 0.0166$), $\text{PR-AUC} = 0.2839$ ($5.93\times\text{ lift}$), $\text{F1-Score} = 0.3974$, $\text{Accuracy} = 92.45\%$** in 12D space ($3.00\text{ ms}$ inference).
   - **Co-Selected Screening Companion (`SVC` RBF, `class_weight='balanced'`):** Achieves the lowest Brier calibration error (**$0.0415\text{–}0.0427$**) and captures **$77.71\%\text{ to }79.43\%$** of all test wildfires 48–72h prior to ignition.
6. **Unsupervised K-Means Intelligent Segmentation & Dual-Regime Hazard Discovery (Activity 8):**
   - Clustering in the orthogonal 3D PCA subspace ($\text{VIF} = 1.00$) eliminates 12D collinear Euclidean distance distortion ($\text{Max VIF} = 80.10$), lifting the $K^* = 4$ Silhouette Score by **+21.7% ($0.2496 \to 0.3039$)** while achieving the **global minimum Davies-Bouldin Index ($1.0474$)**, **global maximum Calinski-Harabasz Index ($9,844.88$)**, and **$61.78\%$ explained 3D variance ($\text{WCSS} = 73,645.77$)**.
   - Unsupervised $K$-Means ($K^* = 4$) reveals that regional wildfire risk bifurcates into **Cluster 2 (*Extreme Spring-Summer Heatwave & Drought Desiccation Apex*: $N = 4,695 [25.69\%], \text{PC1} = +2.66, T_{\max} = 28.62^\circ\text{C}, \text{VPD} = 2.89\text{ kPa}$, capturing $44.29\%$ of all fires [$388$] at an $8.26\%$ hazard rate)** and **Cluster 0 (*Dormant Cured-Biomass Agricultural Valley*: $N = 5,815 [31.82\%], \text{PC3} = -1.19, \text{NDMI} = -0.007, T_{\max} = 22.17^\circ\text{C}$, capturing $38.93\%$ of all fires [$341$] at a $5.86\%$ hazard rate)**. Together, Clusters 2 and 0 account for **83.22% of all regional wildfires ($729 / 876$)**, while **Cluster 3 (*Warm Monsoon Hydrated Valley*: $N = 4,723 [25.84\%], \text{Precip}_{72\text{h}} = 14.46\text{ mm}, 2.69\%$ hazard)** demonstrates monsoon suppression inside agricultural valleys, and **Cluster 1 (*High-Altitude Sierra Fría Sanctuary*: $N = 3,042 [16.65\%], \text{PC2} = +3.30, 100\%$ temperate *Bosque de Pino / Pino-Encino*, $0.66\%$ hazard)** is recovered purely from unlabeled coordinates.
7. **Readiness Classification:** **`READY FOR ADVANCED ANALYTICS`**.

---

## 🗺️ 7. Project Roadmap & Upcoming Steps
1. **Activity 3 (Completed):** Data Engineering Lab — ETL pipeline, Parquet Feature Store, and 7-gate validation suite (**READY FOR ANALYSIS**).
2. **Activity 4 (Completed):** Data Intelligence Lab — Statistical profiling, dispersion analysis, bivariate group contrasts, and visual dashboards (**READY FOR ADVANCED ANALYSIS**).
3. **Activity 5 (Completed):** Variable Impact Analysis — Correlation matrices, OLS regression modeling, VIF multicollinearity diagnostics, confounding analysis, and Variable Impact Ranking (**READY FOR PATTERN DISCOVERY**).
4. **Activity 6 (Completed):** Pattern Discovery Challenge — Multivariate 12-to-3 PCA dimensionality reduction (`X_pca_reduced` $18,275 \times 3$), 2D/3D biplots, seasonal hysteresis loop, and Hotelling's $T^2$ outlier diagnostics (`fig21`–`fig27`, **READY FOR SEGMENTATION**).
5. **Activity 7 (Completed):** Machine Learning Challenge — Supervised classification (`GaussianNB`, `SVC` RBF, `MLPClassifier` $64 \times 32$) across PCA 3D & Original 12D, serialized models (`models/*.pkl`), and decision boundary cartography (`fig28`–`fig33`, **READY FOR MODEL VALIDATION**).
6. **Activity 8 (Completed):** Intelligent Segmentation Lab — Unsupervised $K$-Means ($K^* = 4$, `k-means++`) in 3D orthogonal PCA space ($\text{WCSS} = 73,645.77, \text{Silhouette} = 0.3039$), 4 pyro-climatic regime profiles, serialized bundle (`models/kmeans_segmentation_model.pkl`), and centroid dashboards (`fig34`–`fig37`, **READY FOR ADVANCED ANALYTICS**).
7. **Upcoming Stages:** *Model Performance Validation* (5-Fold Spatially Blocked `GroupKFold` by `voxel_id` & regime-stratified calibration), *Explainable AI Workshop* (SHAP/LIME), *Deep Learning Innovation Lab* (Temporal 1D-CNN/LSTM), and *Predictive Intelligence Engine*.

---

## 👥 Academic Credits
- **Angel Emilio Gutierrez Lozano** (ID: 4341)
- **Emiliano Betanzos Valtierra** (ID: 4795)
- **Docente Asesor:** Dr. Sergio Valadez Godínez
- **Carrera:** Ingeniería en Tecnologías de la Información e Innovación Digital (`LITD9A`)
- **Universidad Tecnológica El Retoño** — 2026
