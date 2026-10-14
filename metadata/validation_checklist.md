# ✅ Final Data Validation Checklist (TERRA-FIRE)

**Artifact Tested:** `data/processed/terra_fire_spatiotemporal_matrix_v1.parquet`  
**Execution Timestamp:** Automated validation suite executed via `src/validate.py`  
**Overall Validation Result:** **ALL 7 QUALITY GATES PASSED (100% SUCCESS)**

| Validation Check | Expected Condition | Empirical Result | Status |
| :--- | :--- | :--- | :---: |
| **1. Record Integrity** | Exactly 18,275 rows produced (25 monitoring voxels $\times$ 731 calendar days in 2023–2024). | 18,275 records verified. No truncation, row drop, or accidental loss occurred during pipeline execution. | **PASS** |
| **2. Duplicate Control** | Primary composite key uniqueness: 0 duplicate `(voxel_id, timestamp)` observations. | Exactly 0 duplicate pairs detected across the entire dataset. 100% key uniqueness guaranteed. | **PASS** |
| **3. Data Types Compliance** | Rigid embedded Arrow bit-width typing: `string`, `int32`, `float32`, `category`. | 22 out of 22 columns match schema specifications exactly without implicit object/string casting. | **PASS** |
| **4. Missing Values Control** | Zero unhandled or silent nulls ($NaN / None$) across all 22 feature attributes. | 0 null cells detected across all $18,275 \times 22 = 402,050$ matrix data points. | **PASS** |
| **5. Physical Range Boundaries** | All variables strictly respect thermodynamic and geospatial boundaries: $VPD \in [0, 10]$ kPa, Indices $\in [-1, 1]$, Slope $\in [0, 90]^\circ$, Target $\in \{0, 1\}$. | 100% of observations reside within physical laws. No negative moisture or out-of-bounds probabilities. | **PASS** |
| **6. Integration Consistency** | Unbroken temporal continuity across all 731 days of 2023 and 2024 leap year. | Seamless daily progression from 2023-01-01 to 2024-12-31 without temporal gaps. | **PASS** |
| **7. Variable Traceability** | Every feature must trace back to an audited primary data provider and explicit formula. | 100% traceability confirmed across 5 primary sources (CONAFOR, FIRMS, ERA5, Copernicus DEM, Sentinel-2). | **PASS** |

---

### Pipeline Quality Diagnostic Questions

1. **Did records disappear?**
   - *No.* The Cartesian product of 25 spatial voxels across 731 continuous days yields exactly 18,275 observations, completely preserved in the final Parquet file.
2. **Were records duplicated?**
   - *No.* Composite key evaluation `(voxel_id, timestamp)` reveals 0 duplicate occurrences.
3. **Did variable types change correctly?**
   - *Yes.* High-precision `float64` and generic objects were cast to memory-efficient `float32`, `int32`, and categorical encodings.
4. **Did integration introduce unexpected missing values?**
   - *No.* All spatial and temporal joins were fully resolved. Zero `NaN` values exist in the final matrix.
5. **Are transformation rules correctly reflected?**
   - *Yes.* The Tetens equation, vector decomposition, and adiabatic lapse-rate adjustments match theoretical physics.
6. **Can every final variable be traced to its source?**
   - *Yes.* Audited in `metadata/data_dictionary.md` and `metadata/transformation_dictionary.md`.
7. **Does the complete pipeline apply the same transformation logic validated using the sample?**
   - *Yes.* Shared functions in `src/clean.py`, `src/transform.py`, and `src/integrate.py` are executed identically on both full and sample data streams.
