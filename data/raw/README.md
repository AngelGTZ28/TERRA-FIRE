# 📦 Raw Data Storage Instructions (TERRA-FIRE)

Due to GitHub repository file size considerations and academic Data Engineering best practices, the full raw datasets (>72 MB combined) are preserved unmodified in the external storage directory (`TERRAFIRE-db` / Institutional Lakehouse) rather than committed directly to version control.

### Candidate Raw Datasets Inventory

| Dataset | Provider | Format | Records | Size | External Storage Path |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CONAFOR Incendios 2012–2025** | CONAFOR / SEMARNAT | `.xlsx` | 101,003 | 18.6 MB | `TERRAFIRE-db/CONAFOR_Incendios_2012_2025.xlsx` |
| **NASA FIRMS VIIRS S-NPP 2023** | NASA LANCE / FIRMS | `.csv` | 327,657 | 26.1 MB | `TERRAFIRE-db/NASA_FIRMS_VIIRS_SNPP_2023_Mexico.csv` |
| **NASA FIRMS VIIRS S-NPP 2024** | NASA LANCE / FIRMS | `.csv` | 330,134 | 26.3 MB | `TERRAFIRE-db/NASA_FIRMS_VIIRS_SNPP_2024_Mexico.csv` |
| **ECMWF ERA5-Land Hourly** | ECMWF / Copernicus C3S | `.csv` | 17,544 | 0.93 MB | `TERRAFIRE-db/ERA5_Land_Hourly_SierraFria_2023_2024.csv` |
| **Sentinel-2 L2A STAC Item** | ESA / Microsoft Planetary | `.json` | 1 scene | 0.03 MB | `TERRAFIRE-db/Sentinel2_L2A_STAC_Item_SierraFria_20240411.json` |

### How to obtain the raw data:
1. Place the acquired raw files inside `../TERRAFIRE-db/` or `data/raw/`.
2. For testing and lightweight reproduction, use the certified representative sample in `data/sample/`.
3. Execute the ETL pipeline using `python -m src.pipeline` or via `notebooks/data_engineering_lab.ipynb`.
