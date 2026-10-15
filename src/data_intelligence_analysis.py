"""
==============================================================================
TERRA-FIRE: Predictive Wildfire Intelligence Engine
Activity 4: Data Intelligence Lab - Comprehensive Statistical Analysis & Visual Analytics
Authors: Angel Emilio Gutierrez Lozano (ID: 4341) & Emiliano Betanzos Valtierra (ID: 4795)
Course: Data Science (LITD9A) - Universidad Tecnologica El Retono
Advisor: Dr. Sergio Valadez Godinez
Term: September - December 2026
==============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for publication-ready visual analytics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14

PRIMARY_DATA_PATH = "/home/eduardo/Workspaces/TERRA-FIRE/TERRA-FIRE/data/processed/terra_fire_spatiotemporal_matrix_v1.parquet"
OUTPUT_FIG_DIR_1 = "/home/eduardo/Downloads/Terra-fire/Work/figures"
OUTPUT_FIG_DIR_2 = "/home/eduardo/Workspaces/TERRA-FIRE/TERRA-FIRE/reports/figures"

os.makedirs(OUTPUT_FIG_DIR_1, exist_ok=True)
os.makedirs(OUTPUT_FIG_DIR_2, exist_ok=True)

def load_dataset(parquet_path=PRIMARY_DATA_PATH):
    """Loads the analysis-ready Parquet dataset and performs initial verification."""
    if not os.path.exists(parquet_path):
        # Fallback to local sample if processed matrix is not found
        sample_path = "/home/eduardo/Workspaces/TERRA-FIRE/TERRA-FIRE/data/sample/terra_fire_matrix_sample.csv"
        print(f"Warning: Primary dataset not found at {parquet_path}. Falling back to {sample_path}")
        df = pd.read_csv(sample_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    else:
        df = pd.read_parquet(parquet_path)
    print(f"[LOAD] Successfully loaded dataset with {df.shape[0]:,} rows and {df.shape[1]} columns.")
    return df

def compute_dataset_profile(df):
    """Generates the high-level dataset structural profile."""
    profile = {
        "Total Observations": len(df),
        "Total Variables": len(df.columns),
        "Quantitative Variables": len(df.select_dtypes(include=[np.number]).columns),
        "Qualitative Variables": len(df.select_dtypes(include=['category', 'object', 'string']).columns),
        "Temporal Variables": len(df.select_dtypes(include=['datetime64', 'datetime', 'datetimetz']).columns),
        "Geographic Variables": sum(col in ['elevation_m', 'slope_deg', 'aspect_rad', 'dist_roads_m', 'dist_agri_m', 'voxel_id'] for col in df.columns),
        "Remaining Missing Values": df.isnull().sum().sum(),
        "Date Range Start": df['timestamp'].min().strftime('%Y-%m-%d'),
        "Date Range End": df['timestamp'].max().strftime('%Y-%m-%d'),
        "Unique Monitoring Voxels": df['voxel_id'].nunique(),
        "Memory Usage (MB)": df.memory_usage(deep=True).sum() / (1024 * 1024)
    }
    return profile

def compute_central_tendency(df, quantitative_cols):
    """Computes mean, median, mode, and skewness for quantitative attributes."""
    records = []
    for col in quantitative_cols:
        mean_val = df[col].mean()
        median_val = df[col].median()
        # Mode might return multiple values; take the primary mode
        mode_series = df[col].mode()
        mode_val = mode_series.iloc[0] if not mode_series.empty else np.nan
        skew_val = df[col].skew()
        
        # Determine asymmetry direction
        if abs(skew_val) < 0.2:
            skew_desc = "Simetrica (Normalizada)"
        elif skew_val > 0.2:
            skew_desc = f"Sesgo Positivo / Derecha ({skew_val:.2f})"
        else:
            skew_desc = f"Sesgo Negativo / Izquierda ({skew_val:.2f})"
            
        records.append({
            "Variable": col,
            "Mean": mean_val,
            "Median": median_val,
            "Mode": mode_val,
            "Skewness": skew_val,
            "Distribution_Shape": skew_desc
        })
    return pd.DataFrame(records)

def compute_dispersion(df, quantitative_cols):
    """Computes minimum, maximum, range, variance, standard deviation, and IQR."""
    records = []
    for col in quantitative_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        range_val = max_val - min_val
        var_val = df[col].var()
        std_val = df[col].std()
        q1 = df[col].quantile(0.25)
        q2 = df[col].quantile(0.50)
        q3 = df[col].quantile(0.75)
        iqr_val = q3 - q1
        cv = (std_val / mean_val * 100) if (mean_val := df[col].mean()) != 0 else np.nan
        
        records.append({
            "Variable": col,
            "Min": min_val,
            "Max": max_val,
            "Range": range_val,
            "Variance": var_val,
            "StdDev": std_val,
            "Q1_25%": q1,
            "Q2_50%": q2,
            "Q3_75%": q3,
            "IQR": iqr_val,
            "CV_Pct": cv
        })
    return pd.DataFrame(records)

def compute_comparative_stats(df, features):
    """Calculates grouped statistics contrasting fire ignition (1) versus baseline (0)."""
    grouped_mean = df.groupby('fire_ignition_48h')[features].mean().T
    grouped_median = df.groupby('fire_ignition_48h')[features].median().T
    grouped_std = df.groupby('fire_ignition_48h')[features].std().T
    
    comparative = pd.DataFrame({
        "Mean_NonIgnition_0": grouped_mean[0],
        "Mean_Ignition_1": grouped_mean[1],
        "Median_NonIgnition_0": grouped_median[0],
        "Median_Ignition_1": grouped_median[1],
        "Std_NonIgnition_0": grouped_std[0],
        "Std_Ignition_1": grouped_std[1],
    })
    
    # Calculate percentage shift in mean
    comparative["Shift_Pct"] = ((comparative["Mean_Ignition_1"] - comparative["Mean_NonIgnition_0"]) / 
                                np.abs(comparative["Mean_NonIgnition_0"])) * 100
    
    # Statistical significance t-test
    p_values = []
    for feat in features:
        group0 = df[df['fire_ignition_48h'] == 0][feat].dropna()
        group1 = df[df['fire_ignition_48h'] == 1][feat].dropna()
        t_stat, p_val = stats.ttest_ind(group0, group1, equal_var=False)
        p_values.append(p_val)
    comparative["p_value"] = p_values
    comparative["Stat_Significant"] = comparative["p_value"] < 0.001
    
    return comparative

def save_and_duplicate(fig, filename):
    """Saves the figure in both target output directories."""
    path1 = os.path.join(OUTPUT_FIG_DIR_1, filename)
    path2 = os.path.join(OUTPUT_FIG_DIR_2, filename)
    fig.savefig(path1, dpi=300, bbox_inches='tight')
    fig.savefig(path2, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[CHART] Generated: {filename}")

def generate_distribution_charts(df):
    """Generates Phase 3 Step 5 distribution charts for individual variables."""
    palette = sns.color_palette("deep")

    # 1. Target Variable Class Distribution
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    counts = df['fire_ignition_48h'].value_counts()
    labels = ['No Incendio (0)', 'Ignicion 48-72h (1)']
    colors = ['#2b5c8f', '#d95f02']
    
    ax[0].bar(labels, counts, color=colors, edgecolor='black', width=0.5)
    ax[0].set_title('Distribucion de Clases: Ventana de Ignicion 48-72h', fontweight='bold')
    ax[0].set_ylabel('Numero de Observaciones')
    for i, v in enumerate(counts):
        ax[0].text(i, v + 250, f"{v:,} ({v/len(df)*100:.2f}%)", ha='center', fontweight='bold')
        
    ax[1].pie(counts, labels=labels, autopct='%1.2f%%', startangle=140, colors=colors,
              explode=(0, 0.15), shadow=True, textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax[1].set_title('Desbalance de Clases en Deteccion de Ignicion', fontweight='bold')
    plt.tight_layout()
    save_and_duplicate(fig, 'fig01_target_distribution.png')

    # 2. Atmospheric Desiccation (VPD and 72h VPD)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.histplot(df['vpd_max_kpa'], kde=True, ax=axes[0, 0], color='#d95f02', bins=40)
    axes[0, 0].set_title('Distribucion del Deficit de Presion de Vapor Diario (VPD Max)', fontweight='bold')
    axes[0, 0].set_xlabel('VPD Maximo Diario (kPa)')
    axes[0, 0].set_ylabel('Frecuencia')
    axes[0, 0].axvline(df['vpd_max_kpa'].mean(), color='blue', linestyle='--', label=f'Media: {df["vpd_max_kpa"].mean():.2f}')
    axes[0, 0].axvline(df['vpd_max_kpa'].median(), color='green', linestyle='-', label=f'Mediana: {df["vpd_max_kpa"].median():.2f}')
    axes[0, 0].legend()

    sns.boxplot(x=df['vpd_max_kpa'], ax=axes[0, 1], color='#e6ab02')
    axes[0, 1].set_title('Diagrama de Caja: VPD Maximo Diario', fontweight='bold')
    axes[0, 1].set_xlabel('VPD Maximo (kPa)')

    sns.histplot(df['vpd_72h_max_kpa'], kde=True, ax=axes[1, 0], color='#7570b3', bins=40)
    axes[1, 0].set_title('Distribucion del VPD Antecedente 72 Horas (VPD 72h Max)', fontweight='bold')
    axes[1, 0].set_xlabel('VPD Maximo 72h (kPa)')
    axes[1, 0].set_ylabel('Frecuencia')
    axes[1, 0].axvline(df['vpd_72h_max_kpa'].mean(), color='blue', linestyle='--', label=f'Media: {df["vpd_72h_max_kpa"].mean():.2f}')
    axes[1, 0].axvline(df['vpd_72h_max_kpa'].median(), color='green', linestyle='-', label=f'Mediana: {df["vpd_72h_max_kpa"].median():.2f}')
    axes[1, 0].legend()

    sns.boxplot(x=df['vpd_72h_max_kpa'], ax=axes[1, 1], color='#1b9e77')
    axes[1, 1].set_title('Diagrama de Caja: VPD 72h Maximo', fontweight='bold')
    axes[1, 1].set_xlabel('VPD 72h (kPa)')
    plt.tight_layout()
    save_and_duplicate(fig, 'fig02_vpd_distributions.png')

    # 3. Microclimatic Surface Dynamics (Temperature & Relative Humidity)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.histplot(df['temp_2m_max_c'], kde=True, ax=axes[0, 0], color='#e41a1c', bins=35)
    axes[0, 0].set_title('Distribucion de Temperatura Maxima a 2m (Ajuste Adiabatico)', fontweight='bold')
    axes[0, 0].set_xlabel('Temperatura Maxima (°C)')
    axes[0, 0].axvline(df['temp_2m_max_c'].mean(), color='black', linestyle='--', label=f'Media: {df["temp_2m_max_c"].mean():.1f}°C')
    axes[0, 0].legend()

    sns.boxplot(x=df['temp_2m_max_c'], ax=axes[0, 1], color='#fc8d62')
    axes[0, 1].set_title('Dispersion y Outliers: Temperatura Maxima (°C)', fontweight='bold')
    axes[0, 1].set_xlabel('Temperatura Maxima (°C)')

    sns.histplot(df['rh_2m_min_pct'], kde=True, ax=axes[1, 0], color='#377eb8', bins=35)
    axes[1, 0].set_title('Distribucion de Humedad Relativa Minima Diaria', fontweight='bold')
    axes[1, 0].set_xlabel('Humedad Relativa Minima (%)')
    axes[1, 0].axvline(df['rh_2m_min_pct'].mean(), color='black', linestyle='--', label=f'Media: {df["rh_2m_min_pct"].mean():.1f}%')
    axes[1, 0].legend()

    sns.boxplot(x=df['rh_2m_min_pct'], ax=axes[1, 1], color='#8da0cb')
    axes[1, 1].set_title('Dispersion: Humedad Relativa Minima (%)', fontweight='bold')
    axes[1, 1].set_xlabel('Humedad Relativa (%)')
    plt.tight_layout()
    save_and_duplicate(fig, 'fig03_temp_humidity_distributions.png')

    # 4. Biophysical Remote Sensing Indices (NDMI, SAVI, NBR)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    sns.histplot(df['ndmi'], kde=True, ax=axes[0], color='#2b83ba', bins=30)
    axes[0].set_title('Indice de Humedad de Canopia (NDMI)', fontweight='bold')
    axes[0].set_xlabel('NDMI [-1, 1]')
    axes[0].axvline(df['ndmi'].mean(), color='red', linestyle='--', label=f'Media: {df["ndmi"].mean():.3f}')
    axes[0].legend()

    sns.histplot(df['savi'], kde=True, ax=axes[1], color='#4dac26', bins=30)
    axes[1].set_title('Indice de Vegetacion Ajustado a Suelo (SAVI)', fontweight='bold')
    axes[1].set_xlabel('SAVI [-1, 1]')
    axes[1].axvline(df['savi'].mean(), color='red', linestyle='--', label=f'Media: {df["savi"].mean():.3f}')
    axes[1].legend()

    sns.histplot(df['nbr'], kde=True, ax=axes[2], color='#fdae61', bins=30)
    axes[2].set_title('Indice de Quema Normalizado (NBR)', fontweight='bold')
    axes[2].set_xlabel('NBR [-1, 1]')
    axes[2].axvline(df['nbr'].mean(), color='red', linestyle='--', label=f'Media: {df["nbr"].mean():.3f}')
    axes[2].legend()
    plt.tight_layout()
    save_and_duplicate(fig, 'fig04_biophysical_indices.png')

    # 5. Topography and Anthropogenic Distances
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.histplot(df['elevation_m'], kde=True, ax=axes[0, 0], color='#8c510a', bins=25)
    axes[0, 0].set_title('Elevacion Topografica (Copernicus DEM)', fontweight='bold')
    axes[0, 0].set_xlabel('Elevacion (msnm)')

    sns.histplot(df['slope_deg'], kde=True, ax=axes[0, 1], color='#dfc27d', bins=25)
    axes[0, 1].set_title('Pendiente del Terreno (Grados)', fontweight='bold')
    axes[0, 1].set_xlabel('Pendiente (°)')

    sns.histplot(df['dist_roads_m'], kde=True, ax=axes[1, 0], color='#542788', bins=30)
    axes[1, 0].set_title('Distancia Euclideana a Red Vial (Metros)', fontweight='bold')
    axes[1, 0].set_xlabel('Distancia a Caminos (m)')

    sns.histplot(df['dist_agri_m'], kde=True, ax=axes[1, 1], color='#b35806', bins=30)
    axes[1, 1].set_title('Distancia Euclideana a Parcelas Agricolas (Metros)', fontweight='bold')
    axes[1, 1].set_xlabel('Distancia a Agricultura (m)')
    plt.tight_layout()
    save_and_duplicate(fig, 'fig05_topography_anthropogenic.png')

    # 6. Fuel Type Distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    ft_counts = df['fuel_type'].value_counts()
    sns.barplot(x=ft_counts.index, y=ft_counts.values, ax=ax, palette='viridis', edgecolor='black')
    ax.set_title('Distribucion de Tipos de Combustible Forestal y Cobertura', fontweight='bold')
    ax.set_ylabel('Frecuencia de Observaciones')
    ax.set_xlabel('Tipo de Estrato de Combustible')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha='right')
    for i, v in enumerate(ft_counts.values):
        ax.text(i, v + 50, f"{v:,} ({v/len(df)*100:.1f}%)", ha='center', fontweight='bold')
    plt.tight_layout()
    save_and_duplicate(fig, 'fig06_fuel_types.png')

def generate_comparative_charts(df):
    """Generates Phase 3 Step 6 comparative visualizations."""
    
    # 7. Multi-variable Biophysical & Climate Comparison by Ignition Status
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    features_to_plot = [
        ('vpd_max_kpa', 'VPD Maximo (kPa)', axes[0, 0]),
        ('vpd_72h_max_kpa', 'VPD 72h Maximo (kPa)', axes[0, 1]),
        ('temp_2m_max_c', 'Temperatura Maxima (°C)', axes[0, 2]),
        ('rh_2m_min_pct', 'Humedad Relativa Minima (%)', axes[1, 0]),
        ('soil_moist_mean', 'Humedad Volumetrica del Suelo (m³/m³)', axes[1, 1]),
        ('ndmi', 'Humedad de Canopia (NDMI)', axes[1, 2]),
    ]
    
    for feat, label, ax in features_to_plot:
        sns.boxplot(x='fire_ignition_48h', y=feat, data=df, ax=ax, 
                    palette=['#2b5c8f', '#d95f02'], notch=True)
        ax.set_title(f'Comparacion: {label}', fontweight='bold')
        ax.set_xlabel('Condicion de Ignicion (0 = Normal, 1 = Ignicion 48-72h)')
        ax.set_xticklabels(['Sin Fuego (0)', 'Ignicion (1)'])
        ax.set_ylabel(label)
        
    plt.tight_layout()
    save_and_duplicate(fig, 'fig07_comparative_boxplots.png')

    # 8. Anthropogenic Vector: Distance to Agriculture and Roads
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.kdeplot(data=df, x='dist_agri_m', hue='fire_ignition_48h', ax=axes[0],
                common_norm=False, fill=True, palette=['#2b5c8f', '#d95f02'], alpha=0.4)
    axes[0].set_title('Densidad de Probabilidad: Distancia a Parcelas Agricolas', fontweight='bold')
    axes[0].set_xlabel('Distancia a Agricultura (m)')
    axes[0].set_ylabel('Densidad')
    axes[0].axvline(df[df['fire_ignition_48h']==1]['dist_agri_m'].median(), color='#d95f02', 
                    linestyle='--', label=f'Mediana Ignicion: {df[df["fire_ignition_48h"]==1]["dist_agri_m"].median():.0f}m')
    axes[0].axvline(df[df['fire_ignition_48h']==0]['dist_agri_m'].median(), color='#2b5c8f', 
                    linestyle='--', label=f'Mediana Normal: {df[df["fire_ignition_48h"]==0]["dist_agri_m"].median():.0f}m')
    axes[0].legend()

    sns.kdeplot(data=df, x='dist_roads_m', hue='fire_ignition_48h', ax=axes[1],
                common_norm=False, fill=True, palette=['#2b5c8f', '#d95f02'], alpha=0.4)
    axes[1].set_title('Densidad de Probabilidad: Distancia a Red Vial', fontweight='bold')
    axes[1].set_xlabel('Distancia a Caminos (m)')
    axes[1].set_ylabel('Densidad')
    axes[1].axvline(df[df['fire_ignition_48h']==1]['dist_roads_m'].median(), color='#d95f02', 
                    linestyle='--', label=f'Mediana Ignicion: {df[df["fire_ignition_48h"]==1]["dist_roads_m"].median():.0f}m')
    axes[1].axvline(df[df['fire_ignition_48h']==0]['dist_roads_m'].median(), color='#2b5c8f', 
                    linestyle='--', label=f'Mediana Normal: {df[df["fire_ignition_48h"]==0]["dist_roads_m"].median():.0f}m')
    axes[1].legend()
    plt.tight_layout()
    save_and_duplicate(fig, 'fig08_anthropogenic_proximity.png')

    # 9. Fire Rate by Fuel Type
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    ft_stats = df.groupby('fuel_type', observed=False)['fire_ignition_48h'].agg(['count', 'sum', 'mean']).reset_index()
    ft_stats['rate_pct'] = ft_stats['mean'] * 100
    ft_stats['pct_total_ignitions'] = (ft_stats['sum'] / df['fire_ignition_48h'].sum()) * 100
    
    sns.barplot(x='fuel_type', y='rate_pct', data=ft_stats, ax=axes[0], palette='Reds_r', edgecolor='black')
    axes[0].set_title('Tasa de Incidencia de Ignicion dentro de Cada Combustible (%)', fontweight='bold')
    axes[0].set_ylabel('Tasa de Ignicion (%)')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=25, ha='right')
    for i, row in ft_stats.iterrows():
        axes[0].text(i, row['rate_pct'] + 0.3, f"{row['rate_pct']:.1f}%", ha='center', fontweight='bold')

    sns.barplot(x='fuel_type', y='pct_total_ignitions', data=ft_stats, ax=axes[1], palette='Oranges_r', edgecolor='black')
    axes[1].set_title('Contribucion Porcentual al Total de Igniciones Estatales (%)', fontweight='bold')
    axes[1].set_ylabel('Participacion en Total de Focos (%)')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=25, ha='right')
    for i, row in ft_stats.iterrows():
        axes[1].text(i, row['pct_total_ignitions'] + 1.0, f"{row['pct_total_ignitions']:.1f}%", ha='center', fontweight='bold')

    plt.tight_layout()
    save_and_duplicate(fig, 'fig09_fuel_type_comparative.png')

    # 10. Temporal Progression: Seasonality of VPD and Wildfires
    df_temp = df.copy()
    df_temp['month'] = df_temp['timestamp'].dt.month
    monthly = df_temp.groupby('month').agg({
        'fire_ignition_48h': 'sum',
        'vpd_max_kpa': 'mean',
        'temp_2m_max_c': 'mean',
        'rh_2m_min_pct': 'mean'
    }).reset_index()

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    months_labels = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    ax1.bar(monthly['month'] - 0.2, monthly['fire_ignition_48h'], width=0.4, color='#d95f02', 
            alpha=0.85, label='Eventos de Ignicion Totales', edgecolor='black')
    ax2.plot(monthly['month'] + 0.2, monthly['vpd_max_kpa'], color='#2b5c8f', 
             marker='o', linewidth=3, label='VPD Maximo Promedio (kPa)')

    ax1.set_xlabel('Mes del Año (Ciclo 2023-2024)', fontweight='bold')
    ax1.set_ylabel('Numero de Eventos de Ignicion Acumulados', color='#d95f02', fontweight='bold')
    ax2.set_ylabel('VPD Maximo Atmosferico (kPa)', color='#2b5c8f', fontweight='bold')
    ax1.set_xticks(range(1, 13))
    ax1.set_xticklabels(months_labels)
    ax1.set_title('Comportamiento Estacional: Coincidencia de Igniciones con el Pico de Desecacion (VPD)', fontweight='bold')
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    plt.tight_layout()
    save_and_duplicate(fig, 'fig10_seasonality_timeline.png')

    # 11. Bivariate Phase Space: VPD vs Relative Humidity
    fig, ax = plt.subplots(figsize=(10, 6))
    sample_sub = df.sample(n=3500, random_state=42)
    sns.scatterplot(x='rh_2m_min_pct', y='vpd_max_kpa', hue='fire_ignition_48h', data=sample_sub,
                    palette=['#2b5c8f', '#d95f02'], alpha=0.7, ax=ax, s=35, edgecolor=None)
    ax.set_title('Espacio de Fase Termodinamico: VPD vs Humedad Relativa Minima', fontweight='bold')
    ax.set_xlabel('Humedad Relativa Minima Diaria (%)')
    ax.set_ylabel('Deficit de Presion de Vapor Maximo (kPa)')
    ax.axhline(2.0, color='red', linestyle='--', label='Umbral Critico de Succion Atmosferica (VPD > 2.0 kPa)')
    ax.axvline(20.0, color='orange', linestyle='--', label='Regla del 30-30-30: RH < 20%')
    ax.legend(title='Condicion de Ignicion', loc='upper right')
    plt.tight_layout()
    save_and_duplicate(fig, 'fig11_thermodynamic_phase_space.png')

    # 12. Topographic Niche: Elevation vs Slope by Ignition
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='elevation_m', y='slope_deg', hue='fire_ignition_48h', data=sample_sub,
                    palette=['#2b5c8f', '#d95f02'], alpha=0.7, ax=ax, s=35)
    ax.set_title('Nicho Topografico de Ignicion: Elevacion vs Pendiente', fontweight='bold')
    ax.set_xlabel('Elevacion sobre el Nivel del Mar (msnm)')
    ax.set_ylabel('Pendiente del Terreno (Grados)')
    ax.legend(title='Condicion de Ignicion', loc='upper right')
    plt.tight_layout()
    save_and_duplicate(fig, 'fig12_topographic_niche.png')

def main():
    print("=" * 80)
    print("TERRA-FIRE: DATA INTELLIGENCE LAB EXECUTION")
    print("=" * 80)
    
    df = load_dataset()
    
    quant_cols = [
        'ndmi', 'savi', 'nbr', 'vpd_max_kpa', 'vpd_72h_max_kpa', 
        'temp_2m_max_c', 'rh_2m_min_pct', 'wind_u10_mean', 'wind_v10_mean', 
        'wind_speed_max_kmh', 'soil_moist_mean', 'days_since_rain', 
        'precip_72h_sum_mm', 'elevation_m', 'slope_deg', 'aspect_rad', 
        'dist_roads_m', 'dist_agri_m'
    ]
    
    print("\n--- 1. DATASET PROFILE ---")
    profile = compute_dataset_profile(df)
    for k, v in profile.items():
        print(f"  {k:30s}: {v}")
        
    print("\n--- 2. CENTRAL TENDENCY METRICS ---")
    ct_df = compute_central_tendency(df, quant_cols)
    print(ct_df.to_string(index=False))
    
    print("\n--- 3. DISPERSION METRICS ---")
    disp_df = compute_dispersion(df, quant_cols)
    print(disp_df.to_string(index=False))
    
    print("\n--- 4. COMPARATIVE ANALYSIS (IGNITION VS NON-IGNITION) ---")
    comp_df = compute_comparative_stats(df, quant_cols)
    print(comp_df.to_string())
    
    print("\n--- 5. GENERATING VISUAL ANALYTICS DASHBOARDS ---")
    generate_distribution_charts(df)
    generate_comparative_charts(df)
    
    print("\n[COMPLETE] All statistical analyses and 12 publication charts generated successfully.")

if __name__ == "__main__":
    main()
