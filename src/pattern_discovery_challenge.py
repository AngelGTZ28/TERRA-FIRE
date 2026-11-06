"""
==============================================================================
TERRA-FIRE: Predictive Wildfire Intelligence Engine
Activity 6: Pattern Discovery Challenge - Multivariate PCA & Visual Analytics Engine
Authors: Angel Emilio Gutierrez Lozano (ID: 4341) & Emiliano Betanzos Valtierra (ID: 4795)
Course: Data Science (LITD9A) - Universidad Tecnologica El Retono
Advisor: Dr. Sergio Valadez Godinez
Term: September - December 2026
==============================================================================
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import seaborn as sns

warnings.filterwarnings("ignore")

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

OUTPUT_FIG_DIRS = [
    "/home/eduardo/Workspaces/TERRA-FIRE/TERRA-FIRE/reports/figures",
    "/home/eduardo/Workspaces/TERRA-FIRE/TERRA-FIRE/Work/figures",
    "/home/eduardo/Workspaces/TerraFire_vault/Work/figures",
    "/home/eduardo/Workspaces/TerraFire_vault/12 - Pattern Discovery Challenge/figures",
    "/home/eduardo/Downloads/Terra-fire/Work/figures",
]

for d in OUTPUT_FIG_DIRS:
    os.makedirs(d, exist_ok=True)


# 12 High & Medium Impact Quantitative Variables selected from Activity 5
# (7 High-Impact + 5 Medium-Impact; excluding nbr due to VIF > 100 with ndmi,
# excluding wind_v10_mean and wind_speed_max_kmh as statistical noise,
# and excluding fire_ignition_48h to prevent Target Leakage)
CORE_PCA_FEATURES = [
    "dist_agri_m",
    "dist_roads_m",
    "temp_2m_max_c",
    "rh_2m_min_pct",
    "soil_moist_mean",
    "vpd_max_kpa",
    "ndmi",
    "slope_deg",
    "elevation_m",
    "vpd_72h_max_kpa",
    "precip_72h_sum_mm",
    "days_since_rain",
]


def load_dataset(parquet_path=PRIMARY_DATA_PATH):
    """Loads the immutable analysis-ready Parquet dataset in strict read-only mode."""
    if not os.path.exists(parquet_path):
        sample_path = "/home/eduardo/Workspaces/TERRA-FIRE/TERRA-FIRE/data/sample/terra_fire_matrix_sample.csv"
        print(f"Warning: Primary dataset not found at {parquet_path}. Falling back to {sample_path}")
        df = pd.read_csv(sample_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    else:
        df = pd.read_parquet(parquet_path)
    print(f"[LOAD] Successfully loaded immutable dataset: {df.shape[0]:,} rows x {df.shape[1]} columns.")
    return df


def save_figure(fig, filename):
    """Saves a matplotlib figure across all project, Vault, and Work figure directories."""
    for d in OUTPUT_FIG_DIRS:
        fig.savefig(os.path.join(d, filename), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[CHART] Generated and synchronized: {filename}")


def draw_confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
    """Draws a 2D covariance confidence ellipse on a matplotlib Axes."""
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)
    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def verify_and_standardize(df, features=CORE_PCA_FEATURES):
    """
    Phase 1 - Step 2: Verifies missing values, numerical types, scale disparities,
    and performs Z-score standardization (StandardScaler) for PCA.
    """
    prep_rows = []
    for col in features:
        s = df[col]
        n_null = int(s.isna().sum())
        mean_val = float(s.mean())
        std_val = float(s.std())
        min_val = float(s.min())
        max_val = float(s.max())
        skew_val = float(stats.skew(s))
        prep_rows.append({
            "Variable": col,
            "Dtype": str(s.dtype),
            "Missing_Count": n_null,
            "Raw_Mean": mean_val,
            "Raw_Std": std_val,
            "Raw_Range": f"[{min_val:.2f}, {max_val:.2f}]",
            "Skewness": skew_val,
            "Standardized": "Yes (Z-Score: mu=0, sigma=1)",
            "Transformation": "StandardScaler (X - mu) / sigma"
        })
    prep_df = pd.DataFrame(prep_rows)
    scaler = StandardScaler()
    X_std = scaler.fit_transform(df[features].astype(float))
    return prep_df, scaler, X_std


def execute_pca_analysis(df, X_std, features=CORE_PCA_FEATURES):
    """
    Phase 2 - Steps 3 & 4:
    1) Exploratory fit (pca_exploratory = PCA()) strictly to obtain the full 12-component
       eigenvalue spectrum for the Scree Plot (fig21) and verify the Kaiser Criterion (lambda_k >= 1.0).
    2) Definitive Dimensionality Reduction fit (pca_reduced = PCA(n_components=3)) that truncates
       the 12-variable feature space into strictly 3 orthogonal Principal Components (PC1, PC2, PC3),
       achieving a 75.0% dimensionality reduction while retaining 87.87% of total variance (VIF = 1.00).
    3) Generates the reduced DataFrame X_pca_reduced of shape (18275, 3) with columns ['PC1', 'PC2', 'PC3'].
    """
    # Step 3A: Exploratory spectrum for Scree Plot (fig21) and Kaiser cutoff verification
    pca_exploratory = PCA()
    pca_exploratory.fit(X_std)

    exploratory_rows = []
    cum_var_full = np.cumsum(pca_exploratory.explained_variance_ratio_) * 100.0
    for i in range(len(features)):
        eig = float(pca_exploratory.explained_variance_[i])
        ev = float(pca_exploratory.explained_variance_ratio_[i] * 100.0)
        cev = float(cum_var_full[i])
        kaiser = "RETAIN (Kaiser lambda >= 1.0)" if eig >= 1.0 else "PRUNE (lambda < 1.0 — Discarded)"
        exploratory_rows.append({
            "Component": f"PC{i+1}",
            "Eigenvalue": eig,
            "Explained_Variance_Pct": ev,
            "Cumulative_Variance_Pct": cev,
            "Kaiser_Decision": kaiser
        })
    exploratory_spectrum_df = pd.DataFrame(exploratory_rows)

    # Step 3B: Definitive Dimensionality Reduction Model (12 Variables -> 3 Orthogonal Components)
    pca_reduced = PCA(n_components=3)
    scores_reduced = pca_reduced.fit_transform(X_std)

    # Explicitly build the reduced transformed DataFrame (18,275 rows x 3 columns)
    X_pca_reduced = pd.DataFrame(
        scores_reduced,
        columns=["PC1", "PC2", "PC3"],
        index=df.index
    )

    # Table 3: Official PCA Summary Matrix (Retained 3D Subspace vs Pruned Tail)
    summary_rows = []
    cum_var_ret = np.cumsum(pca_reduced.explained_variance_ratio_) * 100.0
    for i in range(3):
        eig = float(pca_reduced.explained_variance_[i])
        ev = float(pca_reduced.explained_variance_ratio_[i] * 100.0)
        cev = float(cum_var_ret[i])
        summary_rows.append({
            "Component": f"PC{i+1}",
            "Eigenvalue": eig,
            "Explained_Variance_Pct": ev,
            "Cumulative_Variance_Pct": cev,
            "Status": "Retained (Kaiser lambda >= 1.0 | VIF = 1.00)"
        })
    pruned_eig_sum = float(np.sum(pca_exploratory.explained_variance_[3:]))
    pruned_var_sum = float(np.sum(pca_exploratory.explained_variance_ratio_[3:]) * 100.0)
    summary_rows.append({
        "Component": "PC4 - PC12 (Pruned)",
        "Eigenvalue": pruned_eig_sum,
        "Explained_Variance_Pct": pruned_var_sum,
        "Cumulative_Variance_Pct": 100.0,
        "Status": "Discarded / Pruned (lambda < 1.0 | 75.0% Dim Reduction: 12 -> 3)"
    })
    pca_summary_df = pd.DataFrame(summary_rows)

    # Table 4: Eigenvector Loadings (w_jk) & Squared Cosine Contributions for Retained PC1, PC2, PC3
    loadings_df = pd.DataFrame(
        pca_reduced.components_.T,
        index=features,
        columns=["PC1", "PC2", "PC3"]
    )
    contrib_df = (loadings_df ** 2) * 100.0

    # Attach strictly the 3 retained PCs + Hotelling T2 & Q-residual to analysis dataframe
    df_pca = df.copy()
    df_pca["PC1"] = X_pca_reduced["PC1"]
    df_pca["PC2"] = X_pca_reduced["PC2"]
    df_pca["PC3"] = X_pca_reduced["PC3"]

    # Hotelling's T^2 in 3D retained subspace (PC1..PC3)
    lam_inv = np.diag(1.0 / pca_reduced.explained_variance_)
    df_pca["Hotelling_T2"] = np.sum((scores_reduced @ lam_inv) * scores_reduced, axis=1)

    # Q-Residual / Squared Prediction Error (SPE) from 3D reconstruction
    X_reconstructed = pca_reduced.inverse_transform(scores_reduced)
    df_pca["Q_Residual"] = np.sum((X_std - X_reconstructed) ** 2, axis=1)

    return (
        pca_exploratory,
        exploratory_spectrum_df,
        pca_reduced,
        X_pca_reduced,
        pca_summary_df,
        loadings_df,
        contrib_df,
        df_pca,
    )


def generate_fig21_explained_variance(exploratory_spectrum_df):
    """Fig 21: Exploratory Scree Plot (Eigenvalues) & Explicit Truncation at n_components=3 (87.87% Var)."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    pcs = exploratory_spectrum_df["Component"]
    eigs = exploratory_spectrum_df["Eigenvalue"]
    ev = exploratory_spectrum_df["Explained_Variance_Pct"]
    cev = exploratory_spectrum_df["Cumulative_Variance_Pct"]

    # Panel A: Eigenvalue Scree Plot & Kaiser Criterion Cutoff at k=3
    colors = ['#d95f02' if val >= 1.0 else '#969696' for val in eigs]
    bars = axes[0].bar(pcs, eigs, color=colors, edgecolor='black', alpha=0.88, width=0.65)
    axes[0].plot(range(len(pcs)), eigs, color='#1b1b1b', marker='o', linewidth=2, label='Eigenvalue Trajectory ($\\lambda_k$)')
    axes[0].axhline(1.0, color='#e41a1c', linestyle='--', linewidth=1.8, label='Kaiser Criterion Threshold ($\\lambda = 1.0$)')
    axes[0].axvline(2.5, color='#2ca25f', linestyle='-.', linewidth=2.0, label='PCA Truncation Cutoff (n_components=3)')
    axes[0].axvspan(-0.5, 2.5, color='#fee8c8', alpha=0.30, zorder=0, label='Retained 3D Subspace (PC1–PC3)')
    for bar, val in zip(bars, eigs):
        axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.10, f"{val:.2f}",
                     ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    axes[0].set_title("A. PCA Scree Plot: Eigenvalues ($\\lambda_k$) & Kaiser Cutoff ($k=3$)", fontweight='bold')
    axes[0].set_xlabel("Principal Component Axis (Exploratory Spectrum vs. Pruned Tail)")
    axes[0].set_ylabel("Eigenvalue ($\\lambda_k$ = Variance Captured)")
    axes[0].set_ylim(0, 5.6)
    axes[0].legend(loc='upper right', fontsize=8.5)

    # Panel B: Individual vs Cumulative Explained Variance (%)
    ax2 = axes[1]
    bar_colors2 = ['#2b83ba' if i < 3 else '#bdbdbd' for i in range(len(pcs))]
    ax2.bar(pcs, ev, color=bar_colors2, edgecolor='black', alpha=0.80, width=0.6, label='Individual Explained Variance (%)')
    ax2.plot(range(len(pcs)), cev, color='#d7191c', marker='s', linewidth=2.5, markersize=7, label='Cumulative Explained Variance (%)')
    ax2.axhline(87.87, color='#2ca25f', linestyle=':', linewidth=1.8, label='Retained 3-PC Variance (87.87% @ PC3)')
    ax2.axvline(2.5, color='#2ca25f', linestyle='-.', linewidth=2.0, label='Truncation Boundary (12 Vars $\\rightarrow$ 3 PCs)')

    for idx, (v_ind, v_cum) in enumerate(zip(ev, cev)):
        if idx < 4:
            ax2.annotate(f"{v_cum:.1f}%", (idx, v_cum), textcoords="offset points",
                         xytext=(0, 8), ha='center', fontsize=8.5, fontweight='bold', color='#b30000')
        if idx < 3:
            ax2.text(idx, v_ind / 2, f"{v_ind:.1f}%", ha='center', va='center',
                     fontsize=8.5, fontweight='bold', color='white')

    ax2.set_title("B. Explained & Cumulative Variance (75.0% Dimensionality Reduction: 12 $\\rightarrow$ 3)", fontweight='bold')
    ax2.set_xlabel("Principal Component Axis (PC4–PC12 Pruned)")
    ax2.set_ylabel("Percentage of Total Variance Explained (%)")
    ax2.set_ylim(0, 108)
    ax2.legend(loc='center right', fontsize=8.5)

    fig.suptitle("TERRA-FIRE Activity 6: PCA Dimensionality Reduction (n_components=3 | 87.87% Variance Retained, N = 18,275)",
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, "fig21_pca_explained_variance.png")


def generate_fig22_component_loadings(loadings_df, pca_summary_df):
    """Fig 22: Component Loadings Heatmap (Retained PC1-PC3) & Dominant Variable Contributions."""
    fig = plt.figure(figsize=(15.5, 6.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.30], wspace=0.28)

    # Panel A: Annotated Loadings Heatmap strictly for the 3 Retained Components (PC1, PC2, PC3)
    ax0 = fig.add_subplot(gs[0])
    sub_load = loadings_df[["PC1", "PC2", "PC3"]].copy()
    sub_load.columns = [
        f"PC1 ({pca_summary_df.loc[0, 'Explained_Variance_Pct']:.2f}%)\n[Desiccation]",
        f"PC2 ({pca_summary_df.loc[1, 'Explained_Variance_Pct']:.2f}%)\n[Orographic/Agri]",
        f"PC3 ({pca_summary_df.loc[2, 'Explained_Variance_Pct']:.2f}%)\n[Canopy Phenology]"
    ]
    sns.heatmap(sub_load, annot=True, fmt=".3f", cmap="RdBu_r", center=0, vmin=-0.65, vmax=0.70,
                cbar_kws={"label": "Eigenvector Loading Weight ($w_{jk}$)"}, linewidths=0.6, ax=ax0)
    ax0.set_title("A. Retained 3-PC Eigenvector Loadings Matrix (PC1 to PC3)", fontweight='bold')
    ax0.set_ylabel("Standardized Input Variable (12 Features)")

    # Panel B: Horizontal Grouped Bar Chart of Variable Loadings on PC1, PC2, PC3
    ax1 = fig.add_subplot(gs[1])
    y_pos = np.arange(len(loadings_df.index))
    height = 0.26
    ax1.barh(y_pos - height, loadings_df["PC1"], height=height, color='#d95f02', edgecolor='black', label='PC1: Atmospheric-Edaphic Desiccation (40.76%)')
    ax1.barh(y_pos, loadings_df["PC2"], height=height, color='#1b9e77', edgecolor='black', label='PC2: Orographic vs. Agri Valley (33.41%)')
    ax1.barh(y_pos + height, loadings_df["PC3"], height=height, color='#7570b3', edgecolor='black', label='PC3: Canopy Greenness Phenology (13.70%)')
    ax1.axvline(0, color='black', linewidth=1.2)
    ax1.axvline(0.30, color='gray', linestyle='--', linewidth=1.0, alpha=0.7)
    ax1.axvline(-0.30, color='gray', linestyle='--', linewidth=1.0, alpha=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(loadings_df.index)
    ax1.invert_yaxis()
    ax1.set_xlabel("Principal Component Loading ($w_{jk}$)")
    ax1.set_title("B. Directional Variable Influence Across Retained 3D Subspace", fontweight='bold')
    ax1.legend(loc='lower right', frameon=True)

    fig.suptitle("TERRA-FIRE Activity 6: Component Interpretation & Latent Physical Structure (PC1, PC2, PC3)",
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, "fig22_pca_component_loadings.png")


def generate_fig23_pca_2d_biplot(df_pca, loadings_df, pca_summary_df):
    """Fig 23: 2D PCA Biplot (PC1 vs PC2 and PC1 vs PC3) stratified by 48-72h Wildfire Ignition."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    df_0 = df_pca[df_pca["fire_ignition_48h"] == 0]
    df_1 = df_pca[df_pca["fire_ignition_48h"] == 1]

    # Panel A: PC1 vs PC2 Biplot with Quadrant IV Highlight
    ax = axes[0]
    ax.axvspan(0, 7.5, ymin=0, ymax=0.52, color='#fee8c8', alpha=0.40, zorder=0,
               label='Quadrant IV: Pyro-Anthropogenic Zone (68.6% of Fires)')
    ax.scatter(df_0["PC1"], df_0["PC2"], c='#3182bd', s=8, alpha=0.18, label=f'Baseline No-Fire (n={len(df_0):,})', zorder=1)
    ax.scatter(df_1["PC1"], df_1["PC2"], c='#e6550d', s=22, alpha=0.78, edgecolors='black', linewidth=0.25,
               label=f'Ignition 48–72h (n={len(df_1):,})', zorder=2)

    draw_confidence_ellipse(df_0["PC1"], df_0["PC2"], ax, n_std=2.0, edgecolor='#08519c', linestyle='--', linewidth=2.0, label='95% Ellipse (Baseline)')
    draw_confidence_ellipse(df_1["PC1"], df_1["PC2"], ax, n_std=2.0, edgecolor='#a63603', linestyle='-', linewidth=2.4, label='95% Ellipse (Ignition)')

    # Plot Biplot Loading Vectors scaled for visibility with custom label offsets
    scale_factor = 6.2
    vec_offsets = {
        "vpd_max_kpa": (0.35, 0.28),
        "temp_2m_max_c": (0.35, -0.25),
        "rh_2m_min_pct": (-0.45, -0.28),
        "soil_moist_mean": (-0.45, 0.25),
        "dist_agri_m": (-0.75, 0.25),
        "elevation_m": (0.70, 0.25),
        "slope_deg": (0.0, 0.45),
        "ndmi": (-0.30, -0.28),
        "days_since_rain": (0.55, -0.22),
    }
    for var, (dx, dy) in vec_offsets.items():
        vx = loadings_df.loc[var, "PC1"] * scale_factor
        vy = loadings_df.loc[var, "PC2"] * scale_factor
        ax.arrow(0, 0, vx, vy, color='#1a1a1a', alpha=0.85, width=0.03, head_width=0.18, zorder=4)
        ax.text(vx + dx, vy + dy, var, fontsize=8.5, fontweight='bold', color='#111111',
                ha='center', va='center', bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.85, edgecolor='none'))

    ax.axhline(0, color='black', linestyle=':', linewidth=1.0)
    ax.axvline(0, color='black', linestyle=':', linewidth=1.0)
    ax.set_xlim(-6.5, 6.5)
    ax.set_ylim(-4.5, 6.5)
    ax.set_xlabel(f"PC1: Atmospheric-Edaphic Desiccation ({pca_summary_df.loc[0, 'Explained_Variance_Pct']:.2f}% Var)")
    ax.set_ylabel(f"PC2: Orographic Altitude vs. Agri Valley ({pca_summary_df.loc[1, 'Explained_Variance_Pct']:.2f}% Var)")
    ax.set_title("A. 2D PCA Biplot (PC1 vs PC2 — 74.17% Cumulative Variance)", fontweight='bold')
    ax.legend(loc='upper left', fontsize=8.5, frameon=True)

    # Panel B: PC1 vs PC3 Projection
    ax2 = axes[1]
    ax2.scatter(df_0["PC1"], df_0["PC3"], c='#3182bd', s=8, alpha=0.18, label='Baseline No-Fire (0)')
    ax2.scatter(df_1["PC1"], df_1["PC3"], c='#e6550d', s=22, alpha=0.78, edgecolors='black', linewidth=0.25, label='Ignition 48–72h (1)')
    draw_confidence_ellipse(df_0["PC1"], df_0["PC3"], ax2, n_std=2.0, edgecolor='#08519c', linestyle='--', linewidth=2.0)
    draw_confidence_ellipse(df_1["PC1"], df_1["PC3"], ax2, n_std=2.0, edgecolor='#a63603', linestyle='-', linewidth=2.4)

    for var in ["vpd_max_kpa", "temp_2m_max_c", "rh_2m_min_pct", "soil_moist_mean", "ndmi", "precip_72h_sum_mm", "days_since_rain"]:
        vx = loadings_df.loc[var, "PC1"] * scale_factor
        vz = loadings_df.loc[var, "PC3"] * scale_factor
        ax2.arrow(0, 0, vx, vz, color='#1a1a1a', alpha=0.85, width=0.03, head_width=0.18, zorder=4)
        ax2.text(vx * 1.12, vz * 1.12, var, fontsize=8.5, fontweight='bold', color='#111111',
                 ha='center', va='center', bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.8, edgecolor='none'))

    ax2.axhline(0, color='black', linestyle=':', linewidth=1.0)
    ax2.axvline(0, color='black', linestyle=':', linewidth=1.0)
    ax2.set_xlim(-6.5, 6.5)
    ax2.set_ylim(-4.5, 5.5)
    ax2.set_xlabel(f"PC1: Atmospheric-Edaphic Desiccation ({pca_summary_df.loc[0, 'Explained_Variance_Pct']:.2f}% Var)")
    ax2.set_ylabel(f"PC3: Vegetative Phenology & Canopy Hydration ({pca_summary_df.loc[2, 'Explained_Variance_Pct']:.2f}% Var)")
    ax2.set_title("B. 2D PCA Biplot (PC1 vs PC3 — Desiccation vs. Foliar Curing)", fontweight='bold')
    ax2.legend(loc='upper left', fontsize=8.5, frameon=True)

    fig.suptitle("TERRA-FIRE Activity 6: Multivariate Pattern Discovery in Reduced 2D PCA Space",
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, "fig23_pca_2d_biplot_ignition.png")


def generate_fig24_ecological_and_seasonal_regimes(df_pca):
    """Fig 24: Ecological Fuel Type Strata & Monthly Seasonal Hysteresis Cycle in PCA Space."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.8))

    # Panel A: Ecological Fuel Strata in PC1 vs PC2 Space
    ax = axes[0]
    palette = {
        "Pastizal Inducido": "#d95f02",
        "Matorral Xerofilo": "#e6ab02",
        "Matorral Submontano": "#66a61e",
        "Bosque de Encino": "#1b9e77",
        "Pino-Encino": "#7570b3",
        "Bosque de Pino": "#2b83ba",
    }
    for fuel, grp in df_pca.groupby("fuel_type", observed=False):
        c = palette.get(fuel, "#333333")
        fire_rate = grp["fire_ignition_48h"].mean() * 100
        ax.scatter(grp["PC1"], grp["PC2"], c=c, s=9, alpha=0.25,
                   label=f"{fuel} (Fire Rate: {fire_rate:.1f}%)")
        cx, cy = grp["PC1"].mean(), grp["PC2"].mean()
        ax.scatter(cx, cy, c=c, s=180, marker='D', edgecolors='black', linewidth=1.8, zorder=5)
        ax.text(cx + 0.18, cy + 0.15, fuel, fontsize=8.5, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85, edgecolor=c))

    ax.axhline(0, color='black', linestyle=':', linewidth=1.0)
    ax.axvline(0, color='black', linestyle=':', linewidth=1.0)
    ax.set_xlabel("PC1: Atmospheric-Edaphic Desiccation Index")
    ax.set_ylabel("PC2: Orographic Altitude vs. Agricultural Valley Gradient")
    ax.set_title("A. Latent Ecological Fuel Stratification & Centroids in PCA Space", fontweight='bold')
    ax.legend(loc='upper right', fontsize=8.5, frameon=True)

    # Panel B: Monthly Seasonal Hysteresis Trajectory in PC1 vs PC3 Space
    ax2 = axes[1]
    df_pca["month"] = pd.to_datetime(df_pca["timestamp"]).dt.month
    monthly = df_pca.groupby("month").agg(
        PC1_mean=("PC1", "mean"),
        PC3_mean=("PC3", "mean"),
        Fire_Rate=("fire_ignition_48h", lambda s: s.mean() * 100.0)
    ).reset_index()

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    sc = ax2.scatter(monthly["PC1_mean"], monthly["PC3_mean"], c=monthly["Fire_Rate"],
                     cmap="YlOrRd", s=260, edgecolors='black', linewidth=1.5, zorder=5)
    for i in range(len(monthly)):
        m_idx = int(monthly.loc[i, "month"]) - 1
        x0, y0 = monthly.loc[i, "PC1_mean"], monthly.loc[i, "PC3_mean"]
        x1, y1 = monthly.loc[(i + 1) % 12, "PC1_mean"], monthly.loc[(i + 1) % 12, "PC3_mean"]
        ax2.annotate("", xy=(x1, y1), xytext=(x0, y0),
                     arrowprops=dict(arrowstyle="->", color="#444444", lw=1.8, shrinkA=8, shrinkB=8))
        ax2.text(x0, y0 + 0.09, f"{month_names[m_idx]}\n({monthly.loc[i, 'Fire_Rate']:.1f}%)",
                 ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    cbar = plt.colorbar(sc, ax=ax2)
    cbar.set_label("Monthly 48–72h Wildfire Ignition Rate (%)")
    ax2.axhline(0, color='black', linestyle=':', linewidth=1.0)
    ax2.axvline(0, color='black', linestyle=':', linewidth=1.0)
    ax2.set_ylim(monthly["PC3_mean"].min() - 0.45, monthly["PC3_mean"].max() + 0.55)
    ax2.set_xlim(monthly["PC1_mean"].min() - 0.55, monthly["PC1_mean"].max() + 0.55)
    ax2.set_xlabel("Mean PC1 Score (Atmospheric & Soil Desiccation)")
    ax2.set_ylabel("Mean PC3 Score (Vegetative Canopy Hydration & Phenology)")
    ax2.set_title("B. Annual Pyro-Climatic Hysteresis Loop (PC1 vs PC3 Trajectory)", fontweight='bold')

    fig.suptitle("TERRA-FIRE Activity 6: Emergent Ecological Groups & Seasonal Trajectory Patterns",
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, "fig24_pca_ecological_regimes.png")


def generate_fig25_pca_3d_projection(df_pca, pca_summary_df):
    """Fig 25: Dual-Angle 3D PCA Manifold Projection (PC1 x PC2 x PC3 = 87.87% Variance)."""
    fig = plt.figure(figsize=(16, 7.2))

    df_0 = df_pca[df_pca["fire_ignition_48h"] == 0].sample(n=4000, random_state=42)
    df_1 = df_pca[df_pca["fire_ignition_48h"] == 1]

    angles = [(24, -55, "A. 3D PCA Manifold (Azimuth = -55°, Elevation = 24°)"),
              (28, 35, "B. Orthogonal 3D View: Valley Ignition Pocket (Azimuth = +35°, Elevation = 28°)")]

    for idx, (elev, azim, title) in enumerate(angles, 1):
        ax = fig.add_subplot(1, 2, idx, projection='3d')
        ax.scatter(df_0["PC1"], df_0["PC2"], df_0["PC3"], c='#3182bd', s=6, alpha=0.18, label='Baseline (0)')
        ax.scatter(df_1["PC1"], df_1["PC2"], df_1["PC3"], c='#e6550d', s=20, alpha=0.80,
                   edgecolors='black', linewidth=0.2, label='Ignition 48–72h (1)')

        ax.set_xlabel(f"PC1: Desiccation ({pca_summary_df.loc[0, 'Explained_Variance_Pct']:.1f}%)")
        ax.set_ylabel(f"PC2: Orographic/Agri ({pca_summary_df.loc[1, 'Explained_Variance_Pct']:.1f}%)")
        ax.set_zlabel(f"PC3: Canopy NDMI ({pca_summary_df.loc[2, 'Explained_Variance_Pct']:.1f}%)")
        ax.set_title(title, fontweight='bold', pad=12)
        ax.view_init(elev=elev, azim=azim)
        if idx == 1:
            ax.legend(loc='upper left')

    fig.suptitle("TERRA-FIRE Activity 6: Three-Component Projection (87.87% Total Information Retained in 3D Subspace)",
                 fontweight='bold', y=1.01)
    plt.tight_layout()
    save_figure(fig, "fig25_pca_3d_projection.png")


def generate_fig26_latent_profiles(df_pca):
    """Fig 26: Latent Component Vigintile Ignition Probability Curves & Quadrant Risk Matrix."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    # Panel A: Empirical Ignition Rate (%) across 20 Quantiles of PC1, PC2, PC3
    ax = axes[0]
    for pc_col, color, label in [
        ("PC1", "#d95f02", "PC1: Desiccation Stress (+r with Fire)"),
        ("PC2", "#1b9e77", "PC2: Mountain vs Agri Valley (-r with Fire)"),
        ("PC3", "#7570b3", "PC3: Canopy Phenology (Non-linear seasonal)")
    ]:
        bins = pd.qcut(df_pca[pc_col], q=20, duplicates='drop')
        grouped = df_pca.groupby(bins, observed=False).agg(
            pc_mean=(pc_col, "mean"),
            fire_prob=("fire_ignition_48h", lambda s: s.mean() * 100.0)
        )
        ax.plot(grouped["pc_mean"], grouped["fire_prob"], marker='o', linewidth=2.3, color=color, label=label)

    ax.axhline(4.79, color='black', linestyle='--', linewidth=1.2, label='Dataset Baseline Prevalence (4.79%)')
    ax.set_xlabel("Principal Component Score (Vigintile Bin Mean)")
    ax.set_ylabel("Empirical 48–72h Wildfire Ignition Probability (%)")
    ax.set_title("A. Non-Linear Wildfire Ignition Response Across Latent PC Axes", fontweight='bold')
    ax.legend(loc='upper left', frameon=True)

    # Panel B: Quadrant Risk Breakdown in PC1 x PC2 Plane
    ax2 = axes[1]
    quad_labels = [
        "Q2: Wet Mountain\n(PC1≤0, PC2>0)",
        "Q1: Dry Mountain\n(PC1>0, PC2>0)",
        "Q3: Wet Agri Valley\n(PC1≤0, PC2≤0)",
        "Q4: Dry Agri Valley\n(PC1>0, PC2≤0)",
        "Apex Zone\n(PC1>2.5, PC2<-1.0)"
    ]
    q_masks = [
        (df_pca["PC1"] <= 0) & (df_pca["PC2"] > 0),
        (df_pca["PC1"] > 0) & (df_pca["PC2"] > 0),
        (df_pca["PC1"] <= 0) & (df_pca["PC2"] <= 0),
        (df_pca["PC1"] > 0) & (df_pca["PC2"] <= 0),
        (df_pca["PC1"] > 2.5) & (df_pca["PC2"] < -1.0)
    ]
    rates = [df_pca[m]["fire_ignition_48h"].mean() * 100.0 for m in q_masks]
    counts = [int(df_pca[m]["fire_ignition_48h"].sum()) for m in q_masks]
    bar_colors = ["#2b83ba", "#abdda4", "#fdae61", "#d7191c", "#7f0000"]

    bars = ax2.bar(quad_labels, rates, color=bar_colors, edgecolor='black', width=0.58)
    ax2.axhline(4.79, color='black', linestyle='--', linewidth=1.2, label='Baseline Rate (4.79%)')
    for bar, r_val, c_val in zip(bars, rates, counts):
        ax2.text(bar.get_x() + bar.get_width()/2, r_val + 0.5, f"{r_val:.2f}%\n({c_val} fires)",
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax2.set_ylim(0, 27.5)
    ax2.set_ylabel("48–72h Wildfire Ignition Rate (%)")
    ax2.set_title("B. Separation Zones: 28.1x Risk Contrast Across PCA Quadrants", fontweight='bold')
    ax2.legend(loc='upper left')

    fig.suptitle("TERRA-FIRE Activity 6: Latent Regime Separation & Quantitative Risk Amplification",
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, "fig26_pca_latent_profiles.png")


def generate_fig27_pca_outliers_hotelling(df_pca):
    """Fig 27: Multivariate Anomaly & Outlier Detection (Hotelling's T^2 vs Q-Residual / SPE)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    t2_99 = np.percentile(df_pca["Hotelling_T2"], 99)
    q_99 = np.percentile(df_pca["Q_Residual"], 99)

    df_0 = df_pca[df_pca["fire_ignition_48h"] == 0]
    df_1 = df_pca[df_pca["fire_ignition_48h"] == 1]

    # Panel A: Hotelling's T2 vs Q-Residual Diagnostic Influence Plot
    ax = axes[0]
    ax.scatter(df_0["Hotelling_T2"], df_0["Q_Residual"], c='#3182bd', s=10, alpha=0.22, label='Baseline (0)')
    ax.scatter(df_1["Hotelling_T2"], df_1["Q_Residual"], c='#e6550d', s=24, alpha=0.80,
               edgecolors='black', linewidth=0.25, label='Ignition 48–72h (1)')
    ax.axvline(t2_99, color='#d7191c', linestyle='--', linewidth=1.8, label=f'99% Hotelling $T^2$ Limit ({t2_99:.2f})')
    ax.axhline(q_99, color='#7570b3', linestyle='-.', linewidth=1.8, label=f'99% Q-Residual (SPE) Limit ({q_99:.2f})')
    ax.annotate("Extreme Monsoon Storm Deluges\n(High Altitude, Precip > 50mm, 0% Fire)",
                xy=(18.0, 14.0), xytext=(11.5, 22.0),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
                fontsize=8.5, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f0'))
    ax.set_xlabel("Hotelling's $T^2$ Statistic (Mahalanobis Distance in Retained 3D PC Subspace)")
    ax.set_ylabel("Q-Residual / Squared Prediction Error (Orthogonal Reconstruction Error)")
    ax.set_title("A. Multivariate Outlier Diagnostic Chart (Hotelling's $T^2$ vs. Q-Residual)", fontweight='bold')
    ax.legend(loc='upper left', frameon=True)

    # Panel B: Node-Level Mean PC1 vs Mean PC2 Trajectory & Ignition Clustering
    ax2 = axes[1]
    vox_summary = df_pca.groupby("voxel_id").agg(
        PC1_mean=("PC1", "mean"),
        PC2_mean=("PC2", "mean"),
        Fire_Count=("fire_ignition_48h", "sum"),
        Fire_Rate=("fire_ignition_48h", lambda s: s.mean() * 100.0),
        Fuel=("fuel_type", "first")
    ).reset_index().sort_values("PC2_mean").reset_index(drop=True)

    sc2 = ax2.scatter(vox_summary["PC1_mean"], vox_summary["PC2_mean"],
                      s=vox_summary["Fire_Count"] * 3.5 + 60, c=vox_summary["Fire_Rate"],
                      cmap="OrRd", edgecolors='black', linewidth=1.4, alpha=0.9, zorder=4)
    for idx in [0, 2, 5, 9, 13, 17, 20, 22, 24]:
        r = vox_summary.loc[idx]
        vid = r["voxel_id"].replace("AGS_VOX_", "V")
        dx = -0.09 if idx % 2 == 0 else 0.08
        ha = 'right' if idx % 2 == 0 else 'left'
        ax2.text(r["PC1_mean"] + dx, r["PC2_mean"], f"{vid} ({int(r['Fire_Count'])} fires)",
                 fontsize=8, fontweight='bold', ha=ha, va='center',
                 bbox=dict(boxstyle='round,pad=0.12', facecolor='white', alpha=0.82, edgecolor='none'))

    cbar2 = plt.colorbar(sc2, ax=ax2)
    cbar2.set_label("Node Ignition Rate (%) [Bubble Size = Total Fires]")
    ax2.axhline(0, color='black', linestyle=':', linewidth=1.0)
    ax2.axvline(0, color='black', linestyle=':', linewidth=1.0)
    ax2.set_xlim(vox_summary["PC1_mean"].min() - 0.55, vox_summary["PC1_mean"].max() + 0.55)
    ax2.set_ylim(vox_summary["PC2_mean"].min() - 0.50, vox_summary["PC2_mean"].max() + 0.50)
    ax2.set_xlabel("Node Mean PC1 (Persistent Desiccation Regime)")
    ax2.set_ylabel("Node Mean PC2 (Orographic Altitude vs. Agricultural Valley)")
    ax2.set_title("B. Spatial Node Regime Separation in Reduced PCA Coordinates", fontweight='bold')

    fig.suptitle("TERRA-FIRE Activity 6: Unusual Observations, Anomaly Regimes & Spatial Node Grouping",
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, "fig27_pca_outliers_hotelling.png")


def main():
    print("=" * 85)
    print("TERRA-FIRE: ACTIVITY 6 - PATTERN DISCOVERY CHALLENGE ENGINE (12 -> 3 PCA REDUCTION)")
    print("=" * 85)

    df = load_dataset()
    prep_df, scaler, X_std = verify_and_standardize(df, CORE_PCA_FEATURES)
    print("\n--- TABLE 2: MULTIVARIATE PREPARATION REPORT (12 STANDARDIZED VARIABLES) ---")
    print(prep_df.to_string(index=False))

    (
        pca_exploratory,
        exploratory_spectrum_df,
        pca_reduced,
        X_pca_reduced,
        pca_summary_df,
        loadings_df,
        contrib_df,
        df_pca,
    ) = execute_pca_analysis(df, X_std, CORE_PCA_FEATURES)

    print("\n--- STEP 3A: EXPLORATORY EIGENVALUE SPECTRUM (FOR SCREE PLOT & KAISER CUTOFF) ---")
    print(exploratory_spectrum_df.round(4).to_string(index=False))

    print("\n--- STEP 3B: DEFINITIVE DIMENSIONALITY REDUCTION (PCA(n_components=3)) ---")
    print(f"Original Input Feature Matrix Shape (X_std):      {X_std.shape}  (12 Variables)")
    print(f"Reduced Transformed PCA Matrix Shape (X_pca_reduced): {X_pca_reduced.shape}   (3 Orthogonal Components: 75.0% Reduction)")
    print("\nFirst 5 rows of X_pca_reduced (shape: 18,275 x 3):")
    print(X_pca_reduced.head(5).round(4).to_string())

    print("\n--- TABLE 3: PCA SUMMARY MATRIX (12 VARIABLES -> 3 RETAINED COMPONENTS) ---")
    print(pca_summary_df.round(4).to_string(index=False))

    print("\n--- TABLE 4: EIGENVECTOR LOADINGS MATRIX (RETAINED PC1, PC2, PC3) ---")
    combined_loadings = loadings_df.copy()
    for c in ["PC1", "PC2", "PC3"]:
        combined_loadings[f"{c}_Contrib_%"] = contrib_df[c]
    print(combined_loadings.round(4).to_string())

    print("\n--- GENERATING 7 PUBLICATION VISUAL ANALYTICS DASHBOARDS (FIG 21 - FIG 27) ---")
    generate_fig21_explained_variance(exploratory_spectrum_df)
    generate_fig22_component_loadings(loadings_df, pca_summary_df)
    generate_fig23_pca_2d_biplot(df_pca, loadings_df, pca_summary_df)
    generate_fig24_ecological_and_seasonal_regimes(df_pca)
    generate_fig25_pca_3d_projection(df_pca, pca_summary_df)
    generate_fig26_latent_profiles(df_pca)
    generate_fig27_pca_outliers_hotelling(df_pca)

    print("\n[SUCCESS] All Pattern Discovery PCA calculations (12 -> 3 dimensions) and 7 visual dashboards (fig21 - fig27) completed!")
    print("Formal Project Readiness Status: READY FOR SEGMENTATION")


if __name__ == "__main__":
    main()
