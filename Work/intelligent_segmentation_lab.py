"""
==============================================================================
TERRA-FIRE: Predictive Wildfire Intelligence Engine
Activity 8: Intelligent Segmentation Lab - Unsupervised K-Means Clustering Engine
Authors: Angel Emilio Gutierrez Lozano (ID: 4341) & Emiliano Betanzos Valtierra (ID: 4795)
Course: Data Science (LITD9A) - Universidad Tecnologica El Retono
Advisor: Dr. Sergio Valadez Godinez
Term: September - December 2026
==============================================================================
"""

import os
import sys
import time
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.spatial.distance import cdist, pdist, squareform

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    davies_bouldin_score,
    calinski_harabasz_score,
)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import seaborn as sns

warnings.filterwarnings("ignore")

# Set publication-grade visual style matching Activities 4, 5, 6 & 7
plt.style.use(
    "seaborn-v0_8-whitegrid"
    if "seaborn-v0_8-whitegrid" in plt.style.available
    else "default"
)
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 9
plt.rcParams["figure.titlesize"] = 14

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Resolve paths dynamically across Windows & Linux environments
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
WORKSPACE_ROOT = os.path.dirname(PROJECT_ROOT)

PRIMARY_DATA_PATH = os.path.join(
    PROJECT_ROOT, "data", "processed", "terra_fire_spatiotemporal_matrix_v1.parquet"
)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

OUTPUT_FIG_DIRS = [
    os.path.join(PROJECT_ROOT, "reports", "figures"),
    os.path.join(PROJECT_ROOT, "Work", "figures"),
]

VAULT_WORK_FIG = os.path.join(WORKSPACE_ROOT, "TerraFire_vault", "Work", "figures")
if os.path.exists(os.path.join(WORKSPACE_ROOT, "TerraFire_vault")):
    OUTPUT_FIG_DIRS.append(VAULT_WORK_FIG)

for d in OUTPUT_FIG_DIRS:
    os.makedirs(d, exist_ok=True)

# 12 High & Medium Impact Quantitative Variables validated in Activities 5, 6 & 7
# Strictly excluding target `fire_ignition_48h` (unsupervised learning boundary),
# `nbr` (VIF > 100), `savi` (collinear), `wind_v10_mean` (p = 0.621 noise),
# and `wind_speed_max_kmh` (post-ignition spread vector).
CORE_FEATURES_12 = [
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

PCA_COLUMNS_3 = ["PC1", "PC2", "PC3"]
POST_HOC_TARGET = "fire_ignition_48h"
OPTIMAL_K = 4

# Consistent publication color palette and human-readable cluster names for K* = 4
CLUSTER_NAMES = {
    0: "Cluster 0: Dormant Cured-Biomass Agricultural Valley",
    1: "Cluster 1: High-Altitude Sierra Fria Forest Sanctuary",
    2: "Cluster 2: Extreme Heatwave & Drought Desiccation Apex",
    3: "Cluster 3: Warm Monsoon Hydrated & Green-Up Valley",
}

CLUSTER_SHORT_NAMES = {
    0: "C0: Dormant Cured Valley",
    1: "C1: Sierra Fria Sanctuary",
    2: "C2: Extreme Heatwave Apex",
    3: "C3: Monsoon Hydrated Valley",
}

CLUSTER_COLORS = {
    0: "#E67E22",  # Amber/Orange - High Cured-Biomass Winter/Spring Burn Risk (5.86% fire rate)
    1: "#2E86AB",  # Steel Blue - High-Altitude Pine-Oak Sanctuary (0.66% fire rate)
    2: "#D62828",  # Crimson Red - Extreme Spring-Summer Heatwave Apex (8.26% fire rate)
    3: "#2A9D8F",  # Emerald Teal - Warm Monsoon Hydrated Valley (2.69% fire rate)
}


def load_dataset(parquet_path=PRIMARY_DATA_PATH):
    """
    Phase 1 - Step 1: Loads the immutable analysis-ready Parquet dataset in strict read-only mode.
    """
    if not os.path.exists(parquet_path):
        sample_path = os.path.join(
            PROJECT_ROOT, "data", "sample", "terra_fire_matrix_sample.csv"
        )
        print(f"[WARN] Primary dataset not found at {parquet_path}. Using {sample_path}")
        df = pd.read_csv(sample_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    else:
        df = pd.read_parquet(parquet_path)
    print(
        f"[LOAD] Immutable dataset loaded in read-only mode: {df.shape[0]:,} observations x {df.shape[1]} attributes."
    )
    return df


def save_figure(fig, filename):
    """
    Synchronizes and exports a 300 DPI matplotlib figure across all repository report directories.
    """
    for d in OUTPUT_FIG_DIRS:
        fig.savefig(os.path.join(d, filename), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIGURE] Saved 300 DPI visual analytics dashboard: {filename}")


def compute_vif_vector(X_df):
    """
    Computes Variance Inflation Factor (VIF) via inverse correlation matrix diagonal.
    """
    corr = np.corrcoef(X_df.values, rowvar=False)
    inv_corr = np.linalg.inv(corr)
    vifs = np.diag(inv_corr)
    return pd.Series(vifs, index=X_df.columns)


def draw_confidence_ellipse(x, y, ax, n_std=2.0, facecolor="none", **kwargs):
    """
    Draws a 2D covariance confidence ellipse (default 2.0 SD ~ 86.5%-95% contour) on a matplotlib Axes.
    """
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs,
    )
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = float(np.mean(x))
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = float(np.mean(y))
    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def prepare_clustering_inputs(df, features=CORE_FEATURES_12):
    """
    Phase 1 - Step 2: Prepares both standardized 12D physical features (X_scaled_12)
    and the primary 3D Orthogonal PCA Subspace (X_pca_3, retaining 87.87% variance, VIF = 1.00)
    strictly without the target variable `fire_ignition_48h`.
    """
    X_raw_12 = df[features].astype(float).copy()
    scaler = StandardScaler()
    X_scaled_arr = scaler.fit_transform(X_raw_12)
    X_scaled_12 = pd.DataFrame(X_scaled_arr, columns=features, index=df.index)

    pca = PCA(n_components=3, random_state=RANDOM_STATE)
    X_pca_arr = pca.fit_transform(X_scaled_arr)
    X_pca_3 = pd.DataFrame(X_pca_arr, columns=PCA_COLUMNS_3, index=df.index)

    vif_12 = compute_vif_vector(X_scaled_12)
    vif_pca = compute_vif_vector(X_pca_3)

    prep_summary = {
        "n_observations": int(df.shape[0]),
        "n_original_features": len(features),
        "n_pca_components": 3,
        "pca_eigenvalues": [float(v) for v in pca.explained_variance_],
        "pca_explained_variance_pct": [
            float(v * 100.0) for v in pca.explained_variance_ratio_
        ],
        "pca_cumulative_variance_pct": float(
            np.sum(pca.explained_variance_ratio_) * 100.0
        ),
        "total_ss_pca_3d": float(np.sum(X_pca_arr ** 2)),
        "total_ss_orig_12d": float(np.sum(X_scaled_arr ** 2)),
        "vif_12_max": float(vif_12.max()),
        "vif_12_mean": float(vif_12.mean()),
        "vif_12_dict": {k: float(v) for k, v in vif_12.items()},
        "vif_pca_max": float(vif_pca.max()),
        "vif_pca_dict": {k: float(v) for k, v in vif_pca.items()},
    }
    print(
        f"[PREP] X_pca_3 shape: {X_pca_3.shape} | Cumulative Variance: {prep_summary['pca_cumulative_variance_pct']:.2f}% | PCA Max VIF: {prep_summary['vif_pca_max']:.2f} vs 12D Max VIF: {prep_summary['vif_12_max']:.2f}"
    )
    return scaler, pca, X_raw_12, X_scaled_12, X_pca_3, prep_summary


def evaluate_candidate_k(X_pca_3, X_scaled_12, k_range=range(2, 9)):
    """
    Phase 2 - Step 3: Evaluates KMeans(n_clusters=K, init='k-means++', n_init=20, max_iter=300, random_state=42)
    across K in {2..8} on both the primary 3D PCA subspace and the 12D standardized space.
    """
    tss_pca = float(np.sum(X_pca_3.values ** 2))
    tss_12 = float(np.sum(X_scaled_12.values ** 2))

    domain_interpretations = {
        2: "Coarse binary split (Humid/Mountain vs. Dry/Valley); under-segments seasonal canopy phenology (PC3) and merges moderate winter dryness with extreme spring heatwaves.",
        3: "Global Silhouette maximum (0.3208); separates Moist Valley, Dry Valley, and Sierra Fria Sanctuary in 2D (PC1-PC2), but leaves PC3 centroids near zero (ignores foliar senescence vs. heatwave).",
        4: "Optimal Elbow inflection & physical segmentation (Silhouette = 0.3039, 61.78% 3D variance explained); unlocks PC3 to separate Dormant Cured-Biomass Valley (C0) from Extreme Heatwave Apex (C2), Mountain Sanctuary (C1), and Monsoon Valley (C3).",
        5: "Diminishing marginal WCSS gain (-12.26% vs -21.18% at K=4); splits the small Sierra Fria mountain sanctuary (16.6% of data) into two sub-altitude bands with negligible hazard distinction.",
        6: "Over-segmentation of baseline valley transitions; Silhouette drops below 0.295 as intermediate weather days fragment into overlapping boundaries.",
        7: "Excessive cluster fragmentation (Silhouette = 0.2868); induces artificial boundaries across continuous meteorological gradients without actionable CONAFOR dispatch value.",
        8: "Severe over-partitioning (Silhouette = 0.2821); high centroid redundancy and reduced operational interpretability.",
    }

    eval_rows = []
    models_pca = {}

    for k in k_range:
        t0 = time.perf_counter()
        km_pca = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=20,
            max_iter=300,
            random_state=RANDOM_STATE,
        )
        labels_pca = km_pca.fit_predict(X_pca_3.values)
        fit_ms = (time.perf_counter() - t0) * 1000.0

        inertia_pca = float(km_pca.inertia_)
        var_exp_pca = float((1.0 - inertia_pca / tss_pca) * 100.0)
        sil_pca = float(silhouette_score(X_pca_3.values, labels_pca))
        db_pca = float(davies_bouldin_score(X_pca_3.values, labels_pca))
        ch_pca = float(calinski_harabasz_score(X_pca_3.values, labels_pca))

        # Benchmark against 12D collinear space
        km_12 = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=20,
            max_iter=300,
            random_state=RANDOM_STATE,
        )
        labels_12 = km_12.fit_predict(X_scaled_12.values)
        inertia_12 = float(km_12.inertia_)
        var_exp_12 = float((1.0 - inertia_12 / tss_12) * 100.0)
        sil_12 = float(silhouette_score(X_scaled_12.values, labels_12))

        models_pca[k] = km_pca
        eval_rows.append(
            {
                "K": int(k),
                "Inertia_PCA_3D": inertia_pca,
                "Var_Explained_PCA_Pct": var_exp_pca,
                "Silhouette_PCA_3D": sil_pca,
                "Davies_Bouldin_PCA_3D": db_pca,
                "Calinski_Harabasz_PCA_3D": ch_pca,
                "Inertia_Orig_12D": inertia_12,
                "Var_Explained_12D_Pct": var_exp_12,
                "Silhouette_Orig_12D": sil_12,
                "Silhouette_Lift_Pct": float((sil_pca - sil_12) / sil_12 * 100.0),
                "Fit_Time_ms": fit_ms,
                "Interpretation": domain_interpretations[k],
            }
        )
        print(
            f"[EVAL K={k}] PCA WCSS={inertia_pca:,.2f} ({var_exp_pca:.2f}% Var) | PCA Sil={sil_pca:.4f} vs 12D Sil={sil_12:.4f} (+{(sil_pca - sil_12)/sil_12*100:.1f}%)"
        )

    eval_df = pd.DataFrame(eval_rows)
    # Compute marginal WCSS drops
    eval_df["WCSS_Drop_PCA"] = eval_df["Inertia_PCA_3D"].diff().fillna(0.0)
    eval_df["WCSS_Drop_Pct"] = (
        eval_df["Inertia_PCA_3D"].pct_change().fillna(0.0) * 100.0
    )
    return eval_df, models_pca


def characterize_optimal_clusters(
    df, X_raw_12, X_scaled_12, X_pca_3, km_opt, k_opt=OPTIMAL_K
):
    """
    Phase 3 - Steps 5 & 6: Characterizes the final selected K-Means model (K* = 4),
    computing PCA centroids, Z-score physical centroids, raw physical means/SDs,
    intra-cluster cohesion, inter-centroid separation distances, and post-hoc validation
    against `fire_ignition_48h`, `fuel_type`, and Quadrant IV overlap.
    """
    labels = km_opt.labels_
    df_seg = df.copy()
    df_seg["PC1"] = X_pca_3["PC1"]
    df_seg["PC2"] = X_pca_3["PC2"]
    df_seg["PC3"] = X_pca_3["PC3"]
    df_seg["Pyro_PC_Index"] = df_seg["PC1"] - df_seg["PC2"]
    df_seg["cluster_id"] = labels
    df_seg["cluster_name"] = df_seg["cluster_id"].map(CLUSTER_NAMES)

    # Distance of each observation to its assigned centroid in 3D PCA space
    centroids_pca = km_opt.cluster_centers_
    dist_matrix = cdist(X_pca_3.values, centroids_pca, metric="euclidean")
    df_seg["dist_to_centroid"] = dist_matrix[np.arange(len(df_seg)), labels]

    # Sample-level silhouette coefficients for K* = 4
    sil_samples = silhouette_samples(X_pca_3.values, labels)
    df_seg["silhouette_val"] = sil_samples

    # Inter-centroid pairwise Euclidean distance matrix in 3D PCA space
    inter_centroid_dist = squareform(pdist(centroids_pca, metric="euclidean"))

    total_n = len(df_seg)
    total_fires = int(df_seg[POST_HOC_TARGET].sum())
    baseline_fire_rate = float(df_seg[POST_HOC_TARGET].mean() * 100.0)

    cluster_profiles = []
    for c in range(k_opt):
        mask = df_seg["cluster_id"] == c
        sub = df_seg[mask]
        sub_raw = X_raw_12[mask]
        sub_std = X_scaled_12[mask]

        n_obs = int(mask.sum())
        pct_obs = float(n_obs / total_n * 100.0)
        n_fires = int(sub[POST_HOC_TARGET].sum())
        fire_rate_pct = float(sub[POST_HOC_TARGET].mean() * 100.0)
        share_of_fires_pct = float(n_fires / total_fires * 100.0)
        risk_lift = float(fire_rate_pct / baseline_fire_rate)

        # Quadrant IV overlap (PC1 > 0 and PC2 <= 0)
        q4_mask = (sub["PC1"] > 0) & (sub["PC2"] <= 0)
        q4_pct = float(q4_mask.mean() * 100.0)

        # Fuel type composition
        fuel_counts = sub["fuel_type"].value_counts()
        fuel_pcts = (fuel_counts / n_obs * 100.0).to_dict()
        top_fuels_str = ", ".join(
            [f"{k} ({v:.1f}%)" for k, v in list(fuel_pcts.items())[:3]]
        )

        # Dominant months
        months = pd.to_datetime(sub["timestamp"]).dt.month
        top_months = months.value_counts().head(4).index.tolist()

        # Min distance to another centroid
        other_dists = [inter_centroid_dist[c, j] for j in range(k_opt) if j != c]
        min_inter_dist = float(np.min(other_dists))
        mean_inter_dist = float(np.mean(other_dists))

        profile = {
            "cluster_id": int(c),
            "cluster_name": CLUSTER_NAMES[c],
            "short_name": CLUSTER_SHORT_NAMES[c],
            "n_observations": n_obs,
            "pct_of_total": pct_obs,
            "centroid_PC1": float(centroids_pca[c, 0]),
            "centroid_PC2": float(centroids_pca[c, 1]),
            "centroid_PC3": float(centroids_pca[c, 2]),
            "mean_pyro_index": float(sub["Pyro_PC_Index"].mean()),
            "wcss_cluster": float(np.sum(sub["dist_to_centroid"] ** 2)),
            "mean_dist_to_centroid": float(sub["dist_to_centroid"].mean()),
            "std_dist_to_centroid": float(sub["dist_to_centroid"].std()),
            "mean_silhouette": float(sub["silhouette_val"].mean()),
            "pct_positive_silhouette": float((sub["silhouette_val"] > 0).mean() * 100.0),
            "min_inter_centroid_dist": min_inter_dist,
            "mean_inter_centroid_dist": mean_inter_dist,
            "post_hoc_fires": n_fires,
            "post_hoc_fire_rate_pct": fire_rate_pct,
            "post_hoc_share_of_fires_pct": share_of_fires_pct,
            "post_hoc_risk_lift": risk_lift,
            "quadrant_iv_overlap_pct": q4_pct,
            "fuel_composition_pct": {str(k): float(v) for k, v in fuel_pcts.items()},
            "top_fuels_summary": top_fuels_str,
            "top_months": [int(m) for m in top_months],
            "raw_means": {col: float(sub_raw[col].mean()) for col in CORE_FEATURES_12},
            "raw_stds": {col: float(sub_raw[col].std()) for col in CORE_FEATURES_12},
            "z_means": {col: float(sub_std[col].mean()) for col in CORE_FEATURES_12},
        }
        cluster_profiles.append(profile)
        print(
            f"[CLUSTER {c}] N={n_obs:,} ({pct_obs:.2f}%) | Centroid=({centroids_pca[c,0]:+.3f}, {centroids_pca[c,1]:+.3f}, {centroids_pca[c,2]:+.3f}) | Sil={profile['mean_silhouette']:.4f} | Post-Hoc Fires={n_fires} ({fire_rate_pct:.2f}%, {share_of_fires_pct:.2f}% share)"
        )

    # One-way ANOVA F-statistics across the 4 clusters for PC1..PC3 and 12 features
    anova_stats = {}
    for col in PCA_COLUMNS_3 + CORE_FEATURES_12:
        groups = [
            df_seg.loc[df_seg["cluster_id"] == c, col].values
            if col in PCA_COLUMNS_3
            else X_raw_12.loc[df_seg["cluster_id"] == c, col].values
            for c in range(k_opt)
        ]
        f_stat, p_val = stats.f_oneway(*groups)
        anova_stats[col] = {"f_stat": float(f_stat), "p_val": float(p_val)}

    # Chi-square test of independence between cluster_id and post-hoc fire_ignition_48h
    contingency_fire = pd.crosstab(df_seg["cluster_id"], df_seg[POST_HOC_TARGET])
    chi2_fire, p_fire, dof_fire, _ = stats.chi2_contingency(contingency_fire)

    # Chi-square test between cluster_id and fuel_type
    contingency_fuel = pd.crosstab(df_seg["cluster_id"], df_seg["fuel_type"])
    chi2_fuel, p_fuel, dof_fuel, _ = stats.chi2_contingency(contingency_fuel)

    validation_stats = {
        "inter_centroid_distance_matrix": inter_centroid_dist.tolist(),
        "min_inter_centroid_distance": float(
            np.min(inter_centroid_dist[np.triu_indices(k_opt, k=1)])
        ),
        "max_inter_centroid_distance": float(
            np.max(inter_centroid_dist[np.triu_indices(k_opt, k=1)])
        ),
        "mean_inter_centroid_distance": float(
            np.mean(inter_centroid_dist[np.triu_indices(k_opt, k=1)])
        ),
        "anova_f_stats": anova_stats,
        "post_hoc_fire_chi2": float(chi2_fire),
        "post_hoc_fire_pval": float(p_fire),
        "post_hoc_fuel_chi2": float(chi2_fuel),
        "post_hoc_fuel_pval": float(p_fuel),
    }

    return df_seg, cluster_profiles, validation_stats


def generate_fig34_elbow_and_silhouette(eval_df, df_seg):
    """
    Fig 34: K-Means Model Selection Dashboard (Phase 2 - Step 3):
      - Panel A: Elbow Method (Inertia WCSS & % Variance Explained across K=2..8 in 3D PCA vs 12D Space)
      - Panel B: Silhouette Score Curve across K=2..8 (3D Orthogonal PCA VIF=1.00 vs 12D Collinear Space)
      - Panel C: Per-Cluster Silhouette Knife Plot for Optimal K* = 4 (N = 18,275 observations)
    """
    fig = plt.figure(figsize=(19.5, 6.4))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.15, 1.15, 1.25], wspace=0.34)

    k_vals = eval_df["K"].values
    wcss_pca = eval_df["Inertia_PCA_3D"].values / 1000.0
    wcss_12 = eval_df["Inertia_Orig_12D"].values / 1000.0
    var_pca = eval_df["Var_Explained_PCA_Pct"].values
    sil_pca = eval_df["Silhouette_PCA_3D"].values
    sil_12 = eval_df["Silhouette_Orig_12D"].values

    # Panel A: Elbow Method (WCSS & Explained Variance)
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(
        k_vals,
        wcss_pca,
        marker="o",
        linewidth=2.6,
        markersize=8,
        color="#1D3557",
        label="3D PCA Space WCSS ($\\times 10^3$)",
    )
    ax1.plot(
        k_vals,
        wcss_12,
        marker="s",
        linewidth=2.0,
        linestyle="--",
        markersize=6.5,
        color="#8D99AE",
        label="12D Standardized WCSS ($\\times 10^3$)",
    )
    ax1.axvline(OPTIMAL_K, color="#D62828", linestyle=":", linewidth=2.0)
    k4_wcss = wcss_pca[k_vals == OPTIMAL_K][0]
    k4_var = var_pca[k_vals == OPTIMAL_K][0]
    ax1.scatter(
        [OPTIMAL_K],
        [k4_wcss],
        s=150,
        color="#D62828",
        zorder=6,
        edgecolor="black",
        linewidth=1.2,
    )
    ax1.annotate(
        f"Optimal Elbow $K^* = 4$\nWCSS = {k4_wcss*1000:,.0f}\n({k4_var:.1f}% 3D Var Explained)",
        xy=(OPTIMAL_K, k4_wcss),
        xytext=(OPTIMAL_K + 0.55, k4_wcss + 22),
        fontsize=9,
        fontweight="bold",
        color="#D62828",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF3F3", edgecolor="#D62828"),
        arrowprops=dict(arrowstyle="->", color="#D62828", lw=1.5),
    )
    for k, w, v in zip(k_vals, wcss_pca, var_pca):
        if k != OPTIMAL_K:
            ax1.text(
                k,
                w - 6.5,
                f"{v:.0f}%",
                ha="center",
                va="top",
                fontsize=8,
                color="#1D3557",
                fontweight="bold",
            )
    ax1.set_title(
        "A. Elbow Method: WCSS & Variance Explained",
        fontweight="bold",
        fontsize=11,
        pad=10,
    )
    ax1.set_xlabel("Number of Clusters ($K$)", fontweight="bold")
    ax1.set_ylabel("Within-Cluster Sum of Squares (WCSS $\\times 10^3$)", fontweight="bold")
    ax1.set_xticks(k_vals)
    ax1.set_ylim(35, 175)
    ax1.legend(loc="upper right", frameon=True)

    # Panel B: Silhouette Score Comparison (PCA 3D vs Original 12D)
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(
        k_vals,
        sil_pca,
        marker="o",
        linewidth=2.6,
        markersize=8,
        color="#2A9D8F",
        label="3D Orthogonal PCA ($\\mathrm{VIF}=1.00$)",
    )
    ax2.plot(
        k_vals,
        sil_12,
        marker="^",
        linewidth=2.0,
        linestyle="--",
        markersize=6.5,
        color="#E76F51",
        label="12D Collinear Space ($\\mathrm{Max\\ VIF}=80.05$)",
    )
    ax2.axvline(OPTIMAL_K, color="#D62828", linestyle=":", linewidth=2.0)
    ax2.axhline(0.30, color="#6C757D", linestyle="-.", linewidth=1.1, label="Strong Structure Threshold ($s = 0.30$)")

    k3_sil = sil_pca[k_vals == 3][0]
    k4_sil = sil_pca[k_vals == OPTIMAL_K][0]
    ax2.scatter([3], [k3_sil], s=110, color="#1D3557", zorder=5)
    ax2.scatter([OPTIMAL_K], [k4_sil], s=150, color="#D62828", zorder=6, edgecolor="black")

    ax2.annotate(
        f"$K=3$ Max Sil = {k3_sil:.4f}\n(2D only; ignores PC3)",
        xy=(3, k3_sil),
        xytext=(3.3, k3_sil + 0.012),
        fontsize=8.5,
        color="#1D3557",
        arrowprops=dict(arrowstyle="->", color="#1D3557", lw=1.2),
    )
    ax2.annotate(
        f"Selected $K^* = 4$: Sil = {k4_sil:.4f}\n(+21.8% vs 12D Sil = 0.2496)\nSeparates PC3 Regimes",
        xy=(OPTIMAL_K, k4_sil),
        xytext=(4.35, k4_sil - 0.032),
        fontsize=8.8,
        fontweight="bold",
        color="#D62828",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF3F3", edgecolor="#D62828"),
        arrowprops=dict(arrowstyle="->", color="#D62828", lw=1.5),
    )
    ax2.set_title(
        "B. Silhouette Score: PCA 3D vs. 12D Raw Space",
        fontweight="bold",
        fontsize=11,
        pad=10,
    )
    ax2.set_xlabel("Number of Clusters ($K$)", fontweight="bold")
    ax2.set_ylabel("Mean Silhouette Coefficient ($\\bar{s}$)", fontweight="bold")
    ax2.set_xticks(k_vals)
    ax2.set_ylim(0.21, 0.35)
    ax2.legend(loc="lower left", frameon=True, fontsize=8.2)

    # Panel C: Silhouette Knife Plot for K* = 4
    ax3 = fig.add_subplot(gs[2])
    y_lower = 150
    mean_sil = float(df_seg["silhouette_val"].mean())

    for c in range(OPTIMAL_K):
        c_sil = np.sort(
            df_seg.loc[df_seg["cluster_id"] == c, "silhouette_val"].values
        )
        size_c = len(c_sil)
        y_upper = y_lower + size_c
        ax3.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            c_sil,
            facecolor=CLUSTER_COLORS[c],
            edgecolor=CLUSTER_COLORS[c],
            alpha=0.85,
            label=f"{CLUSTER_SHORT_NAMES[c]} ($\\bar{{s}}={c_sil.mean():.3f}, N={size_c:,}$)",
        )
        ax3.text(
            -0.06,
            y_lower + 0.5 * size_c,
            f"C{c}",
            fontsize=9.5,
            fontweight="bold",
            va="center",
            color=CLUSTER_COLORS[c],
        )
        y_lower = y_upper + 350

    ax3.axvline(
        mean_sil,
        color="#D62828",
        linestyle="--",
        linewidth=2.0,
        label=f"Global Mean Silhouette ($\\bar{{s}} = {mean_sil:.4f}$)",
    )
    ax3.set_title(
        f"C. Silhouette Knife Plot ($K^* = {OPTIMAL_K}, N = 18,275$)",
        fontweight="bold",
        fontsize=11,
        pad=10,
    )
    ax3.set_xlabel("Sample Silhouette Coefficient $s(i)$", fontweight="bold")
    ax3.set_ylabel("Clustered Observations (Ordered by $s(i)$)", fontweight="bold")
    ax3.set_xlim(-0.15, 0.72)
    ax3.set_yticks([])
    ax3.legend(loc="lower right", frameon=True, fontsize=8.0)

    fig.suptitle(
        "TERRA-FIRE Activity 8: K-Means Configuration, Elbow Inflection & Orthogonal PCA Silhouette Analysis",
        fontweight="bold",
        y=1.02,
    )
    save_figure(fig, "fig34_kmeans_elbow_and_silhouette.png")


def generate_fig35_pca_clusters_2d_and_3d(df_seg, km_opt):
    """
    Fig 35: 2D & 3D Orthogonal PCA Cluster Geometry and Centroid Cartography (Phase 2 - Step 4):
      - Panel A: 2D PCA Subspace (PC1 x PC2) colored by K-Means Cluster + 95% Ellipses + Centroids
      - Panel B: 2D PCA Subspace (PC1 x PC3) showing how K*=4 resolves Canopy Phenology & Monsoon vs Cured Biomass
      - Panel C: 3D Orthogonal PCA Subspace (PC1 x PC2 x PC3) with Centroid Vectors
    """
    fig = plt.figure(figsize=(19.5, 6.5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.1, 1.1, 1.25], wspace=0.33)

    centroids = km_opt.cluster_centers_

    # Subsample slightly for clean scatter rendering while plotting 100% of centroids & ellipses
    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = rng.choice(len(df_seg), size=min(6000, len(df_seg)), replace=False)
    df_sub = df_seg.iloc[sample_idx]

    # Panel A: PC1 vs PC2
    ax1 = fig.add_subplot(gs[0])
    for c in range(OPTIMAL_K):
        sub_c = df_sub[df_sub["cluster_id"] == c]
        full_c = df_seg[df_seg["cluster_id"] == c]
        ax1.scatter(
            sub_c["PC1"],
            sub_c["PC2"],
            s=10,
            alpha=0.35,
            color=CLUSTER_COLORS[c],
            label=f"{CLUSTER_SHORT_NAMES[c]} ($N={len(full_c):,}$)",
        )
        draw_confidence_ellipse(
            full_c["PC1"].values,
            full_c["PC2"].values,
            ax1,
            n_std=2.0,
            edgecolor=CLUSTER_COLORS[c],
            linewidth=2.0,
            linestyle="--",
        )
        ax1.scatter(
            centroids[c, 0],
            centroids[c, 1],
            s=260,
            marker="*",
            color=CLUSTER_COLORS[c],
            edgecolor="black",
            linewidth=1.5,
            zorder=10,
        )
        ax1.annotate(
            f"$\\mu_{c}$\n({centroids[c,0]:+.2f}, {centroids[c,1]:+.2f})",
            xy=(centroids[c, 0], centroids[c, 1]),
            xytext=(centroids[c, 0] + 0.25, centroids[c, 1] + 0.35),
            fontsize=8.5,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=CLUSTER_COLORS[c], alpha=0.92),
            zorder=11,
        )
    ax1.axhline(0, color="gray", linestyle=":", linewidth=1.0)
    ax1.axvline(0, color="gray", linestyle=":", linewidth=1.0)
    ax1.set_title(
        "A. 2D PCA ($\\mathrm{PC1} \\times \\mathrm{PC2}$, 74.17% Var) & Centroids",
        fontweight="bold",
        fontsize=11,
    )
    ax1.set_xlabel("PC1: Atmospheric-Edaphic Desiccation (40.76% Var)", fontweight="bold")
    ax1.set_ylabel("PC2: Orographic Altitude vs. Agri Valley (33.41% Var)", fontweight="bold")
    ax1.legend(loc="upper right", frameon=True, fontsize=8.0, markerscale=2.0)

    # Panel B: PC1 vs PC3 (Highlighting why K*=4 is essential)
    ax2 = fig.add_subplot(gs[1])
    for c in range(OPTIMAL_K):
        sub_c = df_sub[df_sub["cluster_id"] == c]
        full_c = df_seg[df_seg["cluster_id"] == c]
        ax2.scatter(
            sub_c["PC1"],
            sub_c["PC3"],
            s=10,
            alpha=0.35,
            color=CLUSTER_COLORS[c],
            label=CLUSTER_SHORT_NAMES[c],
        )
        draw_confidence_ellipse(
            full_c["PC1"].values,
            full_c["PC3"].values,
            ax2,
            n_std=2.0,
            edgecolor=CLUSTER_COLORS[c],
            linewidth=2.0,
            linestyle="--",
        )
        ax2.scatter(
            centroids[c, 0],
            centroids[c, 2],
            s=260,
            marker="*",
            color=CLUSTER_COLORS[c],
            edgecolor="black",
            linewidth=1.5,
            zorder=10,
        )
        ax2.annotate(
            f"$\\mu_{c}$\n({centroids[c,0]:+.2f}, {centroids[c,2]:+.2f})",
            xy=(centroids[c, 0], centroids[c, 2]),
            xytext=(centroids[c, 0] + 0.25, centroids[c, 2] + 0.30),
            fontsize=8.5,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=CLUSTER_COLORS[c], alpha=0.92),
            zorder=11,
        )
    ax2.axhline(0, color="gray", linestyle=":", linewidth=1.0)
    ax2.axvline(0, color="gray", linestyle=":", linewidth=1.0)
    ax2.set_title(
        "B. 2D PCA ($\\mathrm{PC1} \\times \\mathrm{PC3}$, 54.46% Var) — Phenology Split",
        fontweight="bold",
        fontsize=11,
    )
    ax2.set_xlabel("PC1: Atmospheric-Edaphic Desiccation (40.76% Var)", fontweight="bold")
    ax2.set_ylabel("PC3: Canopy Phenology NDMI & Warm Monsoon (13.70% Var)", fontweight="bold")

    # Panel C: 3D PCA Scatter & Centroids
    ax3 = fig.add_subplot(gs[2], projection="3d")
    sub_3d = df_seg.iloc[rng.choice(len(df_seg), size=3200, replace=False)]
    for c in range(OPTIMAL_K):
        sc = sub_3d[sub_3d["cluster_id"] == c]
        ax3.scatter(
            sc["PC1"],
            sc["PC2"],
            sc["PC3"],
            s=8,
            alpha=0.28,
            color=CLUSTER_COLORS[c],
            label=CLUSTER_SHORT_NAMES[c],
        )
        ax3.scatter(
            [centroids[c, 0]],
            [centroids[c, 1]],
            [centroids[c, 2]],
            s=280,
            marker="*",
            color=CLUSTER_COLORS[c],
            edgecolor="black",
            linewidth=1.5,
            zorder=20,
        )
        ax3.text(
            centroids[c, 0] + 0.2,
            centroids[c, 1] + 0.2,
            centroids[c, 2] + 0.25,
            f"C{c}",
            fontsize=10,
            fontweight="bold",
            color="black",
        )

    # Connect centroids with dashed lines to illustrate tetrahedral geometry in 3D space
    for i in range(OPTIMAL_K):
        for j in range(i + 1, OPTIMAL_K):
            ax3.plot(
                [centroids[i, 0], centroids[j, 0]],
                [centroids[i, 1], centroids[j, 1]],
                [centroids[i, 2], centroids[j, 2]],
                color="#495057",
                linestyle=":",
                linewidth=1.1,
                alpha=0.75,
            )

    ax3.view_init(elev=23, azim=-52)
    ax3.set_title(
        "C. 3D PCA Subspace ($\\mathrm{PC1} \\times \\mathrm{PC2} \\times \\mathrm{PC3}$, 87.87% Var)",
        fontweight="bold",
        fontsize=11,
        pad=10,
    )
    ax3.set_xlabel("PC1 (40.8%)", fontsize=9, fontweight="bold")
    ax3.set_ylabel("PC2 (33.4%)", fontsize=9, fontweight="bold")
    ax3.set_zlabel("PC3 (13.7%)", fontsize=9, fontweight="bold")

    fig.suptitle(
        "TERRA-FIRE Activity 8: 2D & 3D PCA Cluster Separation and Centroid Geometry (K* = 4)",
        fontweight="bold",
        y=1.02,
    )
    save_figure(fig, "fig35_pca_clusters_2d_and_3d_centroids.png")


def generate_fig36_centroid_profiles_and_pairplot(df_seg, X_scaled_12, km_opt):
    """
    Fig 36: Cluster Centroid Profiles & Multivariate Pairwise Separation Dashboard (Phase 2 - Step 4 & Phase 3 - Step 5):
      - Panel A: Standardized Centroid Profile Heatmap across PC1-PC3 & all 12 Physical Variables
      - Panel B: Multivariate Parallel Coordinate / Centroid Signature Trajectories across the 4 Archetypes
      - Panels C, D, E: Pairwise Physical & Latent Projections (VPD vs Temp, Dist_Agri vs Elevation, NDMI vs Soil Moisture)
        colored by K-Means Cluster with plotted Centroids
    """
    fig = plt.figure(figsize=(19, 11.5))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.1, 1.0], wspace=0.26, hspace=0.32)

    # Build Centroid Matrix (4 clusters x [PC1, PC2, PC3 + 12 Z-scored features])
    all_cols = PCA_COLUMNS_3 + CORE_FEATURES_12
    centroid_mat = np.zeros((OPTIMAL_K, len(all_cols)))
    for c in range(OPTIMAL_K):
        mask = df_seg["cluster_id"] == c
        for j, col in enumerate(PCA_COLUMNS_3):
            centroid_mat[c, j] = df_seg.loc[mask, col].mean()
        for j, col in enumerate(CORE_FEATURES_12):
            centroid_mat[c, 3 + j] = X_scaled_12.loc[mask, col].mean()

    centroid_df = pd.DataFrame(
        centroid_mat,
        index=[CLUSTER_SHORT_NAMES[c] for c in range(OPTIMAL_K)],
        columns=all_cols,
    )

    # Panel A (Spans columns 0 & 1 of top row): Centroid Heatmap
    ax1 = fig.add_subplot(gs[0, :2])
    sns.heatmap(
        centroid_df,
        annot=True,
        fmt="+.2f",
        cmap="RdBu_r",
        center=0.0,
        vmin=-2.2,
        vmax=3.2,
        cbar_kws={"label": "Centroid Coordinate / Standardized Z-Score ($\\mu_{k,j}$)"},
        linewidths=1.0,
        ax=ax1,
        annot_kws={"size": 8.8, "weight": "bold"},
    )
    ax1.axvline(3, color="black", linewidth=2.5)
    ax1.set_title(
        "A. Cluster Centroid Profile Matrix: Orthogonal PCA Coordinates (PC1–PC3) & 12 Standardized Physical Variables (Z-Scores)",
        fontweight="bold",
        pad=10,
    )
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=28, ha="right", fontweight="bold")
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, fontweight="bold")

    # Panel B (Column 2 of top row): Centroid Coordinate Bar Chart in 3D PCA Space + Pyro_PC_Index
    ax2 = fig.add_subplot(gs[0, 2])
    x_pos = np.arange(4)
    width = 0.20
    metrics_pca = ["PC1", "PC2", "PC3", "Pyro_PC_Index"]
    labels_pca = ["PC1 (Drought)", "PC2 (Altitude)", "PC3 (Canopy/Monsoon)", "Pyro Index (PC1-PC2)"]
    bar_colors = ["#E63946", "#1D3557", "#2A9D8F", "#F4A261"]

    for m_idx, (m_col, m_lbl) in enumerate(zip(metrics_pca, labels_pca)):
        vals = [df_seg.loc[df_seg["cluster_id"] == c, m_col].mean() for c in range(OPTIMAL_K)]
        ax2.bar(
            x_pos + (m_idx - 1.5) * width,
            vals,
            width=width,
            color=bar_colors[m_idx],
            edgecolor="black",
            linewidth=0.7,
            label=m_lbl,
        )
    ax2.axhline(0, color="black", linewidth=1.0)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"Cluster {c}" for c in range(OPTIMAL_K)], fontweight="bold")
    ax2.set_ylabel("Mean Latent Coordinate Score", fontweight="bold")
    ax2.set_title("B. Latent PCA Centroid Signatures by Cluster", fontweight="bold", pad=10)
    ax2.legend(loc="upper left", frameon=True, fontsize=8.0)

    # Bottom Row Panels C, D, E: Pairwise Feature Projections Colored by Cluster
    rng = np.random.default_rng(RANDOM_STATE)
    sub_idx = rng.choice(len(df_seg), size=4500, replace=False)
    df_sub = df_seg.iloc[sub_idx]

    pairs = [
        ("temp_2m_max_c", "vpd_max_kpa", "C. Thermodynamic: Max Temp vs. VPD", "Max Temperature at 2m ($^\\circ$C)", "Peak Vapor Pressure Deficit (kPa)"),
        ("soil_moist_mean", "ndmi", "D. Eco-Hydrological: Soil Moisture vs. NDMI", "Volumetric Soil Moisture ($\\mathrm{m^3/m^3}$)", "Sentinel-2 Canopy Moisture (NDMI)"),
        ("dist_agri_m", "vpd_max_kpa", "E. Pyro-Anthropogenic: Agri Distance vs. VPD", "Distance to Agricultural Parcels (m)", "Peak Vapor Pressure Deficit (kPa)"),
    ]

    for idx, (x_col, y_col, title_str, x_lbl, y_lbl) in enumerate(pairs):
        ax = fig.add_subplot(gs[1, idx])
        for c in range(OPTIMAL_K):
            sc = df_sub[df_sub["cluster_id"] == c]
            full_c = df_seg[df_seg["cluster_id"] == c]
            ax.scatter(
                sc[x_col],
                sc[y_col],
                s=10,
                alpha=0.32,
                color=CLUSTER_COLORS[c],
                label=CLUSTER_SHORT_NAMES[c],
            )
            cx = full_c[x_col].mean()
            cy = full_c[y_col].mean()
            ax.scatter(
                [cx],
                [cy],
                s=230,
                marker="*",
                color=CLUSTER_COLORS[c],
                edgecolor="black",
                linewidth=1.4,
                zorder=10,
            )
            ax.annotate(
                f"C{c}",
                xy=(cx, cy),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor=CLUSTER_COLORS[c], alpha=0.9),
                zorder=11,
            )
        ax.set_title(title_str, fontweight="bold")
        ax.set_xlabel(x_lbl, fontweight="bold")
        ax.set_ylabel(y_lbl, fontweight="bold")
        if idx == 0:
            ax.legend(loc="upper left", frameon=True, fontsize=8.0, markerscale=2.0)

    fig.suptitle(
        "TERRA-FIRE Activity 8: Cluster Centroid Profiles & Multivariate Feature Space Pair Plots",
        fontweight="bold",
        y=1.01,
    )
    save_figure(fig, "fig36_cluster_centroid_profiles_and_pairplot.png")


def generate_fig37_distributions_and_domain_characterization(df_seg, cluster_profiles):
    """
    Fig 37: Cluster Distribution & Post-Hoc Domain Characterization Dashboard (Phase 2 - Step 4 & Phase 3 - Step 5):
      - Panel A: Cluster Size Distribution (Observations & % of Total N = 18,275)
      - Panel B: Post-Hoc Ecological `fuel_type` Stratum Composition per Cluster (%)
      - Panel C: Post-Hoc Empirical 48-72h Wildfire Hazard Rate (%) & Share of Total Fires (876 ignitions)
      - Panel D: Monthly Seasonality Profile of Cluster Prevalence across the Annual Cycle (Jan-Dec)
    """
    fig = plt.figure(figsize=(18.5, 11.0))
    gs = gridspec.GridSpec(2, 2, wspace=0.25, hspace=0.32)

    clusters = [p["cluster_id"] for p in cluster_profiles]
    short_names = [p["short_name"] for p in cluster_profiles]
    counts = [p["n_observations"] for p in cluster_profiles]
    pcts = [p["pct_of_total"] for p in cluster_profiles]
    colors = [CLUSTER_COLORS[c] for c in clusters]

    # Panel A: Cluster Size Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    bars1 = ax1.bar(
        short_names,
        counts,
        color=colors,
        edgecolor="black",
        linewidth=1.1,
        width=0.58,
    )
    for bar, cnt, pct in zip(bars1, counts, pcts):
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 140,
            f"{cnt:,} rows\n({pct:.2f}%)",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )
    ax1.axhline(
        len(df_seg) / OPTIMAL_K,
        color="#495057",
        linestyle="--",
        linewidth=1.3,
        label=f"Uniform Reference ({len(df_seg)//OPTIMAL_K:,} rows / 25.0%)",
    )
    ax1.set_title(
        "A. Cluster Assignment Distribution ($N = 18,275$ Daily Voxel Observations)",
        fontweight="bold",
        pad=10,
    )
    ax1.set_ylabel("Number of Spatiotemporal Observations", fontweight="bold")
    ax1.set_ylim(0, 7000)
    ax1.tick_params(axis="x", rotation=12)
    ax1.legend(loc="upper right", frameon=True)

    # Panel B: Ecological Fuel Type Composition per Cluster (100% Stacked Bar)
    ax2 = fig.add_subplot(gs[0, 1])
    ct_fuel = (
        pd.crosstab(df_seg["cluster_id"], df_seg["fuel_type"], normalize="index") * 100.0
    )
    fuel_order = [
        "Pastizal Inducido",
        "Matorral Xerofilo",
        "Matorral Submontano",
        "Bosque de Encino",
        "Pino-Encino",
        "Bosque de Pino",
    ]
    fuel_order = [f for f in fuel_order if f in ct_fuel.columns]
    fuel_palette = {
        "Pastizal Inducido": "#E9C46A",
        "Matorral Xerofilo": "#F4A261",
        "Matorral Submontano": "#E76F51",
        "Bosque de Encino": "#8AB17D",
        "Pino-Encino": "#2A9D8F",
        "Bosque de Pino": "#264653",
    }

    bottom = np.zeros(OPTIMAL_K)
    for fuel in fuel_order:
        vals = [ct_fuel.loc[c, fuel] if fuel in ct_fuel.columns else 0.0 for c in clusters]
        ax2.bar(
            short_names,
            vals,
            bottom=bottom,
            color=fuel_palette.get(fuel, "#999999"),
            edgecolor="white",
            linewidth=0.8,
            width=0.58,
            label=fuel,
        )
        for i, v in enumerate(vals):
            if v >= 12.0:
                ax2.text(
                    i,
                    bottom[i] + v / 2.0,
                    f"{v:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    fontweight="bold",
                    color="white" if fuel in ["Bosque de Pino", "Pino-Encino", "Matorral Submontano"] else "black",
                )
        bottom += np.array(vals)

    ax2.set_title(
        "B. Post-Hoc Ecological Fuel Stratum (`fuel_type`) Composition per Cluster",
        fontweight="bold",
        pad=10,
    )
    ax2.set_ylabel("Percentage of Cluster Observations (%)", fontweight="bold")
    ax2.set_ylim(0, 105)
    ax2.tick_params(axis="x", rotation=12)
    ax2.legend(loc="upper right", bbox_to_anchor=(1.02, 1.0), frameon=True, fontsize=8.2)

    # Panel C: Post-Hoc Wildfire Hazard Rate (%) & Share of All 876 Ignitions
    ax3 = fig.add_subplot(gs[1, 0])
    fire_rates = [p["post_hoc_fire_rate_pct"] for p in cluster_profiles]
    fire_shares = [p["post_hoc_share_of_fires_pct"] for p in cluster_profiles]
    fire_counts = [p["post_hoc_fires"] for p in cluster_profiles]
    baseline_rate = float(df_seg[POST_HOC_TARGET].mean() * 100.0)

    bars3 = ax3.bar(
        short_names,
        fire_rates,
        color=colors,
        edgecolor="black",
        linewidth=1.1,
        width=0.58,
    )
    ax3.axhline(
        baseline_rate,
        color="#D62828",
        linestyle="--",
        linewidth=1.8,
        label=f"Regional Baseline Hazard ({baseline_rate:.2f}%)",
    )
    for bar, rate, share, fcnt in zip(bars3, fire_rates, fire_shares, fire_counts):
        lift = rate / baseline_rate
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.25,
            f"{rate:.2f}% Hazard ({lift:.2f}x)\n[{fcnt} fires | {share:.1f}% of total]",
            ha="center",
            va="bottom",
            fontsize=8.8,
            fontweight="bold",
        )
    ax3.set_title(
        "C. Post-Hoc 48–72h Wildfire Ignition Rate (`fire_ignition_48h`) by Cluster",
        fontweight="bold",
        pad=10,
    )
    ax3.set_ylabel("Empirical Wildfire Ignition Rate (%)", fontweight="bold")
    ax3.set_ylim(0, 10.5)
    ax3.tick_params(axis="x", rotation=12)
    ax3.legend(loc="upper left", frameon=True)

    # Panel D: Monthly Seasonal Phenology & Pyro-Climatic Regime Progression (Jan-Dec)
    ax4 = fig.add_subplot(gs[1, 1])
    df_seg_month = df_seg.copy()
    df_seg_month["month"] = pd.to_datetime(df_seg_month["timestamp"]).dt.month
    ct_month = (
        pd.crosstab(df_seg_month["month"], df_seg_month["cluster_id"], normalize="index")
        * 100.0
    )
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    bottom_m = np.zeros(12)
    for c in clusters:
        vals_m = ct_month[c].values
        ax4.bar(
            np.arange(12),
            vals_m,
            bottom=bottom_m,
            color=CLUSTER_COLORS[c],
            edgecolor="white",
            linewidth=0.6,
            width=0.72,
            label=CLUSTER_SHORT_NAMES[c],
        )
        bottom_m += vals_m

    ax4.set_xticks(np.arange(12))
    ax4.set_xticklabels(month_labels, fontweight="bold")
    ax4.set_ylim(0, 100)
    ax4.set_ylabel("Monthly Regime Share (%)", fontweight="bold")
    ax4.set_xlabel("Calendar Month (2023–2024 Hydrological Cycle)", fontweight="bold")
    ax4.set_title(
        "D. Seasonal Regime Evolution Across the Annual Hydrological Cycle",
        fontweight="bold",
        pad=10,
    )
    ax4.legend(loc="upper right", frameon=True, fontsize=8.0)

    fig.suptitle(
        "TERRA-FIRE Activity 8: Cluster Size Distribution, Ecological Composition & Post-Hoc Wildfire Hazard Validation",
        fontweight="bold",
        y=1.01,
    )
    save_figure(fig, "fig37_cluster_distributions_and_domain_characterization.png")


def main():
    print("=" * 80)
    print("TERRA-FIRE: ACTIVITY 8 - INTELLIGENT SEGMENTATION LAB ENGINE")
    print("=" * 80)

    # Step 1 & Step 2: Load dataset and prepare clustering representations
    df = load_dataset()
    scaler, pca, X_raw_12, X_scaled_12, X_pca_3, prep_summary = (
        prepare_clustering_inputs(df)
    )

    # Step 3: Evaluate candidate values of K in {2..8}
    eval_df, models_pca = evaluate_candidate_k(X_pca_3, X_scaled_12, k_range=range(2, 9))

    # Select Optimal K* = 4 and characterize clusters
    km_opt = models_pca[OPTIMAL_K]
    df_seg, cluster_profiles, validation_stats = characterize_optimal_clusters(
        df, X_raw_12, X_scaled_12, X_pca_3, km_opt, k_opt=OPTIMAL_K
    )

    # Also record K=3 summary for comparative documentation
    km_k3 = models_pca[3]

    # Step 4: Generate 300 DPI Visual Analytics Dashboards (fig34 - fig37)
    generate_fig34_elbow_and_silhouette(eval_df, df_seg)
    generate_fig35_pca_clusters_2d_and_3d(df_seg, km_opt)
    generate_fig36_centroid_profiles_and_pairplot(df_seg, X_scaled_12, km_opt)
    generate_fig37_distributions_and_domain_characterization(df_seg, cluster_profiles)

    # Persist trained KMeans segmentation bundle to models/kmeans_segmentation_model.pkl
    model_bundle = {
        "model_name": "KMeans_PCA3D_Optimal_K4",
        "optimal_k": OPTIMAL_K,
        "kmeans_model": km_opt,
        "kmeans_k3_reference": km_k3,
        "scaler_12d": scaler,
        "pca_3d": pca,
        "core_features_12": CORE_FEATURES_12,
        "pca_columns_3": PCA_COLUMNS_3,
        "cluster_names": CLUSTER_NAMES,
        "cluster_short_names": CLUSTER_SHORT_NAMES,
        "centroids_pca_3d": km_opt.cluster_centers_.tolist(),
        "random_state": RANDOM_STATE,
    }
    model_path = os.path.join(MODELS_DIR, "kmeans_segmentation_model.pkl")
    joblib.dump(model_bundle, model_path)
    model_size_kb = os.path.getsize(model_path) / 1024.0
    print(
        f"[SAVE] Serialized K-Means model bundle saved to {model_path} ({model_size_kb:.1f} KB)"
    )

    # Persist complete metrics & summary JSON to models/activity8_segmentation_summary.json
    summary_payload = {
        "activity": "Activity 8 - Intelligent Segmentation Lab",
        "readiness_decision": "READY FOR ADVANCED ANALYTICS",
        "optimal_k": OPTIMAL_K,
        "model_artifact_path": "models/kmeans_segmentation_model.pkl",
        "model_artifact_size_kb": float(model_size_kb),
        "dataset_preparation": prep_summary,
        "k_evaluation_matrix": eval_df.to_dict(orient="records"),
        "cluster_profiles": cluster_profiles,
        "validation_statistics": validation_stats,
    }
    json_path = os.path.join(MODELS_DIR, "activity8_segmentation_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
    print(f"[SAVE] Programmatic segmentation summary saved to {json_path}")
    print("=" * 80)
    print("ACTIVITY 8 SEGMENTATION PIPELINE COMPLETED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    main()
