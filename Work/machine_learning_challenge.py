"""
==============================================================================
TERRA-FIRE: Predictive Wildfire Intelligence Engine
Activity 7: Machine Learning Challenge - Supervised Classification Engine
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

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.utils import resample

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

warnings.filterwarnings("ignore")

# Set publication-grade visual style matching Activities 4, 5 & 6
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

# 12 High & Medium Impact Quantitative Variables validated in Activity 5 & Activity 6
# (Excluding nbr [VIF > 100], savi [collinear], wind_v10_mean [p = 0.621 noise],
# and wind_speed_max_kmh [post-ignition spread vector]).
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
TARGET_COL = "fire_ignition_48h"


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
        f"[LOAD] Immutable dataset loaded: {df.shape[0]:,} observations x {df.shape[1]} attributes."
    )
    return df


def save_figure(fig, filename):
    """
    Synchronizes and exports a 300 DPI matplotlib figure across all repository report directories.
    """
    for d in OUTPUT_FIG_DIRS:
        fig.savefig(os.path.join(d, filename), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIGURE] Saved 300 DPI visual analytics chart: {filename}")


def compute_vif_vector(X_df):
    """
    Computes Variance Inflation Factor (VIF) via inverse correlation matrix diagonal.
    """
    corr = np.corrcoef(X_df.values, rowvar=False)
    try:
        inv_corr = np.linalg.inv(corr)
        vifs = np.diag(inv_corr)
    except np.linalg.LinAlgError:
        inv_corr = np.linalg.pinv(corr)
        vifs = np.diag(inv_corr)
    return pd.Series(vifs, index=X_df.columns)


def prepare_datasets(df, features=CORE_FEATURES_12, target=TARGET_COL, test_size=0.20):
    """
    Phase 1 - Step 2: Executes leakage-free 80/20 stratified train/test partitioning,
    fitting StandardScaler and PCA(n_components=3) strictly on X_train.
    Also enforces sign alignment on PC1 (+vpd_max_kpa), PC2 (+slope_deg), and PC3 (+ndmi)
    for 100% mathematical sign continuity with Activity 6.
    """
    X_raw = df[features].copy()
    y = df[target].astype(int).values

    # Stratified 80/20 Train-Test Split (preserving 4.79% positive ignition prevalence)
    X_train_raw, X_test_raw, y_train, y_test, idx_train, idx_test = train_test_split(
        X_raw,
        y,
        df.index.values,
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # Fit StandardScaler strictly on Training partition (Zero Data Leakage)
    scaler = StandardScaler()
    X_train_std_arr = scaler.fit_transform(X_train_raw)
    X_test_std_arr = scaler.transform(X_test_raw)

    X_train_12 = pd.DataFrame(X_train_std_arr, columns=features, index=idx_train)
    X_test_12 = pd.DataFrame(X_test_std_arr, columns=features, index=idx_test)

    # Fit PCA(n_components=3) strictly on Standardized Training partition
    pca = PCA(n_components=3, random_state=RANDOM_STATE)
    X_train_pca_arr = pca.fit_transform(X_train_std_arr)
    X_test_pca_arr = pca.transform(X_test_std_arr)

    # Align eigenvector signs with Activity 6 physical convention:
    # PC1 positive with vpd_max_kpa, PC2 positive with slope_deg, PC3 positive with ndmi
    anchor_cols = ["vpd_max_kpa", "slope_deg", "ndmi"]
    for k, anchor in enumerate(anchor_cols):
        col_idx = features.index(anchor)
        if pca.components_[k, col_idx] < 0:
            pca.components_[k, :] *= -1.0
            X_train_pca_arr[:, k] *= -1.0
            X_test_pca_arr[:, k] *= -1.0

    X_train_pca = pd.DataFrame(X_train_pca_arr, columns=PCA_COLUMNS_3, index=idx_train)
    X_test_pca = pd.DataFrame(X_test_pca_arr, columns=PCA_COLUMNS_3, index=idx_test)

    # Also fit full-dataset reference PCA (from Activity 6) to verify stability
    scaler_full = StandardScaler()
    X_full_std = scaler_full.fit_transform(X_raw)
    pca_full = PCA(n_components=3, random_state=RANDOM_STATE)
    pca_full.fit(X_full_std)

    vif_12 = compute_vif_vector(X_train_12)
    vif_pca = compute_vif_vector(X_train_pca)

    prep_summary = {
        "n_total": int(len(df)),
        "n_train": int(len(X_train_raw)),
        "n_test": int(len(X_test_raw)),
        "pos_total": int(y.sum()),
        "pos_train": int(y_train.sum()),
        "pos_test": int(y_test.sum()),
        "prev_total_pct": float(y.mean() * 100.0),
        "prev_train_pct": float(y_train.mean() * 100.0),
        "prev_test_pct": float(y_test.mean() * 100.0),
        "pca_train_eigenvalues": [float(v) for v in pca.explained_variance_],
        "pca_train_var_ratio_pct": [
            float(v * 100.0) for v in pca.explained_variance_ratio_
        ],
        "pca_train_cum_var_pct": float( np.sum(pca.explained_variance_ratio_) * 100.0 ),
        "pca_full_cum_var_pct": float( np.sum(pca_full.explained_variance_ratio_) * 100.0 ),
        "vif_12_max": float(vif_12.max()),
        "vif_12_mean": float(vif_12.mean()),
        "vif_pca_max": float(vif_pca.max()),
        "vif_12_dict": {k: float(v) for k, v in vif_12.items()},
        "vif_pca_dict": {k: float(v) for k, v in vif_pca.items()},
    }

    print(
        f"[PREP] Stratified Split -> Train: {prep_summary['n_train']:,} rows "
        f"({prep_summary['pos_train']} ignitions, {prep_summary['prev_train_pct']:.2f}%) | "
        f"Test: {prep_summary['n_test']:,} rows ({prep_summary['pos_test']} ignitions, {prep_summary['prev_test_pct']:.2f}%)"
    )
    print(
        f"[PCA] Train Fit Explained Variance (k=3): {prep_summary['pca_train_cum_var_pct']:.2f}% "
        f"(Full Matrix Ref: {prep_summary['pca_full_cum_var_pct']:.2f}%) | "
        f"Max VIF reduced from {prep_summary['vif_12_max']:.2f} -> {prep_summary['vif_pca_max']:.2f}"
    )

    return {
        "X_train_pca": X_train_pca,
        "X_test_pca": X_test_pca,
        "X_train_12": X_train_12,
        "X_test_12": X_test_12,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "pca": pca,
        "summary": prep_summary,
    }


def find_optimal_f1_threshold(y_true, y_prob):
    """
    Finds the probability decision threshold tau* in [0.05, 0.90] that maximizes F1-score
    strictly on training probabilities (never touching X_test/y_test).
    """
    thresholds = np.linspace(0.05, 0.85, 161)
    best_tau = 0.50
    best_f1 = -1.0
    for tau in thresholds:
        preds = (y_prob >= tau).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_tau = float(tau)
    return best_tau, best_f1


def evaluate_predictions(y_true, y_prob, threshold=0.50):
    """
    Computes complete classification metrics at the specified probability threshold.
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def build_models():
    """
    Instantiates the 3 mandatory supervised classification models required by Activity 7:
    1. Naive Bayes (GaussianNB)
    2. Support Vector Machine (SVC with RBF Kernel & cost-sensitive balanced weights)
    3. Artificial Neural Network (Basic Multi-Layer Perceptron MLPClassifier)
    """
    models = {
        "Naive Bayes": GaussianNB(var_smoothing=1e-9),
        "SVM (RBF)": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=RANDOM_STATE,
        ),
        "Neural Network (MLP)": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-3,
            batch_size=128,
            learning_rate_init=1e-3,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            random_state=RANDOM_STATE,
        ),
    }
    return models


def fit_model_with_imbalance_handling(name, model, X_tr, y_tr):
    """
    Fits a supervised classifier on (X_tr, y_tr).
    - SVC natively incorporates cost-sensitive learning via class_weight='balanced'.
    - GaussianNB operates on exact empirical class priors P(y=1)=0.0479 with threshold calibration.
    - MLPClassifier in scikit-learn lacks a native class_weight parameter and uses raw validation
      accuracy when early_stopping=True; therefore, we apply deterministic 1:3 minority-class
      oversampling strictly on the training fold (X_tr, y_tr) before fitting MLPClassifier,
      preventing Epoch-1 majority-class collapse while preserving 100% zero leakage on X_test.
    """
    if "Neural Network" in name:
        idx_0 = np.where(y_tr == 0)[0]
        idx_1 = np.where(y_tr == 1)[0]
        idx_1_up = resample(
            idx_1,
            replace=True,
            n_samples=len(idx_0) // 3,
            random_state=RANDOM_STATE,
        )
        idx_bal = np.concatenate([idx_0, idx_1_up])
        # Shuffle deterministically
        rng = np.random.RandomState(RANDOM_STATE)
        rng.shuffle(idx_bal)
        X_tr_fit = X_tr.iloc[idx_bal]
        y_tr_fit = y_tr[idx_bal]
        model.fit(X_tr_fit, y_tr_fit)
    else:
        model.fit(X_tr, y_tr)
    return model


def train_and_evaluate_suite(X_train, X_test, y_train, y_test, approach_label="PCA_3D"):
    """
    Phase 2 - Steps 3 & 4: Trains Naive Bayes, SVM (RBF), and Neural Network (MLP),
    performs 5-Fold Stratified Cross-Validation on X_train, determines both standard (0.50)
    and prevalence-calibrated (tau*) decision thresholds on X_train out-of-fold probabilities,
    and evaluates generalization on X_test.
    """
    models = build_models()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    for name, model in models.items():
        print(f"\n[TRAIN - {approach_label}] Fitting {name}...")
        t0 = time.perf_counter()
        fit_model_with_imbalance_handling(name, model, X_train, y_train)
        train_time_ms = (time.perf_counter() - t0) * 1000.0

        # Measure inference latency on Test partition
        t_inf0 = time.perf_counter()
        y_test_prob = model.predict_proba(X_test)[:, 1]
        inf_time_ms = (time.perf_counter() - t_inf0) * 1000.0
        y_train_prob = model.predict_proba(X_train)[:, 1]

        # 5-Fold Stratified Cross-Validation & Out-of-Fold Probability Estimation on X_train
        oof_probs = np.zeros(len(y_train))
        cv_roc_aucs, cv_pr_aucs, cv_f1_cal, cv_rec_cal, cv_prec_cal, cv_acc_cal = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        cv_f1_raw, cv_rec_raw = [], []

        for tr_idx, val_idx in skf.split(X_train, y_train):
            m_clone = build_models()[name]
            fit_model_with_imbalance_handling(
                name, m_clone, X_train.iloc[tr_idx], y_train[tr_idx]
            )
            val_probs = m_clone.predict_proba(X_train.iloc[val_idx])[:, 1]
            tr_probs_fold = m_clone.predict_proba(X_train.iloc[tr_idx])[:, 1]
            oof_probs[val_idx] = val_probs

            # Fold threshold learned strictly on tr_probs_fold
            tau_fold, _ = find_optimal_f1_threshold(y_train[tr_idx], tr_probs_fold)
            val_preds_cal = (val_probs >= tau_fold).astype(int)
            # Also evaluate default predict()
            val_preds_raw = m_clone.predict(X_train.iloc[val_idx])

            cv_roc_aucs.append(roc_auc_score(y_train[val_idx], val_probs))
            cv_pr_aucs.append(average_precision_score(y_train[val_idx], val_probs))
            cv_f1_cal.append(f1_score(y_train[val_idx], val_preds_cal, zero_division=0))
            cv_rec_cal.append(recall_score(y_train[val_idx], val_preds_cal, zero_division=0))
            cv_prec_cal.append(
                precision_score(y_train[val_idx], val_preds_cal, zero_division=0)
            )
            cv_acc_cal.append(accuracy_score(y_train[val_idx], val_preds_cal))

            cv_f1_raw.append(f1_score(y_train[val_idx], val_preds_raw, zero_division=0))
            cv_rec_raw.append(recall_score(y_train[val_idx], val_preds_raw, zero_division=0))

        # Optimal threshold learned on Out-of-Fold training probabilities
        tau_opt, oof_best_f1 = find_optimal_f1_threshold(y_train, oof_probs)

        # Evaluate Default predict() on Test and Train (using model.predict directly,
        # which uses class_weight='balanced' hyperplane for SVC and 0.50 posterior for NB/MLP)
        y_train_pred_default = model.predict(X_train)
        y_test_pred_default = model.predict(X_test)

        tn_d, fp_d, fn_d, tp_d = confusion_matrix(y_test, y_test_pred_default).ravel()
        default_test_metrics = {
            "threshold": 0.50,
            "accuracy": float(accuracy_score(y_test, y_test_pred_default)),
            "precision": float(
                precision_score(y_test, y_test_pred_default, zero_division=0)
            ),
            "recall": float(recall_score(y_test, y_test_pred_default, zero_division=0)),
            "specificity": float(tn_d / (tn_d + fp_d)),
            "f1": float(f1_score(y_test, y_test_pred_default, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_test_prob)),
            "pr_auc": float(average_precision_score(y_test, y_test_prob)),
            "brier": float(brier_score_loss(y_test, y_test_prob)),
            "tn": int(tn_d),
            "fp": int(fp_d),
            "fn": int(fn_d),
            "tp": int(tp_d),
        }

        # Evaluate Calibrated Threshold (tau_opt) on Train and Test
        train_metrics_cal = evaluate_predictions(y_train, y_train_prob, threshold=tau_opt)
        test_metrics_cal = evaluate_predictions(y_test, y_test_prob, threshold=tau_opt)

        # Evaluate High-Sensitivity Operational Threshold (Target Recall ~ 0.80 on Train)
        # corresponding to the native balanced hyperplane regime
        tau_sens = float(np.percentile(y_train_prob[y_train == 1], 22.0))
        test_metrics_sens = evaluate_predictions(y_test, y_test_prob, threshold=tau_sens)

        # Permutation Feature Importance on Test Set (ROC-AUC metric)
        perm = permutation_importance(
            model,
            X_test,
            y_test,
            scoring="roc_auc",
            n_repeats=10,
            random_state=RANDOM_STATE,
        )
        perm_dict = {
            col: {
                "mean": float(perm.importances_mean[i]),
                "std": float(perm.importances_std[i]),
            }
            for i, col in enumerate(X_test.columns)
        }

        results[name] = {
            "model": model,
            "train_time_ms": float(train_time_ms),
            "inf_time_ms": float(inf_time_ms),
            "tau_opt": float(tau_opt),
            "tau_sens": float(tau_sens),
            "train_cal": train_metrics_cal,
            "test_cal": test_metrics_cal,
            "test_default": default_test_metrics,
            "test_sens": test_metrics_sens,
            "y_train_prob": y_train_prob,
            "y_test_prob": y_test_prob,
            "oof_probs": oof_probs,
            "cv": {
                "roc_auc_mean": float(np.mean(cv_roc_aucs)),
                "roc_auc_std": float(np.std(cv_roc_aucs)),
                "pr_auc_mean": float(np.mean(cv_pr_aucs)),
                "pr_auc_std": float(np.std(cv_pr_aucs)),
                "f1_cal_mean": float(np.mean(cv_f1_cal)),
                "f1_cal_std": float(np.std(cv_f1_cal)),
                "rec_cal_mean": float(np.mean(cv_rec_cal)),
                "rec_cal_std": float(np.std(cv_rec_cal)),
                "prec_cal_mean": float(np.mean(cv_prec_cal)),
                "prec_cal_std": float(np.std(cv_prec_cal)),
                "acc_cal_mean": float(np.mean(cv_acc_cal)),
                "acc_cal_std": float(np.std(cv_acc_cal)),
                "f1_raw_mean": float(np.mean(cv_f1_raw)),
                "rec_raw_mean": float(np.mean(cv_rec_raw)),
            },
            "permutation_importance": perm_dict,
        }

        print(
            f"  -> [{name}] Train Time: {train_time_ms:.1f} ms | "
            f"Test ROC-AUC: {test_metrics_cal['roc_auc']:.4f} | PR-AUC: {test_metrics_cal['pr_auc']:.4f} | "
            f"Calibrated (tau={tau_opt:.3f}) -> Acc: {test_metrics_cal['accuracy']:.4f}, "
            f"Prec: {test_metrics_cal['precision']:.4f}, Rec: {test_metrics_cal['recall']:.4f}, F1: {test_metrics_cal['f1']:.4f} | "
            f"Default -> Acc: {default_test_metrics['accuracy']:.4f}, Prec: {default_test_metrics['precision']:.4f}, "
            f"Rec: {default_test_metrics['recall']:.4f}, F1: {default_test_metrics['f1']:.4f}"
        )

    return results


def save_serialized_models(pca_results, orig_results, prep_data):
    """
    Serializes trained models and preprocessing transformers into models/*.pkl
    for reproducible persistence and downstream deployment validation.
    """
    model_file_map = {
        "Naive Bayes": "naive_bayes_model.pkl",
        "SVM (RBF)": "svm_rbf_model.pkl",
        "Neural Network (MLP)": "mlp_ann_model.pkl",
    }
    saved_paths = {}
    for name, fname in model_file_map.items():
        out_path = os.path.join(MODELS_DIR, fname)
        artifact_bundle = {
            "model_name": name,
            "primary_model_pca3": pca_results[name]["model"],
            "benchmark_model_orig12": orig_results[name]["model"],
            "scaler": prep_data["scaler"],
            "pca_3d": prep_data["pca"],
            "tau_opt_pca3": pca_results[name]["tau_opt"],
            "tau_sens_pca3": pca_results[name]["tau_sens"],
            "tau_opt_orig12": orig_results[name]["tau_opt"],
            "features_12": CORE_FEATURES_12,
            "pca_columns": PCA_COLUMNS_3,
            "test_metrics_pca3": pca_results[name]["test_cal"],
            "test_metrics_orig12": orig_results[name]["test_cal"],
        }
        joblib.dump(artifact_bundle, out_path)
        size_kb = os.path.getsize(out_path) / 1024.0
        saved_paths[name] = {"path": out_path, "size_kb": float(size_kb)}
        print(f"[SERIALIZE] Saved {fname} ({size_kb:.1f} KB)")
    return saved_paths


def plot_fig28_partition_and_feature_strategy(prep_data):
    """
    Generates fig28_ml_dataset_partition_and_feature_strategy.png:
    Panel A: Stratified 80/20 Train/Test Partitioning & Class Imbalance (4.79% positive ignitions).
    Panel B: Feature Strategy Comparison (VIF Multicollinearity in 12 Original Features vs PCA 3D Subspace).
    """
    summary = prep_data["summary"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.2))
    fig.suptitle(
        "TERRA-FIRE Activity 7: Stratified Dataset Partitioning & Feature Selection Strategy (Original 12D vs. PCA 3D)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    # Panel A: Stratified Train / Test Split
    partitions = [
        f"Training Set (80%)\n(N = {summary['n_train']:,})",
        f"Testing Set (20%)\n(N = {summary['n_test']:,})",
        f"Total Dataset (100%)\n(N = {summary['n_total']:,})",
    ]
    neg_counts = [
        summary["n_train"] - summary["pos_train"],
        summary["n_test"] - summary["pos_test"],
        summary["n_total"] - summary["pos_total"],
    ]
    pos_counts = [summary["pos_train"], summary["pos_test"], summary["pos_total"]]
    prev_pcts = [
        summary["prev_train_pct"],
        summary["prev_test_pct"],
        summary["prev_total_pct"],
    ]

    x = np.arange(len(partitions))
    width = 0.36
    b1 = ax1.bar(
        x - width / 2,
        neg_counts,
        width,
        label="Class 0: Baseline No-Ignition (95.21%)",
        color="#2b5c8f",
        alpha=0.9,
        edgecolor="black",
    )
    b2 = ax1.bar(
        x + width / 2,
        pos_counts,
        width,
        label="Class 1: Wildfire Ignition 48–72h (4.79%)",
        color="#d95f02",
        alpha=0.9,
        edgecolor="black",
    )

    ax1.set_yscale("log")
    ax1.set_ylim(100, 65000)
    ax1.set_xticks(x)
    ax1.set_xticklabels(partitions, fontweight="bold")
    ax1.set_ylabel("Number of Spatiotemporal Observations (Log Scale)", fontweight="bold")
    ax1.set_title(
        "A. Leakage-Free Stratified 80/20 Partition (Imbalance Ratio 1 : 19.86)",
        fontweight="bold",
    )
    ax1.legend(loc="upper left", frameon=True)

    for rect in b1:
        h = rect.get_height()
        ax1.annotate(
            f"{int(h):,}\n(95.21%)",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#1b3b5c",
        )
    for idx_b, rect in enumerate(b2):
        h = rect.get_height()
        ax1.annotate(
            f"{int(h):,}\n({prev_pcts[idx_b]:.2f}%)",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#993404",
        )

    # Panel B: VIF comparison across 12 Original Variables vs 3 Orthogonal PCA Components
    vif_12_s = pd.Series(summary["vif_12_dict"]).sort_values(ascending=True)
    vif_pca_s = pd.Series(summary["vif_pca_dict"])
    combined_labels = list(vif_12_s.index) + ["---"] + list(vif_pca_s.index)[::-1]
    combined_vals = list(vif_12_s.values) + [0.0] + list(vif_pca_s.values)[::-1]
    colors_vif = []
    for lbl, val in zip(combined_labels, combined_vals):
        if lbl == "---":
            colors_vif.append("#ffffff")
        elif lbl.startswith("PC"):
            colors_vif.append("#2ca25f")
        elif val > 10.0:
            colors_vif.append("#de2d26")
        elif val > 5.0:
            colors_vif.append("#fe9929")
        else:
            colors_vif.append("#3182bd")

    y_pos = np.arange(len(combined_labels))
    bars_vif = ax2.barh(
        y_pos, combined_vals, color=colors_vif, edgecolor="black", height=0.72, alpha=0.9
    )
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(combined_labels, fontweight="bold", fontsize=8.5)
    ax2.set_xscale("log")
    ax2.set_xlim(0.8, 180.0)
    ax2.axvline(1.0, color="#2ca25f", linestyle="--", linewidth=1.5, label="Orthogonal Ideal (VIF = 1.0)")
    ax2.axvline(10.0, color="#de2d26", linestyle=":", linewidth=1.5, label="Severe Collinearity Threshold (VIF = 10.0)")
    ax2.set_xlabel("Variance Inflation Factor — VIF (Log Scale)", fontweight="bold")
    ax2.set_title(
        f"B. Feature Selection Strategy: 12 Raw Features vs. 3 Orthogonal PCs ({summary['pca_train_cum_var_pct']:.2f}% Var)",
        fontweight="bold",
    )
    ax2.legend(loc="lower right", frameon=True, fontsize=8.5)

    for idx_v, (lbl, val) in enumerate(zip(combined_labels, combined_vals)):
        if lbl == "---":
            continue
        tag = f"VIF = {val:.2f}"
        if lbl == "PC1":
            tag += f" ({summary['pca_train_var_ratio_pct'][0]:.1f}% Var)"
        elif lbl == "PC2":
            tag += f" ({summary['pca_train_var_ratio_pct'][1]:.1f}% Var)"
        elif lbl == "PC3":
            tag += f" ({summary['pca_train_var_ratio_pct'][2]:.1f}% Var)"
        ax2.text(
            val * 1.08,
            idx_v,
            tag,
            va="center",
            fontsize=8,
            fontweight="bold",
            color="#006d2c" if lbl.startswith("PC") else "#252525",
        )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, "fig28_ml_dataset_partition_and_feature_strategy.png")


def plot_fig29_confusion_matrices(pca_results, y_test):
    """
    Generates fig29_ml_confusion_matrices_comparison.png:
    Top Row: F1-Calibrated Operational Decision Threshold (tau*) Confusion Matrices.
    Bottom Row: High-Sensitivity / Balanced Screening Regime Confusion Matrices.
    """
    model_names = ["Naive Bayes", "SVM (RBF)", "Neural Network (MLP)"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.8))
    fig.suptitle(
        "TERRA-FIRE Activity 7: Test Set Confusion Matrices Across Supervised Classifiers (N_test = 3,655 | PCA 3D Space)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    for col_idx, name in enumerate(model_names):
        res = pca_results[name]

        # Row 0: Calibrated F1-Optimal Threshold (tau_opt)
        m_cal = res["test_cal"]
        cm_cal = np.array([[m_cal["tn"], m_cal["fp"]], [m_cal["fn"], m_cal["tp"]]])
        cm_cal_pct = cm_cal / cm_cal.sum(axis=1, keepdims=True) * 100.0

        annot_cal = np.empty_like(cm_cal, dtype=object)
        labels_box = [["True Negative (TN)", "False Positive (FP)"], ["False Negative (FN)", "True Positive (TP)"]]
        for i in range(2):
            for j in range(2):
                annot_cal[i, j] = (
                    f"{labels_box[i][j]}\n{cm_cal[i, j]:,}\n({cm_cal_pct[i, j]:.1f}% of row)"
                )

        ax_top = axes[0, col_idx]
        sns.heatmap(
            cm_cal_pct,
            annot=annot_cal,
            fmt="",
            cmap="Blues",
            cbar=False,
            ax=ax_top,
            vmin=0,
            vmax=100,
            linewidths=1.5,
            linecolor="black",
            annot_kws={"size": 9.5, "weight": "bold"},
        )
        ax_top.set_title(
            f"{name} — F1-Calibrated (tau* = {m_cal['threshold']:.3f})\n"
            f"Acc: {m_cal['accuracy']*100:.1f}% | Prec: {m_cal['precision']*100:.1f}% | "
            f"Rec: {m_cal['recall']*100:.1f}% | F1: {m_cal['f1']:.3f}",
            fontsize=10,
            fontweight="bold",
        )
        ax_top.set_xlabel("Predicted Class", fontweight="bold")
        ax_top.set_ylabel("Actual Class (Ground Truth)", fontweight="bold")
        ax_top.set_xticklabels(["0: Baseline", "1: Ignition (48h)"], fontweight="bold")
        ax_top.set_yticklabels(["0: Baseline\n(3,480)", "1: Ignition\n(175)"], fontweight="bold", rotation=0)

        # Row 1: High-Sensitivity / Balanced Cost-Sensitive Screening Regime
        # For SVM (RBF), test_default uses the exact class_weight='balanced' hyperplane!
        m_bal = res["test_default"] if name == "SVM (RBF)" else res["test_sens"]
        regime_tag = (
            "Balanced Hyperplane (w_1=10.4)"
            if name == "SVM (RBF)"
            else f"High-Sensitivity Alert (tau={m_bal['threshold']:.3f})"
        )
        cm_bal = np.array([[m_bal["tn"], m_bal["fp"]], [m_bal["fn"], m_bal["tp"]]])
        cm_bal_pct = cm_bal / cm_bal.sum(axis=1, keepdims=True) * 100.0

        annot_bal = np.empty_like(cm_bal, dtype=object)
        for i in range(2):
            for j in range(2):
                annot_bal[i, j] = (
                    f"{labels_box[i][j]}\n{cm_bal[i, j]:,}\n({cm_bal_pct[i, j]:.1f}% of row)"
                )

        ax_bot = axes[1, col_idx]
        sns.heatmap(
            cm_bal_pct,
            annot=annot_bal,
            fmt="",
            cmap="Oranges",
            cbar=False,
            ax=ax_bot,
            vmin=0,
            vmax=100,
            linewidths=1.5,
            linecolor="black",
            annot_kws={"size": 9.5, "weight": "bold"},
        )
        ax_bot.set_title(
            f"{name} — {regime_tag}\n"
            f"Acc: {m_bal['accuracy']*100:.1f}% | Prec: {m_bal['precision']*100:.1f}% | "
            f"Rec: {m_bal['recall']*100:.1f}% | F1: {m_bal['f1']:.3f}",
            fontsize=10,
            fontweight="bold",
        )
        ax_bot.set_xlabel("Predicted Class", fontweight="bold")
        ax_bot.set_ylabel("Actual Class (Ground Truth)", fontweight="bold")
        ax_bot.set_xticklabels(["0: Baseline", "1: Ignition (48h)"], fontweight="bold")
        ax_bot.set_yticklabels(["0: Baseline\n(3,480)", "1: Ignition\n(175)"], fontweight="bold", rotation=0)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, "fig29_ml_confusion_matrices_comparison.png")


def plot_fig30_roc_and_pr_curves(pca_results, orig_results, y_test):
    """
    Generates fig30_ml_roc_and_pr_curves.png:
    Panel A: Multi-Model ROC Curves & AUC (comparing PCA 3D solid lines vs Original 12D dashed lines).
    Panel B: Multi-Model Precision-Recall Curves & PR-AUC under 4.79% wildfire class prevalence.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.5))
    fig.suptitle(
        "TERRA-FIRE Activity 7: Receiver Operating Characteristic (ROC) & Precision-Recall (PR) Discrimination Curves",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    palette = {
        "Naive Bayes": "#2b5c8f",
        "SVM (RBF)": "#d95f02",
        "Neural Network (MLP)": "#2ca25f",
    }

    # Panel A: ROC Curves
    for name, color in palette.items():
        # PCA 3D
        prob_pca = pca_results[name]["y_test_prob"]
        auc_pca = pca_results[name]["test_cal"]["roc_auc"]
        fpr_p, tpr_p, _ = roc_curve(y_test, prob_pca)
        ax1.plot(
            fpr_p,
            tpr_p,
            color=color,
            lw=2.5,
            label=f"{name} [PCA 3D] (ROC-AUC = {auc_pca:.4f})",
        )

        # Original 12D
        prob_12 = orig_results[name]["y_test_prob"]
        auc_12 = orig_results[name]["test_cal"]["roc_auc"]
        fpr_12, tpr_12, _ = roc_curve(y_test, prob_12)
        ax1.plot(
            fpr_12,
            tpr_12,
            color=color,
            lw=1.6,
            linestyle="--",
            alpha=0.75,
            label=f"{name} [Orig 12D] (ROC-AUC = {auc_12:.4f})",
        )

    ax1.plot([0, 1], [0, 1], "k:", lw=1.5, label="Random Classifier (AUC = 0.5000)")
    ax1.set_xlim([-0.01, 1.01])
    ax1.set_ylim([-0.01, 1.02])
    ax1.set_xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
    ax1.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontweight="bold")
    ax1.set_title("A. Receiver Operating Characteristic (ROC) Curves (Test Set)", fontweight="bold")
    ax1.legend(loc="lower right", frameon=True, fontsize=8.5)

    # Panel B: Precision-Recall Curves
    prevalence = float(np.mean(y_test))
    for name, color in palette.items():
        prob_pca = pca_results[name]["y_test_prob"]
        pr_auc_pca = pca_results[name]["test_cal"]["pr_auc"]
        prec_p, rec_p, _ = precision_recall_curve(y_test, prob_pca)
        ax2.plot(
            rec_p,
            prec_p,
            color=color,
            lw=2.5,
            label=f"{name} [PCA 3D] (PR-AUC = {pr_auc_pca:.4f} | {pr_auc_pca/prevalence:.1f}x Lift)",
        )

        # Mark F1-Calibrated operating point
        m_cal = pca_results[name]["test_cal"]
        ax2.scatter(
            [m_cal["recall"]],
            [m_cal["precision"]],
            color=color,
            s=85,
            edgecolor="black",
            zorder=6,
            marker="o",
        )

        # Original 12D
        prob_12 = orig_results[name]["y_test_prob"]
        pr_auc_12 = orig_results[name]["test_cal"]["pr_auc"]
        prec_12, rec_12, _ = precision_recall_curve(y_test, prob_12)
        ax2.plot(
            rec_12,
            prec_12,
            color=color,
            lw=1.6,
            linestyle="--",
            alpha=0.75,
            label=f"{name} [Orig 12D] (PR-AUC = {pr_auc_12:.4f})",
        )

    ax2.axhline(
        prevalence,
        color="black",
        linestyle=":",
        lw=1.5,
        label=f"Baseline Wildfire Prevalence ({prevalence*100:.2f}%)",
    )
    ax2.set_xlim([-0.01, 1.01])
    ax2.set_ylim([0.0, 0.85])
    ax2.set_xlabel("Recall (Sensitivity — Fraction of Wildfires Detected)", fontweight="bold")
    ax2.set_ylabel("Precision (Positive Predictive Value)", fontweight="bold")
    ax2.set_title("B. Precision-Recall (PR) Curves & Operational Thresholds (Circles)", fontweight="bold")
    ax2.legend(loc="upper right", frameon=True, fontsize=8.2)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, "fig30_ml_roc_and_pr_curves.png")


def plot_fig31_predicted_vs_actual(pca_results, y_test):
    """
    Generates fig31_ml_predicted_vs_actual_probabilities.png:
    Panel A: Predicted Ignition Probability Distributions by Actual Ground Truth Outcome (Class 0 vs Class 1).
    Panel B: Probability Calibration Curves (Reliability Diagram) & Threshold-F1 Optimization Profile.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.4))
    fig.suptitle(
        "TERRA-FIRE Activity 7: Predicted vs. Actual Ignition Probabilities & Reliability Calibration Analysis",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    records = []
    for name in ["Naive Bayes", "SVM (RBF)", "Neural Network (MLP)"]:
        probs = pca_results[name]["y_test_prob"]
        for p, y_val in zip(probs, y_test):
            records.append(
                {
                    "Model": name,
                    "Predicted Probability": float(p),
                    "Actual Outcome": "1: Wildfire Ignition (48–72h)"
                    if y_val == 1
                    else "0: Baseline No-Ignition",
                }
            )
    df_plot = pd.DataFrame(records)

    sns.violinplot(
        data=df_plot,
        x="Model",
        y="Predicted Probability",
        hue="Actual Outcome",
        split=True,
        inner="quart",
        palette={"0: Baseline No-Ignition": "#4a7bb7", "1: Wildfire Ignition (48–72h)": "#e6550d"},
        cut=0,
        ax=ax1,
    )
    ax1.set_title(
        "A. Predicted Probability Distribution vs. Actual Ground Truth (Test Set)",
        fontweight="bold",
    )
    ax1.set_xlabel("Supervised Classification Family (PCA 3D Subspace)", fontweight="bold")
    ax1.set_ylabel("Predicted 48–72h Ignition Probability P(y=1 | x)", fontweight="bold")
    ax1.legend(loc="upper right", frameon=True)

    # Annotate group medians on Panel A
    for idx_m, name in enumerate(["Naive Bayes", "SVM (RBF)", "Neural Network (MLP)"]):
        probs = pca_results[name]["y_test_prob"]
        med_0 = float(np.median(probs[y_test == 0]))
        med_1 = float(np.median(probs[y_test == 1]))
        mean_1 = float(np.mean(probs[y_test == 1]))
        tau = pca_results[name]["tau_opt"]
        ax1.hlines(
            tau,
            idx_m - 0.35,
            idx_m + 0.35,
            colors="#006d2c",
            linestyles="--",
            linewidth=1.8,
        )
        ax1.text(
            idx_m,
            min(0.93, max(med_1, tau) + 0.06),
            f"tau*={tau:.2f}\nMed(1)={med_1:.2f} | Mean(1)={mean_1:.2f}\nMed(0)={med_0:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="gray"),
        )

    # Panel B: Reliability Diagram (Calibration Curve) & F1 Threshold Curve
    palette = {
        "Naive Bayes": "#2b5c8f",
        "SVM (RBF)": "#d95f02",
        "Neural Network (MLP)": "#2ca25f",
    }
    ax2.plot([0, 0.75], [0, 0.75], "k:", lw=1.5, label="Perfectly Calibrated (Diagonal)")
    for name, color in palette.items():
        probs = pca_results[name]["y_test_prob"]
        brier = pca_results[name]["test_cal"]["brier"]
        prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=8, strategy="quantile")
        ax2.plot(
            prob_pred,
            prob_true,
            marker="s",
            lw=2.2,
            color=color,
            label=f"{name} (Brier Score = {brier:.4f})",
        )

    ax2.set_xlim([-0.01, 0.65])
    ax2.set_ylim([-0.01, 0.65])
    ax2.set_xlabel("Mean Predicted Wildfire Ignition Probability (Quantile Bins)", fontweight="bold")
    ax2.set_ylabel("Empirical Observed Ignition Fraction (Actual Rate)", fontweight="bold")
    ax2.set_title(
        "B. Probability Calibration Curves (Predicted vs. Actual Empirical Rates)",
        fontweight="bold",
    )
    ax2.legend(loc="upper left", frameon=True, fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, "fig31_ml_predicted_vs_actual_probabilities.png")


def plot_fig32_decision_boundaries_pca(pca_results, X_test_pca, y_test):
    """
    Generates fig32_ml_decision_boundaries_pca_space.png:
    2D Decision Boundaries and Probability Contours of Naive Bayes, SVM (RBF), and Neural Network (MLP)
    in the PC1 x PC2 plane (with PC3 fixed at median = 0.0), highlighting Quadrant IV convergence.
    """
    model_names = ["Naive Bayes", "SVM (RBF)", "Neural Network (MLP)"]
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.8), sharey=True)
    plt.subplots_adjust(wspace=0.12)
    fig.suptitle(
        "TERRA-FIRE Activity 7: Supervised Decision Boundaries in Orthogonal PCA Space (PC1 × PC2 Plane | Quadrant IV Focus)",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )

    x_min, x_max = -4.2, 5.2
    y_min, y_max = -3.6, 4.2
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 180), np.linspace(y_min, y_max, 180))
    pc3_median = float(X_test_pca["PC3"].median())
    grid_df = pd.DataFrame(
        {"PC1": xx.ravel(), "PC2": yy.ravel(), "PC3": np.full(xx.size, pc3_median)}
    )

    idx_0 = y_test == 0
    idx_1 = y_test == 1

    for idx_m, name in enumerate(model_names):
        ax = axes[idx_m]
        model = pca_results[name]["model"]
        tau_opt = pca_results[name]["tau_opt"]
        tau_sens = pca_results[name]["tau_sens"]

        zz = model.predict_proba(grid_df)[:, 1].reshape(xx.shape)

        cf = ax.contourf(
            xx,
            yy,
            zz,
            levels=np.linspace(0.0, 0.65, 14),
            cmap="YlOrRd",
            alpha=0.72,
            extend="max",
        )
        # Plot decision threshold contours
        cs_opt = ax.contour(
            xx,
            yy,
            zz,
            levels=[tau_opt],
            colors=["#08306b"],
            linewidths=2.3,
            linestyles="-",
        )
        ax.clabel(cs_opt, fmt={tau_opt: f"tau*={tau_opt:.2f}"}, fontsize=8.5)

        cs_sens = ax.contour(
            xx,
            yy,
            zz,
            levels=[tau_sens],
            colors=["#006d2c"],
            linewidths=1.8,
            linestyles="--",
        )
        ax.clabel(cs_sens, fmt={tau_sens: f"Sens={tau_sens:.2f}"}, fontsize=8)

        # Scatter Test set points
        ax.scatter(
            X_test_pca.loc[idx_0, "PC1"],
            X_test_pca.loc[idx_0, "PC2"],
            c="#2b5c8f",
            s=10,
            alpha=0.22,
            label="Test Baseline (y=0)",
        )
        ax.scatter(
            X_test_pca.loc[idx_1, "PC1"],
            X_test_pca.loc[idx_1, "PC2"],
            c="#d95f02",
            edgecolor="black",
            linewidth=0.5,
            s=32,
            alpha=0.88,
            label="Test Ignition (y=1)",
        )

        # Quadrant Axes
        ax.axvline(0, color="black", linestyle=":", lw=1.1)
        ax.axhline(0, color="black", linestyle=":", lw=1.1)
        ax.text(
            4.95,
            -3.35,
            "Quadrant IV: Pyro-Anthropogenic\nConvergence (68.6% of Fires)",
            ha="right",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#fff7bc", edgecolor="#d95f02"),
        )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("PC1: Atmospheric Desiccation (40.8% Var)", fontweight="bold", fontsize=10)
        if idx_m == 0:
            ax.set_ylabel("PC2: Orographic Altitude vs. Valley Interface (33.4% Var)", fontweight="bold")
        ax.set_title(
            f"{name}\nROC-AUC: {pca_results[name]['test_cal']['roc_auc']:.4f} | PR-AUC: {pca_results[name]['test_cal']['pr_auc']:.4f}",
            fontweight="bold",
        )
        if idx_m == 0:
            ax.legend(loc="upper left", frameon=True, fontsize=8.5)

    cbar = fig.colorbar(cf, ax=axes.ravel().tolist(), shrink=0.86, pad=0.02)
    cbar.set_label("Predicted Probability P(Ignition 48–72h)", fontweight="bold")

    save_figure(fig, "fig32_ml_decision_boundaries_pca_space.png")


def plot_fig33_model_comparison_and_importance(pca_results, orig_results):
    """
    Generates fig33_ml_model_comparison_and_importance.png:
    Panel A: Multi-Metric Performance & Generalization Comparison (PCA 3D vs Original 12D).
    Panel B: Permutation Feature Importance across PC1-PC3 and Top Physical Variables + Training Latency.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.4))
    fig.suptitle(
        "TERRA-FIRE Activity 7: Comprehensive Model Comparison, Feature Space Benchmark & Permutation Importance",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    model_names = ["Naive Bayes", "SVM (RBF)", "Neural Network (MLP)"]
    metrics_keys = [
        ("roc_auc", "ROC-AUC"),
        ("pr_auc", "PR-AUC"),
        ("f1", "F1-Score (Cal)"),
        ("recall", "Recall (Cal)"),
        ("accuracy", "Accuracy (Cal)"),
    ]

    x = np.arange(len(metrics_keys))
    width = 0.13
    offsets = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
    configs = [
        ("Naive Bayes", pca_results, "PCA 3D", "#2b5c8f", 1.0),
        ("Naive Bayes", orig_results, "Orig 12D", "#6baed6", 0.85),
        ("SVM (RBF)", pca_results, "PCA 3D", "#d95f02", 1.0),
        ("SVM (RBF)", orig_results, "Orig 12D", "#fdae6b", 0.85),
        ("Neural Network (MLP)", pca_results, "PCA 3D", "#2ca25f", 1.0),
        ("Neural Network (MLP)", orig_results, "Orig 12D", "#74c476", 0.85),
    ]

    for idx_c, (m_name, res_dict, space_tag, col, alpha_v) in enumerate(configs):
        vals = [res_dict[m_name]["test_cal"][k] for k, _ in metrics_keys]
        rects = ax1.bar(
            x + offsets[idx_c] * width,
            vals,
            width,
            label=f"{m_name} [{space_tag}]",
            color=col,
            alpha=alpha_v,
            edgecolor="black",
        )
        for r in rects:
            h = r.get_height()
            ax1.annotate(
                f"{h:.2f}",
                xy=(r.get_x() + r.get_width() / 2, h),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6.8,
                fontweight="bold",
                rotation=90,
            )

    ax1.set_xticks(x)
    ax1.set_xticklabels([lbl for _, lbl in metrics_keys], fontweight="bold")
    ax1.set_ylim(0, 1.08)
    ax1.set_ylabel("Test Set Evaluation Score [0.00 – 1.00]", fontweight="bold")
    ax1.set_title(
        "A. Performance Benchmark: Orthogonal PCA (p=3) vs. Original Features (p=12)",
        fontweight="bold",
    )
    ax1.legend(loc="upper center", ncol=2, frameon=True, fontsize=8)

    # Panel B: Permutation Feature Importance (Delta ROC-AUC) in PCA 3D + Top Physical Drivers
    pca_imp_rows = []
    for col in PCA_COLUMNS_3:
        for m_name in model_names:
            pca_imp_rows.append(
                {
                    "Feature": f"{col} (PCA 3D)",
                    "Model": m_name,
                    "Importance": pca_results[m_name]["permutation_importance"][col]["mean"],
                }
            )
    # Also add top 5 physical features from Neural Network / SVM in 12D space
    mean_12_imp = {}
    for col in CORE_FEATURES_12:
        mean_12_imp[col] = np.mean(
            [orig_results[m]["permutation_importance"][col]["mean"] for m in model_names]
        )
    top5_phys = sorted(mean_12_imp.keys(), key=lambda c: mean_12_imp[c], reverse=True)[:5]
    for col in top5_phys:
        for m_name in model_names:
            pca_imp_rows.append(
                {
                    "Feature": f"{col} (12D)",
                    "Model": m_name,
                    "Importance": orig_results[m_name]["permutation_importance"][col]["mean"],
                }
            )

    df_imp = pd.DataFrame(pca_imp_rows)
    sns.barplot(
        data=df_imp,
        y="Feature",
        x="Importance",
        hue="Model",
        palette={"Naive Bayes": "#2b5c8f", "SVM (RBF)": "#d95f02", "Neural Network (MLP)": "#2ca25f"},
        edgecolor="black",
        ax=ax2,
    )
    ax2.axvline(0.0, color="black", linestyle="--", lw=1.0)
    ax2.set_xlabel("Permutation Feature Importance (Mean Drop in Test ROC-AUC)", fontweight="bold")
    ax2.set_ylabel("Latent PCA Axes (p=3) & Top Physical Drivers (p=12)", fontweight="bold")
    ax2.set_title(
        "B. Permutation Feature Importance (ΔROC-AUC Across 10 Shuffles)",
        fontweight="bold",
    )
    ax2.legend(loc="lower right", frameon=True, fontsize=8.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, "fig33_ml_model_comparison_and_importance.png")


def build_clean_summary_json(prep_data, pca_results, orig_results, saved_models):
    """
    Exports a complete JSON summary of all metrics, cross-validation scores, confusion matrices,
    and permutation importances for 100% programmatic synchronization with the Markdown report.
    """
    out = {
        "dataset_preparation": prep_data["summary"],
        "pca_3d_models": {},
        "orig_12d_models": {},
        "serialized_models": saved_models,
    }
    for name in ["Naive Bayes", "SVM (RBF)", "Neural Network (MLP)"]:
        for key_target, src_dict in [
            ("pca_3d_models", pca_results),
            ("orig_12d_models", orig_results),
        ]:
            r = src_dict[name]
            out[key_target][name] = {
                "train_time_ms": r["train_time_ms"],
                "inf_time_ms": r["inf_time_ms"],
                "tau_opt": r["tau_opt"],
                "tau_sens": r["tau_sens"],
                "train_cal": r["train_cal"],
                "test_cal": r["test_cal"],
                "test_default": r["test_default"],
                "test_sens": r["test_sens"],
                "cv": r["cv"],
                "permutation_importance": r["permutation_importance"],
            }

    json_path = os.path.join(MODELS_DIR, "activity7_metrics_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[JSON] Exported complete programmatic metrics summary: {json_path}")
    return out


def main():
    print("=" * 80)
    print("TERRA-FIRE ACTIVITY 7: MACHINE LEARNING CHALLENGE ENGINE")
    print("=" * 80)

    df = load_dataset()
    prep_data = prepare_datasets(df)

    # 1. Train & Evaluate Primary Suite on Orthogonal PCA Subspace (p = 3)
    pca_results = train_and_evaluate_suite(
        prep_data["X_train_pca"],
        prep_data["X_test_pca"],
        prep_data["y_train"],
        prep_data["y_test"],
        approach_label="PCA_3D (p=3)",
    )

    # 2. Train & Evaluate Comparative Suite on Standardized Original Features (p = 12)
    orig_results = train_and_evaluate_suite(
        prep_data["X_train_12"],
        prep_data["X_test_12"],
        prep_data["y_train"],
        prep_data["y_test"],
        approach_label="ORIGINAL_12D (p=12)",
    )

    # 3. Serialize Models into models/*.pkl
    saved_models = save_serialized_models(pca_results, orig_results, prep_data)

    # 4. Generate 300 DPI Visual Analytics Figures (fig28 - fig33)
    plot_fig28_partition_and_feature_strategy(prep_data)
    plot_fig29_confusion_matrices(pca_results, prep_data["y_test"])
    plot_fig30_roc_and_pr_curves(pca_results, orig_results, prep_data["y_test"])
    plot_fig31_predicted_vs_actual(pca_results, prep_data["y_test"])
    plot_fig32_decision_boundaries_pca(
        pca_results, prep_data["X_test_pca"], prep_data["y_test"]
    )
    plot_fig33_model_comparison_and_importance(pca_results, orig_results)

    # 5. Export JSON Metrics Summary
    build_clean_summary_json(prep_data, pca_results, orig_results, saved_models)

    print("\n" + "=" * 80)
    print("[SUCCESS] Activity 7 Machine Learning Challenge Pipeline Completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
