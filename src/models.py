"""
Machine learning pipeline for IPO underpricing prediction.

Implements:
  - Time-based train/test split (train < 2024-01-01, test ≥ 2024-01-01).
  - TimeSeriesSplit cross-validation.
  - Baseline, Ridge, Random Forest, and LightGBM models.
  - Optuna hyperparameter tuning for LightGBM (50 trials, CV MAE objective).
  - Evaluation reporting: MAE, RMSE, R², Spearman rank correlation.
  - SHAP value computation and plotting helpers.

The tuned LightGBM model is returned as the primary model for interpretation.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Optional

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

from src.utils import setup_logging

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

log = setup_logging(__name__)

FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Feature selection helpers
# ---------------------------------------------------------------------------

def select_features(df: pd.DataFrame) -> list[str]:
    """Return the list of modelling feature columns present in *df*.

    Excludes the target, raw text, date, and identifier columns.

    Args:
        df: Full IPO features DataFrame.

    Returns:
        List of numeric feature column names suitable for modelling.
    """
    exclude_prefixes = ("winsorized_", "first_day_", "first_week", "first_month")
    exclude_exact = {
        "ticker", "company_name", "ipo_date", "filing_date",
        "underpricing", "offer_price", "status", "scrape_year",
        "lead_underwriter_raw", "sector_raw", "industry_raw",
    }

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    features = [
        c for c in numeric_cols
        if c not in exclude_exact
        and not any(c.startswith(p) for p in exclude_prefixes)
    ]
    return features


# ---------------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------------

def time_split(
    df: pd.DataFrame,
    date_col: str = "ipo_date",
    cutoff: str = "2024-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split *df* into train (before cutoff) and test (on/after cutoff).

    Args:
        df: Full dataset.
        date_col: IPO date column name.
        cutoff: ISO date string for the split boundary.

    Returns:
        Tuple ``(train_df, test_df)``.
    """
    dates = pd.to_datetime(df[date_col], errors="coerce")
    cutoff_ts = pd.Timestamp(cutoff)
    train = df[dates < cutoff_ts].copy()
    test = df[dates >= cutoff_ts].copy()
    log.info("Train: %d rows | Test: %d rows (cutoff: %s)", len(train), len(test), cutoff)
    return train, test


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "",
) -> dict[str, float]:
    """Compute MAE, RMSE, R², and Spearman rank correlation.

    Args:
        y_true: Ground-truth values.
        y_pred: Model predictions.
        model_name: Label used in logging.

    Returns:
        Dict with keys ``mae``, ``rmse``, ``r2``, ``spearman_rho``.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    rho, _ = spearmanr(y_true, y_pred)

    metrics = {"mae": mae, "rmse": rmse, "r2": r2, "spearman_rho": rho}
    log.info(
        "[%s] MAE=%.4f | RMSE=%.4f | R²=%.4f | ρ=%.4f",
        model_name or "model", mae, rmse, r2, rho,
    )
    return metrics


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def baseline_median(
    y_train: np.ndarray, y_test: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    """Predict the training-set median for every test observation.

    Args:
        y_train: Training target values.
        y_test: Test target values.

    Returns:
        Tuple of predictions array and evaluation metrics dict.
    """
    pred = np.full(len(y_test), np.median(y_train))
    metrics = evaluate(y_test, pred, "Baseline-Median")
    return pred, metrics


def fit_ols(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    model_name: str = "OLS",
) -> tuple[Any, np.ndarray, dict[str, float]]:
    """Fit OLS regression and return statsmodels result + metrics.

    Args:
        X_train: Training features.
        y_train: Training target.
        X_test: Test features.
        y_test: Test target.
        model_name: Label for logging.

    Returns:
        Tuple of ``(fitted_model, predictions, metrics)``.
    """
    X_tr = sm.add_constant(X_train.fillna(0).astype(float))
    X_te = sm.add_constant(X_test.fillna(0).astype(float))
    model = sm.OLS(y_train, X_tr).fit()
    pred = model.predict(X_te)
    metrics = evaluate(y_test, pred.values, model_name)
    return model, pred.values, metrics


def fit_ridge(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    n_splits: int = 5,
) -> tuple[Pipeline, np.ndarray, dict[str, float]]:
    """Fit Ridge regression with time-series cross-validated alpha.

    Args:
        X_train: Training features.
        y_train: Training target.
        X_test: Test features.
        y_test: Test target.
        n_splits: Number of TimeSeriesSplit folds.

    Returns:
        Tuple of ``(fitted_pipeline, predictions, metrics)``.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    alphas = np.logspace(-3, 4, 50)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=alphas, cv=tscv)),
    ])
    X_tr = X_train.fillna(0).astype(float)
    X_te = X_test.fillna(0).astype(float)
    pipe.fit(X_tr, y_train)
    pred = pipe.predict(X_te)
    best_alpha = pipe.named_steps["ridge"].alpha_
    log.info("Ridge best α=%.4f", best_alpha)
    metrics = evaluate(y_test, pred, "Ridge")
    return pipe, pred, metrics


def fit_random_forest(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    n_splits: int = 5,
) -> tuple[RandomForestRegressor, np.ndarray, dict[str, float]]:
    """Fit a Random Forest with simple grid hyperparameters.

    Args:
        X_train: Training features.
        y_train: Training target.
        X_test: Test features.
        y_test: Test target.
        n_splits: TimeSeriesSplit folds (used for the best-model selection).

    Returns:
        Tuple of ``(fitted_model, predictions, metrics)``.
    """
    X_tr = X_train.fillna(0).astype(float)
    X_te = X_test.fillna(0).astype(float)

    best_mae, best_model = float("inf"), None
    tscv = TimeSeriesSplit(n_splits=n_splits)

    for n_est in (100, 300):
        for max_d in (5, 10, None):
            rf = RandomForestRegressor(
                n_estimators=n_est, max_depth=max_d,
                min_samples_leaf=5, n_jobs=-1, random_state=42,
            )
            cv_maes = []
            for tr_idx, val_idx in tscv.split(X_tr):
                rf.fit(X_tr.iloc[tr_idx], y_train[tr_idx])
                cv_maes.append(mean_absolute_error(y_train[val_idx], rf.predict(X_tr.iloc[val_idx])))
            cv_mae = np.mean(cv_maes)
            log.debug("RF n=%d depth=%s → CV MAE=%.4f", n_est, max_d, cv_mae)
            if cv_mae < best_mae:
                best_mae, best_model = cv_mae, rf

    best_model.fit(X_tr, y_train)
    pred = best_model.predict(X_te)
    metrics = evaluate(y_test, pred, "RandomForest")
    return best_model, pred, metrics


def fit_lgbm_optuna(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    n_trials: int = 60,
    n_splits: int = 5,
) -> tuple[lgb.LGBMRegressor, np.ndarray, dict[str, float]]:
    """Tune and fit LightGBM with Optuna (objective: CV MAE).

    Args:
        X_train: Training features.
        y_train: Training target.
        X_test: Test features.
        y_test: Test target.
        n_trials: Number of Optuna trials.
        n_splits: TimeSeriesSplit folds for CV.

    Returns:
        Tuple of ``(fitted_model, predictions, metrics)``.
    """
    X_tr = X_train.fillna(0).astype(float)
    X_te = X_test.fillna(0).astype(float)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        cv_maes = []
        for tr_idx, val_idx in tscv.split(X_tr):
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_tr.iloc[tr_idx], y_train[tr_idx],
                eval_set=[(X_tr.iloc[val_idx], y_train[val_idx])],
                callbacks=[lgb.early_stopping(50, verbose=False),
                           lgb.log_evaluation(-1)],
            )
            cv_maes.append(mean_absolute_error(y_train[val_idx], model.predict(X_tr.iloc[val_idx])))
        return float(np.mean(cv_maes))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    log.info("Best LGBM params: %s (CV MAE=%.4f)", best_params, study.best_value)

    best_model = lgb.LGBMRegressor(**best_params, random_state=42, n_jobs=-1, verbose=-1)
    best_model.fit(X_tr, y_train)
    pred = best_model.predict(X_te)
    metrics = evaluate(y_test, pred, "LightGBM-Optuna")
    return best_model, pred, metrics


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------

def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Scatter of predicted vs. actual underpricing.

    Args:
        y_true: Ground-truth values.
        y_pred: Model predictions.
        model_name: Title label.
        save_path: Output PNG path.

    Returns:
        Matplotlib Figure.
    """
    save_path = save_path or FIGURES_DIR / "pred_vs_actual.png"
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_pred, y_true, alpha=0.4, s=18, color="#2563EB")
    lim = max(abs(y_true).max(), abs(y_pred).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], "r--", lw=1.5, label="Perfect fit")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Predicted underpricing")
    ax.set_ylabel("Actual underpricing")
    ax.set_title(f"{model_name}: Predicted vs. Actual")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    log.info("Saved → %s", save_path)
    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: Optional[pd.Series] = None,
    model_name: str = "Model",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Residual plots: residual vs. predicted and optionally vs. time.

    Args:
        y_true: Ground-truth values.
        y_pred: Model predictions.
        dates: Optional date Series aligned with y_true for time-drift plot.
        model_name: Title label.
        save_path: Output PNG path.

    Returns:
        Matplotlib Figure.
    """
    save_path = save_path or FIGURES_DIR / "residuals.png"
    residuals = y_true - y_pred
    n_plots = 2 if dates is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    axes[0].scatter(y_pred, residuals, alpha=0.4, s=18, color="#2563EB")
    axes[0].axhline(0, color="red", linestyle="--", lw=1.5)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Residual")
    axes[0].set_title(f"{model_name}: Residuals vs. Predicted")

    if dates is not None:
        axes[1].scatter(pd.to_datetime(dates), residuals, alpha=0.4, s=18, color="#6B7280")
        axes[1].axhline(0, color="red", linestyle="--", lw=1.5)
        axes[1].set_xlabel("IPO date")
        axes[1].set_ylabel("Residual")
        axes[1].set_title(f"{model_name}: Residuals over Time")
        fig.autofmt_xdate()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    log.info("Saved → %s", save_path)
    return fig


# ---------------------------------------------------------------------------
# SHAP interpretation
# ---------------------------------------------------------------------------

def compute_shap(
    model: Any,
    X: pd.DataFrame,
    max_display: int = 15,
    save_prefix: str = "shap",
) -> shap.Explanation:
    """Compute SHAP values and produce summary and dependence plots.

    Args:
        model: Fitted tree model (LightGBM / XGBoost / RandomForest).
        X: Feature matrix for which to compute SHAP values.
        max_display: Number of features shown in the beeswarm plot.
        save_prefix: Prefix for saved PNG files.

    Returns:
        SHAP :class:`~shap.Explanation` object.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X.fillna(0).astype(float))

    # Beeswarm summary
    fig_beeswarm, ax = plt.subplots(figsize=(10, 7))
    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    plt.tight_layout()
    beeswarm_path = FIGURES_DIR / f"{save_prefix}_beeswarm.png"
    plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("SHAP beeswarm saved → %s", beeswarm_path)

    # Dependence plots for top-3 text features
    text_feature_names = [
        c for c in [
            "lm_negative_ratio", "lm_uncertainty_ratio", "prospectus_uniqueness",
            "fog_index_mda", "lm_litigious_ratio",
        ]
        if c in X.columns
    ][:3]

    for feat in text_feature_names:
        if feat not in X.columns:
            continue
        fig_dep, ax = plt.subplots(figsize=(8, 5))
        shap.plots.scatter(shap_values[:, feat], color=shap_values[:, "hot_market_dummy"]
                           if "hot_market_dummy" in X.columns else None, show=False, ax=ax)
        ax.set_title(f"SHAP dependence: {feat}")
        dep_path = FIGURES_DIR / f"{save_prefix}_dep_{feat}.png"
        fig_dep.savefig(dep_path, dpi=150, bbox_inches="tight")
        plt.close(fig_dep)
        log.info("SHAP dependence saved → %s", dep_path)

    return shap_values


# ---------------------------------------------------------------------------
# Full pipeline entry point
# ---------------------------------------------------------------------------

def run_modelling_pipeline(
    df: pd.DataFrame,
    target_col: str = "winsorized_underpricing",
    date_col: str = "ipo_date",
    n_optuna_trials: int = 60,
) -> dict[str, Any]:
    """Execute the full modelling pipeline end-to-end.

    Args:
        df: Full IPO features DataFrame (preprocessed + feature-engineered).
        target_col: Target variable column name.
        date_col: IPO date column (for time-based split and residual plots).
        n_optuna_trials: Optuna trial budget for LightGBM tuning.

    Returns:
        Dict containing trained models, predictions, metrics, and SHAP values.
    """
    feature_cols = select_features(df)
    df_model = df[feature_cols + [target_col, date_col]].dropna(subset=[target_col])

    train_df, test_df = time_split(df_model, date_col)

    X_train = train_df[feature_cols]
    y_train = train_df[target_col].values
    X_test = test_df[feature_cols]
    y_test = test_df[target_col].values

    results: dict[str, Any] = {}

    # ── Baselines ──────────────────────────────────────────────────────────
    pred_median, metrics_median = baseline_median(y_train, y_test)
    results["baseline_median"] = {"predictions": pred_median, "metrics": metrics_median}

    # Financial features only (for H6 comparison)
    fin_features = [c for c in feature_cols if not any(
        kw in c for kw in ("lm_", "fog_", "uniqueness", "word_count")
    )]
    _, _, metrics_ols_fin = fit_ols(
        X_train[fin_features], y_train, X_test[fin_features], y_test, "OLS-Financial"
    )
    results["ols_financial"] = {"metrics": metrics_ols_fin, "features": fin_features}

    _, _, metrics_ols_full = fit_ols(X_train, y_train, X_test, y_test, "OLS-Full")
    results["ols_full"] = {"metrics": metrics_ols_full}

    # ── Ridge ──────────────────────────────────────────────────────────────
    pipe_ridge, pred_ridge, metrics_ridge = fit_ridge(X_train, y_train, X_test, y_test)
    results["ridge"] = {"model": pipe_ridge, "predictions": pred_ridge, "metrics": metrics_ridge}

    # ── Random Forest ──────────────────────────────────────────────────────
    rf, pred_rf, metrics_rf = fit_random_forest(X_train, y_train, X_test, y_test)
    results["random_forest"] = {"model": rf, "predictions": pred_rf, "metrics": metrics_rf}

    # ── LightGBM + Optuna ──────────────────────────────────────────────────
    lgbm, pred_lgbm, metrics_lgbm = fit_lgbm_optuna(
        X_train, y_train, X_test, y_test, n_trials=n_optuna_trials
    )
    results["lightgbm"] = {"model": lgbm, "predictions": pred_lgbm, "metrics": metrics_lgbm}

    # ── Diagnostic plots ───────────────────────────────────────────────────
    plot_predicted_vs_actual(y_test, pred_lgbm, "LightGBM",
                             FIGURES_DIR / "lgbm_pred_vs_actual.png")
    plot_residuals(y_test, pred_lgbm, test_df[date_col], "LightGBM",
                   FIGURES_DIR / "lgbm_residuals.png")

    # ── SHAP ───────────────────────────────────────────────────────────────
    shap_values = compute_shap(lgbm, X_test, save_prefix="lgbm")
    results["shap_values"] = shap_values

    # ── Metrics summary table ──────────────────────────────────────────────
    metrics_table = pd.DataFrame([
        {"Model": name, **res["metrics"]}
        for name, res in results.items()
        if "metrics" in res
    ])
    results["metrics_table"] = metrics_table
    log.info("\n%s", metrics_table.to_string(index=False))

    return results
