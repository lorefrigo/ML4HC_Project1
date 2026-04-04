from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROCESSED_DIR = Path("output/processed_sets")
SET_A_PATH = PROCESSED_DIR / "processed_set_a.parquet"
SET_B_PATH = PROCESSED_DIR / "processed_set_b.parquet"
SET_C_PATH = PROCESSED_DIR / "processed_set_c.parquet"

META_COLS = ["PatientID", "Timestamp", "RecordID", "Label"]
STATIC_COLS = ["Age", "Gender", "Height", "Weight"]
SEED = 42


def get_dynamic_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS and c not in STATIC_COLS]


def build_simple_features(df: pd.DataFrame, dynamic_cols: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """
    Q2.1(1)-style features: one value per dynamic variable (last) + static variables.
    """
    df = df.sort_values(["PatientID", "Timestamp"])
    grouped = df.groupby("PatientID", sort=False)

    dyn = grouped[dynamic_cols].last().add_suffix("_last")
    stat = grouped[STATIC_COLS].last()
    labels = grouped["Label"].last().astype(int)

    x = pd.concat([dyn, stat], axis=1)
    return x, labels


def _compute_slopes(df: pd.DataFrame, dynamic_cols: list[str]) -> pd.DataFrame:
    """
    Compute linear trend slope for each dynamic variable within each patient stay.
    """

    def slopes_for_group(group: pd.DataFrame) -> pd.Series:
        n = len(group)
        if n < 2:
            return pd.Series(0.0, index=[f"{c}_slope" for c in dynamic_cols])

        t = np.arange(n, dtype=float)
        t_centered = t - t.mean()
        denom = float((t_centered**2).sum())
        if denom <= 1e-12:
            return pd.Series(0.0, index=[f"{c}_slope" for c in dynamic_cols])

        x = group[dynamic_cols].to_numpy(dtype=float)

        # Handle all-NaN columns explicitly to avoid noisy runtime warnings.
        valid_counts = np.sum(~np.isnan(x), axis=0)
        safe_x = np.where(np.isnan(x), 0.0, x)
        x_mean = np.divide(
            np.sum(safe_x, axis=0),
            np.maximum(valid_counts, 1),
        )
        numer = np.nansum((x - x_mean) * t_centered[:, None], axis=0)
        numer = np.where(valid_counts > 0, numer, 0.0)
        slopes = numer / denom
        return pd.Series(slopes, index=[f"{c}_slope" for c in dynamic_cols])

    slopes = df.groupby("PatientID", sort=False).apply(slopes_for_group)
    if "PatientID" in slopes.columns: # Older pandas versions
        slopes = slopes.drop(columns=["PatientID"])
    return slopes


# def build_engineered_features(df: pd.DataFrame, dynamic_cols: list[str]) -> tuple[pd.DataFrame, pd.Series]:
#     """
#     Q2.1(2)-style engineered features from time series per variable:
#     - first, last, mean, min, max, std
#     - range (max-min), delta (last-first)
#     - slope (linear trend over time index)
#     plus static variables.
#     """
#     df = df.sort_values(["PatientID", "Timestamp"])
#     grouped = df.groupby("PatientID", sort=False)

#     agg = grouped[dynamic_cols].agg(["first", "last", "mean", "min", "max", "std"])
#     agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]

#     range_df = grouped[dynamic_cols].max() - grouped[dynamic_cols].min()
#     range_df = range_df.add_suffix("_range")

#     delta_df = grouped[dynamic_cols].last() - grouped[dynamic_cols].first()
#     delta_df = delta_df.add_suffix("_delta")

#     slope_df = _compute_slopes(df, dynamic_cols)

#     stat = grouped[STATIC_COLS].last()
#     labels = grouped["Label"].last().astype(int)

#     x = pd.concat([agg, range_df, delta_df, slope_df, stat], axis=1)
#     return x, labels


def build_windowed_features(
    df: pd.DataFrame,
    dynamic_cols: list[str],
    include_last12h_delta: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    df = df.sort_values(["PatientID", "Timestamp"])
    grouped = df.groupby("PatientID", sort=False)

    labels = grouped["Label"].last().astype(int)
    stat = grouped[STATIC_COLS].last()

    # Features for the whole stay
    overall = grouped[dynamic_cols].agg(["mean", "std"])
    overall.columns = [f"{col}_{agg}" for col, agg in overall.columns]

    # Features for the final 12 hours (The "State of the Patient" at handover)
    last_12h = df.sort_values("Timestamp").groupby("PatientID").tail(12)
    recent = last_12h.groupby("PatientID")[dynamic_cols].agg(["mean", "max", "min"])
    recent.columns = [f"{col}_last12h_{agg}" for col, agg in recent.columns]

    # Measurement counts (Doctor's concern proxy)
    counts = grouped[dynamic_cols].count().add_suffix("_count")

    # Deviation of last value from patient's stay mean (trend direction)
    last_val = grouped[dynamic_cols].last()
    stay_mean = grouped[dynamic_cols].mean()
    deviation = (last_val - stay_mean).add_suffix("_last_vs_mean")

    parts = [overall, recent, counts, deviation]

    if include_last12h_delta:
        # Last 12h mean vs overall mean — less noisy than single last point
        recent_mean = last_12h.groupby("PatientID")[dynamic_cols].mean()
        recent_mean = recent_mean.reindex(grouped.size().index)  # restore all patients
        delta_12h_vs_mean = (recent_mean - stay_mean).add_suffix("_last12h_vs_mean")
        parts.append(delta_12h_vs_mean)

    parts.append(stat)
    x = pd.concat(parts, axis=1)
    return x, labels



def evaluate_scores(model: Pipeline, x: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    probs = model.predict_proba(x)[:, 1]
    return {
        "auroc": float(roc_auc_score(y, probs)),
        "auprc": float(average_precision_score(y, probs)),
    }


def select_on_validation(
    base_model: Pipeline,
    grid: dict[str, list[Any]],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[Pipeline, dict[str, Any], dict[str, float]]:
    best_model: Pipeline | None = None
    best_params: dict[str, Any] = {}
    best_scores: dict[str, float] = {"auroc": -1.0, "auprc": -1.0}

    for params in ParameterGrid(grid):
        candidate = base_model.set_params(**params)
        candidate.fit(x_train, y_train)
        scores = evaluate_scores(candidate, x_val, y_val)

        better = (scores["auprc"] > best_scores["auprc"]) or (
            scores["auprc"] == best_scores["auprc"] and scores["auroc"] > best_scores["auroc"]
        )
        if better:
            best_model = candidate
            best_params = params
            best_scores = scores

    if best_model is None:
        raise RuntimeError("No model configuration was evaluated.")

    return best_model, best_params, best_scores


def build_model_specs() -> dict[str, tuple[Pipeline, dict[str, list[Any]]]]:
    return {
        "logistic_regression": (
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=3000,
                            class_weight="balanced",
                            random_state=SEED,
                        ),
                    ),
                ]
            ),
            {
                "model__C": [0.1, 0.5, 1.0, 3.0],
            },
        ),
        "random_forest": (
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        RandomForestClassifier(
                            class_weight="balanced",
                            random_state=SEED,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            {
                "model__n_estimators": [400, 800],
                "model__max_depth": [None, 12],
                "model__min_samples_leaf": [1, 2],
            },
        ),
    }


def run_experiment(
    feature_set_name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    only_models: list[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Pipeline]]:
    rows: list[dict[str, Any]] = []
    best_models: dict[str, Pipeline] = {}
    specs = build_model_specs()
    if only_models is not None:
        specs = {k: v for k, v in specs.items() if k in only_models}

    print("\n" + "#" * 72)
    print(f"Feature set: {feature_set_name} | n_features={x_train.shape[1]}")
    print("#" * 72)

    for model_name, (base_model, grid) in specs.items():
        best_model, best_params, val_scores = select_on_validation(
            base_model, grid, x_train, y_train, x_val, y_val
        )
        test_scores = evaluate_scores(best_model, x_test, y_test)

        print("\n" + "=" * 48)
        print(f"Model: {model_name}")
        print(f"Best params on Set B: {best_params}")
        print(f"Validation Set B - AuROC: {val_scores['auroc']:.4f} | AuPRC: {val_scores['auprc']:.4f}")
        print(f"Test Set C       - AuROC: {test_scores['auroc']:.4f} | AuPRC: {test_scores['auprc']:.4f}")

        rows.append(
            {
                "model": model_name,
                "feature_set": feature_set_name,
                "n_features": int(x_train.shape[1]),
                "best_params": str(best_params),
                "val_auroc": val_scores["auroc"],
                "val_auprc": val_scores["auprc"],
                "test_auroc": test_scores["auroc"],
                "test_auprc": test_scores["auprc"],
            }
        )
        best_models[model_name] = best_model

    return rows, best_models


def main() -> None:
    df_a = pd.read_parquet(SET_A_PATH)
    df_b = pd.read_parquet(SET_B_PATH)
    df_c = pd.read_parquet(SET_C_PATH)

    dynamic_cols = get_dynamic_columns(df_a)
    print(f"Dynamic variables: {len(dynamic_cols)}")
    print(f"Static variables: {len(STATIC_COLS)}")

    x_train_simple, y_train = build_simple_features(df_a, dynamic_cols)
    x_val_simple, y_val = build_simple_features(df_b, dynamic_cols)
    x_test_simple, y_test = build_simple_features(df_c, dynamic_cols)

    # LR uses 300-feature set (with _last12h_vs_mean) for importance-based pruning
    x_train_eng, _ = build_windowed_features(df_a, dynamic_cols, include_last12h_delta=True)
    x_val_eng, _ = build_windowed_features(df_b, dynamic_cols, include_last12h_delta=True)
    x_test_eng, _ = build_windowed_features(df_c, dynamic_cols, include_last12h_delta=True)

    # RF uses 263-feature set (without _last12h_vs_mean) — previously best AuPRC 0.5159
    x_train_eng_rf, _ = build_windowed_features(df_a, dynamic_cols, include_last12h_delta=False)
    x_val_eng_rf, _ = build_windowed_features(df_b, dynamic_cols, include_last12h_delta=False)
    x_test_eng_rf, _ = build_windowed_features(df_c, dynamic_cols, include_last12h_delta=False)

    print(
        f"Patients Train/Val/Test: {len(y_train)}/{len(y_val)}/{len(y_test)} | "
        f"Simple features: {x_train_simple.shape[1]} | "
        f"Engineered features: {x_train_eng.shape[1]}"
    )

    # --- Slope zero-value diagnostic ---
    slope_cols = [c for c in x_train_eng.columns if c.endswith("_slope")]
    if slope_cols:
        slope_vals = x_train_eng[slope_cols]
        zero_frac = (slope_vals == 0.0).mean()  # fraction of patients with zero slope per variable
        print(f"\n[Slope diagnostic] {len(slope_cols)} slope features")
        print(f"  Mean zero-fraction across variables: {zero_frac.mean():.1%}")
        print(f"  Variables with >50% zero slopes: {(zero_frac > 0.5).sum()}")
        top_zero = zero_frac.nlargest(5)
        print(f"  Top 5 most-zeroed slope features:\n{top_zero.to_string()}")

    rows: list[dict[str, Any]] = []

    t0 = time.perf_counter()
    simple_rows, _ = run_experiment(
            feature_set_name="simple_last(dynamic)+static",
            x_train=x_train_simple,
            y_train=y_train,
            x_val=x_val_simple,
            y_val=y_val,
            x_test=x_test_simple,
            y_test=y_test,
    )
    rows.extend(simple_rows)
    print(f"\n[Timing] simple features experiment: {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    eng_rows, eng_models = run_experiment(
            feature_set_name="engineered(first,last,mean,min,max,std,range,delta,slope)+static",
            x_train=x_train_eng,
            y_train=y_train,
            x_val=x_val_eng,
            y_val=y_val,
            x_test=x_test_eng,
            y_test=y_test,
    )
    rows.extend(eng_rows)
    print(f"\n[Timing] engineered features experiment: {time.perf_counter() - t0:.1f}s")

    # --- Feature importance analysis from the RF trained on engineered features ---
    N_TOP = 20
    rf_pipeline = eng_models["random_forest"]
    rf_model = rf_pipeline.named_steps["model"]
    importances = pd.Series(rf_model.feature_importances_, index=x_train_eng.columns)
    importances_sorted = importances.sort_values(ascending=False)
    cumulative = importances_sorted.cumsum()
    n_for_90pct = int((cumulative <= 0.90).sum()) + 1
    print(f"\n[Feature importance] Top {N_TOP} features (cover {importances_sorted.head(N_TOP).sum():.1%} of RF importance, 90% threshold = {n_for_90pct} features):")
    print(importances_sorted.head(N_TOP).to_string())

    # Use fixed top-20 features from both simple and engineered sets combined
    top_eng_features = importances_sorted.head(N_TOP).index.tolist()
    # Also include the simple last-value features so nothing from the baseline is lost
    all_top_features = list(dict.fromkeys(top_eng_features + x_train_simple.columns.tolist()))

    x_train_pruned = pd.concat([x_train_eng[top_eng_features], x_train_simple], axis=1).loc[:, ~pd.concat([x_train_eng[top_eng_features], x_train_simple], axis=1).columns.duplicated()]
    x_val_pruned   = pd.concat([x_val_eng[top_eng_features],   x_val_simple],   axis=1).loc[:, ~pd.concat([x_val_eng[top_eng_features],   x_val_simple],   axis=1).columns.duplicated()]
    x_test_pruned  = pd.concat([x_test_eng[top_eng_features],  x_test_simple],  axis=1).loc[:, ~pd.concat([x_test_eng[top_eng_features],  x_test_simple],  axis=1).columns.duplicated()]

    print(f"\n[Pruned set] {x_train_pruned.shape[1]} features: top-{N_TOP} engineered + {x_train_simple.shape[1]} simple (deduplicated)")

    # LR on pruned features, RF on full engineered features
    t0 = time.perf_counter()
    lr_pruned_rows, _ = run_experiment(
        feature_set_name=f"top{N_TOP}_engineered+simple(RF_importance)",
        x_train=x_train_pruned,
        y_train=y_train,
        x_val=x_val_pruned,
        y_val=y_val,
        x_test=x_test_pruned,
        y_test=y_test,
        only_models=["logistic_regression"],
    )
    rows.extend(lr_pruned_rows)
    print(f"\n[Timing] LR on pruned features: {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    rf_eng_rows, _ = run_experiment(
        feature_set_name="engineered_263(no_last12h_delta)",
        x_train=x_train_eng_rf,
        y_train=y_train,
        x_val=x_val_eng_rf,
        y_val=y_val,
        x_test=x_test_eng_rf,
        y_test=y_test,
        only_models=["random_forest"],
    )
    rows.extend(rf_eng_rows)
    print(f"\n[Timing] RF on 263-feature engineered set: {time.perf_counter() - t0:.1f}s")

    out_dir = Path("output/classic_ml")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / "task2_1_feature_engineering_results.csv"
    pd.DataFrame(rows).to_csv(out_file, index=False)
    print("\nSaved results to:", out_file)


if __name__ == "__main__":
    main()
