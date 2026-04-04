from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROCESSED_DIR = Path("output/processed_sets")
SET_A_PATH = PROCESSED_DIR / "processed_set_a.parquet"
SET_B_PATH = PROCESSED_DIR / "processed_set_b.parquet"
SET_C_PATH = PROCESSED_DIR / "processed_set_c.parquet"

META_COLS = ["PatientID", "Timestamp", "RecordID", "Label"]
STATIC_COLS = ["Age", "Gender", "Height", "Weight"]
SEED = 42


def _get_dynamic_columns(df: pd.DataFrame) -> list[str]:
	return [c for c in df.columns if c not in META_COLS and c not in STATIC_COLS]


def build_patient_level_features(
	df: pd.DataFrame,
	dynamic_cols: list[str],
	dynamic_aggregation: str = "last",
) -> tuple[pd.DataFrame, pd.Series]:
	"""
	Build one fixed-length feature vector per patient.

	For Q2.1(1), each dynamic variable is summarized with one simple function
	(default: last measured value), and static variables are taken as one value
	per patient. This yields 37 + 4 = 41 features.
	"""
	df = df.sort_values(["PatientID", "Timestamp"])
	grouped = df.groupby("PatientID", sort=False)

	if dynamic_aggregation == "last":
		dyn = grouped[dynamic_cols].last()
	elif dynamic_aggregation == "mean":
		dyn = grouped[dynamic_cols].mean()
	elif dynamic_aggregation == "max":
		dyn = grouped[dynamic_cols].max()
	else:
		raise ValueError("dynamic_aggregation must be one of: last, mean, max")

	stat = grouped[STATIC_COLS].last()
	labels = grouped["Label"].last().astype(int)

	x = pd.concat([dyn, stat], axis=1)
	return x, labels


def evaluate_model(model: Pipeline, x: pd.DataFrame, y: pd.Series) -> dict[str, float]:
	probs = model.predict_proba(x)[:, 1]
	return {
		"auroc": float(roc_auc_score(y, probs)),
		"auprc": float(average_precision_score(y, probs)),
	}


def main() -> None:
	df_a = pd.read_parquet(SET_A_PATH)
	df_b = pd.read_parquet(SET_B_PATH)
	df_c = pd.read_parquet(SET_C_PATH)

	dynamic_cols = _get_dynamic_columns(df_a)
	print(f"Dynamic variables: {len(dynamic_cols)}")
	print(f"Static variables: {len(STATIC_COLS)}")

	x_train, y_train = build_patient_level_features(df_a, dynamic_cols, dynamic_aggregation="last")
	x_val, y_val = build_patient_level_features(df_b, dynamic_cols, dynamic_aggregation="last")
	x_test, y_test = build_patient_level_features(df_c, dynamic_cols, dynamic_aggregation="last")

	print(f"Feature dimension per patient: {x_train.shape[1]}")
	print(f"Train/Val/Test patients: {len(y_train)}/{len(y_val)}/{len(y_test)}")

	models: dict[str, Pipeline] = {
		"logistic_regression": Pipeline(
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
		"random_forest": Pipeline(
			steps=[
				("imputer", SimpleImputer(strategy="median")),
				(
					"model",
					RandomForestClassifier(
						n_estimators=500,
						class_weight="balanced",
						min_samples_leaf=2,
						random_state=SEED,
						n_jobs=-1,
					),
				),
			]
		),
	}

	rows: list[dict[str, float | str]] = []
	for name, model in models.items():
		model.fit(x_train, y_train)

		val_scores = evaluate_model(model, x_val, y_val)
		test_scores = evaluate_model(model, x_test, y_test)

		print("\n" + "=" * 48)
		print(f"Model: {name}")
		print(f"Validation Set B - AuROC: {val_scores['auroc']:.4f} | AuPRC: {val_scores['auprc']:.4f}")
		print(f"Test Set C       - AuROC: {test_scores['auroc']:.4f} | AuPRC: {test_scores['auprc']:.4f}")

		rows.append(
			{
				"model": name,
				"feature_set": "last(dynamic)+static",
				"n_features": int(x_train.shape[1]),
				"val_auroc": val_scores["auroc"],
				"val_auprc": val_scores["auprc"],
				"test_auroc": test_scores["auroc"],
				"test_auprc": test_scores["auprc"],
			}
		)

	out_dir = Path("output/classic_ml")
	out_dir.mkdir(parents=True, exist_ok=True)
	out_file = out_dir / "task2_1_baseline_results.csv"
	pd.DataFrame(rows).to_csv(out_file, index=False)
	print("\nSaved results to:", out_file)


if __name__ == "__main__":
	main()
