"""Q4.1 – LLM-based ICU mortality classification via text prompting.

Pipeline
--------
1. Load Set C (test set) and optionally Set A (few-shot examples).
2. For each patient, build a compact clinical text summary from the 48-h time series.
3. Query a local Ollama model (zero-shot or few-shot with real outcome labels).
4. Parse a 0-100 mortality risk score from the model's text output.
5. Compute AuROC / AuPRC on Set C and save results to output/llm/.

Usage
-----
    python utilities/Task4_1.py                              # zero-shot, llama3 (local)
    python utilities/Task4_1.py --mode few-shot              # few-shot with Set A examples
    python utilities/Task4_1.py --model mistral              # use a different Ollama model
    python utilities/Task4_1.py --limit 50                   # quick test on first 50 patients
    python utilities/Task4_1.py --limit 50 --mode few-shot --model mistral
    python utilities/Task4_1.py --env cluster                # cluster defaults (llama3.1:latest)
    python utilities/Task4_1.py --env cluster --model gemma3:1b  # override model on cluster
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import average_precision_score, roc_auc_score

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROCESSED_DIR = Path("output/processed_sets")
OUTPUT_DIR = Path("output/llm")

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL         = "llama3"          # local Ollama (ollama pull llama3)
CLUSTER_DEFAULT_MODEL = "llama3.1:latest" # /cluster/courses/ml4h/llm/bin/ollama list

META_COLS = {"PatientID", "Timestamp", "RecordID", "Label"}
STATIC_COLS = {"Age", "Gender", "Height", "Weight"}

ICU_TYPE_MAP = {1: "CSRU", 2: "CCU", 3: "MICU", 4: "SICU"}

SEED = 42

# ---------------------------------------------------------------------------
# Variable definitions  (var_col, display_abbrev, unit, decimals)
# ---------------------------------------------------------------------------

VITAL_VARS = [
    ("HR",        "HR",     "bpm",  0),
    ("NISysABP",  "SysBP",  "mmHg", 0),
    ("NIDiasABP", "DiaBP",  "mmHg", 0),
    ("NIMAP",     "MAP",    "mmHg", 0),
    ("SysABP",    "iSysBP", "mmHg", 0),
    ("DiasABP",   "iDiaBP", "mmHg", 0),
    ("MAP",       "iMAP",   "mmHg", 0),
    ("RespRate",  "RR",     "/min", 0),
    ("Temp",      "Temp",   "°C",   1),
    ("GCS",       "GCS",    "/15",  0),
]

RESP_VARS = [
    ("MechVent", "MV",    "",     0),
    ("FiO2",     "FiO2",  "",     2),
    ("SaO2",     "SpO2",  "%",    1),
    ("PaO2",     "PaO2",  "mmHg", 0),
    ("PaCO2",    "PaCO2", "mmHg", 0),
]

METABOLIC_VARS = [
    ("pH",      "pH",  "",        2),
    ("HCO3",    "HCO3","mmol/L",  1),
    ("Glucose", "Glu", "mg/dL",   0),
    ("Lactate", "Lac", "mmol/L",  1),
]

RENAL_VARS = [
    ("BUN",        "BUN",   "mg/dL", 0),
    ("Creatinine", "Cr",    "mg/dL", 1),
    ("Urine",      "Urine", "mL",    0),
]

HEMATO_VARS = [
    ("HCT",      "HCT", "%",    1),
    ("WBC",      "WBC", "K/uL", 1),
    ("Platelets","Plt", "K/uL", 0),
]

ELECTRO_VARS = [
    ("Na", "Na", "mmol/L", 0),
    ("K",  "K",  "mmol/L", 1),
    ("Mg", "Mg", "mmol/L", 1),
]

LAB_VARS = [
    ("Albumin",    "Alb",   "g/dL",  1),
    ("Bilirubin",  "Bili",  "mg/dL", 1),
    ("Cholesterol","Chol",  "mg/dL", 0),
    ("ALP",        "ALP",   "U/L",   0),
    ("ALT",        "ALT",   "U/L",   0),
    ("AST",        "AST",   "U/L",   0),
    ("TroponinI",  "TropI", "ng/mL", 2),
    ("TroponinT",  "TropT", "ng/mL", 2),
]

ALL_SECTIONS = [
    ("Vitals",       VITAL_VARS),
    ("Resp",         RESP_VARS),
    ("Metabolic",    METABOLIC_VARS),
    ("Renal",        RENAL_VARS),
    ("Hematology",   HEMATO_VARS),
    ("Electrolytes", ELECTRO_VARS),
    ("Labs",         LAB_VARS),
]

# (col, abbrev, unit, decimals, direction)  direction = "max" | "min"
EXTREME_VARS = [
    ("HR",         "HR",    "bpm",    0, "max"),
    ("NISysABP",   "SysBP", "mmHg",   0, "min"),
    ("NIMAP",      "MAP",   "mmHg",   0, "min"),
    ("SaO2",       "SpO2",  "%",      1, "min"),
    ("PaO2",       "PaO2",  "mmHg",   0, "min"),
    ("pH",         "pH",    "",       2, "min"),
    ("Lactate",    "Lac",   "mmol/L", 1, "max"),
    ("BUN",        "BUN",   "mg/dL",  0, "max"),
    ("Creatinine", "Cr",    "mg/dL",  1, "max"),
    ("Glucose",    "Glu",   "mg/dL",  0, "max"),
    ("Temp",       "Temp",  "\u00b0C",    1, "max"),
    ("GCS",        "GCS",   "/15",    0, "min"),
]

# ---------------------------------------------------------------------------
# Text summary builder
# ---------------------------------------------------------------------------

def _fmt(val: float, decimals: int = 1) -> str:
    """Format a number; return '?' if NaN."""
    if pd.isna(val):
        return "?"
    return f"{val:.{decimals}f}"


def _trend(df: pd.DataFrame, col: str, t_mid: float) -> str:
    """Return ↑, ↓, or empty string based on first-half vs second-half mean."""
    col_df = df[["Timestamp", col]].dropna(subset=[col])
    if len(col_df) < 4:
        return ""
    first  = col_df[col_df["Timestamp"] <= t_mid][col]
    second = col_df[col_df["Timestamp"] >  t_mid][col]
    if len(first) == 0 or len(second) == 0:
        return ""
    m1, m2 = first.mean(), second.mean()
    if m1 == 0:
        return ""
    change = (m2 - m1) / abs(m1)
    if change > 0.10:
        return "\u2191"   # ↑
    elif change < -0.10:
        return "\u2193"  # ↓
    return ""


def build_short_patient_summary(patient_df: pd.DataFrame) -> str:
    """
    Compact 2-line summary for use as few-shot examples.
    Retains only the strongest mortality predictors; drops trends, labs, and
    redundant variables to minimise context-window usage.

    Format (example):
        68yo M MICU | HR 92bpm | SysBP 115mmHg | RR 18/min | Temp 37.1°C | GCS 12
        MV NO | SpO2 95% | Lac 2.1mmol/L | pH 7.35 | Cr 1.1mg/dL | BUN 28mg/dL
    """
    df = patient_df.sort_values("Timestamp")

    def _s(col: str) -> pd.Series:
        return df[col].dropna() if col in df.columns else pd.Series(dtype=float)

    # Demographics
    age = df["Age"].dropna().iloc[0] if df["Age"].notna().any() else float("nan")
    gender_code = df["Gender"].dropna().iloc[0] if df["Gender"].notna().any() else float("nan")
    gender = "M" if gender_code == 1.0 else ("F" if gender_code == 0.0 else "?")
    icu_code = (
        df["ICUType"].dropna().iloc[0]
        if "ICUType" in df.columns and df["ICUType"].notna().any()
        else float("nan")
    )
    icu_name = ICU_TYPE_MAP.get(int(icu_code), "ICU") if pd.notna(icu_code) else "ICU"
    age_str = f"{int(age)}yo" if pd.notna(age) else "?yo"

    # Line 1 — demographics + core vitals (mean, GCS min)
    parts1 = [f"{age_str} {gender} {icu_name}"]
    for col, abbrev, unit, dec in [
        ("HR",       "HR",    "bpm",  0),
        ("NISysABP", "SysBP", "mmHg", 0),
        ("RespRate", "RR",    "/min", 0),
        ("Temp",     "Temp",  "°C",   1),
    ]:
        s = _s(col)
        if len(s) > 0:
            parts1.append(f"{abbrev} {_fmt(s.mean(), dec)}{unit}")
    gcs = _s("GCS")
    if len(gcs) > 0:
        parts1.append(f"GCS {_fmt(gcs.min(), 0)}")

    # Line 2 — key severity markers (last value for labs, mean for SpO2)
    parts2: list[str] = []
    mv = _s("MechVent")
    if len(mv) > 0:
        parts2.append(f"MV {'YES' if (mv > 0).any() else 'NO'}")
    for col, abbrev, unit, dec, use_last in [
        ("SaO2",       "SpO2", "%",      1, False),
        ("Lactate",    "Lac",  "mmol/L", 1, True),
        ("pH",         "pH",   "",       2, True),
        ("Creatinine", "Cr",   "mg/dL",  1, True),
        ("BUN",        "BUN",  "mg/dL",  0, True),
    ]:
        s = _s(col)
        if len(s) == 0:
            continue
        val = s.iloc[-1] if use_last else s.mean()
        unit_str = unit if unit else ""
        parts2.append(f"{abbrev} {_fmt(val, dec)}{unit_str}")

    line1 = " | ".join(parts1)
    line2 = " | ".join(parts2)
    return "\n".join([l for l in [line1, line2] if l])


def build_patient_summary(patient_df: pd.DataFrame) -> str:
    """
    Convert one patient's 48-h ICU time series into a compact clinical text summary.

    Format (example):
        68yo M BMI=26.1 MICU
        Vitals: HR 88/92bpm↑ | SysBP 118/115mmHg | MAP 80mmHg | RR 18/min | Temp 37.1°C | GCS min12/last14↓
        Resp: MV NO | FiO2=0.45 | SpO2 96%/95% | PaO2 82mmHg
        Metabolic: pH 7.38/7.35↓ | HCO3 24mmol/L | Glu 142mg/dL | Lac 2.1/3.8mmol/L↑
        Renal: BUN 28/32mg/dL↑ | Cr 1.1mg/dL | Urine 1850mL total
        Hematology: HCT 32% | WBC 11.2K/uL | Plt 185K/uL
        Electrolytes: Na 138 | K 4.1 | Mg 1.9mmol/L
        Labs: Alb 3.2g/dL | Bili 0.8mg/dL | TropI 0.04ng/mL
        Extremes: HR peak 145bpm @h12 | SysBP nadir 68mmHg @h18 | Lac peak 4.2mmol/L @h30

    Reporting convention
    --------------------
    - Continuous vitals: mean/last  (e.g. "HR 88/92bpm")
    - Trend indicator appended: ↑ if second-half mean > first-half mean by >10%, ↓ if <-10%.
    - GCS: min/last  (worst is clinically most relevant)
    - MechVent: YES or NO
    - FiO2: last value only (as a fraction)
    - Urine: sum over the whole stay
    - Variables with zero non-NaN observations are omitted entirely.
    - Extremes: worst value (peak or nadir) with ICU hour for key clinical variables.
    """
    df = patient_df.sort_values("Timestamp")

    # Detect timestamp units (minutes vs hours) and compute midpoint for trend
    t_max = df["Timestamp"].max() if len(df) > 0 else 48
    t_scale = 60.0 if t_max > 100 else 1.0   # >100 → assume minutes
    t_mid   = df["Timestamp"].median()

    # --- Demographics ---
    age = df["Age"].dropna().iloc[0] if df["Age"].notna().any() else float("nan")
    gender_code = df["Gender"].dropna().iloc[0] if df["Gender"].notna().any() else float("nan")
    gender = "M" if gender_code == 1.0 else ("F" if gender_code == 0.0 else "?")

    height = df["Height"].dropna().iloc[0] if df["Height"].notna().any() else float("nan")
    weight_series = df["Weight"].dropna()
    weight = weight_series.iloc[-1] if len(weight_series) > 0 else float("nan")

    bmi = (
        weight / ((height / 100.0) ** 2)
        if pd.notna(height) and pd.notna(weight) and height > 0
        else float("nan")
    )

    icu_code = df["ICUType"].dropna().iloc[0] if "ICUType" in df.columns and df["ICUType"].notna().any() else float("nan")
    icu_name = ICU_TYPE_MAP.get(int(icu_code), "ICU") if pd.notna(icu_code) else "ICU"

    age_str = f"{int(age)}yo" if pd.notna(age) else "?yo"
    bmi_str = f" BMI={_fmt(bmi, 1)}" if pd.notna(bmi) else ""
    header = f"{age_str} {gender}{bmi_str} {icu_name}"

    lines = [header]

    # --- Clinical sections ---
    for section_name, var_list in ALL_SECTIONS:
        parts: list[str] = []

        for col, abbrev, unit, decimals in var_list:
            if col not in df.columns:
                continue
            series = df[col].dropna()
            if len(series) == 0:
                continue

            unit_str = unit if unit else ""

            if col == "MechVent":
                on_vent = (series > 0).any()
                parts.append(f"MV={'YES' if on_vent else 'NO'}")

            elif col == "FiO2":
                last_val = series.iloc[-1]
                parts.append(f"FiO2={_fmt(last_val, 2)}")

            elif col == "Urine":
                total = series.sum()
                parts.append(f"Urine {_fmt(total, 0)}{unit_str} total")

            elif col == "GCS":
                min_val = series.min()
                last_val = series.iloc[-1]
                trend = _trend(df, col, t_mid)
                parts.append(f"GCS min{_fmt(min_val, 0)}/last{_fmt(last_val, 0)}{trend}")

            else:
                mean_val = series.mean()
                last_val = series.iloc[-1]
                unit_display = unit_str if unit_str else ""
                trend = _trend(df, col, t_mid)
                parts.append(f"{abbrev} {_fmt(mean_val, decimals)}/{_fmt(last_val, decimals)}{unit_display}{trend}")

        if parts:
            lines.append(f"{section_name}: {' | '.join(parts)}")

    # --- Extremes section: worst value + ICU hour for key clinical variables ---
    extremes_parts: list[str] = []
    for col, abbrev, unit, decimals, direction in EXTREME_VARS:
        if col not in df.columns:
            continue
        col_df = df[["Timestamp", col]].dropna(subset=[col])
        if len(col_df) == 0:
            continue
        idx   = col_df[col].idxmax() if direction == "max" else col_df[col].idxmin()
        val   = col_df.loc[idx, col]
        hour  = int(round(col_df.loc[idx, "Timestamp"] / t_scale))
        label = "peak" if direction == "max" else "nadir"
        unit_str = unit if unit else ""
        extremes_parts.append(f"{abbrev} {label} {_fmt(val, decimals)}{unit_str} @h{hour}")

    if extremes_parts:
        lines.append(f"Extremes: {' | '.join(extremes_parts)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an experienced critical care physician reviewing ICU patient summaries. "
    "Each summary covers the first 48 hours of an ICU stay. "
    "For each patient you must predict the probability of in-hospital mortality. "
    "Reply with ONLY a single integer from 0 (certain survival) to 100 (certain death). "
    "Do not include any explanation, units, or extra text."
)


def build_zero_shot_prompt(summary: str) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"--- Patient 48-h ICU Summary ---\n{summary}\n\n"
        "Mortality risk score (0-100):"
    )


def build_few_shot_prompt(summary: str, examples: list[tuple[str, int]]) -> str:
    """
    examples: list of (patient_summary_text, label) where label ∈ {0, 1}.
    We map label → risk score: 0 → 20, 1 → 80 (softer anchors to reduce bias).
    Examples are reordered so the last one is always a survived case (label=0),
    which counteracts the model's recency bias toward the most recently seen score.
    """
    # Sort: deaths first, survivals last — recency bias then anchors toward low risk
    ordered = sorted(examples, key=lambda x: x[1], reverse=True)
    few_shot_block = ""
    for i, (ex_summary, ex_label) in enumerate(ordered, 1):
        score = 80 if ex_label == 1 else 20
        few_shot_block += (
            f"Example {i}:\n{ex_summary}\nMortality risk score (0-100): {score}\n\n"
        )

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Here are some examples to calibrate your predictions:\n\n"
        f"{few_shot_block}"
        f"Now predict for the following patient:\n"
        f"--- Patient 48-h ICU Summary ---\n{summary}\n\n"
        "Mortality risk score (0-100):"
    )


# ---------------------------------------------------------------------------
# Ollama query & response parsing
# ---------------------------------------------------------------------------

def query_ollama(prompt: str, model: str, timeout: int = 120) -> str:
    """
    Query a local Ollama instance and return the model's text response.
    Raises requests.RequestException on connection failure.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,   # deterministic
            "num_predict": 16,    # we only need a short number
        },
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return data.get("response", "").strip()


def parse_risk_score(response_text: str) -> float:
    """
    Extract a 0-100 risk score from the model's response.
    Falls back to 50.0 if no integer can be found.
    """
    # Look for the first 1-3 digit integer in the response
    match = re.search(r"\b(\d{1,3})\b", response_text)
    if match:
        val = int(match.group(1))
        return float(min(max(val, 0), 100))
    return 50.0  # uncertain fallback


# ---------------------------------------------------------------------------
# Few-shot example selection
# ---------------------------------------------------------------------------

def select_few_shot_examples(
    df_a: pd.DataFrame,
    n_per_class: int = 2,
) -> list[tuple[str, int]]:
    """
    Pick balanced examples from Set A: n_per_class died + n_per_class survived.
    Returns list of (summary_text, label).
    """
    rng = np.random.default_rng(SEED)
    labels = df_a.groupby("PatientID")["Label"].last().astype(int)

    died_ids = labels[labels == 1].index.tolist()
    survived_ids = labels[labels == 0].index.tolist()

    chosen_died = rng.choice(died_ids, size=min(n_per_class, len(died_ids)), replace=False)
    chosen_survived = rng.choice(survived_ids, size=min(n_per_class, len(survived_ids)), replace=False)

    examples = []
    for pid in list(chosen_died) + list(chosen_survived):
        pat_df = df_a[df_a["PatientID"] == pid]
        label = int(labels[pid])
        summary = build_short_patient_summary(pat_df)   # compact: saves context window
        examples.append((summary, label))

    # Shuffle so died/survived don't appear in a predictable block
    rng.shuffle(examples)
    return examples


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

CHECKPOINT_EVERY = 50   # save progress to disk every N patients


def run_llm_evaluation(
    model: str,
    mode: str,          # "zero-shot" or "few-shot"
    limit: int | None,  # max patients to evaluate (None = all)
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Determine output path early (needed for checkpoint resume) ---
    out_suffix = f"{model.replace(':', '_')}_{mode.replace('-', '_')}"
    if limit is not None:
        out_suffix += f"_n{limit}"
    detail_path = OUTPUT_DIR / f"q4_1_results_{out_suffix}.csv"

    # --- Check Ollama is reachable ---
    try:
        requests.get("http://localhost:11434", timeout=5)
    except requests.exceptions.ConnectionError:
        print(
            "ERROR: Cannot connect to Ollama at localhost:11434.\n"
            "  Local:   ollama serve\n"
            "  Cluster: OLLAMA_MODELS=/cluster/courses/ml4h/llm/models "
            "/cluster/courses/ml4h/llm/bin/ollama serve\n"
            f"Then ensure your model is available:  ollama pull {model}"
        )
        return

    # --- Load data ---
    print("Loading datasets...")
    df_c = pd.read_parquet(PROCESSED_DIR / "processed_set_c.parquet")

    few_shot_examples: list[tuple[str, int]] = []
    if mode == "few-shot":
        df_a = pd.read_parquet(PROCESSED_DIR / "processed_set_a.parquet")
        print("Building few-shot examples from Set A...")
        few_shot_examples = select_few_shot_examples(df_a, n_per_class=2)
        print(f"  Selected {len(few_shot_examples)} examples "
              f"({sum(1 for _, l in few_shot_examples if l == 1)} died, "
              f"{sum(1 for _, l in few_shot_examples if l == 0)} survived)")

    # --- Get patient list ---
    all_patient_ids = df_c["PatientID"].unique()
    labels_series = df_c.groupby("PatientID")["Label"].last().astype(int)

    if limit is not None:
        all_patient_ids = all_patient_ids[:limit]

    # --- Resume from checkpoint if it exists ---
    results: list[dict] = []
    done_ids: set = set()
    if detail_path.exists():
        existing_df = pd.read_csv(detail_path)
        done_ids = set(existing_df["patient_id"].tolist())
        results = existing_df.to_dict("records")
        print(f"Checkpoint found: {len(done_ids)} patients already done, resuming...")

    patient_ids = [pid for pid in all_patient_ids if pid not in done_ids]

    print(f"\nMode: {mode} | Model: {model} | Patients: {len(patient_ids)} remaining "
          f"({len(done_ids)} already done)")
    print("-" * 60)

    for i, pid in enumerate(patient_ids):
        pat_df = df_c[df_c["PatientID"] == pid]
        true_label = int(labels_series[pid])

        summary = build_patient_summary(pat_df)

        if mode == "few-shot":
            prompt = build_few_shot_prompt(summary, few_shot_examples)
        else:
            prompt = build_zero_shot_prompt(summary)

        t0 = time.perf_counter()
        try:
            raw_response = query_ollama(prompt, model)
            elapsed = time.perf_counter() - t0
            risk_score = parse_risk_score(raw_response)
            status = "ok"
        except requests.exceptions.RequestException as exc:
            elapsed = time.perf_counter() - t0
            raw_response = str(exc)
            risk_score = 50.0
            status = "error"

        results.append({
            "patient_id": pid,
            "true_label": true_label,
            "risk_score": risk_score,
            "raw_response": raw_response,
            "elapsed_s": round(elapsed, 2),
            "status": status,
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(patient_ids):
            done_so_far = [r for r in results if r["status"] == "ok"]
            if len(done_so_far) >= 2:
                scores = [r["risk_score"] for r in done_so_far]
                labels_so_far = [r["true_label"] for r in done_so_far]
                n_unique = len(set(scores))
                if n_unique > 1:
                    auroc_now = roc_auc_score(labels_so_far, scores)
                    print(f"[{i+1}/{len(patient_ids)}] interim AuROC={auroc_now:.4f} | "
                          f"last response: '{raw_response[:40]}'")
                else:
                    print(f"[{i+1}/{len(patient_ids)}] last response: '{raw_response[:60]}'")

        # --- Checkpoint: save to disk every CHECKPOINT_EVERY patients ---
        if (i + 1) % CHECKPOINT_EVERY == 0 or (i + 1) == len(patient_ids):
            pd.DataFrame(results).to_csv(detail_path, index=False)
            print(f"  [checkpoint] {len(results)} total results saved → {detail_path}")

    # --- Compute final metrics ---
    results_df = pd.read_csv(detail_path)   # read from checkpoint (includes any prior run)
    ok_mask = results_df["status"] == "ok"
    ok_df = results_df[ok_mask]

    print(f"\n{'='*60}")
    print(f"Results  ({mode}, model={model})")
    print(f"{'='*60}")
    print(f"Patients evaluated : {len(results_df)}")
    print(f"Successful queries : {ok_mask.sum()}")
    print(f"Errors             : {(~ok_mask).sum()}")

    if len(ok_df) >= 2 and ok_df["risk_score"].nunique() > 1:
        auroc = roc_auc_score(ok_df["true_label"], ok_df["risk_score"])
        auprc = average_precision_score(ok_df["true_label"], ok_df["risk_score"])
        print(f"AuROC              : {auroc:.4f}")
        print(f"AuPRC              : {auprc:.4f}")
    else:
        auroc = float("nan")
        auprc = float("nan")
        print("AuROC / AuPRC : insufficient data or constant predictions")

    # --- Save results ---
    out_suffix = f"{model.replace(':', '_')}_{mode.replace('-', '_')}"
    if limit is not None:
        out_suffix += f"_n{limit}"

    detail_path.unlink(missing_ok=True)   # remove checkpoint so next run starts fresh
    results_df.to_csv(detail_path, index=False)
    print(f"\nDetailed results saved to: {detail_path}")

    # Append summary row
    summary_path = OUTPUT_DIR / "q4_1_summary.csv"
    summary_row = pd.DataFrame([{
        "model": model,
        "mode": mode,
        "n_patients": len(results_df),
        "n_ok": int(ok_mask.sum()),
        "auroc": auroc,
        "auprc": auprc,
    }])
    if summary_path.exists():
        existing = pd.read_csv(summary_path)
        pd.concat([existing, summary_row], ignore_index=True).to_csv(summary_path, index=False)
    else:
        summary_row.to_csv(summary_path, index=False)
    print(f"Summary appended to : {summary_path}")

    # Print one example summary so you can inspect the text format
    print(f"\n{'='*60}")
    print("Example patient summary (first evaluated patient):")
    print("-" * 60)
    first_pid = all_patient_ids[0]
    print(build_patient_summary(df_c[df_c["PatientID"] == first_pid]))
    print("-" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Q4.1 LLM mortality prediction")
    parser.add_argument(
        "--env",
        default="local",
        choices=["local", "cluster"],
        help="Execution environment. 'cluster' sets default model to llama3.1:latest "
             "(ETH Jupyter cluster). (default: local)",
    )
    parser.add_argument("--model",  default=None,
                        help="Ollama model name. Defaults to llama3 (local) or "
                             "llama3.1:latest (cluster) unless overridden.")
    parser.add_argument("--mode",   default="zero-shot",   choices=["zero-shot", "few-shot"],
                        help="Prompting mode (default: zero-shot)")
    parser.add_argument("--limit",  type=int, default=None,
                        help="Limit to first N patients for quick testing")
    args = parser.parse_args()

    if args.model is None:
        args.model = CLUSTER_DEFAULT_MODEL if args.env == "cluster" else DEFAULT_MODEL
    print(f"[env={args.env}] model={args.model}  mode={args.mode}")

    run_llm_evaluation(model=args.model, mode=args.mode, limit=args.limit)


if __name__ == "__main__":
    main()
