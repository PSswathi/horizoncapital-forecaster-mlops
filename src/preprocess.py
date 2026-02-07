"""
preprocess.py — Data preparation for DeepAR Pipeline
Runs inside a SageMaker Processing container (SKLearnProcessor)

Replicates the manual logic from:
  - Notebook 04 (handle missing values)
  - Notebook 06 (DeepAR JSON Lines conversion)

Input:  /opt/ml/processing/input/  (raw parquet or CSV)
Output:
  - /opt/ml/processing/output/train/train.json          (DeepAR training data)
  - /opt/ml/processing/output/test/test.json             (DeepAR test channel)
  - /opt/ml/processing/output/inference/inference.json   (Batch Transform input)
  - /opt/ml/processing/output/actuals/actuals.csv        (Actual NFCI values for evaluation)
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────
# Configuration (matches Notebook 06)
# ──────────────────────────────────────────────────────────────
TARGET_COLUMN = "NFCI"

# Dynamic features for DeepAR (from NB06 Step 4)
NATIONAL_FEATURES = [
    "UNRATE",
    "FEDFUNDS",
    "BAMLH0A0HYM2",
    "SPREAD_10Y_2Y",
    "CPIAUCSL",
]
STATE_FEATURES = [
    "B19013_001E",
    "B25077_001E",
    "B25064_001E",
    "B17001_002E",
]
DYNAMIC_FEATURES = NATIONAL_FEATURES + STATE_FEATURES

# Production hold-out ratio (from NB04)
PRODUCTION_RATIO = 0.20


# ──────────────────────────────────────────────────────────────
# Step A: Handle Missing Values  (from NB04)
# ──────────────────────────────────────────────────────────────
def handle_missing_values(df):
    """Drop unusable columns and fill missing values."""
    df = df.copy()

    # Drop MSPUS (67% missing) and state_name (categorical, not for modeling)
    for col in ["MSPUS", "state_name"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Forward-fill then back-fill for known sparse columns
    for col in ["PPIFDG", "CPI_YOY"]:
        if col in df.columns:
            df[col] = df.groupby("state_fips")[col].ffill()
            df[col] = df.groupby("state_fips")[col].bfill()

    return df


# ──────────────────────────────────────────────────────────────
# Step B: Fill Dynamic Features  (safety net for DeepAR)
# ──────────────────────────────────────────────────────────────
def fill_dynamic_features(df):
    """Ensure dynamic features have no NaN (DeepAR requirement)."""
    df = df.copy()
    for feat in DYNAMIC_FEATURES:
        if feat in df.columns:
            df[feat] = df.groupby("state_fips")[feat].ffill()
            df[feat] = df.groupby("state_fips")[feat].bfill()
    return df


# ──────────────────────────────────────────────────────────────
# Step C: Convert to DeepAR JSON Lines  (from NB06 Step 4)
# ──────────────────────────────────────────────────────────────
def build_deepar_records(df, states, dates, n_train_months, prediction_length):
    """
    Build three sets of DeepAR JSON records:

    train_records:     target = history only (for training)
    test_records:      target = full series  (for DeepAR test channel)
    inference_records: target = history, dynamic_feat extends into forecast
    """
    start_date = str(dates[0])[:10]  # "YYYY-MM-DD"

    train_records = []
    test_records = []
    inference_records = []

    for state in states:
        state_data = df[df["state_fips"] == state].sort_values("date")
        target_values = state_data[TARGET_COLUMN].tolist()

        # Build dynamic feature arrays for this state
        dyn_feat = []
        for feat in DYNAMIC_FEATURES:
            values = state_data[feat].fillna(method="ffill").fillna(method="bfill").tolist()
            dyn_feat.append(values)

        # 1) Train: history only
        train_records.append({
            "start": start_date,
            "target": target_values[:n_train_months],
            "dynamic_feat": [f[:n_train_months] for f in dyn_feat],
        })

        # 2) Test channel: full series (DeepAR evaluates on the extra months)
        test_records.append({
            "start": start_date,
            "target": target_values,
            "dynamic_feat": dyn_feat,
        })

        # 3) Inference: history in target, dynamic_feat covers forecast horizon
        inference_records.append({
            "start": start_date,
            "target": target_values[:n_train_months],
            "dynamic_feat": [f[: n_train_months + prediction_length] for f in dyn_feat],
        })

    return train_records, test_records, inference_records


def save_json_lines(records, filepath):
    """Write a list of dicts as JSON Lines (one JSON object per line)."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-length", type=int, default=12)
    args = parser.parse_args()

    prediction_length = args.prediction_length

    # ── SageMaker Processing paths ──
    input_dir = "/opt/ml/processing/input"
    output_train = "/opt/ml/processing/output/train"
    output_test = "/opt/ml/processing/output/test"
    output_inference = "/opt/ml/processing/output/inference"
    output_actuals = "/opt/ml/processing/output/actuals"

    # ── 1. Load data ──
    parquet_file = os.path.join(input_dir, "state_month_full.parquet")
    csv_file = os.path.join(input_dir, "state_month_full.csv")

    if os.path.exists(parquet_file):
        df = pd.read_parquet(parquet_file)
    elif os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
    else:
        raise FileNotFoundError(f"No data file found in {input_dir}")

    df["date"] = pd.to_datetime(df["date"])
    print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # ── 2. Handle missing values ──
    df = handle_missing_values(df)
    df = fill_dynamic_features(df)
    print(f"After cleaning: {df.shape}")

    # ── 3. Filter to non-production dates (first 80%) ──
    all_dates = sorted(df["date"].unique())
    cutoff_idx = int(len(all_dates) * (1 - PRODUCTION_RATIO))
    model_dates = all_dates[:cutoff_idx]

    df_model = df[df["date"].isin(model_dates)].copy()

    states = sorted(df_model["state_fips"].unique())
    dates = sorted(df_model["date"].unique())
    n_months = len(dates)
    n_train_months = n_months - prediction_length

    print(f"\nStates: {len(states)}")
    print(f"Months: {n_months}  (train={n_train_months}, test={prediction_length})")
    print(f"Date range: {str(dates[0])[:10]} to {str(dates[-1])[:10]}")

    # ── 4. Convert to DeepAR JSON Lines ──
    train_recs, test_recs, inference_recs = build_deepar_records(
        df_model, states, dates, n_train_months, prediction_length
    )

    # ── 5. Save outputs ──
    save_json_lines(train_recs, os.path.join(output_train, "train.json"))
    save_json_lines(test_recs, os.path.join(output_test, "test.json"))
    save_json_lines(inference_recs, os.path.join(output_inference, "inference.json"))

    # Save actual NFCI values for the test period (used by evaluate.py)
    one_state = df_model[df_model["state_fips"] == states[0]].sort_values("date")
    actual_values = one_state[TARGET_COLUMN].tolist()[n_train_months:]
    actual_dates = [str(d)[:10] for d in dates[n_train_months:]]

    os.makedirs(output_actuals, exist_ok=True)
    pd.DataFrame({"date": actual_dates, "actual_nfci": actual_values}).to_csv(
        os.path.join(output_actuals, "actuals.csv"), index=False
    )

    print(f"\n✓ train.json      ({len(train_recs)} records, {n_train_months} months each)")
    print(f"✓ test.json       ({len(test_recs)} records, {n_months} months each)")
    print(f"✓ inference.json  ({len(inference_recs)} records)")
    print(f"✓ actuals.csv     ({len(actual_values)} values)")
    print("Preprocessing complete.")
