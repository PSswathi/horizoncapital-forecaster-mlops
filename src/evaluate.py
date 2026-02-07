"""
evaluate.py — Evaluate DeepAR Batch Transform predictions
Runs inside a SageMaker Processing container (SKLearnProcessor)

Input:
  - /opt/ml/processing/input/predictions/  (Batch Transform output — JSON Lines)
  - /opt/ml/processing/input/actuals/      (actuals.csv from preprocess step)
Output:
  - /opt/ml/processing/output/evaluation/evaluation.json

The evaluation.json is read by the pipeline's ConditionStep
to decide whether to register the model.
"""

import json
import os

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────
# Metrics  (same formulas as NB06 Step 7)
# ──────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    """Compute RMSE, MAE, and R²."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {"rmse": rmse, "mae": mae, "r2": r2}


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":

    predictions_dir = "/opt/ml/processing/input/predictions"
    actuals_dir = "/opt/ml/processing/input/actuals"
    output_dir = "/opt/ml/processing/output/evaluation"

    # ── 1. Load Batch Transform predictions ──
    #    Each line is one state's forecast: {"quantiles": {...}, "mean": [...]}
    pred_file = None
    for fname in sorted(os.listdir(predictions_dir)):
        if fname.endswith(".out") or fname.endswith(".json"):
            pred_file = os.path.join(predictions_dir, fname)
            break

    if pred_file is None:
        # Fallback: take the first file found
        files = os.listdir(predictions_dir)
        if files:
            pred_file = os.path.join(predictions_dir, files[0])
        else:
            raise FileNotFoundError(f"No prediction files in {predictions_dir}")

    print(f"Reading predictions from: {pred_file}")

    # Read the file
    with open(pred_file, "r") as f:
        content = f.read().strip()

    # Handle concatenated JSON objects (}{ without newline separator)
    if content.startswith("{") and "}{" in content:
        json_strings = content.replace("}{", "}\n{").split("\n")
    else:
        json_strings = content.split("\n")

    all_means = []
    for line in json_strings:
        line = line.strip()
        if not line:
            continue
        pred = json.loads(line)
        if "mean" in pred:
            all_means.append(pred["mean"])

    print(f"Loaded predictions for {len(all_means)} states")

    # Average across all states (NFCI is a national index — same target)
    pred_mean = np.mean(all_means, axis=0).tolist()

    # ── 2. Load actual values ──
    actuals_file = os.path.join(actuals_dir, "actuals.csv")
    actuals_df = pd.read_csv(actuals_file)
    y_actual = actuals_df["actual_nfci"].tolist()

    # Match lengths (use the shorter one)
    n = min(len(pred_mean), len(y_actual))
    pred_mean = pred_mean[:n]
    y_actual = y_actual[:n]

    print(f"Comparing {n} forecast months")

    # ── 3. Compute metrics ──
    metrics = compute_metrics(y_actual, pred_mean)

    print(f"\nEvaluation Results:")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  R²:   {metrics['r2']:.4f}")

    # ── 4. Write evaluation report ──
    #    The ConditionStep reads metrics.rmse from this file
    os.makedirs(output_dir, exist_ok=True)

    report = {
        "metrics": metrics,
        "num_states_averaged": len(all_means),
        "num_forecast_months": n,
    }

    eval_path = os.path.join(output_dir, "evaluation.json")
    with open(eval_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n✓ Saved: {eval_path}")
    print("Evaluation complete.")
