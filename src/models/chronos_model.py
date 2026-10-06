import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from chronos import ChronosPipeline
from sklearn.metrics import r2_score

# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "chronos.yaml"

with open(CONFIG_PATH, encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

DATA_PATH = PROJECT_ROOT / CONFIG["data"]["panel_path"]
REPORTS_DIR = PROJECT_ROOT / "reports"

TARGET = CONFIG["target"]

TRAIN_START = CONFIG["train"]["start_year"]
TRAIN_END = CONFIG["train"]["end_year"]
TEST_START = CONFIG["test"]["start_year"]
TEST_END = CONFIG["test"]["end_year"]

CHRONOS_CONFIG = CONFIG["chronos"]


# ============================================================
# METRICS
# ============================================================

def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom != 0
    if not np.any(mask):
        return 0.0
    return np.mean(2 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100


def r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return r2_score(y_true, y_pred)


# ============================================================
# MAIN
# ============================================================

def run_chronos():
    print("Loading panel...")
    df = pd.read_parquet(DATA_PATH)
    df = df[["region", "year", TARGET]].copy()

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
    }

    print("Loading Chronos pipeline...")
    pipeline = ChronosPipeline.from_pretrained(
        CHRONOS_CONFIG["model_name"],
        device_map=CHRONOS_CONFIG["device_map"],
        torch_dtype=dtype_map[CHRONOS_CONFIG["torch_dtype"]],
    )

    regions = sorted(df["region"].dropna().unique())
    print(f"Regions: {len(regions)}")

    all_preds = []
    all_metrics = []

    for i, region in enumerate(regions, start=1):
        print(f"[{i}/{len(regions)}] {region}")

        region_df = df[df["region"] == region].sort_values("year")

        train = region_df[
            (region_df["year"] >= TRAIN_START)
            & (region_df["year"] <= TRAIN_END)
            & region_df[TARGET].notna()
        ]
        test = region_df[
            (region_df["year"] >= TEST_START)
            & (region_df["year"] <= TEST_END)
            & region_df[TARGET].notna()
        ]

        if train.empty or test.empty or len(train) < 10:
            continue

        context = torch.tensor(train[TARGET].values, dtype=torch.float32)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = pipeline.predict(
                context,
                prediction_length=len(test),
            )

        # Медиана прогноза
        prediction = forecast[0].median(dim=0).values.numpy()

        preds_df = test[["region", "year", TARGET]].copy()
        preds_df["prediction"] = prediction
        all_preds.append(preds_df)

        all_metrics.append({
            "region": region,
            "model": "Chronos",
            "n_test": len(test),
            "MAE": mae(test[TARGET], prediction),
            "RMSE": rmse(test[TARGET], prediction),
            "sMAPE": smape(test[TARGET], prediction),
            "R2": r2(test[TARGET], prediction),
        })

    metrics_df = pd.DataFrame(all_metrics)
    preds_df = pd.concat(all_preds, ignore_index=True)

    summary = pd.DataFrame([{
        "model": "Chronos",
        "regions": len(metrics_df),
        "MAE_mean": metrics_df["MAE"].mean(),
        "MAE_median": metrics_df["MAE"].median(),
        "RMSE_mean": metrics_df["RMSE"].mean(),
        "RMSE_median": metrics_df["RMSE"].median(),
        "sMAPE_mean": metrics_df["sMAPE"].mean(),
        "sMAPE_median": metrics_df["sMAPE"].median(),
        "R2_mean": metrics_df["R2"].mean(),
        "R2_median": metrics_df["R2"].median(),
    }])

    metrics_df.to_csv(REPORTS_DIR / "chronos_metrics_by_region.csv", index=False)
    summary.to_csv(REPORTS_DIR / "chronos_metrics_summary.csv", index=False)
    preds_df.to_csv(REPORTS_DIR / "chronos_predictions.csv", index=False)

    print("\n=== CHRONOS SUMMARY ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    run_chronos()