import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from prophet import Prophet
from sklearn.metrics import r2_score

# ============================================================
# CONFIG
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "prophet.yaml"

with open(CONFIG_PATH, encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

DATA_PATH = PROJECT_ROOT / CONFIG["data"]["panel_path"]
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TARGET = CONFIG["target"]
TRAIN_START = CONFIG["train"]["start_year"]
TRAIN_END = CONFIG["train"]["end_year"]
TEST_START = CONFIG["test"]["start_year"]
TEST_END = CONFIG["test"]["end_year"]

PROPHET_PARAMS = CONFIG["prophet"]


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


def forecast_region(region_df):
    region_df = region_df.sort_values("year").copy()

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

    if train.empty or test.empty or len(train) < 8:
        return None

    train_prophet = pd.DataFrame({
        "ds": pd.to_datetime(train["year"].astype(int).astype(str), format="%Y"),
        "y": train[TARGET].values,
    })

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = Prophet(**PROPHET_PARAMS)
        model.fit(train_prophet)

    future = pd.DataFrame({
        "ds": pd.to_datetime(test["year"].astype(int).astype(str), format="%Y"),
    })
    forecast = model.predict(future)

    test = test.copy()
    test["prediction"] = forecast["yhat"].values

    return test


def run_prophet():
    print("Loading panel...")
    df = pd.read_parquet(DATA_PATH)
    df = df[["region", "year", TARGET]].copy()

    regions = sorted(df["region"].dropna().unique())
    print(f"Regions: {len(regions)}")

    all_preds = []
    all_metrics = []

    for i, region in enumerate(regions, start=1):
        print(f"[{i}/{len(regions)}] {region}")

        region_df = df[df["region"] == region]
        result = forecast_region(region_df)

        if result is None:
            continue

        all_preds.append(result[["region", "year", TARGET, "prediction"]])

        all_metrics.append({
            "region": region,
            "model": "Prophet",
            "n_test": len(result),
            "MAE": mae(result[TARGET], result["prediction"]),
            "RMSE": rmse(result[TARGET], result["prediction"]),
            "sMAPE": smape(result[TARGET], result["prediction"]),
            "R2": r2(result[TARGET], result["prediction"]),
        })

    preds_df = pd.concat(all_preds, ignore_index=True)
    metrics_df = pd.DataFrame(all_metrics)

    summary = pd.DataFrame([{
        "model": "Prophet",
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

    metrics_df.to_csv(REPORTS_DIR / "prophet_metrics_by_region.csv", index=False)
    summary.to_csv(REPORTS_DIR / "prophet_metrics_summary.csv", index=False)
    preds_df.to_csv(REPORTS_DIR / "prophet_predictions.csv", index=False)

    print("\n=== PROPHET SUMMARY ===")
    print(summary.to_string(index=False))

    return summary


if __name__ == "__main__":
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    run_prophet()