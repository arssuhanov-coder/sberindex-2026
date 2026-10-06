from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import r2_score

# =========================
# CONFIG
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = PROJECT_ROOT / "configs" / "baseline.yaml"

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

MA_WINDOWS = CONFIG["ma_windows"]


# =========================
# METRICS
# =========================

def mae(y_true, y_pred):
    """Mean Absolute Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def smape(y_true, y_pred):
    """
    Symmetric Mean Absolute Percentage Error, %.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    denominator = np.abs(y_true) + np.abs(y_pred)

    mask = denominator != 0

    if not np.any(mask):
        return 0.0

    return np.mean(
        2 * np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]
    ) * 100
    
    
def r2(y_true, y_pred):
    """R-squared."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return r2_score(y_true, y_pred)


# =========================
# BASELINE MODELS
# =========================

def naive_forecast(train_values, horizon):
    """
    Naive forecast:
    every future value equals the last observed training value.
    """
    train_values = pd.Series(train_values).dropna()

    if train_values.empty:
        return np.full(horizon, np.nan)

    last_value = train_values.iloc[-1]

    return np.repeat(last_value, horizon)


def moving_average_forecast(train_values, horizon, window):
    """
    Moving average forecast:
    every future value equals the mean of the last `window`
    observed training values.
    """
    train_values = pd.Series(train_values).dropna()

    if train_values.empty:
        return np.full(horizon, np.nan)

    values = train_values.tail(window)

    return np.repeat(values.mean(), horizon)


# =========================
# REGION EVALUATION
# =========================

def evaluate_region(region_df):
    """
    Evaluate all baseline models for one region.
    """

    region_df = (
        region_df
        .sort_values("year")
        .copy()
    )

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

    if train.empty or test.empty:
        return [], []

    y_train = train[TARGET].values
    y_test = test[TARGET].values

    years = test["year"].values
    region = region_df["region"].iloc[0]

    forecasts = []

    # -------------------------
    # Naive
    # -------------------------

    naive_pred = naive_forecast(
        y_train,
        len(y_test)
    )

    for year, actual, prediction in zip(
        years,
        y_test,
        naive_pred
    ):
        forecasts.append({
            "region": region,
            "year": year,
            "actual": actual,
            "model": "Naive",
            "prediction": prediction,
        })

    # -------------------------
    # Moving Average 3
    # -------------------------

    ma3_pred = moving_average_forecast(
        y_train,
        len(y_test),
        window=3
    )

    for year, actual, prediction in zip(
        years,
        y_test,
        ma3_pred
    ):
        forecasts.append({
            "region": region,
            "year": year,
            "actual": actual,
            "model": "MA3",
            "prediction": prediction,
        })

    # -------------------------
    # Moving Average 5
    # -------------------------

    ma5_pred = moving_average_forecast(
        y_train,
        len(y_test),
        window=5
    )

    for year, actual, prediction in zip(
        years,
        y_test,
        ma5_pred
    ):
        forecasts.append({
            "region": region,
            "year": year,
            "actual": actual,
            "model": "MA5",
            "prediction": prediction,
        })

    # -------------------------
    # Metrics
    # -------------------------

    metrics = []

    predictions_dict = {
        "Naive": naive_pred,
        "MA3": ma3_pred,
        "MA5": ma5_pred,
    }

    for model_name, predictions in predictions_dict.items():

        metrics.append({
            "region": region,
            "model": model_name,
            "train_start": TRAIN_START,
            "train_end": TRAIN_END,
            "test_start": TEST_START,
            "test_end": TEST_END,
            "n_train": len(y_train),
            "n_test": len(y_test),
            "MAE": mae(y_test, predictions),
            "RMSE": rmse(y_test, predictions),
            "sMAPE": smape(y_test, predictions),
            "R2": r2(y_test, predictions),
        })

    return metrics, forecasts


# =========================
# MAIN EVALUATION
# =========================

def run_baseline():

    print("Loading panel...")

    df = pd.read_parquet(DATA_PATH)

    print(f"Dataset shape: {df.shape}")
    print(f"Regions: {df['region'].nunique()}")
    print(f"Years: {df['year'].min()}–{df['year'].max()}")

    all_metrics = []
    all_forecasts = []

    for region, region_df in df.groupby("region"):

        metrics, forecasts = evaluate_region(region_df)

        all_metrics.extend(metrics)
        all_forecasts.extend(forecasts)

    metrics_df = pd.DataFrame(all_metrics)
    forecasts_df = pd.DataFrame(all_forecasts)

    # =========================
    # REGION-LEVEL METRICS
    # =========================

    metrics_path = REPORTS_DIR / "baseline_metrics_by_region.csv"

    metrics_df.to_csv(
        metrics_path,
        index=False
    )

    # =========================
    # AGGREGATED METRICS
    # =========================

    summary = (
        metrics_df
        .groupby("model")
        .agg(
            regions=("region", "nunique"),
            MAE_mean=("MAE", "mean"),
            MAE_median=("MAE", "median"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_median=("RMSE", "median"),
            sMAPE_mean=("sMAPE", "mean"),
            sMAPE_median=("sMAPE", "median"),
            R2_mean=("R2", "mean"),
            R2_median=("R2", "median"),
        )
        .reset_index()
    )

    summary_path = REPORTS_DIR / "baseline_metrics_summary.csv"

    summary.to_csv(
        summary_path,
        index=False
    )

    # =========================
    # PREDICTIONS
    # =========================

    predictions_path = REPORTS_DIR / "baseline_predictions.csv"

    forecasts_df.to_csv(
        predictions_path,
        index=False
    )

    # =========================
    # PRINT RESULTS
    # =========================

    print("\n=== BASELINE METRICS ===")

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:,.3f}"
        )
    )

    print("\n=== OUTPUT FILES ===")
    print(f"Metrics by region: {metrics_path}")
    print(f"Metrics summary:   {summary_path}")
    print(f"Predictions:       {predictions_path}")

    return metrics_df, forecasts_df, summary


# =========================
# PLOT SELECTED REGIONS
# =========================

def plot_baseline_forecasts(
    forecasts_df,
    regions=None
):
    """
    Plot actual vs baseline forecasts
    for selected regions.
    """

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if regions is None:

        regions = (
            forecasts_df["region"]
            .drop_duplicates()
            .head(5)
            .tolist()
        )

    for region in regions:

        region_forecasts = forecasts_df[
            forecasts_df["region"] == region
        ]

        actual = (
            region_forecasts[
                ["year", "actual"]
            ]
            .drop_duplicates()
            .sort_values("year")
        )

        plt.figure(figsize=(11, 6))

        plt.plot(
            actual["year"],
            actual["actual"],
            marker="o",
            label="Actual"
        )

        for model in ["Naive", "MA3", "MA5"]:

            model_data = (
                region_forecasts[
                    region_forecasts["model"] == model
                ]
                .sort_values("year")
            )

            plt.plot(
                model_data["year"],
                model_data["prediction"],
                marker="o",
                label=model
            )

        plt.axvline(
            TRAIN_END,
            linestyle="--",
            label="Train/Test boundary"
        )

        plt.title(
            f"Baseline forecasts — {region}"
        )

        plt.xlabel("Year")
        plt.ylabel("Investments, million RUB")

        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        safe_region = (
            region
            .replace("/", "_")
            .replace(" ", "_")
        )

        output_path = (
            FIGURES_DIR
            / f"baseline_{safe_region}.png"
        )

        plt.savefig(
            output_path,
            dpi=150
        )

        plt.close()

        print(f"Plot saved: {output_path}")


# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    metrics_df, forecasts_df, summary = run_baseline()

    plot_baseline_forecasts(
        forecasts_df
    )