import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import r2_score
from statsmodels.tsa.arima.model import ARIMA

# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = PROJECT_ROOT / "configs" / "arima.yaml"

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


# ============================================================
# ARIMA ORDERS
# ============================================================

ARIMA_ORDERS = [
    tuple(order)
    for order in CONFIG["candidate_orders"]
]


# ============================================================
# VALIDATION SPLITS
# ============================================================

VALIDATION_SPLITS = CONFIG["validation"]["splits"]


# ============================================================
# METRICS
# ============================================================

def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return np.mean(
        np.abs(y_true - y_pred)
    )


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return np.sqrt(
        np.mean(
            (y_true - y_pred) ** 2
        )
    )


def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    denominator = (
        np.abs(y_true)
        + np.abs(y_pred)
    )

    mask = denominator != 0

    if not np.any(mask):
        return 0.0

    return np.mean(
        2
        * np.abs(
            y_true[mask]
            - y_pred[mask]
        )
        / denominator[mask]
    ) * 100


def r2(y_true, y_pred):
    """R-squared."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return r2_score(y_true, y_pred)


# ============================================================
# ARIMA FIT
# ============================================================

def fit_arima_forecast(
    train_values,
    forecast_steps,
    order,
):
    """
    Fit ARIMA on a single regional time series
    and produce a multi-step forecast.
    """

    if len(train_values) < 8:
        return None

    if not np.all(
        np.isfinite(train_values)
    ):
        return None

    try:

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            model = ARIMA(
                train_values,
                order=order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )

            fitted = model.fit()

            forecast = fitted.forecast(
                steps=forecast_steps
            )

        return np.asarray(
            forecast,
            dtype=float
        )

    except Exception:
        return None


# ============================================================
# VALIDATION FOR ONE REGION
# ============================================================

def evaluate_order_for_region(
    region_df,
    order,
):
    """
    Evaluate one ARIMA order using only
    historical rolling validation.

    The final test period 2021–2024
    is never used here.
    """

    fold_results = []

    for fold_number, split in enumerate(
        VALIDATION_SPLITS,
        start=1
    ):

        train_df = region_df[
            (region_df["year"] >= TRAIN_START)
            & (region_df["year"] <= split["train_end"])
        ].copy()

        validation_df = region_df[
            (region_df["year"] >= split["validation_start"])
            & (region_df["year"] <= split["validation_end"])
        ].copy()

        # ARIMA requires a continuous target
        # for the training segment.
        train_df = train_df.dropna(
            subset=[TARGET]
        )

        validation_df = validation_df.dropna(
            subset=[TARGET]
        )

        if train_df.empty or validation_df.empty:
            continue

        # If there are missing years inside training,
        # don't silently interpolate them.
        expected_train_years = set(
            range(
                TRAIN_START,
                split["train_end"] + 1
            )
        )

        actual_train_years = set(
            train_df["year"].astype(int)
        )

        if not expected_train_years.issubset(
            actual_train_years
        ):
            continue

        # Validation years must also be available.
        expected_validation_years = set(
            range(
                split["validation_start"],
                split["validation_end"] + 1
            )
        )

        actual_validation_years = set(
            validation_df["year"].astype(int)
        )

        if not expected_validation_years.issubset(
            actual_validation_years
        ):
            continue

        train_values = (
            train_df
            .sort_values("year")[TARGET]
            .astype(float)
            .values
        )

        validation_values = (
            validation_df
            .sort_values("year")[TARGET]
            .astype(float)
            .values
        )

        forecast = fit_arima_forecast(
            train_values,
            len(validation_values),
            order,
        )

        if forecast is None:
            continue

        fold_results.append({
            "fold": fold_number,
            "train_end": split["train_end"],
            "validation_start": split["validation_start"],
            "validation_end": split["validation_end"],
            "MAE": mae(
                validation_values,
                forecast
            ),
            "RMSE": rmse(
                validation_values,
                forecast
            ),
            "sMAPE": smape(
                validation_values,
                forecast
            ),
        })

    if not fold_results:
        return None

    return pd.DataFrame(
        fold_results
    )


# ============================================================
# SELECT ORDER FOR ONE REGION
# ============================================================

def select_order_for_region(
    region_df,
):
    """
    Select ARIMA order using historical
    validation only.
    """

    order_results = []

    for order in ARIMA_ORDERS:

        folds_df = evaluate_order_for_region(
            region_df,
            order,
        )

        if folds_df is None:
            continue

        order_results.append({
            "order": order,
            "MAE": folds_df["MAE"].mean(),
            "RMSE": folds_df["RMSE"].mean(),
            "sMAPE": folds_df["sMAPE"].mean(),
            "folds": len(folds_df),
        })

    if not order_results:
        return None

    results_df = pd.DataFrame(
        order_results
    )

    # Primary criterion: MAE
    best_row = (
        results_df
        .sort_values(
            ["MAE", "RMSE"]
        )
        .iloc[0]
    )

    return {
        "order": tuple(
            best_row["order"]
        ),
        "validation_MAE": best_row["MAE"],
        "validation_RMSE": best_row["RMSE"],
        "validation_sMAPE": best_row["sMAPE"],
        "validation_folds": int(
            best_row["folds"]
        ),
    }


# ============================================================
# FINAL FORECAST FOR ONE REGION
# ============================================================

def final_forecast_region(
    region_df,
    order,
):
    """
    Fit selected ARIMA order on 2002–2020
    and forecast 2021–2024.

    Test years are not used for order selection.
    """

    train_df = region_df[
        (region_df["year"] >= TRAIN_START)
        & (region_df["year"] <= TRAIN_END)
    ].copy()

    test_df = region_df[
        (region_df["year"] >= TEST_START)
        & (region_df["year"] <= TEST_END)
    ].copy()

    train_df = train_df.dropna(
        subset=[TARGET]
    )

    test_df = test_df.dropna(
        subset=[TARGET]
    )

    if train_df.empty or test_df.empty:
        return None

    expected_train_years = set(
        range(
            TRAIN_START,
            TRAIN_END + 1
        )
    )

    actual_train_years = set(
        train_df["year"].astype(int)
    )

    if not expected_train_years.issubset(
        actual_train_years
    ):
        return None

    train_values = (
        train_df
        .sort_values("year")[TARGET]
        .astype(float)
        .values
    )

    test_df = test_df.sort_values(
        "year"
    )

    forecast = fit_arima_forecast(
        train_values,
        len(test_df),
        order,
    )

    if forecast is None:
        return None

    test_df = test_df.copy()

    test_df["prediction"] = forecast

    return test_df


# ============================================================
# MAIN ARIMA PIPELINE
# ============================================================

def run_arima():

    print("Loading panel...")

    df = pd.read_parquet(
        DATA_PATH
    )

    print(
        f"Dataset shape: {df.shape}"
    )

    print(
        f"Regions: {df['region'].nunique()}"
    )

    print(
        f"Years: "
        f"{df['year'].min()}–"
        f"{df['year'].max()}"
    )

    df = df[
        [
            "region",
            "year",
            TARGET,
        ]
    ].copy()

    df = df.sort_values(
        ["region", "year"]
    )

    # -----------------------------------------
    # SELECT ARIMA ORDER FOR EACH REGION
    # -----------------------------------------

    print(
        "\n=== SELECTING ARIMA ORDERS ==="
    )

    selected_orders = []
    final_predictions = []
    regional_metrics = []

    regions = sorted(
        df["region"]
        .dropna()
        .unique()
    )

    for index, region in enumerate(
        regions,
        start=1
    ):

        print(
            f"[{index}/{len(regions)}] "
            f"{region}"
        )

        region_df = (
            df[
                df["region"] == region
            ]
            .sort_values("year")
            .copy()
        )

        selected = select_order_for_region(
            region_df
        )

        if selected is None:
            print(
                "  No valid ARIMA order."
            )
            continue

        order = selected["order"]

        print(
            f"  Selected: ARIMA{order}"
        )

        print(
            f"  CV MAE: "
            f"{selected['validation_MAE']:,.2f}"
        )

        print(
            f"  CV RMSE: "
            f"{selected['validation_RMSE']:,.2f}"
        )

        print(
            f"  CV sMAPE: "
            f"{selected['validation_sMAPE']:.2f}%"
        )

        selected_orders.append({
            "region": region,
            "order": str(order),
            "p": order[0],
            "d": order[1],
            "q": order[2],
            "validation_MAE": selected[
                "validation_MAE"
            ],
            "validation_RMSE": selected[
                "validation_RMSE"
            ],
            "validation_sMAPE": selected[
                "validation_sMAPE"
            ],
            "validation_folds": selected[
                "validation_folds"
            ],
        })

        # -----------------------------------------
        # FINAL TEST
        # -----------------------------------------

        prediction_df = final_forecast_region(
            region_df,
            order,
        )

        if prediction_df is None:
            continue

        final_predictions.append(
            prediction_df[
                [
                    "region",
                    "year",
                    TARGET,
                    "prediction",
                ]
            ]
        )

        regional_metrics.append({
            "region": region,
            "model": "ARIMA",
            "order": str(order),
            "n_test": len(prediction_df),
            "MAE": mae(
                prediction_df[TARGET],
                prediction_df["prediction"],
            ),
            "RMSE": rmse(
                prediction_df[TARGET],
                prediction_df["prediction"],
            ),
            "sMAPE": smape(
                prediction_df[TARGET],
                prediction_df["prediction"],
            ),
            "R2": r2(
                prediction_df[TARGET],
                prediction_df["prediction"],
            ),
        })

    # -----------------------------------------
    # CHECK RESULTS
    # -----------------------------------------

    if not regional_metrics:
        raise RuntimeError(
            "No ARIMA results were produced."
        )

    selected_orders_df = pd.DataFrame(
        selected_orders
    )

    predictions_df = pd.concat(
        final_predictions,
        ignore_index=True,
    )

    metrics_df = pd.DataFrame(
        regional_metrics
    )

    # -----------------------------------------
    # SUMMARY
    # -----------------------------------------

    summary_df = pd.DataFrame([
        {
            "model": "ARIMA",
            "regions": len(metrics_df),
            "MAE_mean": metrics_df[
                "MAE"
            ].mean(),
            "MAE_median": metrics_df[
                "MAE"
            ].median(),
            "RMSE_mean": metrics_df[
                "RMSE"
            ].mean(),
            "RMSE_median": metrics_df[
                "RMSE"
            ].median(),
            "sMAPE_mean": metrics_df[
                "sMAPE"
            ].mean(),
            "sMAPE_median": metrics_df[
                "sMAPE"
            ].median(),
            "R2_mean": metrics_df[
                "R2"
            ].mean(),
            "R2_median": metrics_df[
                "R2"
            ].median(),
        }
    ])

    # -----------------------------------------
    # SAVE
    # -----------------------------------------

    metrics_path = (
        REPORTS_DIR
        / "arima_metrics_by_region.csv"
    )

    summary_path = (
        REPORTS_DIR
        / "arima_metrics_summary.csv"
    )

    orders_path = (
        REPORTS_DIR
        / "arima_selected_orders.csv"
    )

    predictions_path = (
        REPORTS_DIR
        / "arima_predictions.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    selected_orders_df.to_csv(
        orders_path,
        index=False
    )

    predictions_df.to_csv(
        predictions_path,
        index=False
    )

    # -----------------------------------------
    # PRINT SUMMARY
    # -----------------------------------------

    print(
        "\n=== ARIMA FINAL TEST SUMMARY ==="
    )

    print(
        summary_df[
            [
                "MAE_mean",
                "MAE_median",
                "RMSE_mean",
                "RMSE_median",
                "sMAPE_mean",
                "sMAPE_median",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:,.3f}"
        )
    )

    print(
        "\n=== SELECTED ARIMA ORDERS ==="
    )

    print(
        selected_orders_df[
            [
                "order",
                "region",
            ]
        ]
        .groupby("order")
        .size()
        .reset_index(
            name="regions"
        )
        .sort_values(
            "regions",
            ascending=False
        )
        .to_string(
            index=False
        )
    )

    print(
        "\n=== OUTPUT FILES ==="
    )

    print(
        f"Metrics:     {metrics_path}"
    )

    print(
        f"Summary:     {summary_path}"
    )

    print(
        f"Orders:      {orders_path}"
    )

    print(
        f"Predictions: {predictions_path}"
    )

    # -----------------------------------------
    # REGIONS WITHOUT RESULT
    # -----------------------------------------

    result_regions = set(
        metrics_df["region"]
    )

    missing_regions = sorted(
        set(regions)
        - result_regions
    )

    print(
        f"\nRegions without ARIMA result: "
        f"{len(missing_regions)}"
    )

    for region in missing_regions:
        print(
            f"  - {region}"
        )

    return (
        summary_df,
        metrics_df,
        selected_orders_df,
        predictions_df,
    )


# ============================================================
# PLOTS
# ============================================================

def plot_arima_forecasts(
    predictions_df,
    regions=None,
):

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if predictions_df.empty:
        return

    if regions is None:

        regions = (
            predictions_df["region"]
            .drop_duplicates()
            .head(5)
            .tolist()
        )

    for region in regions:

        region_df = (
            predictions_df[
                predictions_df["region"] == region
            ]
            .sort_values("year")
        )

        if region_df.empty:
            continue

        plt.figure(
            figsize=(11, 6)
        )

        plt.plot(
            region_df["year"],
            region_df[TARGET],
            marker="o",
            label="Actual",
        )

        plt.plot(
            region_df["year"],
            region_df["prediction"],
            marker="o",
            label="ARIMA",
        )

        plt.axvline(
            TRAIN_END,
            linestyle="--",
            label="Train/Test boundary",
        )

        plt.title(
            f"ARIMA forecast — {region}"
        )

        plt.xlabel("Year")

        plt.ylabel(
            "Investments, million RUB"
        )

        plt.legend()

        plt.grid(
            alpha=0.3
        )

        plt.tight_layout()

        safe_region = (
            region
            .replace("/", "_")
            .replace(" ", "_")
        )

        output_path = (
            FIGURES_DIR
            / f"arima_{safe_region}.png"
        )

        plt.savefig(
            output_path,
            dpi=150
        )

        plt.close()

        print(
            f"Plot saved: {output_path}"
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    (
        summary_df,
        metrics_df,
        selected_orders_df,
        predictions_df,
    ) = run_arima()

    plot_arima_forecasts(
        predictions_df
    )