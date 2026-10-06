from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"

BASELINE_PATH = REPORTS_DIR / "baseline_metrics_by_region.csv"
ARIMA_PATH = REPORTS_DIR / "arima_metrics_by_region.csv"
XGBOOST_PATH = REPORTS_DIR / "xgboost_metrics_by_region.csv"
PROPHET_PATH = REPORTS_DIR / "prophet_metrics_by_region.csv"
CHRONOS_PATH = REPORTS_DIR / "chronos_metrics_by_region.csv"
CPD_PATH = REPORTS_DIR / "cpd_candidates.csv"

OUTPUT_PATH = REPORTS_DIR / "unified_results.csv"


# ============================================================
# HELPERS
# ============================================================

def load_csv(path):
    print(f"Loading {path.name}...")
    df = pd.read_csv(path)

    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    return df


def select_best_baseline(df):
    """
    Select the best baseline model for each region
    using minimum sMAPE.
    """

    required = {
        "region",
        "model",
        "MAE",
        "RMSE",
        "sMAPE",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing baseline columns: {sorted(missing)}"
        )

    df = df.copy()

    df = df.sort_values(
        ["region", "sMAPE"]
    )

    best = (
        df
        .groupby("region", as_index=False)
        .first()
    )

    return best.rename(
        columns={
            "model": "best_baseline",
            "MAE": "baseline_MAE",
            "RMSE": "baseline_RMSE",
            "sMAPE": "baseline_sMAPE",
        }
    )[
        [
            "region",
            "best_baseline",
            "baseline_MAE",
            "baseline_RMSE",
            "baseline_sMAPE",
        ]
    ]


def prepare_forecasting_results(
    baseline,
    arima,
    xgboost,
    prophet=None,
    chronos=None,
):

    baseline_best = select_best_baseline(
        baseline
    )

    arima = arima[
        [
            "region",
            "MAE",
            "RMSE",
            "sMAPE",
            "R2",
        ]
    ].rename(
        columns={
            "MAE": "arima_MAE",
            "RMSE": "arima_RMSE",
            "sMAPE": "arima_sMAPE",
            "R2": "arima_R2",
        }
    )

    xgboost = xgboost[
        [
            "region",
            "MAE",
            "RMSE",
            "sMAPE",
            "R2",
        ]
    ].rename(
        columns={
            "MAE": "xgboost_MAE",
            "RMSE": "xgboost_RMSE",
            "sMAPE": "xgboost_sMAPE",
            "R2": "xgboost_R2",
        }
    )

    result = baseline_best.merge(
        arima,
        on="region",
        how="outer",
    )

    result = result.merge(
        xgboost,
        on="region",
        how="outer",
    )

    # --------------------------------------------------------
    # Prophet (optional)
    # --------------------------------------------------------

    if prophet is not None:

        prophet = prophet[
            [
                "region",
                "MAE",
                "RMSE",
                "sMAPE",
                "R2",
            ]
        ].rename(
            columns={
                "MAE": "prophet_MAE",
                "RMSE": "prophet_RMSE",
                "sMAPE": "prophet_sMAPE",
                "R2": "prophet_R2",
            }
        )

        result = result.merge(
            prophet,
            on="region",
            how="outer",
        )
        
    # --------------------------------------------------------
    # Chronos (optional)
    # --------------------------------------------------------

    if chronos is not None:

        chronos = chronos[
            [
                "region",
                "MAE",
                "RMSE",
                "sMAPE",
                "R2",
            ]
        ].rename(
            columns={
                "MAE": "chronos_MAE",
                "RMSE": "chronos_RMSE",
                "sMAPE": "chronos_sMAPE",
                "R2": "chronos_R2",
            }
        )

        result = result.merge(
            chronos,
            on="region",
            how="outer",
        )

    # --------------------------------------------------------
    # Best forecasting model
    # --------------------------------------------------------

        model_columns = {
        "Baseline": "baseline_sMAPE",
        "ARIMA": "arima_sMAPE",
        "XGBoost": "xgboost_sMAPE",
    }

    if prophet is not None:
        model_columns["Prophet"] = "prophet_sMAPE"

    if chronos is not None:
        model_columns["Chronos"] = "chronos_sMAPE"

    def choose_model(row):

        available = {
            name: row[column]
            for name, column in model_columns.items()
            if pd.notna(row[column])
        }

        if not available:
            return np.nan

        return min(
            available,
            key=available.get,
        )

    result["best_model"] = result.apply(
        choose_model,
        axis=1,
    )

    # --------------------------------------------------------
    # Improvement relative to best baseline
    # --------------------------------------------------------

    result["arima_improvement_vs_baseline"] = (
        (
            result["baseline_sMAPE"]
            - result["arima_sMAPE"]
        )
        / result["baseline_sMAPE"]
    )

    result["xgboost_improvement_vs_baseline"] = (
        (
            result["baseline_sMAPE"]
            - result["xgboost_sMAPE"]
        )
        / result["baseline_sMAPE"]
    )

    if prophet is not None:

        result["prophet_improvement_vs_baseline"] = (
            (
                result["baseline_sMAPE"]
                - result["prophet_sMAPE"]
            )
            / result["baseline_sMAPE"]
        )

    if chronos is not None:

        result["chronos_improvement_vs_baseline"] = (
            (
                result["baseline_sMAPE"]
                - result["chronos_sMAPE"]
            )
            / result["baseline_sMAPE"]
        )

    return result


# ============================================================
# CPD AGGREGATION
# ============================================================

def prepare_cpd(cpd):

    required = {
        "region",
        "year",
        "methods",
        "n_methods",
        "specificity_score",
        "cpd_signal",
        "direction",
        "event_type",
    }

    missing = required - set(cpd.columns)

    if missing:
        raise ValueError(
            f"Missing CPD columns: {sorted(missing)}"
        )

    cpd = cpd.copy()

    # --------------------------------------------------------
    # Number of CPD points per region
    # --------------------------------------------------------

    counts = (
        cpd
        .groupby("region")
        .size()
        .rename("cpd_points")
        .reset_index()
    )

    # --------------------------------------------------------
    # Maximum CPD signal
    # --------------------------------------------------------

    max_signal = (
        cpd
        .groupby("region")["cpd_signal"]
        .max()
        .rename("cpd_max_signal")
        .reset_index()
    )

    # --------------------------------------------------------
    # Mean CPD signal
    # --------------------------------------------------------

    mean_signal = (
        cpd
        .groupby("region")["cpd_signal"]
        .mean()
        .rename("cpd_mean_signal")
        .reset_index()
    )

    # --------------------------------------------------------
    # Strongest CPD point per region
    # --------------------------------------------------------

    strongest = (
        cpd
        .sort_values(
            [
                "region",
                "cpd_signal",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .groupby("region", as_index=False)
        .first()
    )

    strongest = strongest[
        [
            "region",
            "year",
            "methods",
            "n_methods",
            "specificity_score",
            "cpd_signal",
            "direction",
            "event_type",
        ]
    ].rename(
        columns={
            "year": "strongest_cpd_year",
            "methods": "strongest_cpd_methods",
            "n_methods": "strongest_cpd_n_methods",
            "specificity_score": "strongest_cpd_specificity",
            "direction": "strongest_cpd_direction",
            "event_type": "strongest_cpd_event_type",
        }
    )

    result = counts.merge(
        max_signal,
        on="region",
        how="left",
    )

    result = result.merge(
        mean_signal,
        on="region",
        how="left",
    )

    result = result.merge(
        strongest,
        on="region",
        how="left",
    )

    return result


# ============================================================
# MAIN
# ============================================================

def run():

    print("=" * 60)
    print("UNIFIED MODEL COMPARISON")
    print("=" * 60)

    baseline = load_csv(
        BASELINE_PATH
    )

    arima = load_csv(
        ARIMA_PATH
    )

    xgboost = load_csv(
        XGBOOST_PATH
    )

    prophet = None
    if PROPHET_PATH.exists():
        prophet = load_csv(
            PROPHET_PATH
        )
    else:
        print(
            f"Prophet file not found: {PROPHET_PATH}"
        )
        
    chronos = None
    if CHRONOS_PATH.exists():
        chronos = load_csv(
            CHRONOS_PATH
        )
    else:
        print(
            f"Chronos file not found: {CHRONOS_PATH}"
        )

    cpd = load_csv(
        CPD_PATH
    )

    # --------------------------------------------------------
    # Forecasting
    # --------------------------------------------------------

    forecasting = prepare_forecasting_results(
        baseline,
        arima,
        xgboost,
        prophet,
        chronos,
    )


    print(
        f"\nForecasting regions: "
        f"{len(forecasting)}"
    )

    # --------------------------------------------------------
    # CPD
    # --------------------------------------------------------

    cpd_result = prepare_cpd(
        cpd
    )

    print(
        f"CPD regions: "
        f"{len(cpd_result)}"
    )

    # --------------------------------------------------------
    # Unified result
    # --------------------------------------------------------

    result = forecasting.merge(
        cpd_result,
        on="region",
        how="outer",
    )

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    result = result.sort_values(
        "region"
    ).reset_index(
        drop=True
    )

    if result["region"].duplicated().any():
        raise ValueError(
            "Duplicate regions found "
            "in unified results."
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    result.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n=== UNIFIED SUMMARY ===")

    print(
        f"Regions: {len(result)}"
    )

    print(
        "\nBest forecasting models:"
    )

    print(
        result["best_model"]
        .value_counts(dropna=False)
    )

    print(
        "\nCPD coverage:"
    )

    print(
        result["cpd_points"]
        .notna()
        .value_counts()
    )

    print(
        "\nOutput:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    run()