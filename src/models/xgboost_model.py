import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = PROJECT_ROOT / "configs" / "xgboost.yaml"

with open(CONFIG_PATH, encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

DATA_PATH = PROJECT_ROOT / CONFIG["data"]["features_path"]

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TARGET = CONFIG["target"]

TRAIN_START = CONFIG["train"]["start_year"]
TRAIN_END = CONFIG["train"]["end_year"]

TEST_START = CONFIG["test"]["start_year"]
TEST_END = CONFIG["test"]["end_year"]


# ============================================================
# FEATURES
# ============================================================

FEATURES = CONFIG["features"]


# ============================================================
# PARAMETERS
# ============================================================

PARAMETER_GRID = CONFIG["hyperparameters_grid"]


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
# MODEL
# ============================================================

def create_model(params):

    return XGBRegressor(
        objective="reg:squarederror",

        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],

        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],

        min_child_weight=3,
        reg_alpha=0.0,
        reg_lambda=1.0,

        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )


# ============================================================
# YEAR-BASED TIME SERIES SPLITS
# ============================================================

def create_time_splits(years):
    """
    Rolling validation by complete years.

    Each validation fold contains complete years
    across all available regions.
    """

    years = sorted(
        pd.Series(years)
        .dropna()
        .unique()
    )

    if not years:
        raise ValueError(
            "No valid years available for validation."
        )

    splits = CONFIG["validation"]["splits"]

    valid_splits = []

    for split in splits:

        if (
            split["train_end"] < min(years)
            or split["validation_end"] > max(years)
        ):
            continue

        valid_splits.append(split)

    if not valid_splits:
        raise ValueError(
            "No valid time-series validation splits."
        )

    return valid_splits


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    required_columns = (
        ["region", "year", TARGET]
        + FEATURES
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing columns: "
            + ", ".join(missing_columns)
        )

    data = (
        df[
            required_columns
        ]
        .sort_values(
            ["year", "region"]
        )
        .copy()
    )

    # XGBoost не должен получать inf.
    data[FEATURES] = (
        data[FEATURES]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )

    return data


# ============================================================
# VALIDATION
# ============================================================

def evaluate_params(
    data,
    params,
    splits,
):
    """
    Evaluate one parameter configuration
    using rolling year-based validation.
    """

    fold_results = []

    for fold_number, split in enumerate(
        splits,
        start=1
    ):

        train = data[
            (data["year"] >= TRAIN_START)
            & (data["year"] <= split["train_end"])
            & data[TARGET].notna()
        ].copy()

        validation = data[
            (data["year"] >= split["validation_start"])
            & (data["year"] <= split["validation_end"])
            & data[TARGET].notna()
        ].copy()

        if train.empty or validation.empty:
            continue

        # Удаляем строки, в которых отсутствуют признаки.
        train = train.dropna(
            subset=FEATURES
        )

        validation = validation.dropna(
            subset=FEATURES
        )

        if train.empty or validation.empty:
            continue

        X_train = train[FEATURES]
        y_train = train[TARGET]

        X_val = validation[FEATURES]
        y_val = validation[TARGET]

        model = create_model(params)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            model.fit(
                X_train,
                y_train,
                verbose=False,
            )

        prediction = model.predict(
            X_val
        )

        fold_results.append({
            "fold": fold_number,
            "train_end": split["train_end"],
            "validation_start": split["validation_start"],
            "validation_end": split["validation_end"],
            "n_train": len(train),
            "n_validation": len(validation),
            "MAE": mae(
                y_val,
                prediction
            ),
            "RMSE": rmse(
                y_val,
                prediction
            ),
            "sMAPE": smape(
                y_val,
                prediction
            ),
        })

    if not fold_results:
        return None, None

    folds_df = pd.DataFrame(
        fold_results
    )

    summary = {
        "MAE": folds_df["MAE"].mean(),
        "RMSE": folds_df["RMSE"].mean(),
        "sMAPE": folds_df["sMAPE"].mean(),
    }

    return summary, folds_df


# ============================================================
# PARAMETER SELECTION
# ============================================================

def select_parameters(
    data,
    splits,
):

    print("\n=== XGBOOST PARAMETER VALIDATION ===")

    results = []

    for parameter_id, params in enumerate(
        PARAMETER_GRID,
        start=1
    ):

        print(
            f"\nConfiguration {parameter_id}/"
            f"{len(PARAMETER_GRID)}"
        )

        print(params)

        summary, folds_df = evaluate_params(
            data,
            params,
            splits,
        )

        if summary is None:
            print("No valid folds.")
            continue

        print(
            f"MAE:   {summary['MAE']:,.2f}"
        )

        print(
            f"RMSE:  {summary['RMSE']:,.2f}"
        )

        print(
            f"sMAPE: {summary['sMAPE']:.2f}%"
        )

        results.append({
            "parameter_id": parameter_id,
            **params,
            "MAE": summary["MAE"],
            "RMSE": summary["RMSE"],
            "sMAPE": summary["sMAPE"],
        })

    results_df = pd.DataFrame(
        results
    )

    if results_df.empty:
        raise RuntimeError(
            "No valid XGBoost configurations."
        )

    # Критерий выбора совпадает
    # с основным критерием сравнения моделей.
    best_row = (
        results_df
        .sort_values(
            ["sMAPE", "RMSE", "MAE"],
            ascending=True,
        )
        .iloc[0]
    )

    best_params = {
        "n_estimators": int(
            best_row["n_estimators"]
        ),
        "max_depth": int(
            best_row["max_depth"]
        ),
        "learning_rate": float(
            best_row["learning_rate"]
        ),
        "subsample": float(
            best_row["subsample"]
        ),
        "colsample_bytree": float(
            best_row["colsample_bytree"]
        ),
    }

    return (
        best_params,
        results_df,
    )


# ============================================================
# FINAL TEST
# ============================================================

def final_test(
    data,
    best_params,
):

    train = data[
        (data["year"] >= TRAIN_START)
        & (data["year"] <= TRAIN_END)
        & data[TARGET].notna()
    ].copy()

    test = data[
        (data["year"] >= TEST_START)
        & (data["year"] <= TEST_END)
        & data[TARGET].notna()
    ].copy()

    # XGBoost требует доступных признаков.
    train = train.dropna(
        subset=FEATURES
    )

    test = test.dropna(
        subset=FEATURES
    )

    if train.empty:
        raise RuntimeError(
            "Training dataset is empty after feature filtering."
        )

    if test.empty:
        raise RuntimeError(
            "Test dataset is empty after feature filtering."
        )

    print("\n=== FINAL TRAIN / TEST ===")

    print(
        f"Train years: "
        f"{TRAIN_START}–{TRAIN_END}"
    )

    print(
        f"Test years: "
        f"{TEST_START}–{TEST_END}"
    )

    print(
        f"Train rows: {len(train)}"
    )

    print(
        f"Test rows:  {len(test)}"
    )

    model = create_model(
        best_params
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        model.fit(
            train[FEATURES],
            train[TARGET],
            verbose=False,
        )

    test["prediction"] = model.predict(
        test[FEATURES]
    )

    # ---------------------------------
    # Overall metrics
    # ---------------------------------

    overall = {
        "model": "XGBoost",
        "MAE": mae(
            test[TARGET],
            test["prediction"]
        ),
        "RMSE": rmse(
            test[TARGET],
            test["prediction"]
        ),
        "sMAPE": smape(
            test[TARGET],
            test["prediction"]
        ),
        "R2": r2(
            test[TARGET],
            test["prediction"]
        ),
    }

    # ---------------------------------
    # Regional metrics
    # ---------------------------------

    regional_metrics = []

    for region, region_df in test.groupby(
        "region"
    ):

        regional_metrics.append({
            "region": region,
            "model": "XGBoost",
            "n_test": len(region_df),
            "MAE": mae(
                region_df[TARGET],
                region_df["prediction"]
            ),
            "RMSE": rmse(
                region_df[TARGET],
                region_df["prediction"]
            ),
            "sMAPE": smape(
                region_df[TARGET],
                region_df["prediction"]
            ),
            "R2": r2(
                region_df[TARGET],
                region_df["prediction"]
            ),
        })

    regional_metrics_df = pd.DataFrame(
        regional_metrics
    )

    # ---------------------------------
    # Feature importance
    # ---------------------------------

    importance_df = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.feature_importances_,
    }).sort_values(
        "importance",
        ascending=False
    )

    return (
        model,
        test,
        overall,
        regional_metrics_df,
        importance_df,
    )


# ============================================================
# MAIN
# ============================================================

def run_xgboost():

    print("Loading feature dataset...")

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

    data = prepare_data(df)

    # ---------------------------------
    # Validation splits
    # ---------------------------------

    validation_data = data[
        (data["year"] >= TRAIN_START)
        & (data["year"] <= TRAIN_END)
    ]

    splits = create_time_splits(
        validation_data["year"]
    )

    print("\n=== VALIDATION SPLITS ===")

    for split in splits:
        print(
            f"Train <= {split['train_end']} "
            f"| Validation "
            f"{split['validation_start']}–"
            f"{split['validation_end']}"
        )

    # ---------------------------------
    # Parameter selection
    # ---------------------------------

    best_params, tuning_results = (
        select_parameters(
            data,
            splits,
        )
    )

    print("\n=== BEST PARAMETERS ===")

    print(best_params)

    # ---------------------------------
    # Final test
    # ---------------------------------

    (
        model,
        predictions,
        overall,
        regional_metrics,
        importance,
    ) = final_test(
        data,
        best_params,
    )

    # ---------------------------------
    # Save tuning
    # ---------------------------------

    tuning_path = (
        REPORTS_DIR
        / "xgboost_tuning.csv"
    )

    tuning_results.to_csv(
        tuning_path,
        index=False
    )

    # ---------------------------------
    # Save predictions
    # ---------------------------------

    predictions_output = predictions[
        [
            "region",
            "year",
            TARGET,
            "prediction",
        ]
    ].rename(
        columns={
            TARGET: "actual"
        }
    )

    predictions_path = (
        REPORTS_DIR
        / "xgboost_predictions.csv"
    )

    predictions_output.to_csv(
        predictions_path,
        index=False
    )

    # ---------------------------------
    # Save regional metrics
    # ---------------------------------

    regional_path = (
        REPORTS_DIR
        / "xgboost_metrics_by_region.csv"
    )

    regional_metrics.to_csv(
        regional_path,
        index=False
    )

    # ---------------------------------
    # Save overall metrics
    # ---------------------------------

    overall_df = pd.DataFrame(
        [overall]
    )

    overall_path = (
        REPORTS_DIR
        / "xgboost_metrics_summary.csv"
    )

    overall_df.to_csv(
        overall_path,
        index=False
    )

    # ---------------------------------
    # Save feature importance
    # ---------------------------------

    importance_path = (
        REPORTS_DIR
        / "xgboost_feature_importance.csv"
    )

    importance.to_csv(
        importance_path,
        index=False
    )

    # ---------------------------------
    # Print results
    # ---------------------------------

    print("\n=== XGBOOST FINAL TEST ===")

    print(
        f"MAE:   {overall['MAE']:,.3f}"
    )

    print(
        f"RMSE:  {overall['RMSE']:,.3f}"
    )

    print(
        f"sMAPE: {overall['sMAPE']:.3f}%"
    )

    print("\n=== TOP FEATURES ===")

    print(
        importance.head(15)
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    print("\n=== OUTPUT FILES ===")

    print(
        f"Tuning:             {tuning_path}"
    )

    print(
        f"Predictions:        {predictions_path}"
    )

    print(
        f"Regional metrics:   {regional_path}"
    )

    print(
        f"Summary:            {overall_path}"
    )

    print(
        f"Feature importance: {importance_path}"
    )

    return (
        model,
        predictions,
        overall,
        regional_metrics,
        importance,
    )


# ============================================================
# PLOTS
# ============================================================

def plot_xgboost_forecasts(
    predictions,
    regions=None,
):

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if predictions.empty:
        return

    if regions is None:

        regions = (
            predictions["region"]
            .drop_duplicates()
            .head(5)
            .tolist()
        )

    for region in regions:

        region_df = (
            predictions[
                predictions["region"] == region
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
            region_df["investments"],
            marker="o",
            label="Actual",
        )

        plt.plot(
            region_df["year"],
            region_df["prediction"],
            marker="o",
            label="XGBoost",
        )

        plt.axvline(
            TRAIN_END,
            linestyle="--",
            label="Train/Test boundary",
        )

        plt.title(
            f"XGBoost forecast — {region}"
        )

        plt.xlabel("Year")
        plt.ylabel(
            "Investments, million RUB"
        )

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
            / f"xgboost_{safe_region}.png"
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
        model,
        predictions,
        overall,
        regional_metrics,
        importance,
    ) = run_xgboost()

    plot_xgboost_forecasts(
        predictions
    )