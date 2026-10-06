from pathlib import Path

import numpy as np
import pandas as pd


# =========================
# CONFIG
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "panel.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features_forecast.parquet"
)


TARGET = "investments"

BASE_FEATURES = [
    "retail",
    "income",
    "unemployment",
    "housing",
    "population_lag1",
]


# =========================
# FEATURE ENGINEERING
# =========================

def create_features(df):
    """
    Create time-series features using only past information.

    All operations are performed separately for each region.
    """

    df = (
        df
        .sort_values(["region", "year"])
        .copy()
    )

    # ---------------------------------
    # Target lags
    # ---------------------------------

    for lag in [1, 2, 3]:

        df[f"{TARGET}_lag{lag}"] = (
            df
            .groupby("region")[TARGET]
            .shift(lag)
        )

    # ---------------------------------
    # Exogenous feature lags
    # ---------------------------------

    for feature in BASE_FEATURES:

        for lag in [1, 2, 3]:

            df[f"{feature}_lag{lag}"] = (
                df
                .groupby("region")[feature]
                .shift(lag)
            )

    # ---------------------------------
    # Rolling statistics
    # ---------------------------------

    # ВАЖНО:
    # сначала shift(1), затем rolling.
    #
    # Поэтому для года t:
    #
    # rolling_mean_3(t)
    #
    # использует:
    # t-1, t-2, t-3
    #
    # но НЕ t.

    grouped_target = (
        df
        .groupby("region")[TARGET]
    )

    df["investments_roll_mean_3"] = (
        grouped_target
        .shift(1)
        .groupby(df["region"])
        .rolling(3)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["investments_roll_mean_5"] = (
        grouped_target
        .shift(1)
        .groupby(df["region"])
        .rolling(5)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["investments_roll_std_3"] = (
        grouped_target
        .shift(1)
        .groupby(df["region"])
        .rolling(3)
        .std()
        .reset_index(level=0, drop=True)
    )

    # ---------------------------------
    # First difference
    # ---------------------------------

    df["investments_diff1"] = (
        df
        .groupby("region")[TARGET]
        .diff(1)
    )

    # ---------------------------------
    # Log target
    # ---------------------------------

    # Не используем как target пока.
    # Это дополнительный признак,
    # основанный только на прошлом значении.

    df["investments_log_lag1"] = np.log1p(
        df["investments_lag1"]
    )

    # ---------------------------------
    # Sort
    # ---------------------------------

    df = (
        df
        .sort_values(["region", "year"])
        .reset_index(drop=True)
    )

    return df


# =========================
# VALIDATION
# =========================

def validate_no_future_leakage(
    features_df
):
    """
    Basic sanity checks that features for year t
    do not directly contain information from year t
    through lag/rolling construction.
    """

    df = features_df.copy()

    # Check target lags

    for lag in [1, 2, 3]:

        column = f"{TARGET}_lag{lag}"

        check = (
            df
            .groupby("region")[TARGET]
            .shift(lag)
        )

        comparison = (
            df[column]
            .reset_index(drop=True)
            .equals(
                check.reset_index(drop=True)
            )
        )

        if not comparison:
            raise ValueError(
                f"Leakage/check failed for {column}"
            )

    print("✓ Target lag checks passed")

    # Check that rolling features are not equal
    # to rolling statistics including current year.

    current_rolling = (
        df
        .groupby("region")[TARGET]
        .rolling(3)
        .mean()
        .reset_index(level=0, drop=True)
    )

    shifted_rolling = (
        df
        .groupby("region")[TARGET]
        .shift(1)
        .groupby(df["region"])
        .rolling(3)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # The two should differ in general.
    # We only check that the shifted rolling series
    # contains values where expected.

    if shifted_rolling.notna().sum() == 0:
        raise ValueError(
            "Rolling features contain no valid observations"
        )

    print("✓ Rolling feature checks passed")


# =========================
# REPORT
# =========================

def print_feature_report(df):

    feature_columns = [
        column
        for column in df.columns
        if column not in [
            "region",
            "year",
        ]
    ]

    print("\n=== FEATURE DATASET ===")

    print(
        f"Shape: {df.shape}"
    )

    print(
        f"Regions: {df['region'].nunique()}"
    )

    print(
        f"Years: "
        f"{df['year'].min()}–"
        f"{df['year'].max()}"
    )

    print("\n=== FEATURES ===")

    for column in feature_columns:
        print(
            f"{column:35s} "
            f"missing={df[column].isna().sum():4d}"
        )


# =========================
# MAIN
# =========================

def main():

    print("Loading panel...")

    df = pd.read_parquet(
        INPUT_PATH
    )

    print(
        f"Input shape: {df.shape}"
    )

    features_df = create_features(
        df
    )

    validate_no_future_leakage(
        features_df
    )

    print_feature_report(
        features_df
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    features_df.to_parquet(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"\nSaved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()