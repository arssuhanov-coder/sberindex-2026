import pandas as pd
import numpy as np

PATH = "reports/unified_results.csv"

df = pd.read_csv(PATH)

df = df[df["cpd_points"].notna()].copy()

print("=" * 70)
print("CPD EVENT ANALYSIS")
print("=" * 70)

# ---------------------------------------------------------
# 1. EVENT TYPE
# ---------------------------------------------------------

print("\n=== EVENT TYPE ===")

event_summary = (
    df.groupby("strongest_cpd_event_type")
    .agg(
        regions=("region", "count"),
        mean_xgb_sMAPE=("xgboost_sMAPE", "mean"),
        median_xgb_sMAPE=("xgboost_sMAPE", "median"),
        mean_improvement=("xgboost_improvement_vs_baseline", "mean"),
        median_improvement=("xgboost_improvement_vs_baseline", "median"),
    )
    .sort_values("mean_improvement", ascending=False)
)

print(event_summary.round(3).to_string())


# ---------------------------------------------------------
# 2. DIRECTION
# ---------------------------------------------------------

print("\n=== CPD DIRECTION ===")

direction_summary = (
    df.groupby("strongest_cpd_direction")
    .agg(
        regions=("region", "count"),
        mean_xgb_sMAPE=("xgboost_sMAPE", "mean"),
        median_xgb_sMAPE=("xgboost_sMAPE", "median"),
        mean_improvement=("xgboost_improvement_vs_baseline", "mean"),
        median_improvement=("xgboost_improvement_vs_baseline", "median"),
    )
    .sort_values("mean_improvement", ascending=False)
)

print(direction_summary.round(3).to_string())


# ---------------------------------------------------------
# 3. EVENT TYPE × DIRECTION
# ---------------------------------------------------------

print("\n=== EVENT TYPE × DIRECTION ===")

cross = (
    df.groupby(
        [
            "strongest_cpd_event_type",
            "strongest_cpd_direction",
        ]
    )
    .agg(
        regions=("region", "count"),
        mean_xgb_sMAPE=("xgboost_sMAPE", "mean"),
        mean_improvement=("xgboost_improvement_vs_baseline", "mean"),
        median_improvement=("xgboost_improvement_vs_baseline", "median"),
    )
    .sort_values("mean_improvement", ascending=False)
)

print(cross.round(3).to_string())


# ---------------------------------------------------------
# 4. BEST / WORST CASES BY EVENT
# ---------------------------------------------------------

print("\n=== REGIONAL DETAILS ===")

cols = [
    "region",
    "strongest_cpd_year",
    "strongest_cpd_event_type",
    "strongest_cpd_direction",
    "cpd_max_signal",
    "strongest_cpd_specificity",
    "xgboost_sMAPE",
    "xgboost_improvement_vs_baseline",
    "best_model",
]

print(
    df[cols]
    .sort_values(
        "xgboost_improvement_vs_baseline",
        ascending=False
    )
    .head(30)
    .to_string(index=False)
)


# ---------------------------------------------------------
# 5. XGBOOST FAILURE CASES
# ---------------------------------------------------------

print("\n=== XGBOOST FAILURE CASES ===")

print(
    df[
        df["xgboost_improvement_vs_baseline"] < 0
    ][cols]
    .sort_values(
        "xgboost_improvement_vs_baseline"
    )
    .to_string(index=False)
)