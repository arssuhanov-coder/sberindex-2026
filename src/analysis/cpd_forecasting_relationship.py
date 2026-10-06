import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

PATH = "reports/unified_results.csv"

df = pd.read_csv(PATH)


# ============================================================
# PREPARATION
# ============================================================
#
# После tolerance=1 год CPD есть у всех 85 регионов.
# Делим по силе сигнала (медиана cpd_max_signal).
#

median_signal = df["cpd_max_signal"].median()
df["strong_cpd"] = df["cpd_max_signal"] > median_signal


# ============================================================
# 1. STRONG CPD VS WEAK CPD
# ============================================================

print("=" * 70)
print("STRONG CPD VS WEAK CPD")
print("=" * 70)
print(f"Median cpd_max_signal: {median_signal:.4f}\n")

summary = (
    df.groupby("strong_cpd")
    .agg(
        regions=("region", "count"),
        mean_xgb_sMAPE=("xgboost_sMAPE", "mean"),
        median_xgb_sMAPE=("xgboost_sMAPE", "median"),
        mean_xgb_improvement=("xgboost_improvement_vs_baseline", "mean"),
        median_xgb_improvement=("xgboost_improvement_vs_baseline", "median"),
    )
)

print(summary.to_string())


# ============================================================
# 2. CORRELATIONS
# ============================================================

print("\n" + "=" * 70)
print("CORRELATIONS WITH XGBOOST")
print("=" * 70)

corr_columns = [
    "cpd_points",
    "cpd_max_signal",
    "cpd_mean_signal",
    "strongest_cpd_specificity",
    "xgboost_sMAPE",
    "xgboost_improvement_vs_baseline",
]

print(
    df[corr_columns]
    .corr(numeric_only=True)
    .round(3)
    .to_string()
)


# ============================================================
# 3. STRONGEST CPD SIGNALS
# ============================================================

print("\n" + "=" * 70)
print("STRONGEST CPD SIGNALS")
print("=" * 70)

cols = [
    "region",
    "cpd_points",
    "cpd_max_signal",
    "cpd_mean_signal",
    "strongest_cpd_year",
    "strongest_cpd_methods",
    "strongest_cpd_specificity",
    "xgboost_sMAPE",
    "xgboost_improvement_vs_baseline",
    "best_model",
]

print(
    df
    .sort_values("cpd_max_signal", ascending=False)
    [cols]
    .head(20)
    .to_string(index=False)
)


# ============================================================
# 4. STRONG CPD + STRONG XGBOOST IMPROVEMENT
# ============================================================

print("\n" + "=" * 70)
print("STRONG CPD + STRONG XGBOOST IMPROVEMENT")
print("=" * 70)

interesting = df[
    df["strong_cpd"]
    & df["xgboost_improvement_vs_baseline"].notna()
].copy()

interesting["combined_score"] = (
    interesting["cpd_max_signal"]
    * interesting["xgboost_improvement_vs_baseline"]
)

print(
    interesting
    .sort_values("combined_score", ascending=False)
    [
        [
            "region",
            "cpd_max_signal",
            "strongest_cpd_year",
            "strongest_cpd_specificity",
            "xgboost_sMAPE",
            "xgboost_improvement_vs_baseline",
            "combined_score",
        ]
    ]
    .head(20)
    .to_string(index=False)
)


# ============================================================
# 5. REGIONS WHERE XGBOOST IS WORSE THAN BASELINE
# ============================================================

print("\n" + "=" * 70)
print("REGIONS WHERE XGBOOST IS WORSE THAN BASELINE")
print("=" * 70)

print(
    df[df["xgboost_improvement_vs_baseline"] < 0]
    [
        [
            "region",
            "cpd_points",
            "cpd_max_signal",
            "strongest_cpd_year",
            "strongest_cpd_specificity",
            "xgboost_sMAPE",
            "baseline_sMAPE",
            "xgboost_improvement_vs_baseline",
            "best_model",
        ]
    ]
    .sort_values("xgboost_improvement_vs_baseline")
    .to_string(index=False)
)