from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

INPUT = REPORTS_DIR / "unified_results.csv"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT)

print("Loading unified results...")
print(f"Shape: {df.shape}")
print(f"Regions: {df['region'].nunique()}")


# ============================================================
# 1. MODEL SUMMARY
# ============================================================

model_summary = pd.DataFrame({
    "model": [
        "Baseline",
        "ARIMA",
        "Prophet",
        "Chronos",
        "XGBoost",
    ],
    "mean_sMAPE": [
        df["baseline_sMAPE"].mean(),
        df["arima_sMAPE"].mean(),
        df["prophet_sMAPE"].mean(),
        df["chronos_sMAPE"].mean(),
        df["xgboost_sMAPE"].mean(),
    ],
    "median_sMAPE": [
        df["baseline_sMAPE"].median(),
        df["arima_sMAPE"].median(),
        df["prophet_sMAPE"].median(),
        df["chronos_sMAPE"].median(),
        df["xgboost_sMAPE"].median(),
    ],
})

model_summary.to_csv(
    REPORTS_DIR / "final_model_summary.csv",
    index=False,
)


# ============================================================
# 2. BEST MODEL DISTRIBUTION
# ============================================================

best_model = (
    df["best_model"]
    .value_counts()
    .rename_axis("model")
    .reset_index(name="regions")
)

best_model.to_csv(
    REPORTS_DIR / "final_best_model_distribution.csv",
    index=False,
)


# ============================================================
# 3. XGBOOST IMPROVEMENT
# ============================================================

xgb_improvement = pd.DataFrame({
    "region": df["region"],
    "baseline_sMAPE": df["baseline_sMAPE"],
    "xgboost_sMAPE": df["xgboost_sMAPE"],
    "xgboost_R2": df["xgboost_R2"],
    "xgboost_improvement_vs_baseline": (
        df["xgboost_improvement_vs_baseline"]
    ),
    "best_model": df["best_model"],
})

xgb_improvement = xgb_improvement.sort_values(
    "xgboost_improvement_vs_baseline",
    ascending=False,
)

xgb_improvement.to_csv(
    REPORTS_DIR / "final_xgboost_improvement.csv",
    index=False,
)


# ============================================================
# 4. CPD SUMMARY
# ============================================================

cpd = df[
    [
        "region",
        "cpd_points",
        "cpd_max_signal",
        "cpd_mean_signal",
        "strongest_cpd_year",
        "strongest_cpd_methods",
        "strongest_cpd_n_methods",
        "strongest_cpd_specificity",
        "cpd_signal",
        "strongest_cpd_direction",
        "strongest_cpd_event_type",
        "best_model",
    ]
].copy()

cpd.to_csv(
    REPORTS_DIR / "final_cpd_summary.csv",
    index=False,
)


# ============================================================
# 5. CPD × BEST MODEL
# ============================================================

cpd_model = pd.crosstab(
    df["strongest_cpd_event_type"],
    df["best_model"],
)

cpd_model.to_csv(
    REPORTS_DIR / "final_cpd_by_model.csv"
)


# ============================================================
# 6. MODEL × CPD COVERAGE
# ============================================================

df["has_cpd"] = df["cpd_points"].notna()

cpd_coverage = (
    df.groupby("best_model")["has_cpd"]
    .agg(
        regions="count",
        regions_with_cpd="sum",
    )
    .reset_index()
)

cpd_coverage["cpd_share"] = (
    cpd_coverage["regions_with_cpd"]
    / cpd_coverage["regions"]
)

cpd_coverage.to_csv(
    REPORTS_DIR / "final_cpd_coverage_by_model.csv",
    index=False,
)


# ============================================================
# 7. FINAL PROJECT METRICS
# ============================================================


metrics = {
    "regions": len(df),
    "unique_regions": df["region"].nunique(),

    "xgboost_best_regions": (
        df["best_model"] == "XGBoost"
    ).sum(),

    "arima_best_regions": (
        df["best_model"] == "ARIMA"
    ).sum(),

    "baseline_best_regions": (
        df["best_model"] == "Baseline"
    ).sum(),

    "prophet_best_regions": (
        df["best_model"] == "Prophet"
    ).sum(),

    "chronos_best_regions": (
        df["best_model"] == "Chronos"
    ).sum(),

    "baseline_mean_sMAPE": (
        df["baseline_sMAPE"].mean()
    ),

    "arima_mean_sMAPE": (
        df["arima_sMAPE"].mean()
    ),

    "prophet_mean_sMAPE": (
        df["prophet_sMAPE"].mean()
    ),

    "chronos_mean_sMAPE": (
        df["chronos_sMAPE"].mean()
    ),

    "xgboost_mean_sMAPE": (
        df["xgboost_sMAPE"].mean()
    ),

    "baseline_median_sMAPE": (
        df["baseline_sMAPE"].median()
    ),

    "arima_median_sMAPE": (
        df["arima_sMAPE"].median()
    ),

    "prophet_median_sMAPE": (
        df["prophet_sMAPE"].median()
    ),

    "chronos_median_sMAPE": (
        df["chronos_sMAPE"].median()
    ),

    "xgboost_median_sMAPE": (
        df["xgboost_sMAPE"].median()
    ),

    "xgboost_R2_mean": (
        df["xgboost_R2"].mean()
    ),

    "xgboost_R2_median": (
        df["xgboost_R2"].median()
    ),

    "arima_mean_improvement": (
        df["arima_improvement_vs_baseline"].mean()
    ),

    "xgboost_mean_improvement": (
        df["xgboost_improvement_vs_baseline"].mean()
    ),

    "prophet_mean_improvement": (
        df["prophet_improvement_vs_baseline"].mean()
    ),

    "xgboost_improved_regions": (
        df["xgboost_improvement_vs_baseline"] > 0
    ).sum(),

    "cpd_regions": df["has_cpd"].sum(),

    "cpd_coverage": df["has_cpd"].mean(),
}

metrics_df = pd.DataFrame(
    [metrics]
)

metrics_df.to_csv(
    REPORTS_DIR / "final_project_metrics.csv",
    index=False,
)


# ============================================================
# 8. FIGURE: MODEL COMPARISON
# ============================================================

fig, ax = plt.subplots(figsize=(9, 6))

models = ["Baseline", "ARIMA", "Prophet", "Chronos", "XGBoost"]
values = [
    df["baseline_sMAPE"].mean(),
    df["arima_sMAPE"].mean(),
    df["prophet_sMAPE"].mean(),
    df["chronos_sMAPE"].mean(),
    df["xgboost_sMAPE"].mean(),
]

ax.bar(models, values)

ax.set_title(
    "Mean sMAPE by forecasting model"
)

ax.set_ylabel("sMAPE, %")

ax.grid(
    axis="y",
    alpha=0.25,
)

fig.tight_layout()

fig.savefig(
    FIGURES_DIR / "final_model_comparison.png",
    dpi=200,
)

plt.close(fig)


# ============================================================
# 9. FIGURE: BEST MODEL DISTRIBUTION
# ============================================================

fig, ax = plt.subplots(figsize=(9, 6))

counts = (
    df["best_model"]
    .value_counts()
)

ax.bar(
    counts.index,
    counts.values,
)

ax.set_title(
    "Best-performing model by region"
)

ax.set_ylabel("Number of regions")

ax.grid(
    axis="y",
    alpha=0.25,
)

fig.tight_layout()

fig.savefig(
    FIGURES_DIR / "final_best_model_distribution.png",
    dpi=200,
)

plt.close(fig)


# ============================================================
# 10. FIGURE: XGBOOST IMPROVEMENT
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

improvement = (
    df["xgboost_improvement_vs_baseline"]
    .sort_values()
)

ax.hist(
    improvement,
    bins=15,
)

ax.axvline(
    0,
    linestyle="--",
)

ax.set_title(
    "XGBoost improvement vs baseline"
)

ax.set_xlabel(
    "Relative improvement"
)

ax.set_ylabel(
    "Number of regions"
)

ax.grid(
    axis="y",
    alpha=0.25,
)

fig.tight_layout()

fig.savefig(
    FIGURES_DIR / "final_xgboost_improvement_distribution.png",
    dpi=200,
)

plt.close(fig)


# ============================================================
# 11. FIGURE: CPD YEARS
# ============================================================

cpd_years = (
    df["strongest_cpd_year"]
    .dropna()
    .astype(int)
    .value_counts()
    .sort_index()
)

fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(
    cpd_years.index.astype(str),
    cpd_years.values,
)

ax.set_title(
    "Strongest CPD points by year"
)

ax.set_xlabel("Year")
ax.set_ylabel("Number of regions")

ax.grid(
    axis="y",
    alpha=0.25,
)

fig.tight_layout()

fig.savefig(
    FIGURES_DIR / "final_cpd_years.png",
    dpi=200,
)

plt.close(fig)


# ============================================================
# FINAL CONSOLE SUMMARY
# ============================================================

print()
print("=== FINAL PROJECT METRICS ===")

for key, value in metrics.items():

    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")

print()
print("Output files:")
print(REPORTS_DIR / "final_project_metrics.csv")
print(REPORTS_DIR / "final_model_summary.csv")
print(REPORTS_DIR / "final_best_model_distribution.csv")
print(REPORTS_DIR / "final_xgboost_improvement.csv")
print(REPORTS_DIR / "final_cpd_summary.csv")
print(REPORTS_DIR / "final_cpd_by_model.csv")
print(REPORTS_DIR / "final_cpd_coverage_by_model.csv")

print()
print("Figures:")
print(FIGURES_DIR / "final_model_comparison.png")
print(FIGURES_DIR / "final_best_model_distribution.png")
print(FIGURES_DIR / "final_xgboost_improvement_distribution.png")
print(FIGURES_DIR / "final_cpd_years.png")

print()
print("=== FINAL ANALYSIS COMPLETE ===")