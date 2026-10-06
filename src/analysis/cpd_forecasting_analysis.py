import pandas as pd

df = pd.read_csv(
    "reports/unified_results.csv"
)

print("=== FORECASTING ===")

print("\nSMAPE:")
print(
    df[
    [
        "baseline_sMAPE",
        "arima_sMAPE",
        "prophet_sMAPE",
        "chronos_sMAPE",
        "xgboost_sMAPE",
    ]
].describe()
)

print("\n=== XGBOOST IMPROVEMENT ===")

print(
    df["xgboost_improvement_vs_baseline"]
    .describe()
)

print("\n=== WORST XGBOOST ===")

print(
    df[
        [
            "region",
            "xgboost_sMAPE",
            "baseline_sMAPE",
            "xgboost_improvement_vs_baseline",
            "best_model",
        ]
    ]
    .sort_values(
        "xgboost_sMAPE",
        ascending=False
    )
    .head(20)
    .to_string(index=False)
)

print("\n=== BEST XGBOOST ===")

print(
    df[
        [
            "region",
            "xgboost_sMAPE",
            "baseline_sMAPE",
            "xgboost_improvement_vs_baseline",
            "best_model",
        ]
    ]
    .sort_values(
        "xgboost_sMAPE"
    )
    .head(20)
    .to_string(index=False)
)