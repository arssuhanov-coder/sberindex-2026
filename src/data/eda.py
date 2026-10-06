import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path


PANEL_PATH = Path("data/processed/panel.parquet")

REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"


TARGET = "investments"

FEATURES = [
    "retail",
    "income",
    "unemployment",
    "housing",
    "population_lag1",
]

ALL_INDICATORS = [
    "investments",
    "retail",
    "income",
    "unemployment",
    "housing",
    "population_lag1",
]


def main():

    # =========================================================
    # 1. Загрузка данных
    # =========================================================

    print("Loading panel...")

    df = pd.read_parquet(PANEL_PATH)

    print("\n=== DATASET INFO ===")
    print(f"Shape: {df.shape}")
    print(f"Regions: {df['region'].nunique()}")
    print(f"Years: {df['year'].min()}–{df['year'].max()}")

    # =========================================================
    # 2. Базовые проверки
    # =========================================================

    print("\n=== COLUMNS ===")
    print(df.columns.tolist())

    print("\n=== DUPLICATES ===")
    duplicates = df.duplicated(
        ["region", "year"]
    ).sum()

    print(f"Duplicate region-year rows: {duplicates}")

    print("\n=== MISSING VALUES ===")
    print(df.isna().sum())

    print("\n=== DATA TYPES ===")
    print(df.dtypes)

    # =========================================================
    # 3. Описательная статистика
    # =========================================================

    print("\n=== DESCRIPTIVE STATISTICS ===")

    descriptive = (
        df[ALL_INDICATORS]
        .describe()
        .T
    )

    print(descriptive.to_string())

    descriptive.to_csv(
        REPORTS_DIR / "descriptive_statistics.csv"
    )

    # =========================================================
    # 4. Количество наблюдений по регионам
    # =========================================================

    observations_by_region = (
        df.groupby("region")
        .size()
        .sort_values()
    )

    print("\n=== OBSERVATIONS BY REGION ===")
    print(observations_by_region.to_string())

    observations_by_region.to_csv(
        REPORTS_DIR / "observations_by_region.csv"
    )

    # =========================================================
    # 5. Пропуски
    # =========================================================

    missing = (
        df[ALL_INDICATORS]
        .isna()
        .sum()
        .sort_values(ascending=False)
    )

    print("\n=== MISSING VALUES BY INDICATOR ===")
    print(missing.to_string())

    # График пропусков
    plt.figure(figsize=(10, 6))

    missing.plot(kind="bar")

    plt.title("Missing values by indicator")
    plt.xlabel("Indicator")
    plt.ylabel("Number of missing observations")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "missing_values.png",
        dpi=200,
    )

    plt.close()

    # =========================================================
    # 6. Корреляционная матрица
    # =========================================================

    print("\n=== CORRELATION MATRIX ===")

    correlation = (
        df[ALL_INDICATORS]
        .corr()
    )

    print(correlation.round(3).to_string())

    correlation.to_csv(
        REPORTS_DIR / "correlation_matrix.csv"
    )

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
    )

    plt.title("Correlation matrix")
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "correlation_matrix.png",
        dpi=200,
    )

    plt.close()

    # =========================================================
    # 7. Динамика медианы по России
    # =========================================================

    yearly_median = (
        df.groupby("year")[ALL_INDICATORS]
        .median()
    )

    yearly_median.to_csv(
        REPORTS_DIR / "yearly_median.csv"
    )

    # ---------------------------------------------------------
    # Инвестиции
    # ---------------------------------------------------------

    plt.figure(figsize=(12, 6))

    plt.plot(
        yearly_median.index,
        yearly_median["investments"],
        marker="o",
    )

    plt.title(
        "Median regional investments in fixed capital"
    )

    plt.xlabel("Year")
    plt.ylabel("Investments, million RUB")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "investments_median_dynamics.png",
        dpi=200,
    )

    plt.close()

    # ---------------------------------------------------------
    # Все показатели
    # ---------------------------------------------------------

    for indicator in ALL_INDICATORS:

        plt.figure(figsize=(12, 6))

        plt.plot(
            yearly_median.index,
            yearly_median[indicator],
            marker="o",
        )

        plt.title(
            f"Median regional {indicator} dynamics"
        )

        plt.xlabel("Year")
        plt.ylabel(indicator)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            FIGURES_DIR / f"{indicator}_median_dynamics.png",
            dpi=200,
        )

        plt.close()

    # =========================================================
    # 8. Динамика инвестиций отдельных регионов
    # =========================================================

    # Берём несколько регионов с максимальным числом
    # наблюдений, чтобы график был читаемым.

    regions_by_observations = (
        df.groupby("region")
        .size()
        .sort_values(ascending=False)
    )

    selected_regions = (
        regions_by_observations
        .head(10)
        .index
        .tolist()
    )

    print("\n=== SELECTED REGIONS FOR INVESTMENT DYNAMICS ===")

    for region in selected_regions:
        print(region)

    plt.figure(figsize=(14, 8))

    for region in selected_regions:

        region_data = (
            df[df["region"] == region]
            .sort_values("year")
        )

        plt.plot(
            region_data["year"],
            region_data[TARGET],
            marker="o",
            linewidth=1.5,
            label=region,
        )

    plt.title(
        "Investment dynamics in selected regions"
    )

    plt.xlabel("Year")
    plt.ylabel(
        "Investments in fixed capital, million RUB"
    )

    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=8,
    )

    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "investment_regional_dynamics.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    # =========================================================
    # 9. Годовые темпы роста инвестиций
    # =========================================================

    df = df.sort_values(
        ["region", "year"]
    ).reset_index(drop=True)

    df["investments_growth"] = (
        df.groupby("region")[TARGET]
        .pct_change()
        * 100
    )

    growth_statistics = (
        df["investments_growth"]
        .describe()
    )

    print("\n=== INVESTMENTS GROWTH ===")
    print(growth_statistics.to_string())

    growth_statistics.to_csv(
        REPORTS_DIR / "investments_growth_statistics.csv"
    )

    # Медианный темп роста по годам
    yearly_growth = (
        df.groupby("year")["investments_growth"]
        .median()
    )

    yearly_growth.to_csv(
        REPORTS_DIR / "yearly_investments_growth.csv"
    )

    plt.figure(figsize=(12, 6))

    plt.plot(
        yearly_growth.index,
        yearly_growth.values,
        marker="o",
    )

    plt.axhline(
        0,
        linestyle="--",
        linewidth=1,
    )

    plt.title(
        "Median annual investment growth by region"
    )

    plt.xlabel("Year")
    plt.ylabel("Growth, %")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "investments_growth.png",
        dpi=200,
    )

    plt.close()

    # =========================================================
    # 10. Логарифмическое преобразование
    # =========================================================

    # Нужно для визуального анализа распределения.
    # Исходные значения НЕ заменяем.

    log_statistics = []

    for indicator in ALL_INDICATORS:

        values = df[indicator].dropna()

        # log1p безопасен для нулевых значений
        # и не изменяет исходный столбец.

        log_values = np.log1p(values)

        log_statistics.append({
            "indicator": indicator,
            "mean_log1p": log_values.mean(),
            "std_log1p": log_values.std(),
            "min_log1p": log_values.min(),
            "max_log1p": log_values.max(),
        })

    log_statistics = pd.DataFrame(
        log_statistics
    )

    print("\n=== LOG1P STATISTICS ===")
    print(log_statistics.to_string(index=False))

    log_statistics.to_csv(
        REPORTS_DIR / "log1p_statistics.csv",
        index=False,
    )

    # =========================================================
    # 11. Распределение target
    # =========================================================

    plt.figure(figsize=(10, 6))

    sns.histplot(
        df[TARGET].dropna(),
        bins=50,
        kde=True,
    )

    plt.title(
        "Distribution of investments"
    )

    plt.xlabel(
        "Investments, million RUB"
    )

    plt.ylabel("Frequency")
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "investments_distribution.png",
        dpi=200,
    )

    plt.close()

    # ---------------------------------------------------------
    # Логарифмическое распределение
    # ---------------------------------------------------------

    plt.figure(figsize=(10, 6))

    sns.histplot(
        np.log1p(
            df[TARGET].dropna()
        ),
        bins=50,
        kde=True,
    )

    plt.title(
        "Log1p distribution of investments"
    )

    plt.xlabel(
        "log1p(investments)"
    )

    plt.ylabel("Frequency")
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "investments_log_distribution.png",
        dpi=200,
    )

    plt.close()

    # =========================================================
    # 12. Выбросы target
    # =========================================================

    target_values = df[TARGET].dropna()

    q1 = target_values.quantile(0.25)
    q3 = target_values.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = df[
        (df[TARGET] < lower_bound)
        | (df[TARGET] > upper_bound)
    ][
        ["region", "year", TARGET]
    ].copy()

    print("\n=== INVESTMENT OUTLIERS ===")
    print(
        f"Q1: {q1:,.2f}"
    )
    print(
        f"Q3: {q3:,.2f}"
    )
    print(
        f"IQR: {iqr:,.2f}"
    )
    print(
        f"Lower bound: {lower_bound:,.2f}"
    )
    print(
        f"Upper bound: {upper_bound:,.2f}"
    )
    print(
        f"Number of outliers: {len(outliers)}"
    )

    outliers.to_csv(
        REPORTS_DIR / "investment_outliers.csv",
        index=False,
    )

    # =========================================================
    # 13. Итоговая информация
    # =========================================================

    print("\n=== EDA COMPLETED ===")

    print(
        f"Reports saved to: {REPORTS_DIR}"
    )

    print(
        f"Figures saved to: {FIGURES_DIR}"
    )


if __name__ == "__main__":
    main()