import pandas as pd
from pathlib import Path


RAW_PATH = Path("data/raw/data_regions_collection_102_v20260313.parquet")
OUTPUT_PATH = Path("data/processed/panel.parquet")


INDICATORS = {
    "Y477110107": "investments",
    "Y477110223": "retail",
    "Y477110374": "income",
    "Y477110418": "unemployment",
    "Y477110373": "population",
    "Y477110016": "housing",
}


def main():
    print("Loading raw data...")

    columns = [
        "indicator_code",
        "indicator_name",
        "indicator_unit",
        "object_name",
        "object_level",
        "year",
        "indicator_value",
    ]

    df = pd.read_parquet(RAW_PATH, columns=columns)

    # Оставляем только утверждённые показатели
    df = df[df["indicator_code"].isin(INDICATORS)].copy()

    # Только региональный уровень
    df = df[df["object_level"] == "Регион"].copy()
    EXCLUDED_REGIONS = [
        "Архангельская область (с автономным округом)",
        "Тюменская область (с автономными округами)",
    ]

    df = df[~df["object_name"].isin(EXCLUDED_REGIONS)].copy()
    print(f"Rows after filtering: {len(df):,}")

    # ---------------------------------------------------------
    # Обработка специальных кодов пропущенных значений
    # ---------------------------------------------------------

    SENTINELS_MISSING = [
        -99999999,
        -77777777,
    ]

    for sentinel in SENTINELS_MISSING:
        count = (df["indicator_value"] == sentinel).sum()
        print(f"Source sentinel values ({sentinel}): {count:,}")

    df["indicator_value"] = df["indicator_value"].replace(
        SENTINELS_MISSING,
        pd.NA,
    )

    # ---------------------------------------------------------
    # Проверка дублей
    # ---------------------------------------------------------

    duplicates = (
        df.groupby(["object_name", "year", "indicator_code"])
        .size()
        .reset_index(name="n")
    )

    duplicates = duplicates[duplicates["n"] > 1]

    if not duplicates.empty:
        raise ValueError(
            "Found duplicate region-year-indicator rows:\n"
            + duplicates.to_string(index=False)
        )

    # ---------------------------------------------------------
    # Long -> Wide
    # ---------------------------------------------------------

    panel = (
        df.pivot(
            index=["object_name", "year"],
            columns="indicator_code",
            values="indicator_value",
        )
        .reset_index()
    )

    panel.columns.name = None

    # ---------------------------------------------------------
    # Переименование показателей
    # ---------------------------------------------------------

    panel = panel.rename(columns={
        "object_name": "region",
        "Y477110107": "investments",
        "Y477110223": "retail",
        "Y477110374": "income",
        "Y477110418": "unemployment",
        "Y477110373": "population",
        "Y477110016": "housing",
    })

    # ---------------------------------------------------------
    # Сортировка перед созданием лага
    # ---------------------------------------------------------

    panel = (
        panel
        .sort_values(["region", "year"])
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------
    # Население используем с лагом 1
    # ---------------------------------------------------------

    panel["population_lag1"] = (
        panel.groupby("region")["population"].shift(1)
    )

    # Population в исходном виде больше не нужен
    panel = panel.drop(columns=["population"])

    # ---------------------------------------------------------
    # Проверка структуры
    # ---------------------------------------------------------

    expected_columns = [
        "region",
        "year",
        "investments",
        "retail",
        "income",
        "unemployment",
        "housing",
        "population_lag1",
    ]

    missing_columns = [
        col
        for col in expected_columns
        if col not in panel.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing expected columns: {missing_columns}"
        )

    panel = panel[expected_columns]

    # ---------------------------------------------------------
    # Проверка уникальности ключа панели
    # ---------------------------------------------------------

    if panel.duplicated(["region", "year"]).any():
        raise ValueError(
            "Duplicate region-year pairs found in final panel."
        )

    # ---------------------------------------------------------
    # Анализ пропусков ДО удаления строк по population_lag1
    # ---------------------------------------------------------

    print("\n=== MISSING VALUES BEFORE POPULATION LAG FILTER ===")
    print(panel.isna().sum())

    print("\n=== MISSING VALUES BY INDICATOR ===")

    for col in expected_columns[2:]:
        missing = panel[col].isna().sum()

        if missing > 0:
            print(f"{col}: {missing}")

    # ---------------------------------------------------------
    # Удаляем строки без population_lag1
    # ---------------------------------------------------------

    before = len(panel)

    panel = panel.dropna(
        subset=["population_lag1"]
    ).copy()

    removed = before - len(panel)

    # ---------------------------------------------------------
    # Строки, где остались пропуски
    # ---------------------------------------------------------

    indicator_columns = expected_columns[2:]

    print("\n=== ROWS WITH ANY MISSING INDICATOR ===")

    rows_with_missing = panel[
        panel[indicator_columns].isna().any(axis=1)
    ]

    print(
        f"Rows with missing values: "
        f"{len(rows_with_missing)}"
    )

    if len(rows_with_missing) > 0:
        print(
            rows_with_missing[
                ["region", "year"] + indicator_columns
            ]
            .head(30)
            .to_string(index=False)
        )

    # ---------------------------------------------------------
    # Подробная диагностика пропусков
    # по регионам и показателям
    # ---------------------------------------------------------

    print(
        "\n=== MISSING VALUES BY REGION AND INDICATOR ==="
    )

    missing_by_region = (
        panel
        .groupby("region")[indicator_columns]
        .apply(lambda x: x.isna().sum())
    )

    missing_by_region = missing_by_region[
        missing_by_region.sum(axis=1) > 0
    ]

    if missing_by_region.empty:
        print("No missing values found.")
    else:
        print(
            missing_by_region.to_string()
        )

    # ---------------------------------------------------------
    # Конкретные годы пропусков
    # ---------------------------------------------------------

    print("\n=== MISSING YEARS BY REGION ===")

    if missing_by_region.empty:
        print("No missing values found.")
    else:
        for region in missing_by_region.index:
            print(f"\n{region}")

            for indicator in indicator_columns:
                years = panel.loc[
                    panel["region"].eq(region)
                    & panel[indicator].isna(),
                    "year",
                ].tolist()

                if years:
                    print(
                        f"  {indicator}: {years}"
                    )

    # ---------------------------------------------------------
    # Финальная сортировка
    # ---------------------------------------------------------

    panel = (
        panel
        .sort_values(["region", "year"])
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------
    # Создание директории
    # ---------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Сохранение
    # ---------------------------------------------------------

    panel.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    # ---------------------------------------------------------
    # Финальная информация
    # ---------------------------------------------------------

    print("\n=== PANEL CREATED ===")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Shape: {panel.shape}")
    print(f"Regions: {panel['region'].nunique()}")
    print(
        f"Years: "
        f"{panel['year'].min()}–{panel['year'].max()}"
    )
    print(
        f"Removed rows without population lag: "
        f"{removed}"
    )

    print("\n=== MISSING VALUES ===")
    print(panel.isna().sum())

    print("\n=== SAMPLE ===")
    print(
        panel.head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()