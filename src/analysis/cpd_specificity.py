from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONSENSUS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "cpd_consensus.csv"
)

DETECTIONS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "cpd_detections.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "cpd_specificity.csv"
)


# ============================================================
# LOAD
# ============================================================

def load_data():

    consensus = pd.read_csv(
        CONSENSUS_PATH
    )

    detections = pd.read_csv(
        DETECTIONS_PATH
    )

    return consensus, detections


# ============================================================
# YEAR FREQUENCY
# ============================================================

def calculate_year_frequency(
    consensus
):

    year_frequency = (
        consensus
        .groupby("year")
        .agg(
            consensus_points=(
                "region",
                "nunique"
            )
        )
        .reset_index()
    )

    return year_frequency


# ============================================================
# REGION FREQUENCY
# ============================================================

def calculate_region_frequency(
    consensus
):

    region_frequency = (
        consensus
        .groupby("region")
        .agg(
            total_consensus_points=(
                "year",
                "count"
            )
        )
        .reset_index()
    )

    return region_frequency


# ============================================================
# SPECIFICITY
# ============================================================

def calculate_specificity(
    consensus
):

    year_frequency = (
        calculate_year_frequency(
            consensus
        )
    )

    region_frequency = (
        calculate_region_frequency(
            consensus
        )
    )

    result = (
        consensus
        .merge(
            year_frequency,
            on="year",
            how="left"
        )
        .merge(
            region_frequency,
            on="region",
            how="left"
        )
    )

    # --------------------------------------------------------
    # BASIC FREQUENCY
    # --------------------------------------------------------

    max_frequency = (
        result[
            "consensus_points"
        ].max()
    )

    if max_frequency > 0:

        result[
            "year_commonness"
        ] = (
            result[
                "consensus_points"
            ]
            / max_frequency
        )

    else:

        result[
            "year_commonness"
        ] = 0.0

    # --------------------------------------------------------
    # RARITY
    # --------------------------------------------------------
    #
    # Чем меньше регионов имеют перелом в этом году,
    # тем выше specificity.
    #
    # 1 = очень редкий год
    # 0 = самый массовый год
    # --------------------------------------------------------

    result[
        "year_rarity"
    ] = (
        1.0
        - result[
            "year_commonness"
        ]
    )

    # --------------------------------------------------------
    # METHOD AGREEMENT
    # --------------------------------------------------------

    result[
        "method_agreement"
    ] = (
        result[
            "n_methods"
        ] / 3.0
    )

    # --------------------------------------------------------
    # RAW SPECIFICITY SCORE
    # --------------------------------------------------------
    #
    # Нам важны одновременно:
    #
    # 1. согласие методов;
    # 2. редкость года.
    #
    # Это НЕ вероятность и НЕ статистическая значимость.
    # --------------------------------------------------------

    result[
        "specificity_score"
    ] = (
        result[
            "method_agreement"
        ]
        * result[
            "year_rarity"
        ]
    )

    # --------------------------------------------------------
    # CATEGORY
    # --------------------------------------------------------

    def classify(score):

        if score >= 0.50:
            return "high"

        if score >= 0.25:
            return "medium"

        return "low"

    result[
        "specificity_level"
    ] = (
        result[
            "specificity_score"
        ]
        .apply(classify)
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "Loading CPD results..."
    )

    consensus, detections = (
        load_data()
    )

    print(
        f"Consensus rows: "
        f"{len(consensus)}"
    )

    print(
        f"Detection rows: "
        f"{len(detections)}"
    )

    result = (
        calculate_specificity(
            consensus
        )
    )

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    result = (
        result
        .sort_values(
            [
                "specificity_score",
                "n_methods",
                "year"
            ],
            ascending=[
                False,
                False,
                True
            ]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    result.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        "\n=== CPD SPECIFICITY ==="
    )

    print(
        "\nSpecificity levels:"
    )

    print(
        result[
            "specificity_level"
        ]
        .value_counts()
    )

    print(
        "\nTop 30 regional-specific "
        "change points:"
    )

    print(
        result[
            [
                "region",
                "year",
                "methods",
                "n_methods",
                "consensus_points",
                "year_rarity",
                "specificity_score",
                "specificity_level",
            ]
        ]
        .head(30)
        .to_string(
            index=False
        )
    )

    print(
        "\nOutput:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()