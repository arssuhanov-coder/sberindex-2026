from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "panel.parquet"
)

CONSENSUS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "cpd_consensus.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "cpd_candidates.csv"
)

TARGET = "investments"

YEAR_START = 2002
YEAR_END = 2024

MIN_METHODS = 2
WINDOW = 3


# ============================================================
# HELPERS
# ============================================================

def safe_growth(current, previous):
    """
    Year-over-year growth.

    Returns NaN when the previous value is zero or missing.
    """

    if (
        pd.isna(current)
        or pd.isna(previous)
        or previous == 0
    ):
        return np.nan

    return (
        current / previous
    ) - 1.0


def calculate_slope(years, values):
    """
    Linear trend slope.

    Uses log1p(values) to reduce the influence
    of very large regional investment values.
    """

    years = np.asarray(
        years,
        dtype=float
    )

    values = np.asarray(
        values,
        dtype=float
    )

    mask = (
        np.isfinite(years)
        & np.isfinite(values)
        & (values >= 0)
    )

    years = years[mask]
    values = values[mask]

    if len(values) < 3:
        return np.nan

    signal = np.log1p(values)

    try:
        slope = np.polyfit(
            years,
            signal,
            1
        )[0]

        return float(slope)

    except Exception:
        return np.nan


def local_jump_score(
    yoy_change,
    historical_yoy,
):
    """
    Measures how unusual the CPD-year YoY change is
    relative to the region's historical absolute YoY changes.

    This is a descriptive score, not a probability.
    """

    if pd.isna(yoy_change):
        return np.nan

    historical_yoy = np.asarray(
        historical_yoy,
        dtype=float
    )

    historical_yoy = historical_yoy[
        np.isfinite(historical_yoy)
    ]

    if len(historical_yoy) == 0:
        return np.nan

    baseline = np.median(
        np.abs(historical_yoy)
    )

    if baseline == 0:
        return np.nan

    return abs(yoy_change) / baseline


# ============================================================
# SPECIFICITY
# ============================================================

def calculate_specificity(
    consensus
):
    """
    Recalculate specificity only for consensus points
    supported by at least MIN_METHODS methods.

    Specificity combines:

    - method agreement
    - rarity of the calendar year

    It is NOT a probability and NOT an anomaly probability.
    """

    consensus = consensus.copy()

    consensus = consensus[
        consensus["n_methods"] >= MIN_METHODS
    ].copy()

    if consensus.empty:
        return consensus

    year_counts = (
        consensus
        .groupby("year")
        .size()
    )

    total_points = len(consensus)

    consensus["consensus_points"] = (
        consensus["year"]
        .map(year_counts)
        .astype(int)
    )

    consensus["year_rarity"] = (
        1
        - (
            consensus["consensus_points"]
            / total_points
        )
    )

    consensus["specificity_score"] = (
        consensus["n_methods"]
        / 3.0
        * consensus["year_rarity"]
    )

    return consensus


# ============================================================
# MAGNITUDE
# ============================================================

def enrich_candidate(
    candidate,
    region_df,
):
    """
    Add actual investment dynamics around the CPD point.
    """

    region_df = (
        region_df[
            [
                "year",
                TARGET,
            ]
        ]
        .dropna(subset=[TARGET])
        .sort_values("year")
        .copy()
    )

    if region_df.empty:
        return candidate

    region_df["yoy"] = (
        region_df[TARGET]
        .pct_change()
    )

    year = int(
        candidate["year"]
    )

    row_positions = (
        region_df.index[
            region_df["year"] == year
        ]
        .tolist()
    )

    if not row_positions:
        return candidate

    position = region_df.index.get_loc(
        row_positions[0]
    )

    current_row = (
        region_df.iloc[position]
    )

    previous_row = None

    if position > 0:
        previous_row = (
            region_df.iloc[position - 1]
        )

    # --------------------------------------------------------
    # YoY
    # --------------------------------------------------------

    if previous_row is not None:
        yoy_change = safe_growth(
            current_row[TARGET],
            previous_row[TARGET],
        )
    else:
        yoy_change = np.nan

    candidate["investment_value"] = (
        current_row[TARGET]
    )

    candidate["yoy_change"] = (
        yoy_change
    )

    # --------------------------------------------------------
    # Historical YoY
    # --------------------------------------------------------

    historical_yoy = (
        region_df["yoy"]
        .dropna()
        .values
    )

    candidate["local_jump"] = (
        local_jump_score(
            yoy_change,
            historical_yoy,
        )
    )

    # --------------------------------------------------------
    # Before / after trend
    # --------------------------------------------------------

    before = region_df[
        region_df["year"] < year
    ].tail(WINDOW)

    after = region_df[
        region_df["year"] >= year
    ].head(WINDOW)

    trend_before = calculate_slope(
        before["year"].values,
        before[TARGET].values,
    )

    trend_after = calculate_slope(
        after["year"].values,
        after[TARGET].values,
    )

    candidate["trend_before"] = (
        trend_before
    )

    candidate["trend_after"] = (
        trend_after
    )

    if (
        pd.notna(trend_before)
        and pd.notna(trend_after)
    ):
        candidate["trend_change"] = (
            trend_after
            - trend_before
        )
    else:
        candidate["trend_change"] = np.nan

    # --------------------------------------------------------
    # Direction
    # --------------------------------------------------------

    if pd.isna(yoy_change):
        direction = "unknown"

    elif yoy_change > 0:
        direction = "increase"

    elif yoy_change < 0:
        direction = "decrease"

    else:
        direction = "stable"

    candidate["direction"] = direction

    return candidate


# ============================================================
# MAIN
# ============================================================

def run():

    print("Loading panel...")
    
    panel = pd.read_parquet(
        DATA_PATH
    )

    print(
        f"Panel shape: {panel.shape}"
    )

    panel = panel[
        (panel["year"] >= YEAR_START)
        & (panel["year"] <= YEAR_END)
    ].copy()

    print("Loading CPD consensus...")

    consensus = pd.read_csv(
        CONSENSUS_PATH
    )

    print(
        f"Consensus rows: {len(consensus)}"
    )

    # --------------------------------------------------------
    # Only multi-method consensus
    # --------------------------------------------------------

    candidates = calculate_specificity(
        consensus
    )

    print(
        "\nMulti-method CPD points:"
    )

    print(
        len(candidates)
    )

    if candidates.empty:
        print(
            "No multi-method CPD points found."
        )
        return

    # --------------------------------------------------------
    # Enrich every candidate
    # --------------------------------------------------------

    enriched = []

    for _, candidate in candidates.iterrows():

        region = candidate["region"]

        region_df = panel[
            panel["region"] == region
        ].copy()

        candidate = enrich_candidate(
            candidate.to_dict(),
            region_df,
        )

        enriched.append(
            candidate
        )

    result = pd.DataFrame(
        enriched
    )

    # --------------------------------------------------------
    # Magnitude ranking
    # --------------------------------------------------------

    result["abs_yoy_change"] = (
        result["yoy_change"]
        .abs()
    )

    result["abs_trend_change"] = (
        result["trend_change"]
        .abs()
    )

    # Descriptive combined signal.
    #
    # This is intentionally NOT called probability,
    # anomaly probability, or statistical significance.
    result["cpd_signal"] = (
        result["specificity_score"]
        * (
            1
            + np.log1p(
                result["local_jump"]
                .clip(lower=0)
            )
        )
    )

    result = (
        result
        .sort_values(
            [
                "cpd_signal",
                "specificity_score",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    result.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n=== CPD CANDIDATES ==="
    )

    print(
        f"Rows: {len(result)}"
    )

    print(
        "\nTop 30 candidates:"
    )

    columns = [
        "region",
        "year",
        "methods",
        "n_methods",
        "specificity_score",
        "yoy_change",
        "local_jump",
        "trend_before",
        "trend_after",
        "trend_change",
        "direction",
        "cpd_signal",
    ]

    print(
        result[columns]
        .head(30)
        .to_string(index=False)
    )

    print(
        "\nOutput:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    run()