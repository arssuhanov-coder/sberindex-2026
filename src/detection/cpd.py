import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruptures as rpt
import yaml

# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = PROJECT_ROOT / "configs" / "cpd.yaml"

with open(CONFIG_PATH, encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

DATA_PATH = PROJECT_ROOT / CONFIG["data"]["panel_path"]

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TARGET = CONFIG["target"]

YEAR_START = CONFIG["period"]["start_year"]
YEAR_END = CONFIG["period"]["end_year"]

CUSUM_PARAMS = CONFIG["cusum"]
PELT_PARAMS = CONFIG["pelt"]
BINSEG_PARAMS = CONFIG["binseg"]
CONSENSUS_PARAMS = CONFIG["consensus"]


# ============================================================
# CUSUM
# ============================================================

def detect_cusum(
    values,
    threshold=None,
):
    if threshold is None:
        threshold = CUSUM_PARAMS["threshold"]
    """
    Simple standardized CUSUM detector.

    Returns indexes of detected change points.
    """

    values = np.asarray(
        values,
        dtype=float
    )

    if len(values) < 5:
        return []

    mean = np.mean(values)
    std = np.std(values)

    if std == 0:
        return []

    z = (
        values - mean
    ) / std

    positive = 0.0
    negative = 0.0

    change_points = []

    for i, value in enumerate(z):

        positive = max(
            0.0,
            positive + value
        )

        negative = min(
            0.0,
            negative + value
        )

        if positive > threshold or negative < -threshold:
            change_points.append(i)
            positive = 0.0
            negative = 0.0

    return sorted(
        set(change_points)
    )


# ============================================================
# PREPARE SERIES
# ============================================================

def prepare_series(region_df):

    region_df = (
        region_df[
            [
                "region",
                "year",
                TARGET,
            ]
        ]
        .sort_values("year")
        .copy()
    )

    # CPD requires an observed continuous series.
    # We do not interpolate investment values automatically.
    region_df = region_df.dropna(
        subset=[TARGET]
    )

    if region_df.empty:
        return None

    return region_df


# ============================================================
# PELT
# ============================================================

def detect_pelt(values):

    values = np.asarray(
        values,
        dtype=float
    )

    if len(values) < 8:
        return []

    try:
        # Log-transform reduces the effect of very large
        # regional investment values.
        signal = np.log1p(values)

        # Standardization makes the scale comparable.
        std = np.std(signal)

        if std == 0:
            return []

        signal = (
            signal - np.mean(signal)
        ) / std

        algorithm = rpt.Pelt(
            model=PELT_PARAMS["model"],
            min_size=PELT_PARAMS["min_size"],
            jump=PELT_PARAMS["jump"],
        )

        penalty = (
            PELT_PARAMS["penalty_multiplier"]
            * np.log(len(signal))
        )

        breakpoints = algorithm.fit(
            signal
        ).predict(
            pen=penalty
        )

        breakpoints = [
            point
            for point in breakpoints
            if point < len(signal)
        ]

        return breakpoints

    except Exception:
        return []

# ============================================================
# BINSEG
# ============================================================

def detect_binseg(values):

    values = np.asarray(
        values,
        dtype=float
    )

    if len(values) < 8:
        return []

    try:

        signal = np.log1p(values)

        std = np.std(signal)

        if std == 0:
            return []

        signal = (
            signal - np.mean(signal)
        ) / std

        algorithm = rpt.Binseg(
            model=BINSEG_PARAMS["model"],
            min_size=BINSEG_PARAMS["min_size"],
            jump=BINSEG_PARAMS["jump"],
        )

        n_bkps = BINSEG_PARAMS["n_bkps"]

        breakpoints = algorithm.fit(
            signal
        ).predict(
            n_bkps=n_bkps
        )

        breakpoints = [
            point
            for point in breakpoints
            if point < len(signal)
        ]

        return breakpoints

    except Exception:
        return []

# ============================================================
# INDEX -> YEAR
# ============================================================

def breakpoint_to_year(
    series_df,
    breakpoint,
):
    """
    ruptures breakpoint means the boundary before index `breakpoint`.
    Therefore the detected change is associated with the last
    observation of the preceding segment.
    """

    if breakpoint <= 0:
        return None

    if breakpoint >= len(series_df):
        return None

    return int(
        series_df.iloc[
            breakpoint - 1
        ]["year"]
    )
    


def cusum_index_to_year(
    series_df,
    index,
):

    if index <= 0:
        return None

    if index >= len(series_df):
        return None

    return int(
        series_df.iloc[index]["year"]
    )


# ============================================================
# DETECT ONE REGION
# ============================================================

def detect_region(region_df):

    series_df = prepare_series(
        region_df
    )

    if series_df is None:
        return []

    values = (
        series_df[TARGET]
        .astype(float)
        .values
    )

    region = series_df[
        "region"
    ].iloc[0]

    results = []

    # -----------------------------------------
    # CUSUM
    # -----------------------------------------

    cusum_values = np.log1p(values)

    cusum_points = detect_cusum(
        cusum_values
    )

    for index in cusum_points:

        year = cusum_index_to_year(
            series_df,
            index
        )

        if year is None:
            continue

        results.append({
            "region": region,
            "method": "CUSUM",
            "year": year,
            "index": index,
        })

    # -----------------------------------------
    # PELT
    # -----------------------------------------

    pelt_points = detect_pelt(
        values
    )

    for point in pelt_points:

        year = breakpoint_to_year(
            series_df,
            point
        )

        if year is None:
            continue

        results.append({
            "region": region,
            "method": "PELT",
            "year": year,
            "index": point,
        })

    # -----------------------------------------
    # BINSEG
    # -----------------------------------------

    binseg_points = detect_binseg(
        values
    )

    for point in binseg_points:

        year = breakpoint_to_year(
            series_df,
            point
        )

        if year is None:
            continue

        results.append({
            "region": region,
            "method": "Binseg",
            "year": year,
            "index": point,
        })

    return results


# ============================================================
# CONSENSUS
# ============================================================

def build_consensus(
    detections,
):
    """
    Count how many CPD methods identify
    the same region/year with tolerance.

    Points are grouped within tolerance_years.
    A cluster is kept only if >= min_methods agree.

    This is not a statistical probability.
    It is simply a methodological agreement count.
    """

    tolerance = CONSENSUS_PARAMS.get("tolerance_years", 0)
    min_methods = CONSENSUS_PARAMS.get("min_methods", 2)

    if detections.empty:
        return pd.DataFrame(
            columns=[
                "region",
                "year",
                "years",
                "methods",
                "n_methods",
            ]
        )

    result = []

    for region, group in detections.groupby("region"):

        years_sorted = sorted(group["year"].unique())
        clusters = []
        current = [years_sorted[0]]

        for y in years_sorted[1:]:
            if y - current[-1] <= tolerance:
                current.append(y)
            else:
                clusters.append(current)
                current = [y]
        clusters.append(current)

        for cluster in clusters:
            methods = group[
                group["year"].isin(cluster)
            ]["method"].unique()

            if len(methods) >= min_methods:
                result.append({
                    "region": region,
                    "year": int(np.median(cluster)),
                    "years": ", ".join(map(str, cluster)),
                    "methods": ", ".join(sorted(methods)),
                    "n_methods": len(methods),
                })

    if not result:
        return pd.DataFrame(
            columns=[
                "region",
                "year",
                "years",
                "methods",
                "n_methods",
            ]
        )

    return (
        pd.DataFrame(result)
        .sort_values(["region", "year"])
        .reset_index(drop=True)
    )


# ============================================================
# MAIN
# ============================================================

def run_cpd():

    print("Loading panel...")

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

    df = df[
        (df["year"] >= YEAR_START)
        & (df["year"] <= YEAR_END)
    ].copy()

    all_detections = []

    regions = sorted(
        df["region"]
        .dropna()
        .unique()
    )

    print(
        "\n=== CPD DETECTION ==="
    )

    for index, region in enumerate(
        regions,
        start=1
    ):

        print(
            f"[{index}/{len(regions)}] "
            f"{region}"
        )

        region_df = df[
            df["region"] == region
        ].copy()

        detections = detect_region(
            region_df
        )

        all_detections.extend(
            detections
        )

    detections_df = pd.DataFrame(
        all_detections
    )

    if detections_df.empty:
        print(
            "\nNo change points detected."
        )
        return

    detections_df = (
        detections_df
        .sort_values(
            [
                "region",
                "year",
                "method",
            ]
        )
        .reset_index(drop=True)
    )

    # -----------------------------------------
    # CONSENSUS
    # -----------------------------------------

    consensus_df = build_consensus(
        detections_df
    )

    # -----------------------------------------
    # SAVE
    # -----------------------------------------

    detections_path = (
        REPORTS_DIR
        / "cpd_detections.csv"
    )

    consensus_path = (
        REPORTS_DIR
        / "cpd_consensus.csv"
    )

    detections_df.to_csv(
        detections_path,
        index=False
    )

    consensus_df.to_csv(
        consensus_path,
        index=False
    )

    # -----------------------------------------
    # SUMMARY
    # -----------------------------------------

    print(
        "\n=== CPD SUMMARY ==="
    )

    print(
        f"Total detections: "
        f"{len(detections_df)}"
    )

    print(
        f"Regions with detections: "
        f"{detections_df['region'].nunique()}"
    )

    print(
        f"Consensus points "
        f"(>= 2 methods): "
        f"{(consensus_df['n_methods'] >= 2).sum()}"
    )

    print(
        "\n=== DETECTIONS BY METHOD ==="
    )

    print(
        detections_df[
            "method"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\n=== MOST FREQUENT CHANGE YEARS ==="
    )

    print(
        detections_df[
            "year"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\n=== STRONGEST CONSENSUS POINTS ==="
    )

    strong_consensus = (
        consensus_df[
            consensus_df["n_methods"] >= 2
        ]
        .sort_values(
            [
                "n_methods",
                "year",
            ],
            ascending=[
                False,
                True,
            ],
        )
    )

    if strong_consensus.empty:

        print(
            "No points detected by "
            "at least two methods."
        )

    else:

        print(
            strong_consensus
            .head(50)
            .to_string(index=False)
        )

    print(
        "\n=== OUTPUT FILES ==="
    )

    print(
        f"Detections: "
        f"{detections_path}"
    )

    print(
        f"Consensus:  "
        f"{consensus_path}"
    )

    return (
        detections_df,
        consensus_df,
    )


# ============================================================
# PLOTS
# ============================================================

def plot_region_cpd(
    region_df,
    detections_df,
):

    region = region_df[
        "region"
    ].iloc[0]

    series_df = prepare_series(
        region_df
    )

    if series_df is None:
        return

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        series_df["year"],
        series_df[TARGET],
        marker="o",
        label="Investments",
    )

    region_detections = detections_df[
        detections_df["region"] == region
    ]

    method_markers = {
        "CUSUM": "x",
        "PELT": "s",
        "Binseg": "^",
    }

    for method, marker in method_markers.items():

        points = region_detections[
            region_detections["method"] == method
        ]

        if points.empty:
            continue

        years = points["year"].tolist()

        values = []

        for year in years:

            row = series_df[
                series_df["year"] == year
            ]

            if row.empty:
                continue

            values.append(
                (
                    year,
                    row[TARGET].iloc[0]
                )
            )

        if not values:
            continue

        plot_years = [
            item[0]
            for item in values
        ]

        plot_values = [
            item[1]
            for item in values
        ]

        plt.scatter(
            plot_years,
            plot_values,
            marker=marker,
            s=80,
            label=method,
        )

    plt.title(
        f"Structural changes — {region}"
    )

    plt.xlabel("Year")

    plt.ylabel(
        "Investments, million RUB"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    safe_region = (
        region
        .replace("/", "_")
        .replace(" ", "_")
    )

    output_path = (
        FIGURES_DIR
        / f"cpd_{safe_region}.png"
    )

    plt.savefig(
        output_path,
        dpi=150
    )

    plt.close()

    print(
        f"Plot saved: {output_path}"
    )


def plot_selected_regions(
    df,
    detections_df,
):

    regions = (
        detections_df[
            "region"
        ]
        .drop_duplicates()
        .head(5)
        .tolist()
    )

    for region in regions:

        region_df = df[
            df["region"] == region
        ].copy()

        plot_region_cpd(
            region_df,
            detections_df
        )

def run_cpd_qa(detections_df, consensus_df):
    """
    Technical QA for CPD results.

    Checks:
    - duplicate region/year/method combinations
    - valid change-point years
    - agreement between methods
    - number of 1/2/3-method consensus points
    - most frequent consensus years
    """

    print("\n" + "=" * 60)
    print("CPD QUALITY ASSURANCE")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. DUPLICATES
    # --------------------------------------------------------

    duplicates = detections_df.duplicated(
        subset=[
            "region",
            "year",
            "method",
        ]
    ).sum()

    print(
        f"\nDuplicate region/year/method rows: "
        f"{duplicates}"
    )

    # --------------------------------------------------------
    # 2. VALID YEARS
    # --------------------------------------------------------

    invalid_years = detections_df[
        (detections_df["year"] < YEAR_START)
        | (detections_df["year"] > YEAR_END)
    ]

    print(
        f"Invalid change-point years: "
        f"{len(invalid_years)}"
    )

    # --------------------------------------------------------
    # 3. METHOD COUNTS
    # --------------------------------------------------------

    print("\nDetections by method:")

    print(
        detections_df[
            "method"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # --------------------------------------------------------
    # 4. AGREEMENT DISTRIBUTION
    # --------------------------------------------------------

    agreement_counts = (
        consensus_df[
            "n_methods"
        ]
        .value_counts()
        .sort_index()
    )

    print(
        "\nConsensus distribution:"
    )

    for n_methods in [1, 2, 3]:

        count = int(
            agreement_counts.get(
                n_methods,
                0
            )
        )

        print(
            f"{n_methods}/3 methods: "
            f"{count}"
        )

    # --------------------------------------------------------
    # 5. 3/3 CONSENSUS
    # --------------------------------------------------------

    consensus_3 = consensus_df[
        consensus_df["n_methods"] == 3
    ].copy()

    print(
        "\n3/3 consensus points: "
        f"{len(consensus_3)}"
    )

    print(
        "Regions with at least one 3/3 point: "
        f"{consensus_3['region'].nunique()}"
    )

    if not consensus_3.empty:

        print(
            "\n3/3 consensus:"
        )

        print(
            consensus_3
            .sort_values(
                [
                    "year",
                    "region",
                ]
            )
            .to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # 6. MOST FREQUENT CONSENSUS YEARS
    # --------------------------------------------------------

    print(
        "\nMost frequent consensus years:"
    )

    consensus_years = (
        consensus_df[
            "year"
        ]
        .value_counts()
        .sort_index()
    )

    print(
        consensus_years.to_string()
    )

    # --------------------------------------------------------
    # 7. STRONGEST YEARS
    # --------------------------------------------------------

    print(
        "\nStrongest consensus years:"
    )

    strongest_years = (
        consensus_df
        .groupby("year")
        .agg(
            consensus_points=(
                "region",
                "count",
            ),
            max_methods=(
                "n_methods",
                "max",
            ),
        )
        .sort_values(
            [
                "max_methods",
                "consensus_points",
            ],
            ascending=False,
        )
    )

    print(
        strongest_years.to_string()
    )

    # --------------------------------------------------------
    # 8. SAVE QA
    # --------------------------------------------------------

    qa_path = (
        REPORTS_DIR
        / "cpd_qa.txt"
    )

    with open(
        qa_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "CPD QUALITY ASSURANCE\n"
        )

        f.write(
            f"Duplicate rows: {duplicates}\n"
        )

        f.write(
            f"Invalid years: "
            f"{len(invalid_years)}\n"
        )

        f.write(
            f"Total detections: "
            f"{len(detections_df)}\n"
        )

        f.write(
            f"Consensus points: "
            f"{len(consensus_df)}\n"
        )

        f.write(
            f"3/3 consensus: "
            f"{len(consensus_3)}\n"
        )

        f.write(
            f"Regions with 3/3: "
            f"{consensus_3['region'].nunique()}\n"
        )

    print(
        f"\nQA report saved: {qa_path}"
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
        detections_df,
        consensus_df,
    ) = run_cpd()

    if detections_df is not None:

        run_cpd_qa(
            detections_df,
            consensus_df
        )

        df = pd.read_parquet(
            DATA_PATH
        )

        plot_selected_regions(
            df,
            detections_df
        )