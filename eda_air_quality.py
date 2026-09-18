import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# PATHS - MAC VERSION
# ============================================================

BASE_DIR = Path("/Users/aalok_x92/Downloads/Development code")
FIG_DIR = BASE_DIR / "figures"

FIG_DIR.mkdir(parents=True, exist_ok=True)

# Main CSV file that you currently have
GLOBAL_MAIN = BASE_DIR / "global_air_quality_dataset_10000_clean.csv"

# Optional files
GLOBAL_10K = BASE_DIR / "global_air_quality_data_10000_clean.csv"
OPENAQ = BASE_DIR / "world_air_quality_clean.csv"
WHO_2014 = BASE_DIR / "aap_pm_database_may2014_clean.csv"
WHO_2021 = BASE_DIR / "who_aap_2021_clean.csv"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def detect_datetime_col(df):
    """Find the first column that looks like a date/time column."""

    for col in df.columns:
        name = col.lower()

        if any(
            key in name
            for key in ["date", "time", "datetime", "timestamp"]
        ):
            return col

    return None


def detect_pollutant_cols(df):
    """Find numeric pollutant columns."""

    keys = [
        "pm2",
        "pm10",
        "no2",
        "so2",
        "o3",
        "co"
    ]

    pollutant_cols = []

    for col in df.columns:

        col_lower = col.lower()

        if any(key in col_lower for key in keys):

            if pd.api.types.is_numeric_dtype(df[col]):
                pollutant_cols.append(col)

    return pollutant_cols


def detect_location_cols(df):
    """Find country, city and station/location columns."""

    country = next(
        (
            c for c in df.columns
            if "country" in c.lower()
        ),
        None
    )

    city = next(
        (
            c for c in df.columns
            if "city" in c.lower()
        ),
        None
    )

    station = next(
        (
            c for c in df.columns
            if any(
                key in c.lower()
                for key in ["station", "location"]
            )
        ),
        None
    )

    return country, city, station


def show_and_save(filename):
    """
    Save the current figure and display it.
    """

    path = FIG_DIR / filename

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"✓ Saved: {path}")

    # Show graph on screen
    plt.show()

    plt.close()


# ============================================================
# CHECK FILES
# ============================================================

def check_files():

    print("\n" + "=" * 70)
    print("CHECKING INPUT CSV FILES")
    print("=" * 70)

    files = {
        "GLOBAL_MAIN": GLOBAL_MAIN,
        "GLOBAL_10K": GLOBAL_10K,
        "OPENAQ": OPENAQ,
        "WHO_2014": WHO_2014,
        "WHO_2021": WHO_2021,
    }

    for name, path in files.items():

        if path.exists():

            print(f"✓ {name}")
            print(f"  {path}")

        else:

            print(f"⚠ {name} NOT FOUND")
            print(f"  {path}")

    print("=" * 70)


# ============================================================
# 4.4.1 DESCRIPTIVE STATISTICS
# ============================================================

def eda_descriptive_statistics():

    print("\n")
    print("=" * 70)
    print("4.4.1 DESCRIPTIVE STATISTICS")
    print("=" * 70)

    # --------------------------------------------------------
    # Load global dataset
    # --------------------------------------------------------

    if not GLOBAL_MAIN.exists():

        print("\n❌ Global CSV file not found:")
        print(GLOBAL_MAIN)

        return

    df_global = pd.read_csv(GLOBAL_MAIN)

    print("\n" + "-" * 70)
    print("GLOBAL DATASET")
    print("-" * 70)

    print(f"Rows: {df_global.shape[0]}")
    print(f"Columns: {df_global.shape[1]}")
    print(
        f"Missing values: "
        f"{df_global.isna().sum().sum()}"
    )

    print("\nColumns:")

    for col in df_global.columns:

        print(f"  - {col}")

    # --------------------------------------------------------
    # Detect pollutants
    # --------------------------------------------------------

    poll_global = detect_pollutant_cols(df_global)

    print("\nGlobal pollutant columns:")
    print(poll_global)

    if not poll_global:

        print("⚠ No pollutant columns found.")

        return

    # ========================================================
    # GRAPH 1 - HISTOGRAMS
    # ========================================================

    print("\nCreating global pollutant histograms...")

    plt.figure(figsize=(12, 8))

    selected_pollutants = poll_global[:6]

    for i, col in enumerate(
        selected_pollutants,
        start=1
    ):

        plt.subplot(2, 3, i)

        plt.hist(
            df_global[col].dropna(),
            bins=30,
            color="steelblue",
            edgecolor="black",
            alpha=0.8
        )

        plt.title(col)

        plt.xlabel("Concentration")

        plt.ylabel("Frequency")

        plt.grid(
            True,
            linestyle="--",
            alpha=0.3
        )

    plt.suptitle(
        "Distribution of Air Pollutants",
        fontsize=16
    )

    show_and_save(
        "descriptive_global_histograms.png"
    )

    # ========================================================
    # GRAPH 2 - BOXPLOT
    # ========================================================

    print("\nCreating global pollutant boxplot...")

    plt.figure(figsize=(12, 6))

    data = [
        df_global[col].dropna()
        for col in poll_global
    ]

    # IMPORTANT:
    # New Matplotlib uses tick_labels instead of labels

    plt.boxplot(
        data,
        tick_labels=poll_global,
        showfliers=False
    )

    plt.xlabel("Pollutant")

    plt.ylabel("Concentration")

    plt.title(
        "Distribution of Pollutant Concentrations"
    )

    plt.xticks(rotation=45)

    plt.grid(
        True,
        axis="y",
        linestyle="--",
        alpha=0.4
    )

    show_and_save(
        "descriptive_global_boxplot.png"
    )

    # ========================================================
    # SUMMARY STATISTICS
    # ========================================================

    print("\nPollutant summary statistics:")

    print(
        df_global[poll_global].describe()
    )


# ============================================================
# 4.4.2 TEMPORAL PATTERNS
# ============================================================

def eda_temporal_patterns():

    print("\n")
    print("=" * 70)
    print("4.4.2 TEMPORAL PATTERNS")
    print("=" * 70)

    if not GLOBAL_MAIN.exists():

        print("❌ Global CSV not found.")

        return

    df = pd.read_csv(GLOBAL_MAIN)

    # --------------------------------------------------------
    # Detect date column
    # --------------------------------------------------------

    dt_col = detect_datetime_col(df)

    if dt_col is None:

        print(
            "❌ No datetime column detected."
        )

        return

    print(
        f"Datetime column: {dt_col}"
    )

    # Convert to datetime

    df[dt_col] = pd.to_datetime(
        df[dt_col],
        errors="coerce"
    )

    df = df.dropna(
        subset=[dt_col]
    )

    df = df.sort_values(dt_col)

    # --------------------------------------------------------
    # Detect pollutants
    # --------------------------------------------------------

    poll_cols = detect_pollutant_cols(df)

    if not poll_cols:

        print(
            "❌ No pollutant columns found."
        )

        return

    target = poll_cols[0]

    print(
        f"Pollutant used for temporal analysis: {target}"
    )

    # --------------------------------------------------------
    # Find city
    # --------------------------------------------------------

    _, city, _ = detect_location_cols(df)

    if city is not None:

        top_city = (
            df[city]
            .value_counts()
            .idxmax()
        )

        df_city = df[
            df[city] == top_city
        ].copy()

        label = str(top_city)

        print(
            f"City used: {top_city}"
        )

    else:

        df_city = df.copy()

        label = "All locations"

    # --------------------------------------------------------
    # Set datetime index
    # --------------------------------------------------------

    df_city = df_city.set_index(
        dt_col
    )

    # ========================================================
    # GRAPH 3 - DAILY TIME SERIES
    # ========================================================

    print(
        "\nCreating daily time series..."
    )

    daily = (
        df_city[target]
        .resample("D")
        .mean()
    )

    plt.figure(figsize=(12, 5))

    plt.plot(
        daily.index,
        daily.values,
        color="blue",
        linewidth=1.5
    )

    plt.title(
        f"Daily Mean {target} - {label}"
    )

    plt.xlabel("Date")

    plt.ylabel(
        f"{target} concentration"
    )

    plt.grid(
        True,
        linestyle="--",
        alpha=0.4
    )

    show_and_save(
        "temporal_daily_timeseries.png"
    )

    # ========================================================
    # GRAPH 4 - WEEKLY PATTERN
    # ========================================================

    print(
        "\nCreating weekly pattern..."
    )

    df_city["day_of_week"] = (
        df_city.index.dayofweek
    )

    by_dow = (
        df_city
        .groupby("day_of_week")[target]
        .mean()
    )

    days = [
        "Mon",
        "Tue",
        "Wed",
        "Thu",
        "Fri",
        "Sat",
        "Sun"
    ]

    plt.figure(figsize=(8, 5))

    plt.bar(
        range(7),
        by_dow.reindex(range(7)),
        color="teal",
        edgecolor="black"
    )

    plt.xticks(
        range(7),
        days
    )

    plt.xlabel("Day of Week")

    plt.ylabel(
        f"Mean {target}"
    )

    plt.title(
        f"Weekly Pattern of {target} - {label}"
    )

    plt.grid(
        True,
        axis="y",
        linestyle="--",
        alpha=0.4
    )

    show_and_save(
        "temporal_weekly_pattern.png"
    )

    # ========================================================
    # GRAPH 5 - SEASONAL / MONTHLY
    # ========================================================

    print(
        "\nCreating seasonal pattern..."
    )

    df_city["month"] = (
        df_city.index.month
    )

    by_month = (
        df_city
        .groupby("month")[target]
        .mean()
    )

    plt.figure(figsize=(9, 5))

    plt.plot(
        by_month.index,
        by_month.values,
        marker="o",
        linewidth=2,
        color="darkorange"
    )

    plt.xticks(
        range(1, 13)
    )

    plt.xlabel("Month")

    plt.ylabel(
        f"Mean {target}"
    )

    plt.title(
        f"Seasonal Pattern of {target} - {label}"
    )

    plt.grid(
        True,
        linestyle="--",
        alpha=0.4
    )

    show_and_save(
        "temporal_seasonal_pattern.png"
    )


# ============================================================
# 4.4.3 SPATIAL / REGIONAL PATTERNS
# ============================================================

def eda_spatial_patterns():

    print("\n")
    print("=" * 70)
    print("4.4.3 SPATIAL / REGIONAL PATTERNS")
    print("=" * 70)

    if not GLOBAL_MAIN.exists():

        print("❌ Global CSV not found.")

        return

    df = pd.read_csv(
        GLOBAL_MAIN
    )

    country, city, _ = detect_location_cols(
        df
    )

    poll_cols = detect_pollutant_cols(
        df
    )

    if city is None:

        print(
            "⚠ City column not found."
        )

        return

    if not poll_cols:

        print(
            "⚠ Pollutant column not found."
        )

        return

    target = poll_cols[0]

    print(
        f"Pollutant used: {target}"
    )

    # ========================================================
    # GRAPH 6 - TOP 15 CITIES
    # ========================================================

    by_city = (
        df.groupby(city)[target]
        .mean()
        .sort_values(
            ascending=False
        )
        .head(15)
    )

    print(
        "\nTop 15 cities:"
    )

    print(by_city)

    plt.figure(
        figsize=(10, 7)
    )

    plt.barh(
        by_city.index[::-1],
        by_city.values[::-1],
        color="crimson"
    )

    plt.xlabel(
        f"Mean {target}"
    )

    plt.ylabel("City")

    plt.title(
        f"Top 15 Cities by Mean {target}"
    )

    plt.grid(
        True,
        axis="x",
        linestyle="--",
        alpha=0.4
    )

    show_and_save(
        "spatial_global_city_top15.png"
    )


# ============================================================
# 4.4.4 CORRELATION ANALYSIS
# ============================================================

def eda_correlation_analysis():

    print("\n")
    print("=" * 70)
    print("4.4.4 CORRELATION AND DEPENDENCE ANALYSIS")
    print("=" * 70)

    if not GLOBAL_MAIN.exists():

        print(
            "❌ Global CSV not found."
        )

        return

    df = pd.read_csv(
        GLOBAL_MAIN
    )

    poll_cols = detect_pollutant_cols(
        df
    )

    print(
        "\nPollutant columns:"
    )

    print(poll_cols)

    if len(poll_cols) < 2:

        print(
            "❌ Not enough pollutant columns."
        )

        return

    # ========================================================
    # CORRELATION MATRIX
    # ========================================================

    corr = df[
        poll_cols
    ].corr()

    print(
        "\nCorrelation matrix:"
    )

    print(corr)

    # ========================================================
    # GRAPH 7 - HEATMAP
    # ========================================================

    print(
        "\nCreating correlation heatmap..."
    )

    plt.figure(
        figsize=(9, 7)
    )

    im = plt.imshow(
        corr.values,
        cmap="coolwarm",
        vmin=-1,
        vmax=1
    )

    plt.colorbar(
        im,
        label="Correlation"
    )

    plt.xticks(
        range(len(poll_cols)),
        poll_cols,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        range(len(poll_cols)),
        poll_cols
    )

    # Add correlation values

    for i in range(
        len(poll_cols)
    ):

        for j in range(
            len(poll_cols)
        ):

            plt.text(
                j,
                i,
                f"{corr.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                color="black"
            )

    plt.title(
        "Correlation Matrix of Air Pollutants"
    )

    show_and_save(
        "correlation_heatmap_global.png"
    )

    # ========================================================
    # GRAPH 8 - SCATTER PLOT
    # ========================================================

    x_col = poll_cols[0]
    y_col = poll_cols[1]

    pair = df[
        [x_col, y_col]
    ].dropna()

    correlation = pair[
        x_col
    ].corr(
        pair[y_col]
    )

    print(
        f"\n{x_col} vs {y_col} correlation: "
        f"{correlation:.4f}"
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        pair[x_col],
        pair[y_col],
        alpha=0.4,
        color="purple",
        edgecolors="none"
    )

    plt.xlabel(
        x_col
    )

    plt.ylabel(
        y_col
    )

    plt.title(
        f"{x_col} vs {y_col}\n"
        f"Correlation = {correlation:.3f}"
    )

    plt.grid(
        True,
        linestyle="--",
        alpha=0.4
    )

    show_and_save(
        "correlation_scatter_pm25_pm10.png"
    )


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print(
        "       AIR QUALITY EXPLORATORY DATA ANALYSIS"
    )
    print("=" * 70)

    print(
        f"\nInput directory:\n{BASE_DIR}"
    )

    print(
        f"\nOutput figures directory:\n{FIG_DIR}"
    )

    # Check files

    check_files()

    # Run EDA

    eda_descriptive_statistics()

    eda_temporal_patterns()

    eda_spatial_patterns()

    eda_correlation_analysis()

    print("\n")
    print("=" * 70)
    print("EDA COMPLETE")
    print("=" * 70)

    print(
        f"\nAll figures are saved in:\n{FIG_DIR}"
    )

    print(
        "\nGraphs were also displayed on screen."
    )
