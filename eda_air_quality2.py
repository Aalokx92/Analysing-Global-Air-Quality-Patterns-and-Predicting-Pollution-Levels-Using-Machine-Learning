import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# PROJECT PATHS

BASE_DIR = Path(
    "/Users/aalok_x92/Downloads/Development code"
)

# Processed CSV files
PROC_DIR = BASE_DIR

# Output directory for graphs
FIG_DIR = BASE_DIR / "figures"

# Create figures directory
FIG_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# CSV FILE PATHS

# Main dataset
GLOBAL_MAIN = (
    PROC_DIR /
    "global_air_quality_dataset_10000_clean.csv"
)

# Alternative 10K dataset
GLOBAL_10K = (
    PROC_DIR /
    "global_air_quality_data_10000_clean.csv"
)

# World / OpenAQ dataset
OPENAQ = (
    PROC_DIR /
    "world_air_quality_clean.csv"
)

# WHO 2014 dataset
WHO_2014 = (
    PROC_DIR /
    "aap_pm_database_may2014_clean.csv"
)

# WHO 2021 dataset
WHO_2021 = (
    PROC_DIR /
    "who_aap_2021_clean.csv"
)

# FILE CHECK
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

# SAFE CSV READER

def read_csv_safe(path: Path) -> pd.DataFrame:

    if not path.exists():

        raise FileNotFoundError(
            f"\nCSV file not found:\n{path}"
        )

    try:

        df = pd.read_csv(
            path,
            low_memory=False
        )

    except Exception as e:

        raise RuntimeError(
            f"\nCould not read CSV:\n{path}\n"
            f"Error: {e}"
        )

    return df

# DATA SUMMARY

def print_dataset_summary(
    name: str,
    df: pd.DataFrame
):

    print("\n" + "-" * 70)
    print(name)
    print("-" * 70)

    print(
        f"Rows: {df.shape[0]}"
    )

    print(
        f"Columns: {df.shape[1]}"
    )

    print(
        f"Missing values: {df.isna().sum().sum()}"
    )

    print(
        "Columns:"
    )

    for col in df.columns:

        print(
            f"  - {col}"
        )

# DATETIME DETECTION

def detect_datetime_col(
    df: pd.DataFrame
):

    possible_columns = []

    for col in df.columns:

        name = str(col).lower()

        if any(
            key in name
            for key in [
                "date",
                "time",
                "datetime",
                "timestamp"
            ]
        ):

            possible_columns.append(col)

    # First try obvious datetime columns
    for col in possible_columns:

        converted = pd.to_datetime(
            df[col],
            errors="coerce"
        )

        if converted.notna().sum() > 0:

            return col

    return None

# POLLUTANT DETECTION

def detect_pollutant_cols(
    df: pd.DataFrame
):

    pollutant_keywords = [
        "pm2",
        "pm10",
        "no2",
        "so2",
        "o3",
        "co"
    ]

    pollutant_cols = []

    for col in df.columns:

        col_lower = str(col).lower()

        if not any(
            key in col_lower
            for key in pollutant_keywords
        ):

            continue

        # Convert to numeric if necessary
        converted = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        # Use the converted values
        if converted.notna().any():

            df[col] = converted

            pollutant_cols.append(col)

    return pollutant_cols

# LOCATION DETECTION

def detect_location_cols(
    df: pd.DataFrame
):

    country = None
    city = None
    station = None

    for col in df.columns:

        name = str(col).lower()

        if (
            country is None
            and "country" in name
        ):

            country = col

        if (
            city is None
            and "city" in name
        ):

            city = col

        if (
            station is None
            and (
                "station" in name
                or "location" in name
            )
        ):

            station = col

    return country, city, station

# 4.4.1 DESCRIPTIVE STATISTICS

def eda_descriptive_statistics():

    print("\n")
    print("=" * 70)
    print("4.4.1 DESCRIPTIVE STATISTICS")
    print("=" * 70)

    # Load global dataset

    df_global = read_csv_safe(
        GLOBAL_MAIN
    )

    print_dataset_summary(
        "GLOBAL DATASET",
        df_global
    )
    # Load WHO 2021 if available

    if WHO_2021.exists():

        df_who = read_csv_safe(
            WHO_2021
        )

    else:

        print(
            "\n⚠ WHO 2021 CSV not found."
        )

        df_who = None

    # Detect pollutants

    poll_global = detect_pollutant_cols(
        df_global
    )

    print(
        "\nGlobal pollutant columns:"
    )

    print(
        poll_global
    )

    if df_who is not None:

        poll_who = detect_pollutant_cols(
            df_who
        )

        print(
            "WHO pollutant columns:"
        )

        print(
            poll_who
        )

    else:

        poll_who = []
    # HISTOGRAMS

    if poll_global:

        print(
            "\nCreating global pollutant histograms..."
        )

        plt.figure(
            figsize=(12, 8)
        )

        selected_cols = poll_global[:4]

        for i, col in enumerate(
            selected_cols,
            start=1
        ):

            plt.subplot(
                2,
                2,
                i
            )

            values = pd.to_numeric(
                df_global[col],
                errors="coerce"
            ).dropna()

            if values.empty:

                continue

            plt.hist(
                values,
                bins=30,
                color="steelblue",
                edgecolor="black",
                alpha=0.75
            )

            plt.title(
                str(col)
            )

            plt.xlabel(
                "Concentration"
            )

            plt.ylabel(
                "Frequency"
            )

            plt.grid(
                True,
                linestyle="--",
                alpha=0.5
            )

        plt.suptitle(
            "Histograms of Key Pollutant Concentrations",
            fontsize=15
        )

        plt.tight_layout(
            rect=[
                0,
                0,
                1,
                0.95
            ]
        )

        output = (
            FIG_DIR /
            "descriptive_global_histograms.png"
        )

        plt.savefig(
            output,
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()
        plt.close()

        print(
            f"✓ Saved: {output}"
        )
    # GLOBAL BOXPLOT

    if poll_global:

        print(
            "\nCreating global pollutant boxplot..."
        )

        data = []
        labels = []

        for col in poll_global:

            values = pd.to_numeric(
                df_global[col],
                errors="coerce"
            ).dropna()

            if not values.empty:

                data.append(
                    values
                )

                labels.append(
                    str(col)
                )

        if data:

            plt.figure(
                figsize=(10, 6)
            )

            plt.boxplot(
                data,
                tick_labels=labels,
                showfliers=False
            )

            plt.title(
                "Distribution of Pollutant Concentrations"
            )

            plt.ylabel(
                "Concentration"
            )

            plt.xticks(
                rotation=45
            )

            plt.grid(
                True,
                axis="y",
                linestyle="--",
                alpha=0.5
            )

            plt.tight_layout()

            output = (
                FIG_DIR /
                "descriptive_global_boxplot.png"
            )

            plt.savefig(
                output,
                dpi=300,
                bbox_inches="tight"
            )
            plt.show()
            plt.close()

            print(
                f"✓ Saved: {output}"
            )
    # WHO BOXPLOT

    if poll_who:

        print(
            "\nCreating WHO 2021 boxplot..."
        )

        data = []
        labels = []

        for col in poll_who:

            values = pd.to_numeric(
                df_who[col],
                errors="coerce"
            ).dropna()

            if not values.empty:

                data.append(
                    values
                )

                labels.append(
                    str(col)
                )

        if data:

            plt.figure(
                figsize=(10, 6)
            )

            plt.boxplot(
                data,
                tick_labels=labels,
                showfliers=False
            )

            plt.title(
                "WHO 2021 – Pollutant Statistics"
            )

            plt.ylabel(
                "Annual Mean (µg/m³)"
            )

            plt.xticks(
                rotation=45
            )

            plt.grid(
                True,
                axis="y",
                linestyle="--",
                alpha=0.5
            )

            plt.tight_layout()

            output = (
                FIG_DIR /
                "descriptive_who2021_boxplot.png"
            )

            plt.savefig(
                output,
                dpi=300,
                bbox_inches="tight"
            )
            plt.show()
            plt.close()

            print(
                f"✓ Saved: {output}"
            )
# 4.4.2 TEMPORAL PATTERNS

def eda_temporal_patterns():

    print("\n")
    print("=" * 70)
    print("4.4.2 TEMPORAL PATTERNS")
    print("=" * 70)

    df = read_csv_safe(
        GLOBAL_MAIN
    )

    # Detect datetime

    dt_col = detect_datetime_col(
        df
    )

    if dt_col is None:

        print(
            "⚠ No datetime column detected."
        )

        return

    print(
        f"Datetime column: {dt_col}"
    )
    # Convert datetime

    df[dt_col] = pd.to_datetime(
        df[dt_col],
        errors="coerce"
    )

    df = df.dropna(
        subset=[dt_col]
    )

    if df.empty:

        print(
            "⚠ No valid datetime records."
        )

        return
    # Detect pollutants

    poll_cols = detect_pollutant_cols(
        df
    )

    if not poll_cols:

        print(
            "⚠ No pollutant columns found."
        )

        return

    target = poll_cols[0]

    print(
        f"Pollutant used for temporal analysis: {target}"
    )

    # Find city

    country, city, station = (
        detect_location_cols(
            df
        )
    )

    if city is not None:

        city_counts = (
            df[city]
            .dropna()
            .value_counts()
        )

        if not city_counts.empty:

            top_city = (
                city_counts.index[0]
            )

            df_city = df[
                df[city] == top_city
            ].copy()

            label = str(
                top_city
            )

        else:

            df_city = df.copy()
            label = "All Locations"

    else:

        df_city = df.copy()
        label = "All Locations"

    # Datetime index

    df_city = df_city.set_index(
        dt_col
    )

    df_city[target] = pd.to_numeric(
        df_city[target],
        errors="coerce"
    )

    # DAILY

    daily = (
        df_city[target]
        .resample("D")
        .mean()
        .dropna()
    )

    if not daily.empty:

        plt.figure(
            figsize=(12, 5)
        )

        plt.plot(
            daily.index,
            daily.values,
            color="blue",
            linewidth=1.5
        )

        plt.title(
            f"Daily Mean {target} – {label}"
        )

        plt.xlabel(
            "Date"
        )

        plt.ylabel(
            f"{target} (µg/m³)"
        )

        plt.grid(
            True,
            linestyle="--",
            alpha=0.5
        )

        plt.tight_layout()

        output = (
            FIG_DIR /
            "temporal_daily_timeseries.png"
        )

        plt.savefig(
            output,
            dpi=300,
            bbox_inches="tight"
        )
        plt.show()
        plt.close()

        print(
            f"✓ Saved: {output}"
        )

    # WEEKLY

    df_city["day_of_week"] = (
        df_city.index.dayofweek
    )

    weekly = (
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

    if not weekly.empty:

        plt.figure(
            figsize=(8, 5)
        )

        plt.bar(
            weekly.index,
            weekly.values,
            color="orange",
            edgecolor="black"
        )

        plt.xticks(
            range(7),
            days
        )

        plt.title(
            f"Weekly Pattern of {target} – {label}"
        )

        plt.xlabel(
            "Day of Week"
        )

        plt.ylabel(
            f"Mean {target}"
        )

        plt.grid(
            True,
            axis="y",
            linestyle="--",
            alpha=0.5
        )

        plt.tight_layout()

        output = (
            FIG_DIR /
            "temporal_weekly_pattern.png"
        )

        plt.savefig(
            output,
            dpi=300,
            bbox_inches="tight"
        )
        plt.show()
        plt.close()

        print(
            f"✓ Saved: {output}"
        )

    # SEASONAL / MONTHLY 

    df_city["month"] = (
        df_city.index.month
    )

    monthly = (
        df_city
        .groupby("month")[target]
        .mean()
    )

    if not monthly.empty:

        plt.figure(
            figsize=(9, 5)
        )

        plt.plot(
            monthly.index,
            monthly.values,
            marker="o",
            color="green",
            linewidth=2
        )

        plt.xticks(
            range(1, 13)
        )

        plt.title(
            f"Seasonal Pattern of {target} – {label}"
        )

        plt.xlabel(
            "Month"
        )

        plt.ylabel(
            f"Mean {target}"
        )

        plt.grid(
            True,
            linestyle="--",
            alpha=0.5
        )

        plt.tight_layout()

        output = (
            FIG_DIR /
            "temporal_seasonal_pattern.png"
        )

        plt.savefig(
            output,
            dpi=300,
            bbox_inches="tight"
        )
        plt.show()
        plt.close()

        print(
            f"✓ Saved: {output}"
        )

# 4.4.3 SPATIAL / REGIONAL PATTERNS


def eda_spatial_patterns():

    print("\n")
    print("=" * 70)
    print("4.4.3 SPATIAL / REGIONAL PATTERNS")
    print("=" * 70)

    # WHO 2021 COUNTRY ANALYSIS

    if WHO_2021.exists():

        df_who = read_csv_safe(
            WHO_2021
        )

        country_col, city_col, station_col = (
            detect_location_cols(
                df_who
            )
        )

        poll_cols = detect_pollutant_cols(
            df_who
        )

        if country_col and poll_cols:

            target = poll_cols[0]

            df_who[target] = pd.to_numeric(
                df_who[target],
                errors="coerce"
            )

            by_country = (
                df_who
                .groupby(country_col)[target]
                .mean()
                .dropna()
                .sort_values(
                    ascending=False
                )
                .head(15)
            )

            if not by_country.empty:

                plt.figure(
                    figsize=(11, 7)
                )

                plt.barh(
                    by_country.index[::-1],
                    by_country.values[::-1],
                    color="crimson"
                )

                plt.xlabel(
                    f"Mean {target} (µg/m³)"
                )

                plt.title(
                    f"Top 15 Countries by Mean {target} – WHO 2021"
                )

                plt.grid(
                    True,
                    axis="x",
                    linestyle="--",
                    alpha=0.5
                )

                plt.tight_layout()

                output = (
                    FIG_DIR /
                    "spatial_who_country_top15.png"
                )

                plt.savefig(
                    output,
                    dpi=300,
                    bbox_inches="tight"
                )
                plt.show()
                plt.close()

                print(
                    f"✓ Saved: {output}"
                )

        else:

            print(
                "⚠ WHO country/pollutant "
                "columns not detected."
            )

    else:

        print(
            "⚠ WHO 2021 file not available."
        )

    # GLOBAL CITY ANALYSIS

    df_global = read_csv_safe(
        GLOBAL_MAIN
    )

    country_col, city_col, station_col = (
        detect_location_cols(
            df_global
        )
    )

    poll_cols = detect_pollutant_cols(
        df_global
    )

    if city_col and poll_cols:

        target = poll_cols[0]

        df_global[target] = pd.to_numeric(
            df_global[target],
            errors="coerce"
        )

        by_city = (
            df_global
            .groupby(city_col)[target]
            .mean()
            .dropna()
            .sort_values(
                ascending=False
            )
            .head(15)
        )

        if not by_city.empty:

            plt.figure(
                figsize=(11, 7)
            )

            plt.barh(
                by_city.index[::-1],
                by_city.values[::-1],
                color="darkblue"
            )

            plt.xlabel(
                f"Mean {target}"
            )

            plt.title(
                f"Top 15 Cities by Mean {target}"
            )

            plt.grid(
                True,
                axis="x",
                linestyle="--",
                alpha=0.5
            )

            plt.tight_layout()

            output = (
                FIG_DIR /
                "spatial_global_city_top15.png"
            )

            plt.savefig(
                output,
                dpi=300,
                bbox_inches="tight"
            )
            plt.show()
            plt.close()

            print(
                f"✓ Saved: {output}"
            )

    else:

        print(
            "⚠ City or pollutant column "
            "not detected in global dataset."
        )


# PM2.5 / PM10 DETECTION

def pick_pm25_pm10_columns(
    poll_cols
):

    pm25_col = None
    pm10_col = None

    for col in poll_cols:

        name = str(col).lower()

        # PM2.5
        if pm25_col is None:

            if (
                "pm2.5" in name
                or "pm2_5" in name
                or "pm25" in name
            ):

                pm25_col = col

        # PM10
        if pm10_col is None:

            if "pm10" in name:

                pm10_col = col

    # Fallback
    if (
        pm25_col is None
        and len(poll_cols) >= 1
    ):

        pm25_col = poll_cols[0]

    if (
        pm10_col is None
        and len(poll_cols) >= 2
    ):

        for col in poll_cols:

            if col != pm25_col:

                pm10_col = col
                break

    if pm25_col == pm10_col:

        pm10_col = None

    return pm25_col, pm10_col

# 4.4.4 CORRELATION ANALYSIS

def eda_correlation_analysis():

    print("\n")
    print("=" * 70)
    print("4.4.4 CORRELATION AND DEPENDENCE ANALYSIS")
    print("=" * 70)

    df = read_csv_safe(
        GLOBAL_MAIN
    )

    poll_cols = detect_pollutant_cols(
        df
    )

    print(
        "Pollutant columns:"
    )

    print(
        poll_cols
    )

    if len(poll_cols) < 2:

        print(
            "⚠ At least two pollutant columns "
            "are required."
        )

        return

    # CORRELATION MATRIX

    corr = df[
        poll_cols
    ].corr()

    plt.figure(
        figsize=(8, 7)
    )

    image = plt.imshow(
        corr.values,
        cmap="coolwarm",
        vmin=-1,
        vmax=1
    )

    plt.colorbar(
        image,
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

    plt.title(
        "Correlation Matrix of Pollutants"
    )

    plt.tight_layout()

    output = (
        FIG_DIR /
        "correlation_heatmap_global.png"
    )

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()
    plt.close()

    print(
        f"✓ Saved: {output}"
    )

    # PM2.5 VS PM10

    pm25_col, pm10_col = (
        pick_pm25_pm10_columns(
            poll_cols
        )
    )

    print(
        f"PM2.5 column: {pm25_col}"
    )

    print(
        f"PM10 column: {pm10_col}"
    )

    if (
        pm25_col is None
        or pm10_col is None
    ):

        print(
            "⚠ PM2.5 and PM10 could not "
            "be identified."
        )

        return

    df_pair = df[
        [
            pm25_col,
            pm10_col
        ]
    ].copy()

    df_pair[pm25_col] = pd.to_numeric(
        df_pair[pm25_col],
        errors="coerce"
    )

    df_pair[pm10_col] = pd.to_numeric(
        df_pair[pm10_col],
        errors="coerce"
    )

    df_pair = df_pair.dropna()

    if df_pair.empty:

        print(
            "⚠ No overlapping PM2.5 / PM10 data."
        )

        return

    # Correlation
    r = df_pair[
        pm25_col
    ].corr(
        df_pair[pm10_col]
    )

    print(
        f"PM2.5 vs PM10 correlation: {r:.4f}"
    )

    # HEXBIN

    plt.figure(
        figsize=(8, 6)
    )

    hexbin = plt.hexbin(
        df_pair[pm25_col],
        df_pair[pm10_col],
        gridsize=40,
        mincnt=1,
        cmap="viridis"
    )

    plt.colorbar(
        hexbin,
        label="Number of observations"
    )

    # 1:1 line
    minimum = min(
        df_pair[pm25_col].min(),
        df_pair[pm10_col].min()
    )

    maximum = max(
        df_pair[pm25_col].max(),
        df_pair[pm10_col].max()
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        "--",
        color="red",
        linewidth=1.5,
        label="1:1 reference"
    )

    plt.xlabel(
        pm25_col
    )

    plt.ylabel(
        pm10_col
    )

    plt.title(
        f"PM2.5 vs PM10 Relationship "
        f"(r = {r:.2f})"
    )

    plt.legend()

    plt.grid(
        True,
        linestyle="--",
        alpha=0.4
    )

    plt.tight_layout()

    output = (
        FIG_DIR /
        "correlation_scatter_pm25_pm10.png"
    )

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()
    plt.close()

    print(
        f"✓ Saved: {output}"
    )

# MAIN PROGRAM

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("       AIR QUALITY EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    print(
        f"\nInput directory:\n{PROC_DIR}"
    )

    print(
        f"\nOutput figures directory:\n{FIG_DIR}"
    )

    # Check files

    check_files()

    # Run EDA

    try:

        # 4.4.1
        eda_descriptive_statistics()

        # 4.4.2
        eda_temporal_patterns()

        # 4.4.3
        eda_spatial_patterns()

        # 4.4.4
        eda_correlation_analysis()

        # Finished

        print("\n")
        print("=" * 70)
        print("EDA COMPLETE")
        print("=" * 70)

        print(
            f"\nAll generated figures are saved in:"
        )

        print(
            FIG_DIR
        )

        print("\n")

    except FileNotFoundError as e:

        print("\n")
        print("=" * 70)
        print("❌ FILE ERROR")
        print("=" * 70)

        print(e)

    except Exception as e:

        print("\n")
        print("=" * 70)
        print("❌ EDA ERROR")
        print("=" * 70)

        print(
            f"{type(e).__name__}: {e}"
        )
