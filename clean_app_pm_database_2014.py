import pandas as pd
import numpy as np

# Input CSV file
RAW_PATH = "/Users/aalok_x92/Downloads/Development code/global_air_quality_dataset_10000_clean.csv"

# Output CSV file
OUT_PATH = "/Users/aalok_x92/Downloads/Development code/global_air_quality_dataset_10000_cleaned.csv"


def load_data(path):
    """Read CSV file into a pandas DataFrame."""
    return pd.read_csv(path)


def replace_special_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Replace blank cells and special missing-value markers with NaN."""
    df = df.replace(r"^\s*$", np.nan, regex=True)
    df = df.replace(["##", "###"], np.nan)
    return df


def drop_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that contain only missing values."""
    return df.dropna(axis=1, how="all")


def drop_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows that contain only missing values."""
    return df.dropna(axis=0, how="all")


def clean_and_handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows missing essential fields and fill numeric missing values."""

    essential_cols = [
        c for c in df.columns
        if c.lower() in ["country", "city", "station"]
    ]

    if essential_cols:
        df = df.dropna(subset=essential_cols, how="any")

    # Find numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns

    # Fill missing numeric values with median
    for col in num_cols:
        if df[col].notna().any():
            df[col] = df[col].fillna(df[col].median())

    return df


def normalise_and_harmonise_units(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize PM measurement units."""

    unit_cols = [
        c for c in df.columns
        if "unit" in c.lower()
    ]

    for ucol in unit_cols:

        # Convert unit values to strings and normalize
        df[ucol] = (
            df[ucol]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        # Find PM value columns
        value_candidates = [
            c for c in df.columns
            if "pm" in c.lower() and c != ucol
        ]

        if not value_candidates:
            continue

        vcol = value_candidates[0]

        # Convert mg/m3 -> µg/m³
        mask_mg = df[ucol].str.contains(
            "mg/m3",
            na=False
        )

        df.loc[mask_mg, vcol] = (
            pd.to_numeric(
                df.loc[mask_mg, vcol],
                errors="coerce"
            ) * 1000
        )

        df.loc[mask_mg, ucol] = "µg/m³"

        # Normalize ug/m3 and µg/m3
        mask_ug = (
            df[ucol].str.contains("ug/m3", na=False)
            | df[ucol].str.contains("µg/m3", na=False)
        )

        df.loc[mask_ug, ucol] = "µg/m³"

    return df


def save_clean_data(df: pd.DataFrame, path: str) -> None:
    """Save cleaned DataFrame as CSV."""
    df.to_csv(path, index=False)


def main():
    print("Reading CSV file...")

    df = load_data(RAW_PATH)

    print(f"Original rows: {len(df)}")
    print(f"Original columns: {len(df.columns)}")

    df = replace_special_missing(df)
    df = drop_empty_columns(df)
    df = drop_empty_rows(df)
    df = clean_and_handle_missing(df)
    df = normalise_and_harmonise_units(df)

    save_clean_data(df, OUT_PATH)

    print("\nCleaning completed successfully.")
    print(f"Final rows: {len(df)}")
    print(f"Final columns: {len(df.columns)}")
    print(f"Output saved to:\n{OUT_PATH}")


if __name__ == "__main__":
    main()
