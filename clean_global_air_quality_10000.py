import pandas as pd
import numpy as np
import os

# FILE PATHS

RAW_PATH =  "/Users/aalok_x92/Downloads/Development code/global_air_quality_dataset_10000_clean.csv"

OUT_PATH = "/Users/aalok_x92/Downloads/Development code/global_air_quality_dataset_10000_cleaned.csv"

# LOAD CSV


def load_data(path: str) -> pd.DataFrame:
    """Read the raw CSV file."""
    return pd.read_csv(path)

# HANDLE SPECIAL MISSING VALUES


def replace_special_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Replace blank cells and special missing-value markers."""

    # Empty strings / whitespace -> NaN
    df = df.replace(r"^\s*$", np.nan, regex=True)

    # Special missing-value markers -> NaN
    df = df.replace(["##", "###"], np.nan)

    return df

# REMOVE EMPTY COLUMNS


def drop_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns containing only missing values."""

    return df.dropna(axis=1, how="all")

# REMOVE EMPTY ROWS

def drop_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows containing only missing values."""

    return df.dropna(axis=0, how="all")

# CLEAN AND HANDLE MISSING VALUES

def clean_and_handle_missing(df: pd.DataFrame) -> pd.DataFrame:

    # Convert date/time columns

    for col in df.columns:

        if "date" in col.lower() or "time" in col.lower():

            df[col] = pd.to_datetime(
                df[col],
                errors="coerce"
            )

    # Identify pollutant columns

    pollutant_cols = [
        c for c in df.columns
        if any(
            p in c.lower()
            for p in ["pm2", "pm10", "no2", "so2", "o3", "co"]
        )
    ]

    # Remove rows where ALL pollutant values are missing

    if pollutant_cols:
        df = df.dropna(
            subset=pollutant_cols,
            how="all"
        )

    # Fill numeric missing values with median

    num_cols = df.select_dtypes(
        include=[np.number]
    ).columns

    for col in num_cols:

        if df[col].notna().any():

            df[col] = df[col].fillna(
                df[col].median()
            )

    # Fill categorical missing values with mode

    cat_cols = df.select_dtypes(
        exclude=[np.number]
    ).columns

    for col in cat_cols:

        if df[col].notna().any():

            mode_value = df[col].mode()

            if not mode_value.empty:

                df[col] = df[col].fillna(
                    mode_value.iloc[0]
                )

    return df

# NORMALISE AND HARMONISE UNITS


def normalise_and_harmonise_units(
    df: pd.DataFrame
) -> pd.DataFrame:

    # Find columns containing "unit"
    unit_cols = [
        c for c in df.columns
        if "unit" in c.lower()
    ]

    for ucol in unit_cols:

        # Determine the associated measurement column
        

        base = (
            ucol
            .replace("unit", "")
            .replace("_", "")
            .lower()
        )

        val_candidates = [
            c
            for c in df.columns
            if (
                base
                and base in c.replace("_", "").lower()
                and c != ucol
            )
        ]

        if not val_candidates:
            continue

        val_col = val_candidates[0]

        # Normalize unit text

        df[ucol] = (
            df[ucol]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        # Convert mg/m3 -> µg/m³
        # 1 mg/m³ = 1000 µg/m³

        mask_mg = df[ucol].str.contains(
            "mg/m3",
            na=False
        )

        if mask_mg.any():

            df[val_col] = pd.to_numeric(
                df[val_col],
                errors="coerce"
            )

            df.loc[mask_mg, val_col] = (
                df.loc[mask_mg, val_col] * 1000
            )

            df.loc[mask_mg, ucol] = "µg/m³"

    
        # Normalize ug/m3 and µg/m3

        mask_ug = (
            df[ucol].str.contains(
                "ug/m3",
                na=False
            )
            |
            df[ucol].str.contains(
                "µg/m3",
                na=False
            )
        )

        df.loc[mask_ug, ucol] = "µg/m³"

    return df

# SAVE CLEAN CSV

def save_clean_data(
    df: pd.DataFrame,
    path: str
) -> None:

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(path)

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True
        )

    # Save as CSV
    df.to_csv(
        path,
        index=False
    )

# MAIN PIPELINE

def main():

    print(" Air Quality CSV Cleaning Pipeline")
    


    # Load


    print("\n[1/7] Reading CSV...")

    df = load_data(RAW_PATH)

    print(
        f"Loaded {len(df)} rows "
        f"and {len(df.columns)} columns."
    )

    
    # Replace special missing values

    print("[2/7] Handling special missing values...")

    df = replace_special_missing(df)

    
    # Remove empty columns
    

    print("[3/7] Removing empty columns...")

    df = drop_empty_columns(df)

   
    # Remove empty rows

    print("[4/7] Removing empty rows...")

    df = drop_empty_rows(df)

    # Handle missing values

    print("[5/7] Cleaning missing values...")

    df = clean_and_handle_missing(df)

    # Normalize units

    print("[6/7] Normalising measurement units...")

    df = normalise_and_harmonise_units(df)

    # Save


    print("[7/7] Saving cleaned CSV...")

    save_clean_data(
        df,
        OUT_PATH
    )

    # Final information

   
    print(" Cleaning completed successfully!")
   

    print(f"Final rows    : {len(df)}")
    print(f"Final columns : {len(df.columns)}")
    print(f"Output file   : {OUT_PATH}")

# RUN PROGRAM

if __name__ == "__main__":
    main()
