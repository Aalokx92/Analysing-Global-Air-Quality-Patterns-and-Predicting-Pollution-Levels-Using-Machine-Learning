import pandas as pd
import numpy as np
import os

# FILE PATHS

RAW_PATH = "/Users/aalok_x92/Downloads/Development code/global_air_quality_dataset_10000_clean.csv"

OUT_PATH = "/Users/aalok_x92/Downloads/Development code/global_air_quality_dataset_10000_clean.csv"

# LOAD CSV

def load_data(path: str) -> pd.DataFrame:
    """Load world air quality data from CSV."""

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Input CSV file not found:\n{path}"
        )

    df = pd.read_csv(path)

    print(
        f"Loaded CSV: {df.shape[0]} rows × "
        f"{df.shape[1]} columns"
    )

    return df

# REPLACE SPECIAL MISSING VALUES

def replace_special_missing(
    df: pd.DataFrame
) -> pd.DataFrame:

    # Blank strings / whitespace → NaN
    df = df.replace(
        r"^\s*$",
        np.nan,
        regex=True
    )

    # Special missing-value markers → NaN
    df = df.replace(
        ["##", "###"],
        np.nan
    )

    return df

# DROP EMPTY COLUMNS

def drop_empty_columns(
    df: pd.DataFrame
) -> pd.DataFrame:

    return df.dropna(
        axis=1,
        how="all"
    )

# DROP EMPTY ROWS

def drop_empty_rows(
    df: pd.DataFrame
) -> pd.DataFrame:

    return df.dropna(
        axis=0,
        how="all"
    )

# CLEAN AND HANDLE MISSING VALUES

def clean_and_handle_missing(
    df: pd.DataFrame
) -> pd.DataFrame:

    # Convert date/time columns

    for col in df.columns:

        if (
            "date" in col.lower()
            or "time" in col.lower()
        ):

            converted = pd.to_datetime(
                df[col],
                errors="coerce"
            )

            # Only replace if conversion succeeded
            if converted.notna().any():
                df[col] = converted

    # Find value/concentration columns

    value_cols = [
        c
        for c in df.columns
        if c.lower() in [
            "value",
            "concentration"
        ]
    ]

    # Rows must have a measurement value
    if value_cols:

        df = df.dropna(
            subset=value_cols,
            how="any"
        )
    # Convert value columns to numeric

    for col in value_cols:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # Fill numeric missing values with median


    num_cols = df.select_dtypes(
        include=[np.number]
    ).columns

    for col in num_cols:

        if (
            df[col].notna().any()
            and df[col].isna().any()
        ):

            df[col] = df[col].fillna(
                df[col].median()
            )
    # Fill categorical missing values with mode

    cat_cols = df.select_dtypes(
        include=["object", "category"]
    ).columns

    for col in cat_cols:

        if (
            df[col].notna().any()
            and df[col].isna().any()
        ):

            mode_values = df[col].mode()

            if not mode_values.empty:

                df[col] = df[col].fillna(
                    mode_values.iloc[0]
                )

    return df

# NORMALISE AND HARMONISE UNITS

def normalise_and_harmonise_units(
    df: pd.DataFrame
) -> pd.DataFrame:

    # Find Unit column
    unit_col_candidates = [
        c
        for c in df.columns
        if c.lower() == "unit"
    ]

    # Find Value / Concentration column
    value_col_candidates = [
        c
        for c in df.columns
        if c.lower() in [
            "value",
            "concentration"
        ]
    ]

    if (
        not unit_col_candidates
        or not value_col_candidates
    ):
        print(
            "No matching Unit and Value/"
            "Concentration columns found."
        )

        return df

    ucol = unit_col_candidates[0]
    vcol = value_col_candidates[0]

    # Normalize unit text

    df[ucol] = (
        df[ucol]
        .astype("string")
        .str.strip()
        .str.lower()
    ) 
    # Convert mg/m3 → µg/m³
    # 1 mg/m³ = 1000 µg/m³

    mask_mg = df[ucol].str.contains(
        "mg/m3",
        na=False
    )

    if mask_mg.any():

        df[vcol] = pd.to_numeric(
            df[vcol],
            errors="coerce"
        )

        df.loc[
            mask_mg,
            vcol
        ] = (
            df.loc[
                mask_mg,
                vcol
            ] * 1000
        )

        df.loc[
            mask_mg,
            ucol
        ] = "µg/m³"

        print(
            "Converted "
            f"{int(mask_mg.sum())} rows "
            "from mg/m³ to µg/m³"
        )

    # Normalize ug/m3 → µg/m³

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

    if mask_ug.any():

        df.loc[
            mask_ug,
            ucol
        ] = "µg/m³"

    return df

# SAVE CLEAN CSV

def save_clean_data(
    df: pd.DataFrame,
    path: str
) -> None:

    # Create output directory
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

    print(
        f"\nCleaned CSV saved to:\n{path}"
    )

# MAIN PIPELINE

def main():

    print("\n" + "=" * 70)
    print("       WORLD AIR QUALITY CSV CLEANING")
    print("=" * 70)

    
    # 1. Load

    print("\n[1] Loading CSV...")

    df = load_data(
        RAW_PATH
    )

    # 2. Replace special missing values

    print(
        "[2] Replacing special missing values..."
    )

    df = replace_special_missing(
        df
    )

    # 3. Drop empty columns

    print(
        "[3] Removing empty columns..."
    )

    df = drop_empty_columns(
        df
    )

    # 4. Drop empty rows

    print(
        "[4] Removing empty rows..."
    )

    df = drop_empty_rows(
        df
    )

    # 5. Handle missing values


    print(
        "[5] Cleaning missing values..."
    )

    df = clean_and_handle_missing(
        df
    )
    # 6. Normalize units

    print(
        "[6] Normalising units..."
    )

    df = normalise_and_harmonise_units(
        df
    )

    # 7. Preview

    print("\nColumns:")
    print(
        list(df.columns)
    )

    print("\nFirst 5 rows:")
    print(
        df.head(5).to_string(
            index=False
        )
    )

    # 8. Save

    print(
        "\n[7] Saving cleaned CSV..."
    )

    save_clean_data(
        df,
        OUT_PATH
    )
    # Final summary

    print("\n" + "=" * 70)
    print("          CLEANING COMPLETED")
    print("=" * 70)

    print(
        f"Final shape: "
        f"{df.shape[0]} rows × "
        f"{df.shape[1]} columns"
    )

    print("=" * 70)
# RUN

if __name__ == "__main__":
    main()
