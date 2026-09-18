import pandas as pd
import numpy as np
import os

# FILE PATHS

RAW_PATH = "/Users/aalok_x92/Downloads/Development code/global_air_quality_dataset_10000_clean.csv"

OUT_PATH = "/Users/aalok_x92/Downloads/Development code/global_air_quality_dataset_10000_clean.csv"

# LOAD CSV

def load_data(path: str) -> pd.DataFrame:
    """Read WHO air quality data from CSV."""

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Input CSV file not found:\n{path}"
        )

    return pd.read_csv(path)
# REPLACE SPECIAL MISSING VALUES


def replace_special_missing(
    df: pd.DataFrame
) -> pd.DataFrame:

    # Empty strings / whitespace → NaN
    df = df.replace(
        r"^\s*$",
        np.nan,
        regex=True
    )

    # Special missing values → NaN
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

    # Find important location columns
    
    essential_cols = [
        c
        for c in df.columns
        if c.lower() in [
            "country",
            "city",
            "town"
        ]
    ]

    # Remove rows where essential location information
    # is missing
    if essential_cols:

        df = df.dropna(
            subset=essential_cols,
            how="any"
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

    return df

# NORMALISE AND HARMONISE UNITS


def normalise_and_harmonise_units(
    df: pd.DataFrame
) -> pd.DataFrame:

    # Find columns containing "unit"
    unit_cols = [
        c
        for c in df.columns
        if "unit" in c.lower()
    ]

    for ucol in unit_cols:

        # Normalize unit text
        df[ucol] = (
            df[ucol]
            .astype("string")
            .str.strip()
            .str.lower()
        )

        # Find PM value column
        value_candidates = [
            c
            for c in df.columns
            if (
                "pm" in c.lower()
                and c != ucol
            )
        ]

        if not value_candidates:
            continue

        vcol = value_candidates[0]

  
        
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

    # Create output directory if necessary
    output_dir = os.path.dirname(path)

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True
        )

    df.to_csv(
        path,
        index=False
    )

    print(
        f"\nCleaned CSV saved successfully:\n{path}"
    )

# MAIN

def main():

    print("=" * 70)
    print("WHO AIR QUALITY CSV CLEANING")
    print("=" * 70)

    # 1. Load CSV
    print("\n[1] Loading CSV...")
    df = load_data(RAW_PATH)

    print(
        f"Loaded: {df.shape[0]} rows × "
        f"{df.shape[1]} columns"
    )

    # 2. Replace special missing values
    print("\n[2] Replacing special missing values...")
    df = replace_special_missing(df)

    # 3. Remove empty columns
    print("[3] Removing empty columns...")
    df = drop_empty_columns(df)

    # 4. Remove empty rows
    print("[4] Removing empty rows...")
    df = drop_empty_rows(df)

    # 5. Handle missing values
    print("[5] Handling missing values...")
    df = clean_and_handle_missing(df)

    # 6. Normalize units
    print("[6] Normalising measurement units...")
    df = normalise_and_harmonise_units(df)

    # 7. Save
    print("[7] Saving cleaned CSV...")
    save_clean_data(
        df,
        OUT_PATH
    )

    print("\n" + "=" * 70)
    print("CLEANING COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(
        f"Final dataset: "
        f"{df.shape[0]} rows × "
        f"{df.shape[1]} columns"
    )

if __name__ == "__main__":
    main()
