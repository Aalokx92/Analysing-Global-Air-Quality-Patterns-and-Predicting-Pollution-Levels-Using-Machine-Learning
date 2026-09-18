import pandas as pd
import numpy as np
import os

# FILE PATHS

RAW_PATH = "/Users/aalok_x92/Downloads/Development code/global_air_quality_dataset_10000_cleaned.csv"

OUT_PATH = "/Users/aalok_x92/Downloads/Development code/global_air_quality_dataset_10000_clean.csv"

# LOGGING

def log_step(title: str, df: pd.DataFrame) -> None:
    """Print a quick summary to the VS Code terminal."""

    print("\n" + "=" * 70)
    print(title)
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    na_total = int(df.isna().sum().sum())

    print(f"Total missing values: {na_total}")
    print("=" * 70)

# LOAD CSV

def load_data(path: str) -> pd.DataFrame:
    """Load the raw CSV dataset."""

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Input CSV file not found:\n{path}"
        )

    df = pd.read_csv(path)

    log_step(
        "1) Loaded raw CSV",
        df
    )

    return df

# REPLACE SPECIAL MISSING VALUES


def replace_special_missing(
    df: pd.DataFrame
) -> pd.DataFrame:

    before_na = int(
        df.isna().sum().sum()
    )

    # Empty strings / whitespace → NaN
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

    after_na = int(
        df.isna().sum().sum()
    )

    print(
        f"\n[replace_special_missing] "
        f"Missing values: {before_na} → {after_na}"
    )

    return df

# DROP EMPTY COLUMNS


def drop_empty_columns(
    df: pd.DataFrame
) -> pd.DataFrame:

    before_cols = df.shape[1]

    df = df.dropna(
        axis=1,
        how="all"
    )

    after_cols = df.shape[1]

    print(
        f"[drop_empty_columns] "
        f"Columns: {before_cols} → {after_cols}"
    )

    return df

# DROP EMPTY ROWS


def drop_empty_rows(
    df: pd.DataFrame
) -> pd.DataFrame:

    before_rows = df.shape[0]

    df = df.dropna(
        axis=0,
        how="all"
    )

    after_rows = df.shape[0]

    print(
        f"[drop_empty_rows] "
        f"Rows: {before_rows} → {after_rows}"
    )

    return df

# CLEAN AND HANDLE MISSING VALUES

def clean_and_handle_missing(
    df: pd.DataFrame
) -> pd.DataFrame:


    # Parse date/time columns


    parsed_cols = []

    for col in df.columns:

        if (
            "date" in col.lower()
            or "time" in col.lower()
        ):

            new_col = pd.to_datetime(
                df[col],
                errors="coerce"
            )

            # Replace only when at least one value
            # was successfully converted
            if new_col.notna().any():

                df[col] = new_col

                parsed_cols.append(col)

    if parsed_cols:

        print(
            "[clean_and_handle_missing] "
            f"Parsed datetime columns: {parsed_cols}"
        )

    
    # Find pollutant columns
    

    pollutant_cols = [
        c
        for c in df.columns
        if any(
            p in c.lower()
            for p in [
                "pm2",
                "pm10",
                "no2",
                "so2",
                "o3",
                "co"
            ]
        )
    ]

    if pollutant_cols:

        print(
            "[clean_and_handle_missing] "
            f"Pollutant columns: {pollutant_cols}"
        )

        before_rows = df.shape[0]

        # Remove rows where every pollutant is missing
        df = df.dropna(
            subset=pollutant_cols,
            how="all"
        )

        after_rows = df.shape[0]

        print(
            "[clean_and_handle_missing] "
            "Dropped rows with all pollutants missing: "
            f"{before_rows} → {after_rows}"
        )

    # Convert numeric-looking object columns

    for col in df.columns:

        if df[col].dtype == "object":

            converted = pd.to_numeric(
                df[col],
                errors="coerce"
            )

            # Convert only if most non-null values
            # successfully became numeric
            original_non_null = df[col].notna().sum()

            converted_non_null = converted.notna().sum()

            if (
                original_non_null > 0
                and converted_non_null / original_non_null >= 0.8
            ):

                df[col] = converted

   
    # Fill numeric columns with median
   

    num_cols = df.select_dtypes(
        include=[np.number]
    ).columns

    for col in num_cols:

        if (
            df[col].isna().any()
            and df[col].notna().any()
        ):

            median_value = df[col].median()

            df[col] = df[col].fillna(
                median_value
            )

    # Fill categorical columns with mode


    cat_cols = df.select_dtypes(
        include=["object", "category"]
    ).columns

    for col in cat_cols:

        if (
            df[col].isna().any()
            and df[col].notna().any()
        ):

            mode_values = df[col].mode()

            if not mode_values.empty:

                df[col] = df[col].fillna(
                    mode_values.iloc[0]
                )

    print(
        "[clean_and_handle_missing] "
        "Filled missing values: "
        "numeric → median, "
        "categorical → mode"
    )

    return df

# NORMALISE AND HARMONISE UNITS

def normalise_and_harmonise_units(
    df: pd.DataFrame
) -> pd.DataFrame:

    unit_cols = [
        c
        for c in df.columns
        if "unit" in c.lower()
    ]

    if unit_cols:

        print(
            "[normalise_and_harmonise_units] "
            f"Unit columns found: {unit_cols}"
        )

    conversions = 0

    for ucol in unit_cols:

        
        # Find associated value column

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
                and base in c.replace(
                    "_",
                    ""
                ).lower()
                and c != ucol
            )
        ]

        if not val_candidates:
            continue

        val_col = val_candidates[0]

        
        # Normalize unit text
    

        df[ucol] = (
            df[ucol]
            .astype("string")
            .str.strip()
            .str.lower()
        )


        # mg/m3 → µg/m³
        #
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

            df.loc[
                mask_mg,
                val_col
            ] = (
                df.loc[
                    mask_mg,
                    val_col
                ] * 1000
            )

            df.loc[
                mask_mg,
                ucol
            ] = "µg/m³"

            conversions += int(
                mask_mg.sum()
            )


        # ug/m3 → µg/m³

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

    print(
        "[normalise_and_harmonise_units] "
        "Converted mg/m³ → µg/m³ for "
        f"{conversions} rows"
    )

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

    # Save cleaned dataset as CSV
    df.to_csv(
        path,
        index=False
    )

    print(
        "\n✅ Saved cleaned CSV file to:"
    )

    print(path)

# MAIN PIPELINE

def main():

    print("\n")
    print("=" * 70)
    print("        AIR QUALITY CSV CLEANING PIPELINE")
    print("=" * 70)

    # 1. Load CSV


    df = load_data(
        RAW_PATH
    )

    # 2. Replace special missing values
 

    df = replace_special_missing(
        df
    )

    log_step(
        "2) After replacing blanks / ## / ### with NaN",
        df
    )
    # 3. Drop empty rows and columns

    df = drop_empty_columns(
        df
    )

    df = drop_empty_rows(
        df
    )

    log_step(
        "3) After dropping fully empty rows/columns",
        df
    )

    # 4. Handle missing values

    df = clean_and_handle_missing(
        df
    )

    log_step(
        "4) After handling missing values",
        df
    )

    
    # 5. Normalize units


    df = normalise_and_harmonise_units(
        df
    )

    log_step(
        "5) After unit normalisation / harmonisation",
        df
    )

    # 6. Display columns

    print("\nColumns:")
    print(list(df.columns))

   
    # 7. Display sample data

    print("\nSample rows (top 5):")
    print(
        df.head(5).to_string(
            index=False
        )
    )
    # 8. Save cleaned CSV

    save_clean_data(
        df,
        OUT_PATH
    )

    print("\n")
    print("=" * 70)
    print("              CLEANING COMPLETED")
    print("=" * 70)
    print(
        f"Final dataset: "
        f"{df.shape[0]} rows × "
        f"{df.shape[1]} columns"
    )
    print("=" * 70)

# RUN

if __name__ == "__main__":
    main()
