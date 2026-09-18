import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# MAC PROJECT PATHS
# ============================================================

BASE_DIR = Path(
    "/Users/aalok_x92/Downloads/Development code"
)

# Processed/tableau output directory
OUT_DIR = BASE_DIR / "tableau"

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# INPUT DATA
# ============================================================

GLOBAL_DATA = (
    BASE_DIR /
    "global_air_quality_dataset_10000_clean.csv"
)


# ============================================================
# IMPORTANT
# ============================================================
#
# We are NOT using:
#
# from src.models.ml_utils import ...
#
# because your Mac project currently does not have
# a Python package called "src".
#
# Instead, the required feature engineering is included
# directly in this script.
#
# ============================================================


TARGET_COL = "PM2.5 (µg/m³)"


# ============================================================
# METRIC FUNCTION
# ============================================================

def mae_rmse_r2(y_true, y_pred):

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    mse = mean_squared_error(
        y_true,
        y_pred
    )

    rmse = mse ** 0.5

    r2 = r2_score(
        y_true,
        y_pred
    )

    return mae, rmse, r2


# ============================================================
# LOAD DATA
# ============================================================

def load_global_dataset():

    if not GLOBAL_DATA.exists():

        raise FileNotFoundError(
            f"\nCSV file not found:\n{GLOBAL_DATA}"
        )

    print("\nLoading dataset...")

    df = pd.read_csv(
        GLOBAL_DATA
    )

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def make_features_for_regression(df):

    df = df.copy()

    # --------------------------------------------------------
    # Find datetime column
    # --------------------------------------------------------

    datetime_col = None

    for col in df.columns:

        name = col.lower()

        if (
            "date" in name
            or "time" in name
            or "datetime" in name
            or "timestamp" in name
        ):

            datetime_col = col

            break

    if datetime_col is None:

        raise ValueError(
            "No date/time column found in dataset."
        )

    print(
        f"Datetime column: {datetime_col}"
    )

    # --------------------------------------------------------
    # Convert datetime
    # --------------------------------------------------------

    df[datetime_col] = pd.to_datetime(
        df[datetime_col],
        errors="coerce"
    )

    df = df.dropna(
        subset=[datetime_col]
    )

    # --------------------------------------------------------
    # Target column
    # --------------------------------------------------------

    if TARGET_COL not in df.columns:

        # Try to automatically find PM2.5
        candidates = [
            c for c in df.columns
            if "pm2.5" in c.lower()
            or "pm25" in c.lower()
            or "pm2_5" in c.lower()
        ]

        if not candidates:

            raise ValueError(
                "PM2.5 column could not be found."
            )

        target_col = candidates[0]

    else:

        target_col = TARGET_COL

    print(
        f"Target: {target_col}"
    )

    # --------------------------------------------------------
    # Sort by date
    # --------------------------------------------------------

    df = df.sort_values(
        datetime_col
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Create time features
    # --------------------------------------------------------

    df["year"] = (
        df[datetime_col].dt.year
    )

    df["month"] = (
        df[datetime_col].dt.month
    )

    df["day"] = (
        df[datetime_col].dt.day
    )

    df["day_of_week"] = (
        df[datetime_col].dt.dayofweek
    )

    df["hour"] = (
        df[datetime_col].dt.hour
    )

    # --------------------------------------------------------
    # City encoding
    # --------------------------------------------------------

    if "City" in df.columns:

        df["city_code"] = (
            df["City"]
            .astype("category")
            .cat.codes
        )

    elif "city" in df.columns:

        df["city_code"] = (
            df["city"]
            .astype("category")
            .cat.codes
        )

    else:

        df["city_code"] = 0

    # --------------------------------------------------------
    # Country encoding
    # --------------------------------------------------------

    if "Country" in df.columns:

        df["country_code"] = (
            df["Country"]
            .astype("category")
            .cat.codes
        )

    elif "country" in df.columns:

        df["country_code"] = (
            df["country"]
            .astype("category")
            .cat.codes
        )

    else:

        df["country_code"] = 0

    # --------------------------------------------------------
    # Lag feature
    # --------------------------------------------------------

    df[
        f"{target_col}_lag_1h"
    ] = df[target_col].shift(1)

    # --------------------------------------------------------
    # Remove rows where target or lag is missing
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            target_col,
            f"{target_col}_lag_1h"
        ]
    )

    # --------------------------------------------------------
    # Select numeric features
    # --------------------------------------------------------

    excluded = [
        target_col,
        datetime_col
    ]

    numeric_cols = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    feature_cols = [
        c for c in numeric_cols
        if c not in excluded
    ]

    # --------------------------------------------------------
    # Fill missing numeric values
    # --------------------------------------------------------

    for col in feature_cols:

        if df[col].isna().any():

            df[col] = df[col].fillna(
                df[col].median()
            )

    print(
        "\nFeatures used:"
    )

    for col in feature_cols:

        print(
            f"  - {col}"
        )

    return (
        df,
        datetime_col,
        target_col,
        feature_cols
    )


# ============================================================
# TIME-BASED TRAIN / VALIDATION / TEST SPLIT
# ============================================================

def time_based_split(
    df,
    datetime_col,
    target_col,
    feature_cols
):

    df = df.sort_values(
        datetime_col
    ).reset_index(
        drop=True
    )

    n = len(df)

    train_end = int(
        0.70 * n
    )

    val_end = int(
        0.85 * n
    )

    df_train = df.iloc[
        :train_end
    ].copy()

    df_val = df.iloc[
        train_end:val_end
    ].copy()

    df_test = df.iloc[
        val_end:
    ].copy()

    X_train = df_train[
        feature_cols
    ].values

    y_train = df_train[
        target_col
    ].values

    X_val = df_val[
        feature_cols
    ].values

    y_val = df_val[
        target_col
    ].values

    X_test = df_test[
        feature_cols
    ].values

    y_test = df_test[
        target_col
    ].values

    print("\nData split:")

    print(
        f"Training: {len(df_train)} rows"
    )

    print(
        f"Validation: {len(df_val)} rows"
    )

    print(
        f"Testing: {len(df_test)} rows"
    )

    return (
        df,
        df_train,
        df_val,
        df_test,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print(
        "       AIR QUALITY MODEL TABLEAU DATA"
    )
    print("=" * 70)

    print(
        f"\nInput file:\n{GLOBAL_DATA}"
    )

    print(
        f"\nOutput directory:\n{OUT_DIR}"
    )

    # ========================================================
    # LOAD DATA
    # ========================================================

    df = load_global_dataset()

    # ========================================================
    # FEATURE ENGINEERING
    # ========================================================

    (
        df_feat,
        datetime_col,
        target_col,
        feature_cols
    ) = make_features_for_regression(
        df
    )

    # ========================================================
    # SPLIT DATA
    # ========================================================

    (
        df_feat,
        df_train,
        df_val,
        df_test,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    ) = time_based_split(
        df_feat,
        datetime_col,
        target_col,
        feature_cols
    )

    # ========================================================
    # MODEL METRICS
    # ========================================================

    metrics_rows = []

    # ========================================================
    # 1. CITY MEAN BASELINE
    # ========================================================

    print(
        "\nTraining City-Mean baseline..."
    )

    if "city_code" in feature_cols:

        city_idx = feature_cols.index(
            "city_code"
        )

        city_train = X_train[
            :, city_idx
        ]

        city_test = X_test[
            :, city_idx
        ]

        overall_mean = np.mean(
            y_train
        )

        city_mean = {}

        for city_code in np.unique(
            city_train
        ):

            city_values = y_train[
                city_train == city_code
            ]

            city_mean[city_code] = (
                np.mean(city_values)
            )

        y_pred_citymean = np.array(
            [
                city_mean.get(
                    c,
                    overall_mean
                )
                for c in city_test
            ]
        )

    else:

        y_pred_citymean = np.full(
            len(y_test),
            np.mean(y_train)
        )

    mae, rmse, r2 = mae_rmse_r2(
        y_test,
        y_pred_citymean
    )

    metrics_rows.append(
        (
            "City-Mean Baseline",
            mae,
            rmse,
            r2
        )
    )

    # ========================================================
    # 2. PERSISTENCE BASELINE
    # ========================================================

    print(
        "Training Persistence baseline..."
    )

    lag_name = (
        f"{target_col}_lag_1h"
    )

    if lag_name in feature_cols:

        lag_idx = feature_cols.index(
            lag_name
        )

        y_pred_persist = X_test[
            :, lag_idx
        ]

        mae, rmse, r2 = mae_rmse_r2(
            y_test,
            y_pred_persist
        )

        metrics_rows.append(
            (
                "Persistence Baseline",
                mae,
                rmse,
                r2
            )
        )

    # ========================================================
    # 3. LINEAR REGRESSION
    # ========================================================

    print(
        "Training Linear Regression..."
    )

    linear_model = LinearRegression()

    linear_model.fit(
        X_train,
        y_train
    )

    y_pred_linear = (
        linear_model.predict(
            X_test
        )
    )

    mae, rmse, r2 = mae_rmse_r2(
        y_test,
        y_pred_linear
    )

    metrics_rows.append(
        (
            "Linear Regression",
            mae,
            rmse,
            r2
        )
    )

    # ========================================================
    # 4. RANDOM FOREST
    # ========================================================

    print(
        "Training Random Forest..."
    )

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=42
    )

    rf.fit(
        X_train,
        y_train
    )

    y_pred_rf = rf.predict(
        X_test
    )

    mae, rmse, r2 = mae_rmse_r2(
        y_test,
        y_pred_rf
    )

    metrics_rows.append(
        (
            "Random Forest",
            mae,
            rmse,
            r2
        )
    )

    # ========================================================
    # 5. GRADIENT BOOSTING
    # ========================================================

    print(
        "Training Gradient Boosting..."
    )

    gbr = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )

    gbr.fit(
        X_train,
        y_train
    )

    y_pred_gbr = gbr.predict(
        X_test
    )

    mae, rmse, r2 = mae_rmse_r2(
        y_test,
        y_pred_gbr
    )

    metrics_rows.append(
        (
            "Gradient Boosting",
            mae,
            rmse,
            r2
        )
    )

    # ========================================================
    # CREATE MODEL METRICS CSV
    # ========================================================

    metrics_df = pd.DataFrame(
        metrics_rows,
        columns=[
            "Model",
            "MAE",
            "RMSE",
            "R2"
        ]
    )

    metrics_path = (
        OUT_DIR /
        "model_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False
    )

    print(
        "\n✓ Model metrics saved:"
    )

    print(
        metrics_path
    )

    # ========================================================
    # CREATE RF PREDICTIONS CSV
    # ========================================================

    df_test_output = df_test.copy()

    df_test_output[
        "actual_pm25"
    ] = y_test

    df_test_output[
        "predicted_pm25_rf"
    ] = y_pred_rf

    df_test_output[
        "predicted_pm25_linear"
    ] = y_pred_linear

    df_test_output[
        "predicted_pm25_gbr"
    ] = y_pred_gbr

    df_test_output[
        "abs_error_rf"
    ] = (
        df_test_output["actual_pm25"]
        -
        df_test_output["predicted_pm25_rf"]
    ).abs()

    # ========================================================
    # KEEP USEFUL TABLEAU COLUMNS
    # ========================================================

    keep_cols = [
        datetime_col
    ]

    possible_location_cols = [
        "City",
        "city",
        "Country",
        "country",
        "Latitude",
        "Longitude",
        "lat",
        "lon",
        "city_code",
        "country_code"
    ]

    for col in possible_location_cols:

        if col in df_test_output.columns:

            if col not in keep_cols:

                keep_cols.append(
                    col
                )

    prediction_cols = [
        "actual_pm25",
        "predicted_pm25_rf",
        "predicted_pm25_linear",
        "predicted_pm25_gbr",
        "abs_error_rf"
    ]

    keep_cols.extend(
        prediction_cols
    )

    prediction_df = (
        df_test_output[
            keep_cols
        ].copy()
    )

    pred_path = (
        OUT_DIR /
        "rf_test_predictions.csv"
    )

    prediction_df.to_csv(
        pred_path,
        index=False
    )

    print(
        "\n✓ Prediction data saved:"
    )

    print(
        pred_path
    )

    # ========================================================
    # RANDOM FOREST FEATURE IMPORTANCE
    # ========================================================

    importance_df = pd.DataFrame(
        {
            "Feature": feature_cols,
            "Importance": rf.feature_importances_
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            "Importance",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    importance_path = (
        OUT_DIR /
        "rf_feature_importance.csv"
    )

    importance_df.to_csv(
        importance_path,
        index=False
    )

    print(
        "\n✓ Feature importance saved:"
    )

    print(
        importance_path
    )

    # ========================================================
    # PRINT MODEL RESULTS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("MODEL PERFORMANCE")
    print("=" * 70)

    print(
        metrics_df.to_string(
            index=False
        )
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n")
    print("=" * 70)
    print("TABLEAU DATA PREPARATION COMPLETE")
    print("=" * 70)

    print(
        "\nFiles created:"
    )

    print(
        f"1. {metrics_path}"
    )

    print(
        f"2. {pred_path}"
    )

    print(
        f"3. {importance_path}"
    )

    print(
        "\nYou can now import these CSV files into Tableau."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
