import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# PATHS
# ============================================================

DATA_PATH = Path(
    "/Users/aalok_x92/Downloads/Development code/"
    "global_air_quality_dataset_10000_clean.csv"
)

FIG_DIR = Path(
    "/Users/aalok_x92/Downloads/Development code/figures"
)

FIG_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TARGET_COL = "PM2.5 (µg/m³)"


# ============================================================
# LOAD DATA
# ============================================================

def load_global_dataset():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"CSV file not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH
    )

    print(
        f"✓ Loaded dataset:"
    )

    print(
        DATA_PATH
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
    # Check target
    # --------------------------------------------------------

    if TARGET_COL not in df.columns:

        raise ValueError(
            f"Target column not found: "
            f"{TARGET_COL}\n\n"
            f"Available columns:\n"
            f"{list(df.columns)}"
        )

    # --------------------------------------------------------
    # Find datetime column
    # --------------------------------------------------------

    datetime_col = None

    for col in df.columns:

        if any(
            word in col.lower()
            for word in [
                "date",
                "time",
                "datetime",
                "timestamp"
            ]
        ):

            datetime_col = col
            break

    if datetime_col is None:

        raise ValueError(
            "No datetime column found."
        )

    # Convert datetime
    df[datetime_col] = pd.to_datetime(
        df[datetime_col],
        errors="coerce"
    )

    # Remove invalid dates
    df = df.dropna(
        subset=[datetime_col]
    )

    # Sort chronologically
    df = df.sort_values(
        datetime_col
    ).reset_index(
        drop=True
    )

    print(
        f"✓ Datetime column: "
        f"{datetime_col}"
    )

    # --------------------------------------------------------
    # Target numeric conversion
    # --------------------------------------------------------

    df[TARGET_COL] = pd.to_numeric(
        df[TARGET_COL],
        errors="coerce"
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

    else:

        df["city_code"] = 0

    # --------------------------------------------------------
    # Time features
    # --------------------------------------------------------

    df["hour"] = (
        df[datetime_col].dt.hour
    )

    df["day_of_week"] = (
        df[datetime_col].dt.dayofweek
    )

    df["month"] = (
        df[datetime_col].dt.month
    )

    # --------------------------------------------------------
    # Lag feature
    # --------------------------------------------------------

    df["PM2.5_lag_1h"] = (
        df[TARGET_COL].shift(1)
    )

    # --------------------------------------------------------
    # Rolling mean
    # --------------------------------------------------------

    df["PM2.5_rolling_3"] = (
        df[TARGET_COL]
        .rolling(
            window=3,
            min_periods=1
        )
        .mean()
        .shift(1)
    )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    possible_features = [

        "city_code",

        "hour",

        "day_of_week",

        "month",

        "PM2.5_lag_1h",

        "PM2.5_rolling_3",

        "PM10 (µg/m³)",

        "NO2 (ppb)",

        "SO2 (ppb)",

        "CO (ppm)",

        "O3 (ppb)",

        "Temperature (°C)",

        "Humidity (%)",

        "Wind Speed (m/s)"
    ]

    feature_cols = [
        col
        for col in possible_features
        if col in df.columns
    ]

    if not feature_cols:

        raise ValueError(
            "No regression features found."
        )

    # Convert features to numeric
    for col in feature_cols:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Remove missing values
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            TARGET_COL
        ] + feature_cols
    )

    df = df.reset_index(
        drop=True
    )

    print(
        f"✓ Number of features: "
        f"{len(feature_cols)}"
    )

    print(
        "\nFeatures:"
    )

    for col in feature_cols:

        print(
            f"  - {col}"
        )

    print(
        f"\n✓ Final dataset shape: "
        f"{df.shape}"
    )

    return (
        df,
        datetime_col,
        feature_cols
    )


# ============================================================
# TIME-BASED SPLIT
# ============================================================

def time_based_train_val_test_split(
    df,
    datetime_col,
    feature_cols
):

    df = df.sort_values(
        datetime_col
    ).reset_index(
        drop=True
    )

    X = df[
        feature_cols
    ].values

    y = df[
        TARGET_COL
    ].values

    n = len(df)

    train_end = int(
        n * 0.70
    )

    val_end = int(
        n * 0.85
    )

    X_train = X[
        :train_end
    ]

    y_train = y[
        :train_end
    ]

    X_val = X[
        train_end:val_end
    ]

    y_val = y[
        train_end:val_end
    ]

    X_test = X[
        val_end:
    ]

    y_test = y[
        val_end:
    ]

    print(
        "\nTime-based split:"
    )

    print(
        f"Training:   {len(X_train)}"
    )

    print(
        f"Validation: {len(X_val)}"
    )

    print(
        f"Test:       {len(X_test)}"
    )

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    )


# ============================================================
# METRICS
# ============================================================

def mae_rmse_r2(
    y_true,
    y_pred
):

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    mse = mean_squared_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mse
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    return (
        mae,
        rmse,
        r2
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print(
        "       GRADIENT BOOSTING REGRESSION"
    )
    print("=" * 70)

    print(
        "\nInput CSV:"
    )

    print(
        DATA_PATH
    )

    print(
        "\nOutput figures:"
    )

    print(
        FIG_DIR
    )

    # ========================================================
    # 1. LOAD
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "1. LOADING DATA"
    )
    print("=" * 70)

    df = load_global_dataset()

    # ========================================================
    # 2. FEATURES
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "2. FEATURE ENGINEERING"
    )
    print("=" * 70)

    (
        df_feat,
        dt_col,
        feature_cols
    ) = make_features_for_regression(
        df
    )

    # ========================================================
    # 3. SPLIT
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "3. TRAIN / VALIDATION / TEST SPLIT"
    )
    print("=" * 70)

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    ) = time_based_train_val_test_split(
        df_feat,
        dt_col,
        feature_cols
    )

    # ========================================================
    # 4. CREATE GRADIENT BOOSTING MODEL
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "4. GRADIENT BOOSTING MODEL"
    )
    print("=" * 70)

    gbr = GradientBoostingRegressor(

        n_estimators=300,

        learning_rate=0.05,

        max_depth=3,

        random_state=42
    )

    print(
        "\nModel parameters:"
    )

    print(
        "  n_estimators = 300"
    )

    print(
        "  learning_rate = 0.05"
    )

    print(
        "  max_depth = 3"
    )

    # ========================================================
    # 5. MODEL TRAINING
    # ========================================================

    print("\n")
    print(
        "Training Gradient Boosting..."
    )

    gbr.fit(
        X_train,
        y_train
    )

    print(
        "✓ Model training complete."
    )

    # ========================================================
    # 6. VALIDATION PREDICTION
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "5. VALIDATION PERFORMANCE"
    )
    print("=" * 70)

    y_val_pred = gbr.predict(
        X_val
    )

    (
        mae_val,
        rmse_val,
        r2_val
    ) = mae_rmse_r2(
        y_val,
        y_val_pred
    )

    print(
        f"MAE :  {mae_val:.4f}"
    )

    print(
        f"RMSE:  {rmse_val:.4f}"
    )

    print(
        f"R²  :  {r2_val:.4f}"
    )

    # ========================================================
    # 7. TEST PREDICTION
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "6. TEST PERFORMANCE"
    )
    print("=" * 70)

    y_test_pred = gbr.predict(
        X_test
    )

    (
        mae_test,
        rmse_test,
        r2_test
    ) = mae_rmse_r2(
        y_test,
        y_test_pred
    )

    print(
        f"MAE :  {mae_test:.4f}"
    )

    print(
        f"RMSE:  {rmse_test:.4f}"
    )

    print(
        f"R²  :  {r2_test:.4f}"
    )

    # ========================================================
    # 8. OBSERVED VS PREDICTED SCATTER
    # ========================================================

    print("\n")
    print(
        "7. Creating observed vs predicted graph..."
    )

    plt.figure(
        figsize=(7, 7)
    )

    plt.scatter(
        y_test,
        y_test_pred,
        alpha=0.4,
        color="steelblue",
        edgecolors="none"
    )

    # 1:1 reference line
    min_value = min(
        y_test.min(),
        y_test_pred.min()
    )

    max_value = max(
        y_test.max(),
        y_test_pred.max()
    )

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--",
        color="red",
        linewidth=2,
        label="Perfect prediction"
    )

    plt.xlabel(
        "Observed PM2.5 (µg/m³)"
    )

    plt.ylabel(
        "Predicted PM2.5 (µg/m³)"
    )

    plt.title(
        "Gradient Boosting: Observed vs Predicted"
    )

    plt.legend()

    plt.grid(
        True,
        linestyle="--",
        alpha=0.5
    )

    plt.tight_layout()

    output_path = (
        FIG_DIR /
        "gbr_scatter_test.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(
        f"✓ Saved: {output_path}"
    )

    # IMPORTANT:
    # Display graph on screen
    plt.show()

    plt.close()

    # ========================================================
    # 9. ACTUAL VS PREDICTED TIME SERIES
    # ========================================================

    print("\n")
    print(
        "8. Creating prediction time-series graph..."
    )

    n_show = min(
        200,
        len(y_test)
    )

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        range(n_show),
        y_test[:n_show],
        label="Observed",
        color="black",
        linewidth=1.5
    )

    plt.plot(
        range(n_show),
        y_test_pred[:n_show],
        label="Gradient Boosting",
        color="blue",
        linewidth=1.5
    )

    plt.xlabel(
        "Time Step"
    )

    plt.ylabel(
        "PM2.5 (µg/m³)"
    )

    plt.title(
        "Gradient Boosting: Observed vs Predicted PM2.5"
    )

    plt.legend()

    plt.grid(
        True,
        linestyle="--",
        alpha=0.5
    )

    plt.tight_layout()

    output_path_2 = (
        FIG_DIR /
        "gbr_prediction_timeseries.png"
    )

    plt.savefig(
        output_path_2,
        dpi=300,
        bbox_inches="tight"
    )

    print(
        f"✓ Saved: {output_path_2}"
    )

    # Display graph
    plt.show()

    plt.close()

    # ========================================================
    # 10. FEATURE IMPORTANCE
    # ========================================================

    print("\n")
    print(
        "9. Creating feature importance graph..."
    )

    importances = (
        gbr.feature_importances_
    )

    indices = np.argsort(
        importances
    )[::-1]

    sorted_features = [
        feature_cols[i]
        for i in indices
    ]

    sorted_importances = (
        importances[indices]
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.barh(
        sorted_features[::-1],
        sorted_importances[::-1],
        color="seagreen"
    )

    plt.xlabel(
        "Feature Importance"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "Gradient Boosting Feature Importance"
    )

    plt.grid(
        axis="x",
        linestyle="--",
        alpha=0.5
    )

    plt.tight_layout()

    output_path_3 = (
        FIG_DIR /
        "gbr_feature_importance.png"
    )

    plt.savefig(
        output_path_3,
        dpi=300,
        bbox_inches="tight"
    )

    print(
        f"✓ Saved: {output_path_3}"
    )

    # Display graph
    plt.show()

    plt.close()

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "GRADIENT BOOSTING ANALYSIS COMPLETE"
    )
    print("=" * 70)

    print(
        "\nValidation:"
    )

    print(
        f"  MAE  = {mae_val:.4f}"
    )

    print(
        f"  RMSE = {rmse_val:.4f}"
    )

    print(
        f"  R²   = {r2_val:.4f}"
    )

    print(
        "\nTest:"
    )

    print(
        f"  MAE  = {mae_test:.4f}"
    )

    print(
        f"  RMSE = {rmse_test:.4f}"
    )

    print(
        f"  R²   = {r2_test:.4f}"
    )

    print(
        "\nGenerated figures:"
    )

    print(
        f"✓ {output_path}"
    )

    print(
        f"✓ {output_path_2}"
    )

    print(
        f"✓ {output_path_3}"
    )

    print("\n")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
