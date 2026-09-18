import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# PATHS

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

# LOAD DATA

def load_global_dataset():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"CSV file not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print(
        f"✓ Loaded dataset: {DATA_PATH}"
    )

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    return df

# CREATE FEATURES

def make_features_for_regression(df):

    df = df.copy()

    # Check target

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Target column not found: {TARGET_COL}\n"
            f"Available columns:\n{list(df.columns)}"
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
        f"✓ Datetime column: {datetime_col}"
    )

    # --------------------------------------------------------
    # Convert target to numeric
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
    # Select numeric predictors
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

    if len(feature_cols) == 0:
        raise ValueError(
            "No usable regression features found."
        )

    # Convert features to numeric
    for col in feature_cols:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Remove rows with missing target/features
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
        f"✓ Regression features created: "
        f"{len(feature_cols)}"
    )

    print(
        "Features:"
    )

    for col in feature_cols:
        print(
            f"  - {col}"
        )

    print(
        f"✓ Final regression dataset: "
        f"{df.shape}"
    )

    return df, datetime_col, feature_cols


# ============================================================
# TIME-BASED TRAIN / VALIDATION / TEST SPLIT
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
        f"Training:   {len(X_train)} samples"
    )

    print(
        f"Validation: {len(X_val)} samples"
    )

    print(
        f"Test:       {len(X_test)} samples"
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

    return mae, rmse, r2


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print(
        "       BASELINE REGRESSION MODELS"
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
        "2. CREATING FEATURES"
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
    # 4. CITY MEAN BASELINE
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "4. CITY-MEAN BASELINE"
    )
    print("=" * 70)

    city_idx = feature_cols.index(
        "city_code"
    )

    city_train = X_train[
        :, city_idx
    ]

    city_test = X_test[
        :, city_idx
    ]

    city_means = {}

    for city in np.unique(
        city_train
    ):

        city_values = y_train[
            city_train == city
        ]

        city_means[city] = (
            np.mean(
                city_values
            )
        )

    overall_mean = np.mean(
        y_train
    )

    y_pred_mean = np.array(
        [
            city_means.get(
                city,
                overall_mean
            )
            for city in city_test
        ]
    )

    (
        mae_mean,
        rmse_mean,
        r2_mean
    ) = mae_rmse_r2(
        y_test,
        y_pred_mean
    )

    print(
        f"MAE :  {mae_mean:.4f}"
    )

    print(
        f"RMSE:  {rmse_mean:.4f}"
    )

    print(
        f"R²  :  {r2_mean:.4f}"
    )

    # ========================================================
    # 5. PERSISTENCE BASELINE
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "5. PERSISTENCE BASELINE"
    )
    print("=" * 70)

    lag_idx = feature_cols.index(
        "PM2.5_lag_1h"
    )

    y_pred_persist = X_test[
        :, lag_idx
    ]

    (
        mae_pers,
        rmse_pers,
        r2_pers
    ) = mae_rmse_r2(
        y_test,
        y_pred_persist
    )

    print(
        f"MAE :  {mae_pers:.4f}"
    )

    print(
        f"RMSE:  {rmse_pers:.4f}"
    )

    print(
        f"R²  :  {r2_pers:.4f}"
    )

    # ========================================================
    # 6. LINEAR REGRESSION
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "6. LINEAR REGRESSION"
    )
    print("=" * 70)

    model = LinearRegression()

    print(
        "Training model..."
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "✓ Model training complete."
    )

    y_pred_lin = model.predict(
        X_test
    )

    (
        mae_lin,
        rmse_lin,
        r2_lin
    ) = mae_rmse_r2(
        y_test,
        y_pred_lin
    )

    print(
        f"MAE :  {mae_lin:.4f}"
    )

    print(
        f"RMSE:  {rmse_lin:.4f}"
    )

    print(
        f"R²  :  {r2_lin:.4f}"
    )

    # ========================================================
    # 7. COMPARISON TABLE
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "7. MODEL PERFORMANCE COMPARISON"
    )
    print("=" * 70)

    print(
        f"\n{'Model':<22}"
        f"{'MAE':>12}"
        f"{'RMSE':>12}"
        f"{'R²':>12}"
    )

    print(
        "-" * 70
    )

    print(
        f"{'City Mean':<22}"
        f"{mae_mean:>12.4f}"
        f"{rmse_mean:>12.4f}"
        f"{r2_mean:>12.4f}"
    )

    print(
        f"{'Persistence':<22}"
        f"{mae_pers:>12.4f}"
        f"{rmse_pers:>12.4f}"
        f"{r2_pers:>12.4f}"
    )

    print(
        f"{'Linear Regression':<22}"
        f"{mae_lin:>12.4f}"
        f"{rmse_lin:>12.4f}"
        f"{r2_lin:>12.4f}"
    )

    # ========================================================
    # 8. GRAPH - MODEL PERFORMANCE
    # ========================================================

    print("\n")
    print(
        "8. Creating model performance graph..."
    )

    labels = [
        "City Mean",
        "Persistence",
        "Linear Regression"
    ]

    maes = [
        mae_mean,
        mae_pers,
        mae_lin
    ]

    rmses = [
        rmse_mean,
        rmse_pers,
        rmse_lin
    ]

    x = np.arange(
        len(labels)
    )

    width = 0.35

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        x - width / 2,
        maes,
        width,
        label="MAE",
        color="steelblue"
    )

    plt.bar(
        x + width / 2,
        rmses,
        width,
        label="RMSE",
        color="orange"
    )

    plt.xticks(
        x,
        labels,
        rotation=15
    )

    plt.ylabel(
        "Error (µg/m³)"
    )

    plt.xlabel(
        "Model"
    )

    plt.title(
        "Baseline Model Performance on Test Set"
    )

    plt.legend()

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.5
    )

    plt.tight_layout()

    output1 = (
        FIG_DIR /
        "baseline_regression_performance.png"
    )

    plt.savefig(
        output1,
        dpi=300,
        bbox_inches="tight"
    )

    print(
        f"✓ Saved: {output1}"
    )

    # Display graph
    plt.show()

    plt.close()

    # ========================================================
    # 9. GRAPH - OBSERVED VS PREDICTED
    # ========================================================

    print("\n")
    print(
        "9. Creating observed vs predicted graph..."
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
        y_pred_lin[:n_show],
        label="Linear Regression",
        color="red",
        linewidth=1.5
    )

    plt.xlabel(
        "Time Step"
    )

    plt.ylabel(
        TARGET_COL
    )

    plt.title(
        "Observed vs Linear Regression Prediction"
    )

    plt.legend()

    plt.grid(
        True,
        linestyle="--",
        alpha=0.5
    )

    plt.tight_layout()

    output2 = (
        FIG_DIR /
        "baseline_linear_timeseries.png"
    )

    plt.savefig(
        output2,
        dpi=300,
        bbox_inches="tight"
    )

    print(
        f"✓ Saved: {output2}"
    )

    # Display graph
    plt.show()

    plt.close()

    # ========================================================
    # FINISHED
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "BASELINE REGRESSION ANALYSIS COMPLETE"
    )
    print("=" * 70)

    print(
        "\nFigures:"
    )

    print(
        f"✓ {output1}"
    )

    print(
        f"✓ {output2}"
    )

    print(
        "\n"
    )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()
