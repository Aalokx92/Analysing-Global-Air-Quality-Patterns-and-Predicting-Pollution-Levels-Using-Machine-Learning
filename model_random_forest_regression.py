import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PATHS - MAC
# ============================================================

DATA_PATH = Path(
    "/Users/aalok_x92/Downloads/Development code/"
    "global_air_quality_dataset_10000_clean.csv"
)

FIG_DIR = Path(
    "/Users/aalok_x92/Downloads/Development code/figures"
)

FIG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

TARGET_COL = "PM2.5 (µg/m³)"


# ============================================================
# METRICS
# ============================================================

def mae_rmse_r2(y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)

    mse = mean_squared_error(y_true, y_pred)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_true, y_pred)

    return mae, rmse, r2


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"\nCSV file not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("\n" + "=" * 70)
    print("RANDOM FOREST AIR QUALITY MODEL")
    print("=" * 70)

    print(f"\nDataset: {DATA_PATH}")

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\nColumns:")

    for col in df.columns:
        print(f"  - {col}")

    return df


# ============================================================
# FEATURE PREPARATION
# ============================================================

def prepare_features(df):

    df = df.copy()

    # --------------------------------------------------------
    # Convert Date
    # --------------------------------------------------------

    if "Date" in df.columns:

        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce"
        )

        df = df.dropna(subset=["Date"])

        # Time features
        df["year"] = df["Date"].dt.year
        df["month"] = df["Date"].dt.month
        df["day"] = df["Date"].dt.day
        df["day_of_week"] = df["Date"].dt.dayofweek

    # --------------------------------------------------------
    # Check target
    # --------------------------------------------------------

    if TARGET_COL not in df.columns:

        raise ValueError(
            f"\nTarget column not found:\n{TARGET_COL}"
        )

    # --------------------------------------------------------
    # Convert target to numeric
    # --------------------------------------------------------

    df[TARGET_COL] = pd.to_numeric(
        df[TARGET_COL],
        errors="coerce"
    )

    df = df.dropna(subset=[TARGET_COL])

    # --------------------------------------------------------
    # Encode City
    # --------------------------------------------------------

    if "City" in df.columns:

        df["city_code"] = (
            df["City"]
            .astype("category")
            .cat.codes
        )

    # --------------------------------------------------------
    # Encode Country
    # --------------------------------------------------------

    if "Country" in df.columns:

        df["country_code"] = (
            df["Country"]
            .astype("category")
            .cat.codes
        )

    # --------------------------------------------------------
    # Numeric columns
    # --------------------------------------------------------

    numeric_cols = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    # Remove target
    feature_cols = [
        col
        for col in numeric_cols
        if col != TARGET_COL
    ]

    # --------------------------------------------------------
    # Add lag feature
    # --------------------------------------------------------

    df = df.sort_values("Date")

    df[f"{TARGET_COL}_lag_1"] = (
        df[TARGET_COL].shift(1)
    )

    feature_cols.append(
        f"{TARGET_COL}_lag_1"
    )

    # Remove rows where lag doesn't exist
    df = df.dropna(
        subset=[f"{TARGET_COL}_lag_1"]
    )

    # --------------------------------------------------------
    # Fill remaining numeric missing values
    # --------------------------------------------------------

    for col in feature_cols:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

            df[col] = df[col].fillna(
                df[col].median()
            )

    print("\nFeatures used:")

    for col in feature_cols:
        print(f"  - {col}")

    return df, feature_cols


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

def split_data(df, feature_cols):

    X = df[feature_cols]

    y = df[TARGET_COL]

    n = len(df)

    train_end = int(n * 0.70)

    val_end = int(n * 0.85)

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]

    X_val = X.iloc[train_end:val_end]
    y_val = y.iloc[train_end:val_end]

    X_test = X.iloc[val_end:]
    y_test = y.iloc[val_end:]

    print("\n" + "=" * 70)
    print("DATA SPLIT")
    print("=" * 70)

    print(f"\nTraining rows:   {len(X_train)}")
    print(f"Validation rows: {len(X_val)}")
    print(f"Testing rows:    {len(X_test)}")

    return (
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

    # --------------------------------------------------------
    # 1. Load
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # 2. Prepare features
    # --------------------------------------------------------

    df, feature_cols = prepare_features(df)

    # --------------------------------------------------------
    # 3. Split
    # --------------------------------------------------------

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    ) = split_data(
        df,
        feature_cols
    )

    # --------------------------------------------------------
    # 4. Random Forest
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING RANDOM FOREST")
    print("=" * 70)

    rf = RandomForestRegressor(

        n_estimators=200,

        max_depth=None,

        min_samples_split=2,

        min_samples_leaf=1,

        n_jobs=-1,

        random_state=42
    )

    rf.fit(
        X_train,
        y_train
    )

    print("\n✓ Random Forest training complete")

    # --------------------------------------------------------
    # 5. Validation
    # --------------------------------------------------------

    y_val_pred = rf.predict(X_val)

    mae_val, rmse_val, r2_val = mae_rmse_r2(
        y_val,
        y_val_pred
    )

    print("\nValidation performance:")

    print(f"MAE  = {mae_val:.3f}")
    print(f"RMSE = {rmse_val:.3f}")
    print(f"R²   = {r2_val:.3f}")

    # --------------------------------------------------------
    # 6. Test
    # --------------------------------------------------------

    y_test_pred = rf.predict(X_test)

    mae_test, rmse_test, r2_test = mae_rmse_r2(
        y_test,
        y_test_pred
    )

    print("\nTest performance:")

    print(f"MAE  = {mae_test:.3f}")
    print(f"RMSE = {rmse_test:.3f}")
    print(f"R²   = {r2_test:.3f}")

    # ========================================================
    # GRAPH 1
    # Observed vs Predicted
    # ========================================================

    print("\nCreating Graph 1...")

    plt.figure(figsize=(8, 6))

    plt.scatter(
        y_test,
        y_test_pred,
        alpha=0.5
    )

    # Perfect prediction line
    minimum = min(
        y_test.min(),
        y_test_pred.min()
    )

    maximum = max(
        y_test.max(),
        y_test_pred.max()
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        "r--",
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
        "Random Forest: Observed vs Predicted PM2.5"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    graph1 = FIG_DIR / "rf_observed_vs_predicted.png"

    plt.savefig(
        graph1,
        dpi=300
    )

    print(f"✓ Saved: {graph1}")

    # SHOW GRAPH
    plt.show()

    # ========================================================
    # GRAPH 2
    # Time Series
    # ========================================================

    print("\nCreating Graph 2...")

    n_show = min(
        200,
        len(y_test)
    )

    plt.figure(figsize=(12, 5))

    plt.plot(
        y_test.iloc[:n_show].values,
        label="Observed",
        linewidth=2
    )

    plt.plot(
        y_test_pred[:n_show],
        label="Random Forest",
        linewidth=2
    )

    plt.xlabel(
        "Test time step"
    )

    plt.ylabel(
        "PM2.5 (µg/m³)"
    )

    plt.title(
        "Random Forest: Observed vs Predicted PM2.5"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    graph2 = FIG_DIR / "rf_timeseries_test.png"

    plt.savefig(
        graph2,
        dpi=300
    )

    print(f"✓ Saved: {graph2}")

    # SHOW GRAPH
    plt.show()

    # ========================================================
    # GRAPH 3
    # Feature Importance
    # ========================================================

    print("\nCreating Graph 3...")

    importances = rf.feature_importances_

    indices = np.argsort(
        importances
    )[::-1]

    top_n = min(
        15,
        len(feature_cols)
    )

    indices = indices[:top_n]

    top_features = [
        feature_cols[i]
        for i in indices
    ]

    top_importances = [
        importances[i]
        for i in indices
    ]

    plt.figure(figsize=(10, 6))

    plt.barh(
        top_features[::-1],
        top_importances[::-1]
    )

    plt.xlabel(
        "Feature importance"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "Random Forest Feature Importance"
    )

    plt.grid(
        True,
        axis="x",
        alpha=0.3
    )

    plt.tight_layout()

    graph3 = FIG_DIR / "rf_feature_importance.png"

    plt.savefig(
        graph3,
        dpi=300
    )

    print(f"✓ Saved: {graph3}")

    # SHOW GRAPH
    plt.show()

    # ========================================================
    # FINISHED
    # ========================================================

    print("\n" + "=" * 70)
    print("RANDOM FOREST ANALYSIS COMPLETE")
    print("=" * 70)

    print("\nGraphs saved in:")

    print(FIG_DIR)

    print("\nTest metrics:")

    print(f"MAE  : {mae_test:.3f}")
    print(f"RMSE : {rmse_test:.3f}")
    print(f"R²   : {r2_test:.3f}")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
