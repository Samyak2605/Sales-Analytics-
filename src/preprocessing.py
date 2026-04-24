"""
preprocessing.py
=================
Data ingestion and multi-relational structuring for the Sales Analytics Engine.

This module now processes relational database outputs (CRM dumps):
  1. Loads `sales_pipeline.csv` and `accounts.csv`.
  2. Merges them on the `account` key to flatten the relation.
  3. Filters ongoing deals ("Prospecting", "Engaging") to strictly model
     closed deals ("Won", "Lost") as binary labels for Win Probability scoring.
  4. Derives the numerical logic (days active) from engage and close dates.
  5. Performs stratified Train/Test splits to maintain class distributions natively.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

TEST_SIZE = 0.30
RANDOM_STATE = 42

def load_data(data_dir: str) -> pd.DataFrame:
    """
    Load relational CRM tables and join them into a master feature DataFrame.

    Parameters
    ----------
    data_dir : str
        Directory containing `sales_pipeline.csv` and `accounts.csv`.

    Returns
    -------
    pd.DataFrame
        Merged and filtered DataFrame containing only closed deals.
    """
    pipeline_path = os.path.join(data_dir, "sales_pipeline.csv")
    accounts_path = os.path.join(data_dir, "accounts.csv")

    if not os.path.exists(pipeline_path) or not os.path.exists(accounts_path):
        raise FileNotFoundError("Missing required CSV files in data directory.")

    # Load Tables
    pipeline = pd.read_csv(pipeline_path)
    accounts = pd.read_csv(accounts_path)

    # 1. Left join accounts onto pipeline
    df = pipeline.merge(accounts, on="account", how="left")

    # 2. Filter for strictly Closed Deals
    df = df[df["deal_stage"].isin(["Won", "Lost"])].copy()

    # 3. Create Binary Target
    df["IsWon"] = (df["deal_stage"] == "Won").astype(int)

    # 4. Feature Engineering: Deal Duration
    df["engage_date"] = pd.to_datetime(df["engage_date"])
    df["close_date"] = pd.to_datetime(df["close_date"])
    df["sales_cycle_duration_days"] = (df["close_date"] - df["engage_date"]).dt.days
    
    # 5. Clean up unused / purely informational columns
    cols_to_drop = ["opportunity_id", "engage_date", "close_date", "close_value", "deal_stage", "subsidiary_of"]
    df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

    print(f"✅ Master dataset compiled: {df.shape[0]} closed deals × {df.shape[1]} features")
    return df


def inspect_data(df: pd.DataFrame) -> dict:
    inspection = {
        "shape": df.shape,
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "class_distribution": df["IsWon"].value_counts().to_dict(),
    }
    print("\\n" + "=" * 50)
    print("  DATA INSPECTION SUMMARY")
    print("=" * 50)
    print(f"  Shape: {inspection['shape']}")
    print(f"  Missing Values: {inspection['missing_values']}")
    print(f"  Class Distribution (IsWon): {inspection['class_distribution']}")
    print("=" * 50)
    return inspection


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values robustly for tabular data.
    """
    df = df.copy()
    
    # Numeric columns using median
    for col in ["revenue", "employees", "year_established", "sales_cycle_duration_days"]:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
            
    # Categorical columns using string constant
    cat_cols = ["sales_agent", "product", "account", "sector", "office_location"]
    for col in cat_cols:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna("Unknown")

    return df


def split_data(df: pd.DataFrame, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE):
    """
    Split with Stratification to preserve Win Rate representation.
    """
    X = df.drop(columns=["IsWon"])
    y = df["IsWon"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"\\n✅ Stratified Train/Test split complete.")
    print(f"   Train samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test
