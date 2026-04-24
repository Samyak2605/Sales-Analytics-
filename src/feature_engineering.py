"""
feature_engineering.py
======================
Feature transformation pipeline for the Sales Analytics Engine (Phase 1).

This module replaces the old TF-IDF transformers with dense categorical 
OneHotEncoding and robust StandardScaling structures designed for the 
purely relational tabular dataset.
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

NUMERIC_COLUMNS = ["sales_cycle_duration_days", "revenue", "employees", "year_established"]
CATEGORICAL_COLUMNS = ["sales_agent", "product", "sector", "office_location"]

def build_feature_transformer() -> ColumnTransformer:
    """
    Build the pipeline mapping structured data formats.

    Returns
    -------
    sklearn.compose.ColumnTransformer
    """
    numeric_transformer = StandardScaler()

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_COLUMNS),
            ("categorical", categorical_transformer, CATEGORICAL_COLUMNS),
        ],
        remainder="drop", # Implicitly drops fields like 'account' preventing massive cardinality bloat.
        verbose_feature_names_out=False,
    )

    print("✅ Advanced ML Feature transformer built.")
    print(f"   Numeric Scaling: {NUMERIC_COLUMNS}")
    print(f"   Categorical OHE: {CATEGORICAL_COLUMNS}")

    return column_transformer

def get_feature_names(transformer: ColumnTransformer) -> list:
    return list(transformer.get_feature_names_out())
