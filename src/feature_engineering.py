# Feature Engineering for Credit Scoring Model

This file contains functions for feature extraction and transformation. It creates new features from the existing data to improve model performance.

import pandas as pd

def create_feature_age(df):
    """Create an age feature from the birth year."""
    if 'birth_year' in df.columns:
        current_year = pd.Timestamp.now().year
        df['age'] = current_year - df['birth_year']
    return df

def create_feature_income_to_debt_ratio(df):
    """Create an income to debt ratio feature."""
    if 'annual_income' in df.columns and 'debt' in df.columns:
        df['income_to_debt_ratio'] = df['annual_income'] / (df['debt'] + 1e-6)  # Adding a small value to avoid division by zero
    return df

def create_feature_credit_history_length(df):
    """Create a feature for the length of credit history."""
    if 'credit_start_year' in df.columns:
        current_year = pd.Timestamp.now().year
        df['credit_history_length'] = current_year - df['credit_start_year']
    return df

def feature_engineering(df):
    """Apply all feature engineering functions."""
    df = create_feature_age(df)
    df = create_feature_income_to_debt_ratio(df)
    df = create_feature_credit_history_length(df)
    return df