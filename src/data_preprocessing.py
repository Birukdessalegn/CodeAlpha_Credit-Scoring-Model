# src/data_preprocessing.py

import pandas as pd
import numpy as np

def load_data(file_path):
    """Load the dataset from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully from '{file_path}'")
        return df
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return None

def preprocess_data(df):
    """Preprocess the dataset by handling missing values and cleaning data."""
    if df is None:
        print("DataFrame is None. Please load the data first.")
        return None
    
    # Example preprocessing steps
    # Fill missing values
    df.fillna(df.mean(), inplace=True)
    
    # Drop duplicates
    df.drop_duplicates(inplace=True)
    
    # Convert categorical variables to numeric if necessary
    df = pd.get_dummies(df, drop_first=True)
    
    print("Data preprocessing completed.")
    return df

def split_data(df, target_column, test_size=0.2, random_state=42):
    """Split the dataset into training and testing sets."""
    from sklearn.model_selection import train_test_split
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    print("Data split into training and testing sets.")
    return X_train, X_test, y_train, y_test