# Utility functions for the credit scoring model project

import matplotlib.pyplot as plt
import seaborn as sns

def plot_missing_values(df):
    """Plot the missing values in the DataFrame."""
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Values in DataFrame')
    plt.show()

def plot_feature_distribution(df, feature):
    """Plot the distribution of a specified feature."""
    plt.figure(figsize=(10, 6))
    sns.histplot(df[feature], bins=30, kde=True)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()

def save_model(model, filename):
    """Save the trained model to a file."""
    import joblib
    joblib.dump(model, filename)
    print(f'Model saved to {filename}')

def load_model(filename):
    """Load a trained model from a file."""
    import joblib
    model = joblib.load(filename)
    print(f'Model loaded from {filename}')
    return model