# Credit Scoring Model

This project aims to develop a credit scoring model using machine learning techniques. The model will analyze loan data to predict the likelihood of default, helping financial institutions make informed lending decisions.

## Project Structure

- **data/**: Contains the dataset and related documentation.
  - **README.md**: Information about the dataset, including its source, structure, and preprocessing steps.
  
- **notebooks/**: Jupyter notebooks for exploratory data analysis, model training, and evaluation.
  - **credit_scoring.ipynb**: Contains code and visualizations for understanding the data and model performance.
  
- **src/**: Source code for data processing, feature engineering, model training, and evaluation.
  - **data_preprocessing.py**: Functions for loading and preprocessing the dataset.
  - **feature_engineering.py**: Functions for feature extraction and transformation.
  - **model.py**: Defines the machine learning model, including training and prediction.
  - **evaluate.py**: Functions for evaluating model performance using various metrics.
  - **utils.py**: Utility functions for data visualization and manipulation.
  
- **tests/**: Unit tests for the project.
  - **test_model.py**: Tests for the functions in model.py to ensure expected behavior.
  
- **requirements.txt**: Lists the required Python packages for the project.

## Getting Started

1. Clone the repository:
   ```
   git clone <repository-url>
   cd credit-scoring-model
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Run the Jupyter notebook for analysis and model training:
   ```
   jupyter notebook notebooks/credit_scoring.ipynb
   ```

## License

This project is licensed under the MIT License. See the LICENSE file for details.