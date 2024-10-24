import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer

def impute_target(y_train, method='zero', X_train=None):
    """
    Impute missing values in the target variable using specified methods.
    
    Parameters:
    - y_train: The target variable with missing values.
    - method: The imputation method to apply ('mean', 'median', 'ffill', 'zero', or 'knn').
    - X_train: The feature set, required if 'knn' method is selected.
    
    Returns:
    - y_train_imputed: The target variable with missing values imputed.
    """
    print(f'y_train shape: {y_train.shape} \n y_train number of NaN before imputing {y_train.isnull().sum()}')
    y_train_imputed = y_train.copy()
    
    if method == 'mean':
        # Impute missing values using the mean of the target variable
        y_train_imputed.fillna(y_train.mean(), inplace=True)
    elif method == 'median':
        # Impute missing values using the median of the target variable
        y_train_imputed.fillna(y_train.median(), inplace=True)
    elif method == 'ffill':
        # Impute missing values using forward fill
        y_train_imputed.fillna(method='ffill', inplace=True)
    elif method == 'zero':
        # Impute missing values with zeros
        y_train_imputed.fillna(0, inplace=True)
    elif method == 'knn':
        # Impute missing values using KNN based on X_train features
        if X_train is None:
            raise ValueError("X_train must be provided for 'knn' method.")
        # Select only numeric columns for the KNN imputer
        X_train_numeric = detect_type_columns(X_train)
        # Combine X_train_numeric and y_train into a single DataFrame
        data_combined = X_train_numeric.copy()
        data_combined['target'] = y_train
        # Apply KNN imputation
        imputer = KNNImputer(n_neighbors=10)
        data_imputed_array = imputer.fit_transform(data_combined)
        # Reconstruct DataFrame with original indices
        data_imputed = pd.DataFrame(data_imputed_array, columns=data_combined.columns, index=data_combined.index)
        # Extract the imputed target variable
        y_train_imputed = data_imputed['target']
    else:
        # Raise an error if an unknown method is passed
        raise ValueError(f"Unknown method: {method}")
    
    # Output the number of NaN values after imputation
    print("Number of NaN after imputation:", y_train_imputed.isna().sum())
    return y_train_imputed

def detect_type_columns(df, type='numeric'):
    """
    Detects and returns columns of a specified data type from the DataFrame.
    
    Parameters:
    - df: The input DataFrame.
    - type: The type of columns to detect ('numeric' or 'categorical').
    
    Returns:
    - DataFrame containing only the columns of the specified type.
    """
    # Identify numeric features
    numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
    # Identify categorical features
    categoric_features = df.select_dtypes(include=['object', 'category']).columns
    # Return numeric or categorical columns based on the input argument
    return df[numeric_features] if type == 'numeric' else df[categoric_features]