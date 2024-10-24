import joblib
import pandas as pd
import time
from functools import wraps

def save_model(model, filename):
    """
    Save a machine learning model to a file using joblib.
    
    Parameters:
    - model: The model to save.
    - filename: The file path where the model will be saved.
    """
    joblib.dump(model, filename)

def load_model(filename):
    """
    Load a machine learning model from a file using joblib.
    
    Parameters:
    - filename: The file path from which to load the model.
    
    Returns:
    - The loaded model.
    """
    return joblib.load(filename)

def to_dataframe_transformer(X, column_names=None):
    """
    Convert a numpy array back to a pandas DataFrame, inferring numerical and categorical columns.
    
    Parameters:
    - X: The numpy array to convert.
    - column_names: The list of column names for the DataFrame. If not provided, numeric indices will be used.
    
    Returns:
    - A pandas DataFrame with proper data types assigned to numeric and categorical columns.
    """
    if column_names is None:
        # Generate default column names if none are provided
        size_df = X.shape[1]
        col_names = [f'{i}' for i in range(size_df)]
        df = pd.DataFrame(X, columns=col_names)
    else:
        df = pd.DataFrame(X, columns=column_names)
    
    numeric_features_list = []
    categorical_features_list = []
    
    for col in column_names:
        try:
            # Try to convert the column to a numeric type
            pd.to_numeric(df[col], errors='raise')  # Raises error if it cannot be converted
            numeric_features_list.append(col)  # Add as numeric if successful
        except ValueError:
            # If conversion fails, classify as categorical
            categorical_features_list.append(col)
    
    # Convert numeric columns to float type
    df[numeric_features_list] = df[numeric_features_list].astype('float')    
    return df

def timing_decorator(func):
    """
    A decorator to measure the execution time of a function.
    
    Parameters:
    - func: The function to be wrapped by the decorator.
    
    Returns:
    - The result of the function call, with added print statements for timing.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Starting function: {func.__name__}")
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Time taken for {func.__name__}: {end_time - start_time:.2f} seconds")
        return result
    return wrapper