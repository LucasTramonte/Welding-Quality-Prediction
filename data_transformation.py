import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler

# Class for removing outliers based on the interquartile range (IQR)
class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, multiplier=1.5):
        self.multiplier = multiplier
        self.numeric_columns = None
        self.lower_bounds = {}
        self.upper_bounds = {}
    
    # Fit method to calculate the IQR bounds for each numeric feature
    def fit(self, X, y=None):
        self.numeric_columns = X.select_dtypes(include=[np.number]).columns
        for col in self.numeric_columns:
            Q1 = X[col].quantile(0.25)  # First quartile
            Q3 = X[col].quantile(0.75)  # Third quartile
            IQR = Q3 - Q1  # Interquartile range
            self.lower_bounds[col] = Q1 - self.multiplier * IQR  # Lower bound for outliers
            self.upper_bounds[col] = Q3 + self.multiplier * IQR  # Upper bound for outliers
        return self
    
    # Transform method to remove rows with outliers
    def transform(self, X, y=None):
        X_clean = X.copy()
        mask = pd.Series(True, index=X_clean.index)
        for col in self.numeric_columns:
            lower = self.lower_bounds[col]
            upper = self.upper_bounds[col]
            # Mask for rows within bounds
            mask &= (X_clean[col] >= lower) & (X_clean[col] <= upper)
        # Return cleaned DataFrame without outliers
        X_clean = X_clean[mask].reset_index(drop=True)
        return X_clean


# Class for scaling numeric features using different scalers
class ScalerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, scaler='standard'):
        self.scaler_type = scaler
        self.scaler = None
        self.numeric_features = None

    # Fit method to initialize and fit the selected scaler
    def fit(self, X, y=None):
        # Identify continuous numeric features (excluding binary)
        self.numeric_features = [col for col in X.columns if X[col].dtype in ['float64', 'int64'] and X[col].nunique() > 2]
        
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Scaler {self.scaler_type} not recognized.")
        
        # Fit the scaler to the numeric features
        self.scaler.fit(X[self.numeric_features])
        return self

    # Transform method to apply scaling to the numeric features
    def transform(self, X):
        X_scaled = X.copy()
        X_scaled[self.numeric_features] = self.scaler.transform(X[self.numeric_features])  # Apply scaling
        return X_scaled


# Custom OneHotEncoder class to handle categorical encoding manually
class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.categories_ = {}

    # Fit method to capture unique categories in the categorical features
    def fit(self, X, y=None):
        # Store unique categories for each categorical column during training
        for col in X.select_dtypes(include=['object', 'category']).columns:
            self.categories_[col] = X[col].dropna().unique().tolist()
        return self

    # Transform method to encode the categorical columns
    def transform(self, X, y=None):
        X_transformed = X.copy()
        for col, categories in self.categories_.items():
            # Create dummy variables only for the categories seen in training
            dummies = np.zeros((X_transformed.shape[0], len(categories)), dtype=int)
            for i, category in enumerate(categories):
                dummies[:, i] = (X_transformed[col] == category).astype(int)

            # Add the dummy variables to the DataFrame with the correct column names
            dummy_df = pd.DataFrame(dummies, columns=[f"{col}_{cat}" for cat in categories], index=X_transformed.index)
            X_transformed = pd.concat([X_transformed, dummy_df], axis=1)

        # Remove the original categorical columns
        X_transformed = X_transformed.drop(columns=self.categories_.keys())
        return X_transformed