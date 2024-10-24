import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline

# Transformer to fix data types and handle missing value indicators
class FixDatatypeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    # Fit method to identify numeric features
    def fit(self, X, y=None):
        self.numeric_features = ['Carbon concentration (weight%)', 'Silicon concentration (weight%)',
           'Manganese concentration (weight%)', 'Sulphur concentration (weight%)',
           'Phosphorus concentration (weight%)', 'Nickel concentration (weight%)',
           'Chromium concentration (weight%)', 'Molybdenum concentration (weight%)',
           'Vanadium concentration (weight%)', 'Copper concentration (weight%)',
           'Cobalt concentration (weight%)', 'Tungsten concentration (weight%)',
           'Oxygen concentration (ppm by weight)', 'Titanium concentration (ppm by weight)',
           'Nitrogen concentration (ppm by weight)', 'Aluminium concentration (ppm by weight)',
           'Boron concentration (ppm by weight)', 'Niobium concentration (ppm by weight)',
           'Tin concentration (ppm by weight)', 'Arsenic concentration (ppm by weight)',
           'Antimony concentration (ppm by weight)', 'Current (A)', 'Voltage (V)',
           'Heat input (kJ/mm)', 'Interpass temperature (°C)',
           'Post weld heat treatment temperature (°C)', 'Post weld heat treatment time (hours)',
           'Elongation (%)', 'Reduction of Area (%)', 'Charpy temperature (°C)',
           'Charpy impact toughness (J)', 'Hardness (kg/mm2)', '50% FATT',
           'Primary ferrite in microstructure (%)', 'Ferrite with second phase (%)',
           'Acicular ferrite (%)', 'Martensite (%)', 'Ferrite with carbide aggregate (%)']
        return self

    # Transform method to convert columns to numeric and add missing value indicators
    def transform(self, X, y=None):
        X = X.copy()
        # Convert the numeric features to numeric data types
        X[self.numeric_features] = X[self.numeric_features].apply(pd.to_numeric, errors='coerce')

        # Add new columns to indicate missing values for specific features
        X['Missing Elongation (%)'] = X['Elongation (%)'].isna().astype(int)    
        X['Missing Reduction of Area (%)'] = X['Reduction of Area (%)'].isna().astype(int)    
        X['Missing Molybdenum concentration (weight%)'] = X['Molybdenum concentration (weight%)'].isna().astype(int)    
        X['Missing Chromium concentration (weight%)'] = X['Chromium concentration (weight%)'].isna().astype(int)    
        X['Missing Nickel concentration (weight%)'] = X['Nickel concentration (weight%)'].isna().astype(int)    

        return X
    
# Transformer to drop columns with missing values above a certain threshold
class DropNanColsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold):
        self.threshold = threshold

    # Fit method to calculate the percentage of missing values and store columns to drop
    def fit(self, X, y=None):
        percent_n = X.isnull().mean()
        self.cols_to_drop = percent_n[percent_n > self.threshold].index
        return self

    # Transform method to drop columns with too many missing values
    def transform(self, X, y=None):
        return X.drop(columns=self.cols_to_drop)
    

# Function to create the preprocessing pipeline
def create_preprocessing_pipeline():
    # Pipeline for preprocessing (e.g., imputation, etc.)
    preprocessing_pipeline = Pipeline(steps=[
        ('fix_datatype', FixDatatypeTransformer()),  # Fix data types and handle missing indicators
        ('drop_nan_cols', DropNanColsTransformer(threshold=0.7)),  # Drop columns with >70% missing values
    ])
    return preprocessing_pipeline