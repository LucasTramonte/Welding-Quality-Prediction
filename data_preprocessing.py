import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator

class FixDatatypeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.numeric_features = ['Carbon concentration (weight%)', 'Silicon concentration (weight%)',
       'Manganese concentration (weight%)', 'Sulphur concentration (weight%)',
       'Phosphorus concentration (weight%)', 'Nickel concentration (weight%)',
       'Chromium concentration (weight%)',
       'Molybdenum concentration (weight%)',
       'Vanadium concentration (weight%)', 'Copper concentration (weight%)',
       'Cobalt concentration (weight%)', 'Tungsten concentration (weight%)',
       'Oxygen concentration (ppm by weight)',
       'Titanium concentration (ppm by weight)',
       'Nitrogen concentration (ppm by weight)',
       'Aluminium concentration (ppm by weight)',
       'Boron concentration (ppm by weight)',
       'Niobium concentration (ppm by weight)',
       'Tin concentration (ppm by weight)',
       'Arsenic concentration (ppm by weight)',
       'Antimony concentration (ppm by weight)', 'Current (A)', 'Voltage (V)', 
       'Heat input (kJ/mm)',
       'Interpass temperature (°C)',
       'Post weld heat treatment temperature (°C)',
       'Post weld heat treatment time (hours)', 'Elongation (%)',
       'Reduction of Area (%)', 'Charpy temperature (°C)',
       'Charpy impact toughness (J)', 'Hardness (kg/mm2)', '50% FATT',
       'Primary ferrite in microstructure (%)',
       'Ferrite with second phase (%)', 'Acicular ferrite (%)',
       'Martensite (%)', 'Ferrite with carbide aggregate (%)']
        return self

    def transform(self, X, y=None):
        X[self.numeric_features] = X[self.numeric_features].apply(pd.to_numeric, errors='coerce')
        return X


class DropNanColsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold):
        self.threshold = threshold

    def fit(self, X, y=None):
        percent_n = X.isnull().mean()
        self.cols_to_drop = percent_n[percent_n > self.threshold].index
        print(percent_n.sort_values())
        return self

    def transform(self, X, y=None):
        return X.drop(columns=self.cols_to_drop)