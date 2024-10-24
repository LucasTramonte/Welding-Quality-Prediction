import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(target = 'Yield strength (MPa)'):
    data_original = pd.read_csv('Assets/Data/welddb.csv', delimiter='\s+', header=None)

    data = data_original.copy().replace({"N": np.nan})

    # Name the columns
    data.columns = [
        'Carbon concentration (weight%)', 
        'Silicon concentration (weight%)', 
        'Manganese concentration (weight%)', 
        'Sulphur concentration (weight%)', 
        'Phosphorus concentration (weight%)', 
        'Nickel concentration (weight%)', 
        'Chromium concentration (weight%)', 
        'Molybdenum concentration (weight%)', 
        'Vanadium concentration (weight%)', 
        'Copper concentration (weight%)', 
        'Cobalt concentration (weight%)', 
        'Tungsten concentration (weight%)', 
        'Oxygen concentration (ppm by weight)', 
        'Titanium concentration (ppm by weight)', 
        'Nitrogen concentration (ppm by weight)', 
        'Aluminium concentration (ppm by weight)', 
        'Boron concentration (ppm by weight)', 
        'Niobium concentration (ppm by weight)', 
        'Tin concentration (ppm by weight)', 
        'Arsenic concentration (ppm by weight)', 
        'Antimony concentration (ppm by weight)', 
        'Current (A)', 
        'Voltage (V)', 
        'AC or DC', 
        'Electrode positive or negative', 
        'Heat input (kJ/mm)', 
        'Interpass temperature (°C)', 
        'Type of weld', 
        'Post weld heat treatment temperature (°C)', 
        'Post weld heat treatment time (hours)', 
        'Yield strength (MPa)', 
        'Ultimate tensile strength (MPa)', 
        'Elongation (%)', 
        'Reduction of Area (%)', 
        'Charpy temperature (°C)', 
        'Charpy impact toughness (J)', 
        'Hardness (kg/mm2)', 
        '50% FATT', 
        'Primary ferrite in microstructure (%)', 
        'Ferrite with second phase (%)', 
        'Acicular ferrite (%)', 
        'Martensite (%)', 
        'Ferrite with carbide aggregate (%)', 
        'Weld ID'
    ]

    # Splitting data into train and test sets
    X = data.drop(columns = ["Yield strength (MPa)", "Weld ID", "Ultimate tensile strength (MPa)"])
    y = data[target]    
    return X, y

def train_test_split_balance(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Dropping missing values in test set
    X_test_clean = X_test[~y_test.isna()]
    y_test_clean = y_test.dropna().astype(float)

    X_train_full = pd.concat([X_train, X_test[y_test.isna()]], axis=0)
    y_train_full = pd.concat([y_train, y_test[y_test.isna()]], axis=0)

    return X_train_full, X_test_clean, y_train_full, y_test_clean