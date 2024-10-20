import pandas as pd

def load_and_label_data(file_path):
    columns_names = [
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

    data = pd.read_csv('Assets/Data/welddb.csv', delimiter='\s+', header=None)
    data.columns = columns_names

    return data