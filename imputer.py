import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import RobustScaler # it is not affected by outliers.
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import TransformerMixin, BaseEstimator



# %% [markdown]
# # EDA

# %%
data_original = pd.read_csv('Assets/Data/welddb.csv', delimiter='\s+', header=None)

# %%
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

# %% [markdown]
# # Preprocessing

# %%
def drop_nan_cols(data_clean: pd.DataFrame) -> pd.DataFrame:
    percent_n = (data == 'N').mean() 
    percent_n_sorted = percent_n.sort_values(ascending=False)

    # Removing columns with more than 60% missing values
    cols_to_drop = percent_n_sorted[percent_n_sorted > 0.6].index

    data_clean = data_clean.drop(columns = cols_to_drop)
    return data_clean

def fix_datatype(data_fixed: pd.DataFrame) -> tuple[pd.DataFrame, list, list]:
    data_fixed = data_fixed.replace({"N": np.nan})

    categoric_features = ['AC or DC', 'Electrode positive or negative', 'Type of weld']  # Weld ID isn't important

    # Convert columns to numeric
    numeric_features = data_fixed.drop(columns=['Weld ID', 'AC or DC', 'Electrode positive or negative', 'Type of weld']).columns
    data_fixed[numeric_features] = data_fixed[numeric_features].apply(pd.to_numeric, errors='coerce')
    return data_fixed, categoric_features, numeric_features

def apply_log(data_log: pd.DataFrame) -> pd.DataFrame:
    log_columns = ['Silicon concentration (weight%)',
                'Sulphur concentration (weight%)',
                'Phosphorus concentration (weight%)',
                'Nickel concentration (weight%)',
                'Titanium concentration (ppm by weight)',
                'Nitrogen concentration (ppm by weight)',
                'Oxygen concentration (ppm by weight)',
                'Voltage (V)',
                'Heat input (kJ/mm)']
    
    
    # Apply logarithmic transformation (log(x + 1)) to avoid issues with zero values
    data_log[log_columns] = data_log[log_columns].apply(lambda x: np.log(x + 1))
    
    return data_log

def impute_dataframe(data_imputed: pd.DataFrame, categoric_features: list, numeric_features: list) -> pd.DataFrame:
    imputer_median = SimpleImputer(strategy = "median")
    imputer_mode = SimpleImputer(strategy = "most_frequent")

    # for the concentration columns, we will set the values to zero for elements different from Phosphorus and Sulphur
    concentration_features = [col for col in data_imputed.columns if "concentration" in col and not("Phosphorus" in col or "Sulphur" in col)]
    other_numeric_features = [col for col in numeric_features if col not in concentration_features]

    data_imputed[concentration_features] = data_imputed[concentration_features].fillna(0)

    data_imputed[other_numeric_features] = imputer_median.fit_transform(data_imputed[other_numeric_features])
    data_imputed[categoric_features] = imputer_mode.fit_transform(data_imputed[categoric_features])
    return data_imputed

# %%
# 1. Custom transformer to remove columns with many null values
class DropNanColsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.6):
        self.threshold = threshold
        self.cols_to_drop = None
    
    def fit(self, X, y=None):
        percent_n = (X == 'N').mean()
        self.cols_to_drop = percent_n[percent_n > self.threshold].index
        return self
    
    def transform(self, X, y=None):
        return X.drop(columns=self.cols_to_drop)

# 2. Transformer to fix data types
class FixDatatypeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, categoric_features):
        self.categoric_features = categoric_features
        self.numeric_features = None
    
    def fit(self, X, y=None):
        self.numeric_features = X.drop(columns=['Weld ID'] + self.categoric_features).columns
        return self
    
    def transform(self, X, y=None):
        X = X.replace({"N": np.nan})
        X[self.numeric_features] = X[self.numeric_features].apply(pd.to_numeric, errors='coerce')
        return X

# 3. Function to apply the logarithm
def apply_log_transformer(df):
    log_columns = ['Silicon concentration (weight%)',
                   'Sulphur concentration (weight%)',
                   'Phosphorus concentration (weight%)',
                   'Nickel concentration (weight%)',
                   'Titanium concentration (ppm by weight)',
                   'Nitrogen concentration (ppm by weight)',
                   'Oxygen concentration (ppm by weight)',
                   'Voltage (V)',
                   'Heat input (kJ/mm)']
    df[log_columns] = np.log1p(df[log_columns])  # log(x + 1) to avoid problems with zero
    return df

log_transformer = FunctionTransformer(apply_log_transformer)

# 4. Data imputation

def impute_dataframe(numeric_features, categoric_features):
        
    # Define concentration columns (except Phosphorus and Sulphur)

    concentration_features = [col for col in numeric_features if "concentration" in col and not("Phosphorus" in col or "Sulphur" in col)]
    other_numeric_features = [col for col in numeric_features if col not in concentration_features]
    
    # Imputation for concentration columns with zero

    concentration_pipeline = Pipeline(steps=[
        ('fill_zero', SimpleImputer(strategy='constant', fill_value=0))
    ])
    
    # Imputation for numerical columns with median
    numeric_pipeline = Pipeline(steps=[
        ('median_imputer', SimpleImputer(strategy='median'))
    ])
    
    # Imputation for categorical columns with the most frequent value
    categorical_pipeline = Pipeline(steps=[
        ('mode_imputer', SimpleImputer(strategy='most_frequent'))  # Categorical imputation
    ])
    
    # Create a ColumnTransformer to apply the different imputers and OneHotEncoder
    preprocessor = ColumnTransformer(
    transformers=[
        ('concentration', concentration_pipeline, concentration_features),  # Imputação nas colunas de concentração
        ('numeric', numeric_pipeline, other_numeric_features),  # Imputação nas colunas numéricas
        ('categoric_impute', categorical_pipeline, categoric_features),  # Imputação nas colunas categóricas
    ]
    )
    return preprocessor

def to_dataframe_transformer(df):
    return pd.DataFrame(df)

def hot_encoder_last_three_columns():
    # Here we are saying to apply OneHotEncoder to the last 3 columns (negative indexes)
    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(handle_unknown="ignore", sparse_output=False), [-3, -2, -1])  # Last 3 columns
        ],
        remainder='passthrough'  # Keeps the other columns unchanged

    )
    return preprocessor

# Function to identify binary columns

def identify_binary_columns(df):
    binary_cols = [col for col in df.columns if df[col].nunique() == 2]  # Columns with exactly two different values

    non_binary_cols = [col for col in df.columns if col not in binary_cols]
    return binary_cols, non_binary_cols

# Custom transformer to identify and apply the scaler to non-binary columns
class BinaryScalerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.non_binary_cols = None
        self.scaler = None
    
    def fit(self, X, y=None):
        df = pd.DataFrame(X)
        _, self.non_binary_cols = identify_binary_columns(df)
        # Set the scaler for non-binary columns
        self.scaler = StandardScaler()
        # Fit the scaler only to non-binary columns
        self.scaler.fit(df[self.non_binary_cols])
        return self
    
    def transform(self, X, y=None):
        df = pd.DataFrame(X).copy()
        # Apply the scaler to non-binary columns
        df[self.non_binary_cols] = self.scaler.transform(df[self.non_binary_cols])
        return df.values  # Returns the result as a numpy array for compatibility with the rest of the pipeline


#5 Initialization of the complete pipeline

def create_full_pipeline(numeric_features, categoric_features):
    full_pipeline = Pipeline(steps=[
        ('drop_nan_cols', DropNanColsTransformer(threshold=0.6)),  
        ('fix_datatype', FixDatatypeTransformer(categoric_features)),  
        ('apply_log', log_transformer),  
        ('imputer', impute_dataframe(numeric_features, categoric_features)),  
        ('to_dataframe_transformer', FunctionTransformer(to_dataframe_transformer)),
        ('one_hot', hot_encoder_last_three_columns()),  
        ('binary_scaler', BinaryScalerTransformer())  
    ])
    return full_pipeline

categoric_features = ['AC or DC', 'Electrode positive or negative', 'Type of weld']
numeric_features = ['Sulphur concentration (weight%)', 'Nickel concentration (weight%)', 
                    'Silicon concentration (weight%)', 'Phosphorus concentration (weight%)', 
                    'Titanium concentration (ppm by weight)', 'Nitrogen concentration (ppm by weight)', 
                    'Oxygen concentration (ppm by weight)', 'Voltage (V)', 'Heat input (kJ/mm)']

full_pipeline = create_full_pipeline(numeric_features, categoric_features)

X = data.drop(columns = ["Yield strength (MPa)"])
y = data["Yield strength (MPa)"]
# y.fillna(0, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train_transformed = full_pipeline.fit_transform(X_train)
X_test_transformed = full_pipeline.transform(X_test)
