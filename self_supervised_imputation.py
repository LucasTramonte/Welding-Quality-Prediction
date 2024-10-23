import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer

def impute_target(y_train, method='zero', X_train=None):
    """
    Impute missing values in the target variable using specified methods.
    """
    print(f'y_train shape: {y_train.shape} \n y_train number of NaN before imputing {y_train.isnull().sum()}')
    y_train_imputed = y_train.copy()
    
    if method == 'mean':
        y_train_imputed.fillna(y_train.mean(), inplace=True)
    elif method == 'median':
        y_train_imputed.fillna(y_train.median(), inplace=True)
    elif method == 'ffill':
        y_train_imputed.fillna(method='ffill', inplace=True)
    elif method == 'zero':
        y_train_imputed.fillna(0, inplace=True)
    elif method == 'knn':
        if X_train is None:
            raise ValueError("X_train must be provided for 'knn' method.")
        # Selecionar apenas as colunas numéricas
        X_train_numeric = detect_type_columns(X_train)
        # Combinar X_train_numeric e y_train em um único DataFrame
        data_combined = X_train_numeric.copy()
        data_combined['target'] = y_train
        # Aplicar o imputer no DataFrame combinado
        imputer = KNNImputer(n_neighbors=10)
        data_imputed_array = imputer.fit_transform(data_combined)
        # Reconstruir o DataFrame com os índices originais
        data_imputed = pd.DataFrame(data_imputed_array, columns=data_combined.columns, index=data_combined.index)
        # Extrair y_train_imputed
        y_train_imputed = data_imputed['target']
    else:
        raise ValueError(f"Unknown method: {method}")
    
    print("Number of NaN after imputation:", y_train_imputed.isna().sum())
    return y_train_imputed

def detect_type_columns(df, type='numeric'):
    numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
    categoric_features = df.select_dtypes(include=['object', 'category']).columns
    return df[numeric_features] if type == 'numeric' else df[categoric_features]