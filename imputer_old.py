# Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.tree import DecisionTreeRegressor

class DropNanColsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.6):
        self.threshold = threshold
        self.cols_to_drop = None
    
    def fit(self, X, y=None):
        percent_n = (X == np.nan).mean()
        self.cols_to_drop = percent_n[percent_n > self.threshold].index
        return self
    
    def transform(self, X, y=None):
        return X.drop(columns=self.cols_to_drop)

class FixDatatypeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, categoric_features):
        self.categoric_features = categoric_features
        self.numeric_features = None
    
    def fit(self, X, y=None):
        self.numeric_features_corr = X[self.numeric_features].corr()
        return self
    
    def transform(self, X, y=None):
        X[self.numeric_features] = X[self.numeric_features].apply(pd.to_numeric, errors='coerce')
        return X

class RemoveHighCorrelation(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.75, subset_columns=None):
        self.threshold = threshold  # Limiar de correlação
        self.subset_columns = [col for col in subset_columns if col not in [
            'Ultimate tensile strenght (MPa)', 'Yield strenght (MPa)']]  # Colunas específicas para aplicar o filtro de correlação
    
    def fit(self, X, y=None):
        # Se subset_columns for None, aplicar em todas as colunas
        if self.subset_columns is None:
            self.subset_columns = X.columns
        
        # Filtrar o subconjunto de colunas
        subset_data = X[self.subset_columns]
        
        # Calcular a matriz de correlação no subconjunto de colunas
        corr_matrix = np.abs(np.corrcoef(subset_data, rowvar=False))
        
        # Encontrar as colunas no subset que têm alta correlação
        upper_tri = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        self.to_drop_ = [self.subset_columns[col] for col in range(corr_matrix.shape[1])
                         if any(corr_matrix[col, :][upper_tri[col]] > self.threshold)]
        
        return self
    
    def transform(self, X):
        # Remover as colunas identificadas no subset, mantendo as outras
        return X.drop(columns=self.to_drop_, errors='ignore')

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
    df[log_columns] = np.log1p(df[log_columns]+1)  # log(x + 1) para evitar problemas com zero
    return df

def impute_dataframe(numeric_features, categoric_features, percent_n_sorted):
    
    # Definir colunas de concentração (exceto Phosphorus e Sulphur)
    concentration_features = [col for col in numeric_features if "concentration" in col and not("Phosphorus" in col or "Sulphur" in col)]
    
    # Definir as colunas com valores faltantes < 10%
    low_missing = percent_n_sorted[(percent_n_sorted <= 0.1)].index
    low_missing = list(set(low_missing) - set(categoric_features))
    
    # Definir as colunas com valores faltantes entre 10% e 50%
    mid_missing = percent_n_sorted[(percent_n_sorted > 0.1) & (percent_n_sorted <= 0.5)].index
    mid_missing = list(set(mid_missing) - set(categoric_features))
    
    # Definir as colunas com mais de 50% de valores faltantes
    high_missing = percent_n_sorted[(percent_n_sorted > 0.5)].index
    high_missing = list(set(high_missing) - set(categoric_features))
    
    # Pipelines de imputação
    concentration_pipeline = Pipeline(steps=[
        ('fill_zero', SimpleImputer(strategy='constant', fill_value=0))  # Imputação por 0 nas concentrações
    ])
    
    low_missing_pipeline = Pipeline(steps=[
        ('mean_imputer', SimpleImputer(strategy='mean'))  # Imputação por média
    ])
    
    mid_missing_pipeline = Pipeline(steps=[
        ('knn_imputer', KNNImputer(n_neighbors=5))  # Imputação KNN
    ])
    
    high_missing_pipeline = Pipeline(steps=[
        ('iterative_imputer', IterativeImputer(estimator=DecisionTreeRegressor(), max_iter=10, random_state=0))  # Imputação Iterativa
    ])
    
    categorical_pipeline = Pipeline(steps=[
        ('mode_imputer', SimpleImputer(strategy='most_frequent'))  # Imputação pela moda
    ])
    
    # Criar um ColumnTransformer para aplicar os diferentes pipelines de imputação
    preprocessor = ColumnTransformer(
        transformers=[
            ('concentration', concentration_pipeline, concentration_features),  # Imputação nas colunas de concentração
            ('low_missing', low_missing_pipeline, low_missing),  # Imputação nas colunas com menos de 10% de valores faltantes
            ('mid_missing', mid_missing_pipeline, mid_missing),  # Imputação nas colunas com 10-50% de valores faltantes
            ('high_missing', high_missing_pipeline, high_missing),  # Imputação nas colunas com mais de 50% de valores faltantes
            ('categoric_impute', categorical_pipeline, categoric_features),  # Imputação nas colunas categóricas
        ]
    )
    
    return preprocessor

def to_dataframe_transformer(df):
    return pd.DataFrame(df)

def hot_encoder_last_three_columns():
    # Aqui estamos dizendo para aplicar o OneHotEncoder nas últimas 3 colunas (índices negativos)
    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(handle_unknown="ignore", sparse_output=False), [-3, -2, -1])  # Últimas 3 colunas
        ],
        remainder='passthrough'  # Mantém as outras colunas inalteradas
    )
    return preprocessor

def identify_binary_columns(df):
    binary_cols = [col for col in df.columns if df[col].nunique() == 2]  # Colunas com exatamente dois valores distintos
    non_binary_cols = [col for col in df.columns if col not in binary_cols]
    return binary_cols, non_binary_cols

class BinaryScalerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.non_binary_cols = None
        self.scaler = None
    
    def fit(self, X, y=None):
        df = pd.DataFrame(X)
        _, self.non_binary_cols = identify_binary_columns(df)
        # Definir o scaler para as colunas não binárias
        self.scaler = StandardScaler()
        # Fit o scaler apenas nas colunas não binárias
        self.scaler.fit(df[self.non_binary_cols])
        return self
    
    def transform(self, X, y=None):
        df = pd.DataFrame(X).copy()
        # Aplicar o scaler nas colunas não binárias
        df[self.non_binary_cols] = self.scaler.transform(df[self.non_binary_cols])
        return df.values  # Retorna o resultado como numpy array para compatibilidade com o restante do pipeline

def create_full_pipeline(numeric_features, categoric_features, percent_n_sorted):
    log_transformer = FunctionTransformer(apply_log_transformer)
    full_pipeline = Pipeline(steps=[
        ('drop_nan_cols', DropNanColsTransformer(threshold=0.6)),  # Remove colunas com muitos valores nulos
        ('fix_datatype', FixDatatypeTransformer(categoric_features)),  # Corrige tipos de dados
        ('apply_log', log_transformer),  # Aplica logaritmo nas colunas específicas
        ('imputer', impute_dataframe(numeric_features, categoric_features, percent_n_sorted)),  # Imputação
        ('to_dataframe_transformer', FunctionTransformer(to_dataframe_transformer)),
        ('one_hot', hot_encoder_last_three_columns()),  # OneHotEncoding nas últimas 3 colunas
        ('binary_scaler', BinaryScalerTransformer())  # Aplica o StandardScaler nas colunas não binárias
    ])
    return full_pipeline