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
        # Forçar a conversão das colunas identificadas como numéricas
        X[self.numeric_features] = X[self.numeric_features].apply(pd.to_numeric, errors='coerce')
        return X

class DropNanColsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold):
        self.threshold = threshold
        self.cols_to_drop = None
    
    def fit(self, X, y=None):
        percent_n = X.isnull().mean()
        self.cols_to_drop = percent_n[percent_n > self.threshold].index
        return self
    
    def transform(self, X, y=None):
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns
        return X.drop(columns=self.cols_to_drop)
    
class RemoveHighCorrelation(BaseEstimator, TransformerMixin):
    def __init__(self, threshold):
        self.threshold = threshold  # Limiar de correlação
    
    def fit(self, X, y=None):
        self.numeric_features = X.select_dtypes(include=['float64', 'int64']).columns

        # Filtrar o subconjunto de colunas
        subset_data = X[self.numeric_features]
        
        # Calcular a matriz de correlação no subconjunto de colunas
        corr_matrix = subset_data.corr().abs()  # Usar pandas corr() com valor absoluto
        
        # Identificar a parte superior da matriz de correlação
        upper_tri = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)

        # Aplicar a máscara upper_tri na matriz de correlação
        upper_tri_corr = corr_matrix.where(upper_tri)  # Mantém apenas a parte superior
        
        # Encontrar as colunas no subset que têm alta correlação
        self.to_drop_ = [column for column in upper_tri_corr.columns if any(upper_tri_corr[column] > self.threshold)]
        
        return self
    
    def transform(self, X):
        # Remover as colunas identificadas no subset, mantendo as outras
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns
        return X.drop(columns=self.to_drop_, errors='ignore')

class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold_skew=0.5, threshold_outliers=1.5):
        # Definindo thresholds para skewness e outliers
        self.threshold_skew = threshold_skew  # Skewness acima desse valor será transformada
        self.threshold_outliers = threshold_outliers  # Limite baseado no IQR para identificar outliers
        self.columns_to_log = []
    
    def fit(self, X, y=None):
        # Identificar variáveis numéricas e categóricas automaticamente
        self.numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
        self.categoric_features = X.select_dtypes(include=['object', 'category']).columns
        
        # Calcular a porcentagem de valores faltantes
        percent_n_missing = X.isnull().mean()

        # Definir colunas de concentração (exceto Phosphorus e Sulphur)
        self.concentration_features = [col for col in self.numeric_features if "concentration" in col and not("Phosphorus" in col or "Sulphur" in col)]
        
        # Definir as colunas com valores faltantes < 10%
        self.low_missing = percent_n_missing[(percent_n_missing <= 0.1)].index
        self.low_missing = list(set(self.low_missing) - set(self.categoric_features))
        
        # Definir as colunas com valores faltantes entre 10% e 50%
        self.mid_missing = percent_n_missing[(percent_n_missing > 0.1) & (percent_n_missing <= 0.5)].index
        self.mid_missing = list(set(self.mid_missing) - set(self.categoric_features))
        
        # Definir as colunas com mais de 50% de valores faltantes
        self.high_missing = percent_n_missing[(percent_n_missing > 0.5)].index
        self.high_missing = list(set(self.high_missing) - set(self.categoric_features))
        
        return self
    
    def transform(self, X):
        # Aplicar a transformação logarítmica nas colunas identificadas
        X = X.copy()  # Evitar mudanças no dataframe original
        if self.columns_to_log:
            X[self.columns_to_log] = np.log1p(X[self.columns_to_log]+1)  # log(x + 1) para lidar com zeros
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns
        return X

class ImputeDataFrame(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        # Identificar variáveis numéricas e categóricas automaticamente
        self.numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
        self.categoric_features = X.select_dtypes(include=['object', 'category']).columns
        
        # Calcular a porcentagem de valores faltantes
        percent_n_missing = X.isnull().mean()

        # Definir colunas de concentração (exceto Phosphorus e Sulphur)
        self.concentration_features = [col for col in self.numeric_features if "concentration" in col and not("Phosphorus" in col or "Sulphur" in col)]
        
        # Definir as colunas com valores faltantes < 10%
        self.low_missing = percent_n_missing[(percent_n_missing <= 0.1)].index
        self.low_missing = list(set(self.low_missing) - set(self.categoric_features))
        
        # Definir as colunas com valores faltantes entre 10% e 50%
        self.mid_missing = percent_n_missing[(percent_n_missing > 0.1) & (percent_n_missing <= 0.5)].index
        self.mid_missing = list(set(self.mid_missing) - set(self.categoric_features))
        
        # Definir as colunas com mais de 50% de valores faltantes
        self.high_missing = percent_n_missing[(percent_n_missing > 0.5)].index
        self.high_missing = list(set(self.high_missing) - set(self.categoric_features))
        
        # Pipelines de imputação
        concentration_pipeline = Pipeline(steps=[
            ('fill_zero', SimpleImputer(strategy='constant', fill_value=0))  # Imputação por 0 nas concentrações
        ])
        
        low_missing_pipeline = Pipeline(steps=[
            ('median_imputer', SimpleImputer(strategy='median'))  # Imputação por mediana
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
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('concentration', concentration_pipeline, self.concentration_features),  # Imputação nas colunas de concentração
                ('low_missing', low_missing_pipeline, self.low_missing),  # Imputação nas colunas com menos de 10% de valores faltantes
                ('mid_missing', mid_missing_pipeline, self.mid_missing),  # Imputação nas colunas com 10-50% de valores faltantes
                ('high_missing', high_missing_pipeline, self.high_missing),  # Imputação nas colunas com mais de 50% de valores faltantes
                ('categoric_impute', categorical_pipeline, self.categoric_features),  # Imputação nas colunas categóricas
            ],
            remainder='passthrough'
        )

        # Ajustar o imputador nos dados de treino
        self.preprocessor.fit(X)

        return self
    
    def transform(self, X):
        
        # Aplicar a transformação
        col_names = [f'{i}' for i in range(X.shape[1])]
        X_transformed = pd.DataFrame(X, columns=col_names)
        
        return X_transformed

def identify_binary_columns(df):
    binary_cols = [col for col in df.columns if df[col].nunique() == 2]  # Colunas com exatamente dois valores distintos
    non_binary_cols = [col for col in df.columns if col not in binary_cols]
    return binary_cols, non_binary_cols


class ScalerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        self.numeric_features = X.select_dtypes(include=['float64', 'int64']).columns

        # Fit o StandardScaler apenas nas colunas não binárias
        self.scaler = StandardScaler()
        self.scaler.fit(X[self.numeric_features])
        
        return self
    
    def transform(self, X, y=None):

        # Aplicar o scaler nas colunas não binárias
        X[self.numeric_features] = self.scaler.transform(X[self.numeric_features])
        return X
    
# Identifica colunas categóricas dentro do pipeline
def hot_encoder_categorical_columns(X):
    categorical_columns = X.select_dtypes(include=['object', 'category']).columns
    return ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_columns)
        ],
        remainder='passthrough'
    )

def create_full_pipeline():

    full_pipeline = Pipeline(steps=[
        ('fix_datatype', FixDatatypeTransformer()),  # Corrige tipos de dados
        ('drop_nan_cols', DropNanColsTransformer(threshold=0.6)),  # Remove colunas com muitos valores nulos
        ('remove_high_corr', RemoveHighCorrelation(threshold=0.75)),
        ('log_transform', LogTransformer(threshold_skew=0.5, threshold_outliers=1.5)),  # Definir os thresholds
        ('imputer', ImputeDataFrame()),  # Imputação
        ('scaler', ScalerTransformer()),  # Aplica o StandardScaler nas colunas não binárias
        ('one_hot', FunctionTransformer(lambda X: hot_encoder_categorical_columns(X).fit_transform(X))),  # Aplicar o OneHotEncoder
    ])
    
    return full_pipeline
