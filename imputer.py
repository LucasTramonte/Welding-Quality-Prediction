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
        self.concentration_pipeline = Pipeline(steps=[
            ('fill_zero', SimpleImputer(strategy='constant', fill_value=0))  # Imputação por 0 nas concentrações
        ])
        
        self.low_missing_pipeline = Pipeline(steps=[
            ('median_imputer', SimpleImputer(strategy='median'))  # Imputação por mediana
        ])
        
        self.mid_missing_pipeline = Pipeline(steps=[
            ('knn_imputer', KNNImputer(n_neighbors=5))  # Imputação KNN
        ])
        
        self.high_missing_pipeline = Pipeline(steps=[
            ('iterative_imputer', IterativeImputer(estimator=DecisionTreeRegressor(), max_iter=10, random_state=0))  # Imputação Iterativa
        ])
        
        self.categorical_pipeline = Pipeline(steps=[
            ('mode_imputer', SimpleImputer(strategy='most_frequent'))  # Imputação pela moda
        ])
        
        # Criar um ColumnTransformer para aplicar os diferentes pipelines de imputação
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('concentration', self.concentration_pipeline, self.concentration_features),  # Imputação nas colunas de concentração
                ('low_missing', self.low_missing_pipeline, self.low_missing),  # Imputação nas colunas com menos de 10% de valores faltantes
                ('mid_missing', self.mid_missing_pipeline, self.mid_missing),  # Imputação nas colunas com 10-50% de valores faltantes
                ('high_missing', self.high_missing_pipeline, self.high_missing),  # Imputação nas colunas com mais de 50% de valores faltantes
                ('categoric_impute', self.categorical_pipeline, self.categoric_features),  # Imputação nas colunas categóricas
            ],
            remainder='passthrough'  # Garante que as colunas não transformadas sejam preservadas e na mesma ordem
        )
        
        # Ajustar o imputador nos dados de treino
        self.preprocessor.fit(X)
        
        # Guardar as colunas originais na ordem original
        self.original_columns = X.columns
        
        return self
    
    def transform(self, X):
        # Aplicar a transformação (sem ajustar novamente) nos dados de teste ou validação
        X_transformed = self.preprocessor.transform(X)
        
        # Convertendo de volta para DataFrame e restaurando a ordem das colunas originais
        X_transformed = to_dataframe_transformer(X_transformed)

        return X_transformed

def to_dataframe_transformer(X):
    # Converter o np.array de volta para DataFrame
    size_df = X.shape[1]
    col_names = [f'{i}' for i in range(size_df)]
    df = pd.DataFrame(X, columns=col_names)
    
    numeric_features_list = []
    categorical_features_list = []
    
    for col in col_names:
        try:
            # Tenta converter a coluna para numérico
            pd.to_numeric(df[col], errors='raise')  # Levanta erro se não puder converter
            numeric_features_list.append(col)  # Se for possível, adiciona como numérica
        except ValueError:
            # Se houver erro, classificamos como categórica
            categorical_features_list.append(col)
    
    # Convertendo colunas numéricas para float
    df[numeric_features_list] = df[numeric_features_list].astype('float')    
    return df

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
    
class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.categories_ = {}

    def fit(self, X, y=None):
        # Para cada coluna categórica, encontramos as categorias únicas presentes nos dados de treino
        for col in X.select_dtypes(include=['object', 'category']).columns:
            self.categories_[col] = X[col].dropna().unique().tolist()
        return self

    def transform(self, X, y=None):
        # Aplicamos pd.get_dummies apenas nas colunas categóricas
        X_transformed = X.copy()
        for col, categories in self.categories_.items():
            # Criar dummies com as categorias presentes nos dados de treino
            dummies = pd.get_dummies(X_transformed[col], prefix=col)
            # Criar colunas para categorias que não estão presentes nos dados de teste
            for category in categories:
                dummy_col = f"{col}_{category}"
                if dummy_col not in dummies.columns:
                    dummies[dummy_col] = 0
            # Reordenar as colunas para garantir a consistência
            dummies = dummies[[f"{col}_{category}" for category in categories]]
            # Substituir a coluna original pelos dummies
            X_transformed = X_transformed.drop(columns=[col])
            X_transformed = pd.concat([X_transformed, dummies], axis=1)
        
        return X_transformed

def create_full_pipeline():

    full_pipeline = Pipeline(steps=[
        ('fix_datatype', FixDatatypeTransformer()),  # Corrige tipos de dados
        ('drop_nan_cols', DropNanColsTransformer(threshold=0.6)),  # Remove colunas com muitos valores nulos
        ('remove_high_corr', RemoveHighCorrelation(threshold=0.75)),
        ('log_transform', LogTransformer(threshold_skew=0.5, threshold_outliers=1.5)),  # Definir os thresholds
        ('imputer', ImputeDataFrame()),  # Imputação
        ('scaler', ScalerTransformer()),  # Aplica o StandardScaler nas colunas numéricas
        ('hot_encoder', CustomOneHotEncoder()) # Aplica o Hot Encoder nas colunas categóricas
    ])
    
    return full_pipeline

def HotEncoderCategorical(X):
    # Identifica as colunas categóricas
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns

    # Cria um ColumnTransformer que aplica o OneHotEncoder apenas nas colunas categóricas
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)  # OneHot nas categóricas
        ],
        remainder='passthrough'  # Deixa as colunas numéricas inalteradas
    )
    return preprocessor
