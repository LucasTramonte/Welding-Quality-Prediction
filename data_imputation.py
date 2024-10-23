import pandas as pd
from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from utils import to_dataframe_transformer
from sklearn.ensemble import RandomForestRegressor
from feature_engineering import RemoveHighCorrelation, LogTransformer
from data_transformation import ScalerTransformer, CustomOneHotEncoder, OutlierRemover

class ImputeDataFrame(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None, strategy='knn'):
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

        if strategy == 'median':            
            # Definindo o pipeline para colunas com valores faltantes moderados
            self.mid_missing_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median'))  # Escolhe o imputer baseado na estratégia
            ])
            
            # Definindo o pipeline para colunas com muitos valores faltantes
            self.high_missing_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median'))  # Escolhe o imputer baseado na estratégia
            ])
            
        else:
            # Definindo o pipeline para colunas com valores faltantes moderados
            self.mid_missing_pipeline = Pipeline(steps=[
                ('knn_imputer', KNNImputer(n_neighbors=10))  # Imputação KNN
            ])
            
            # Definindo o pipeline para colunas com muitos valores faltantes
            self.high_missing_pipeline = Pipeline(steps=[
                ('knn_imputer', KNNImputer(n_neighbors=10))  # Imputação KNN
            ])

        # Pipelines de imputação
        self.concentration_pipeline = Pipeline(steps=[
            ('fill_zero', SimpleImputer(strategy='constant', fill_value=0))  # Imputação por 0 nas concentrações
        ])
        
        # Definindo o pipeline para colunas com poucos valores faltantes
        self.low_missing_pipeline = Pipeline(steps=[
            ('imputer', IterativeImputer(
                estimator=RandomForestRegressor(n_estimators=100, random_state=42),
                max_iter=10, random_state=42))  # Escolhe o imputer baseado na estratégia
                    ])

        # Definindo o pipeline para colunas numéricas
        self.numeric_pipeline = Pipeline(steps=[
            ('knn_imputer', KNNImputer(n_neighbors=10))
                    ])
        
        # Definindo o pipeline para colunas categóricas
        self.categorical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent'))
                    ])

        # Criar um ColumnTransformer para aplicar os diferentes pipelines de imputação
        self.preprocessor = ColumnTransformer(
            transformers=[
                # ('concentration', self.concentration_pipeline, self.concentration_features),  # Imputação nas colunas de concentração
                ('numeric', self.numeric_pipeline, self.numeric_features),  # Imputação nas colunas com menos de 10% de valores faltantes
                # ('mid_missing', self.mid_missing_pipeline, self.mid_missing),  # Imputação nas colunas com 10-50% de valores faltantes
                # ('high_missing', self.high_missing_pipeline, self.high_missing),  # Imputação nas colunas com mais de 50% de valores faltantes
                ('categoric_imputer', self.categorical_pipeline, self.categoric_features),  # Imputação nas colunas categóricas
            ],
            remainder='passthrough'  # Garante que as colunas não transformadas sejam preservadas e na mesma ordem
        )

        self.preprocessor.fit(X)
        return self

    def transform(self, X):
        X_imputed = self.preprocessor.transform(X)
        
        # Usando get_feature_names_out para garantir que temos os nomes das colunas corretos
        col_names = self.preprocessor.get_feature_names_out()
        
        return to_dataframe_transformer(X_imputed, column_names=col_names)
    
def create_imputation_pipeline(include_outlier_removal=True):
    # Pipeline para a imputação e pré-processamento adicional
    steps = [
        ('imputer', ImputeDataFrame()),
        ('log_transform', LogTransformer()),
        ('one_hot_encoder', CustomOneHotEncoder()),
        ('scaler', ScalerTransformer(scaler='robust')),
    ]
    if include_outlier_removal:
        # Inserir o OutlierRemover após a imputação
        steps.insert(1, ('outlier_removal', OutlierRemover()))

    imputation_pipeline = Pipeline(steps=steps)
    return imputation_pipeline