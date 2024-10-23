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
    def __init__(self, use_all_features=True):
        """
        use_all_features: se True, usa todas as features disponíveis para imputar valores faltantes em cada grupo.
        """
        self.use_all_features = use_all_features

    def fit(self, X, y=None):
        """
        Ajusta os pipelines de imputação para diferentes tipos de features.
        """
        self.numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
        self.categoric_features = X.select_dtypes(include=['object', 'category']).columns

        # Calcular a porcentagem de valores faltantes
        percent_n_missing = X.isnull().mean()

        # Definir colunas de concentração (exceto Phosphorus e Sulphur)
        self.concentration_features = [col for col in self.numeric_features if "concentration" in col and not("Phosphorus" in col or "Sulphur" in col)]
        
        # Definir as colunas com valores faltantes < 10%
        self.low_missing = percent_n_missing[(percent_n_missing <= 0.1)].index
        self.low_missing = list(set(self.low_missing) - set(self.categoric_features))
        
        self.mid_high_missing = percent_n_missing[(percent_n_missing > 0.1)].index
        self.mid_high_missing = list(set(self.mid_high_missing) - set(self.categoric_features))

        # Definir as colunas com valores faltantes entre 10% e 50%
        # self.mid_missing = percent_n_missing[(percent_n_missing > 0.1)].index
        # self.mid_missing = list(set(self.mid_missing) - set(self.categoric_features))
        
        # Definir as colunas com mais de 50% de valores faltantes
        # self.high_missing = percent_n_missing[(percent_n_missing > 0.5)].index
        # self.high_missing = list(set(self.high_missing) - set(self.categoric_features))

        # Pipelines para cada grupo de features
        self.low_missing_pipeline = SimpleImputer(strategy='median')
        self.mid_high_missing_pipeline = KNNImputer(n_neighbors=10)
        # self.mid_missing_pipeline = KNNImputer(n_neighbors=10)
        # self.high_missing_pipeline = KNNImputer(n_neighbors=10)
        # self.high_missing_pipeline = IterativeImputer(estimator=RandomForestRegressor(random_state=42), random_state=42)
        self.concentration_pipeline = SimpleImputer(strategy='constant', fill_value=0)
        self.categorical_pipeline = SimpleImputer(strategy='most_frequent')

        # Ajustar os pipelines (usando todas as features ou apenas as do grupo)
        self.low_missing_pipeline.fit(X[self.low_missing])
        self.mid_high_missing_pipeline.fit(X[self.numeric_features] if self.use_all_features else X[self.mid_high_missing])

        # self.mid_missing_pipeline.fit(X[self.numeric_features] if self.use_all_features else X[self.mid_missing])
        # self.high_missing_pipeline.fit(X[self.numeric_features] if self.use_all_features else X[self.high_missing])
        self.concentration_pipeline.fit(X[self.concentration_features])
        self.categorical_pipeline.fit(X[self.categoric_features])

        return self

    def transform(self, X):
        """
        Aplica a imputação nos grupos de features selecionados.
        """
        X_transformed = X.copy()

        # Imputação nas colunas de baixo missing
        X_transformed[self.low_missing] = pd.DataFrame(self.low_missing_pipeline.transform(X[self.low_missing]), 
                                                       columns=self.low_missing, 
                                                       index=X.index)

        # Imputação nas colunas de mid missing
        if self.use_all_features:
            # mid_missing_transformed = pd.DataFrame(self.mid_missing_pipeline.transform(X[self.numeric_features]), 
            #                                        columns=self.numeric_features, 
            #                                        index=X.index)
            # X_transformed[self.mid_missing] = mid_missing_transformed[self.mid_missing]

            # high_missing_transformed = pd.DataFrame(self.high_missing_pipeline.transform(X[self.numeric_features]), 
            #                                         columns=self.numeric_features, 
            #                                         index=X.index)
            
            # X_transformed[self.high_missing] = high_missing_transformed[self.high_missing]

            mid_high_missing_transformed = pd.DataFrame(self.mid_high_missing_pipeline.transform(X[self.numeric_features]), 
                                                   columns=self.numeric_features, 
                                                   index=X.index)
            X_transformed[self.mid_high_missing] = mid_high_missing_transformed[self.mid_high_missing]

        else:
            X_transformed[self.mid_high_missing] = pd.DataFrame(self.mid_high_missing_pipeline.transform(X[self.mid_high_missing]), 
                                                           columns=self.mid_high_missing, 
                                                           index=X.index)
            # X_transformed[self.mid_missing] = pd.DataFrame(self.mid_missing_pipeline.transform(X[self.mid_missing]), 
            #                                                columns=self.mid_missing, 
            #                                                index=X.index)
            # X_transformed[self.high_missing] = pd.DataFrame(self.high_missing_pipeline.transform(X[self.high_missing]), 
            #                                                 columns=self.high_missing, 
            #                                                 index=X.index)         

        # Imputação nas colunas de concentração
        X_transformed[self.concentration_features] = pd.DataFrame(self.concentration_pipeline.transform(X[self.concentration_features]), 
                                                                  columns=self.concentration_features, 
                                                                  index=X.index)

        # Imputação nas colunas categóricas
        X_transformed[self.categoric_features] = pd.DataFrame(self.categorical_pipeline.transform(X[self.categoric_features]), 
                                                              columns=self.categoric_features, 
                                                              index=X.index)
        
        print(X_transformed.isna().sum().sum())

        return X_transformed
    
def create_imputation_pipeline(include_outlier_removal=True):
    # Pipeline para a imputação e pré-processamento adicional
    steps = [
        ('imputer', ImputeDataFrame()),
        # ('log_transform', LogTransformer()),
        ('one_hot_encoder', CustomOneHotEncoder()),
        ('scaler', ScalerTransformer(scaler='robust')),
    ]
    if include_outlier_removal:
        # Inserir o OutlierRemover após a imputação
        steps.insert(1, ('outlier_removal', OutlierRemover()))

    imputation_pipeline = Pipeline(steps=steps)
    return imputation_pipeline