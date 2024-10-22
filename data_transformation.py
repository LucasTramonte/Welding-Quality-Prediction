import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, multiplier=1.5):
        self.multiplier = multiplier
        self.numeric_columns = None
        self.lower_bounds = {}
        self.upper_bounds = {}
    
    def fit(self, X, y=None):
        self.numeric_columns = X.select_dtypes(include=[np.number]).columns
        for col in self.numeric_columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            self.lower_bounds[col] = Q1 - self.multiplier * IQR
            self.upper_bounds[col] = Q3 + self.multiplier * IQR
        return self
    
    def transform(self, X, y=None):
        X_clean = X.copy()
        mask = pd.Series(True, index=X_clean.index)
        for col in self.numeric_columns:
            lower = self.lower_bounds[col]
            upper = self.upper_bounds[col]
            mask &= (X_clean[col] >= lower) & (X_clean[col] <= upper)
        X_clean = X_clean[mask].reset_index(drop=True)
        return X_clean

class ScalerTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
        self.scaler = StandardScaler().fit(X[self.numeric_features])
        return self

    def transform(self, X, y=None):
        X[self.numeric_features] = self.scaler.transform(X[self.numeric_features])
        return X


class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.categories_ = {}

    def fit(self, X, y=None):
        # Armazena as categorias únicas para cada coluna categórica no treino
        for col in X.select_dtypes(include=['object', 'category']).columns:
            self.categories_[col] = X[col].dropna().unique().tolist()
        return self

    def transform(self, X, y=None):
        X_transformed = X.copy()
        for col, categories in self.categories_.items():
            # Criar dummies apenas para as categorias já presentes no treino
            dummies = np.zeros((X_transformed.shape[0], len(categories)), dtype=int)
            for i, category in enumerate(categories):
                dummies[:, i] = (X_transformed[col] == category).astype(int)

            # Adicionar os dummies ao DataFrame com o nome correto das colunas
            dummy_df = pd.DataFrame(dummies, columns=[f"{col}_{cat}" for cat in categories], index=X_transformed.index)
            X_transformed = pd.concat([X_transformed, dummy_df], axis=1)

        # Remover as colunas categóricas originais
        X_transformed = X_transformed.drop(columns=self.categories_.keys())
        return X_transformed