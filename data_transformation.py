import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

class ScalerTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
        self.scaler = StandardScaler().fit(X[self.numeric_features])
        return self

    def transform(self, X, y=None):
        X[self.numeric_features] = self.scaler.transform(X[self.numeric_features])
        return X

class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.categories_ = {col: X[col].dropna().unique().tolist() for col in X.select_dtypes(include=['object', 'category']).columns}
        return self

    def transform(self, X, y=None):
        X_transformed = X.copy()
        for col, categories in self.categories_.items():
            dummies = pd.get_dummies(X_transformed[col], prefix=col)
            for category in categories:
                dummy_col = f"{col}_{category}"
                if dummy_col not in dummies.columns:
                    dummies[dummy_col] = 0
            X_transformed = X_transformed.drop(columns=[col]).join(dummies)
        return X_transformed