import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator

class RemoveHighCorrelation(BaseEstimator, TransformerMixin):
    def __init__(self, threshold):
        self.threshold = threshold

    def fit(self, X, y=None):
        corr_matrix = X.corr().abs()
        upper_tri = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        upper_tri_corr = corr_matrix.where(upper_tri)
        self.to_drop_ = [column for column in upper_tri_corr.columns if any(upper_tri_corr[column] > self.threshold)]
        return self

    def transform(self, X):
        return X.drop(columns=self.to_drop_, errors='ignore')

class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold_skew=0.5):
        self.threshold_skew = threshold_skew
        self.columns_to_log = []

    def fit(self, X, y=None):
        self.numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
        skewness = X[self.numeric_features].skew()
        positive_cols = [col for col in self.numeric_features if X[col].min() > 0]
        self.columns_to_log = skewness[skewness > self.threshold_skew].index.intersection(positive_cols)
        return self

    def transform(self, X):
        X_transformed = X.copy()
        if not self.columns_to_log.empty:
            X_transformed[self.columns_to_log] = np.log1p(X_transformed[self.columns_to_log])
        return X_transformed
