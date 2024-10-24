import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator

# Transformer to remove highly correlated features
class RemoveHighCorrelation(BaseEstimator, TransformerMixin):
    def __init__(self, threshold):
        self.threshold = threshold

    # Fit method to identify features with high correlation
    def fit(self, X, y=None):
        # Calculate the correlation matrix
        corr_matrix = X.corr().abs()
        # Create a mask for the upper triangle of the correlation matrix
        upper_tri = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        upper_tri_corr = corr_matrix.where(upper_tri)
        # Identify columns that exceed the correlation threshold
        self.to_drop_ = [column for column in upper_tri_corr.columns if any(upper_tri_corr[column] > self.threshold)]
        return self

    # Transform method to drop highly correlated features
    def transform(self, X):
        return X.drop(columns=self.to_drop_, errors='ignore')


# Transformer to apply logarithmic transformation to highly skewed features
class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold_skew=0.5):
        self.threshold_skew = threshold_skew
        self.columns_to_log = []

    # Fit method to identify skewed features that need log transformation
    def fit(self, X, y=None):
        self.numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
        # Calculate skewness for numeric features
        skewness = X[self.numeric_features].skew()
        # Identify columns with only positive values (log transformation only works on positive values)
        positive_cols = [col for col in self.numeric_features if X[col].min() > 0]
        # Select features with skewness above the threshold and positive values
        self.columns_to_log = skewness[skewness > self.threshold_skew].index.intersection(positive_cols)
        return self

    # Transform method to apply log transformation to the selected columns
    def transform(self, X):
        X_transformed = X.copy()
        if not self.columns_to_log.empty:
            # Apply log transformation (log1p adds 1 to avoid log(0) errors)
            X_transformed[self.columns_to_log] = np.log1p(X_transformed[self.columns_to_log])
        return X_transformed