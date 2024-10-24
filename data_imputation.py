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

from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

# Custom class for imputing missing values in a DataFrame
class ImputeDataFrame(BaseEstimator, TransformerMixin):
    def __init__(self, use_all_features=True):
        """
        use_all_features: if True, uses all available features to impute missing values in each group.
        """
        self.use_all_features = use_all_features

    # Fit method to adjust imputation pipelines for different types of features
    def fit(self, X, y=None):
        """
        Fits imputation pipelines for different types of features.
        """
        self.numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
        self.categoric_features = X.select_dtypes(include=['object', 'category']).columns

        # Calculate the percentage of missing values in each column
        percent_n_missing = X.isnull().mean()

        # Define columns with more than 60% missing values for Random Forest imputation
        self.rf_missing = percent_n_missing[percent_n_missing > 0.5].index
        self.rf_missing = list(set(self.rf_missing) - set(self.categoric_features))

        # Define columns with between 10% and 60% missing values for KNN imputation
        self.knn_missing = percent_n_missing[(percent_n_missing > 0.1) & (percent_n_missing <= 0.5)].index
        self.knn_missing = list(set(self.knn_missing) - set(self.categoric_features))

        # Define columns with less than 10% missing values for median imputation
        self.median_missing = percent_n_missing[percent_n_missing <= 0.1].index
        self.median_missing = list(set(self.median_missing) - set(self.categoric_features))

        # Pipelines for the three groups of numerical features
        self.rf_pipeline = IterativeImputer(estimator=RandomForestRegressor(random_state=42), max_iter=10, random_state=42)
        self.knn_pipeline = KNNImputer(n_neighbors=10)
        self.median_pipeline = SimpleImputer(strategy='median')

        # Pipeline for categorical features
        self.categorical_pipeline = SimpleImputer(strategy='most_frequent')

        # Fit the pipelines (using all features or just the group features)
        if self.rf_missing:
            self.rf_pipeline.fit(X[self.rf_missing])
        if self.knn_missing:
            self.knn_pipeline.fit(X[self.knn_missing])
        if self.median_missing:
            self.median_pipeline.fit(X[self.median_missing])
        self.categorical_pipeline.fit(X[self.categoric_features])

        return self

    # Transform method to apply imputation
    def transform(self, X):
        """
        Applies imputation to the selected groups of features.
        """
        X_transformed = X.copy()

        # Impute columns with more than 60% missing values (Random Forest)
        if self.rf_missing:
            X_transformed[self.rf_missing] = pd.DataFrame(self.rf_pipeline.transform(X[self.rf_missing]), 
                                                          columns=self.rf_missing, 
                                                          index=X.index)

        # Impute columns with between 10% and 60% missing values (KNN)
        if self.knn_missing:
            X_transformed[self.knn_missing] = pd.DataFrame(self.knn_pipeline.transform(X[self.knn_missing]), 
                                                           columns=self.knn_missing, 
                                                           index=X.index)

        # Impute columns with less than 10% missing values (Median)
        if self.median_missing:
            X_transformed[self.median_missing] = pd.DataFrame(self.median_pipeline.transform(X[self.median_missing]), 
                                                              columns=self.median_missing, 
                                                              index=X.index)

        # Impute categorical columns
        X_transformed[self.categoric_features] = pd.DataFrame(self.categorical_pipeline.transform(X[self.categoric_features]), 
                                                              columns=self.categoric_features, 
                                                              index=X.index)

        # Print the total number of missing values after imputation
        print("Total number of missing values after imputation:", X_transformed.isna().sum().sum())

        return X_transformed
    

# Function to create a pipeline for imputation and additional preprocessing
def create_imputation_pipeline(include_outlier_removal=True):
    # Steps for imputation and additional preprocessing
    steps = [
        ('imputer', ImputeDataFrame()),  # Impute missing values
        ('log_transform', LogTransformer()),  # Apply log transformation
        ('one_hot_encoder', CustomOneHotEncoder()),  # Apply one-hot encoding
        ('scaler', ScalerTransformer(scaler='robust')),  # Apply scaling (Robust Scaler)
    ]
    if include_outlier_removal:
        # Insert OutlierRemover after imputation
        steps.insert(1, ('outlier_removal', OutlierRemover()))

    # Create and return the imputation pipeline
    imputation_pipeline = Pipeline(steps=steps)
    return imputation_pipeline