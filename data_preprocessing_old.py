import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from imputer_new import impute_data, impute_target, random_perturbation, sample_categories, data_augmentation 
from feature_selection import apply_feature_selection  
from pca import apply_pca  


class DataPreprocessor:
    def __init__(self, data):
        self.data = data.copy()
        self.preprocessor = None
        self.pca_model = None

    def preprocess_data(self, target_column, categoric_features=None, features_to_drop=None, drop_columns=None, y_imputation='zero', feature_selection_methods=None, pca=False, pca_n_components=None, data_augmentation=False):
        """
        Perform data preprocessing including handling categorical features, imputing missing values, and splitting the data.
        """
        # Remove specified columns
        if drop_columns:
            self.data = self.data.drop(columns=drop_columns)
        
        # Convert columns to numeric where possible
        if categoric_features:
            numeric_features = [col for col in self.data.columns if col not in ['Weld ID'] + categoric_features]
        else:
            numeric_features = [col for col in self.data.columns if col != 'Weld ID']
            
        data_categoric = self.data[categoric_features] if categoric_features else pd.DataFrame()
        data_numeric = self.data[numeric_features].apply(pd.to_numeric, errors='coerce')
        
        # Concatenate dataframes and remove duplicates
        self.data = pd.concat([data_numeric, data_categoric], axis=1).drop_duplicates(keep='last')
        
        # Split data into features (X) and target (y)
        X = self.data.drop(columns=features_to_drop)
        y = self.data[target_column]
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Dropping missing values in test
        X_test = X_test[~y_test.isna()]
        y_test = y_test.dropna().astype(float)
        X_test_null = X_test[y_test.isna()]
        y_test_null = y_test[y_test.isna()]

        # Adding old y_test data with missing values to train data
        X_train = pd.concat([X_train, X_test_null], axis=0)
        y_train = pd.concat([y_train, y_test_null], axis=0)

        X_train = X_train.loc[~X_train.index.duplicated(keep='first')]
        X_test = X_test.loc[~X_test.index.duplicated(keep='first')]
                
        # Handle categorical features
        if categoric_features:
            X_train[categoric_features] = X_train[categoric_features].astype(str)
            X_test[categoric_features] = X_test[categoric_features].astype(str)

        # Impute missing values in X using the new imputer function
        X_train, X_test = impute_data(X_train, X_test)

        # Aplicar Data Augmentation no DataFrame original
        if data_augmentation == True:
            X_train = data_augmentation(X_train, perturbation_factor=0.05)

        # Impute missing values for the target variable (y) using the new function
        y_train = impute_target(y_train, method=y_imputation, X_train=X_train)
        
        # Apply feature selection if methods are provided
        if feature_selection_methods:
            X_train, X_test = apply_feature_selection(X_train, X_test, y_train, feature_selection_methods)
        
        # Apply PCA if specified
        if pca:
            X_train, X_test, self.pca_model = apply_pca(X_train, X_test, n_components=pca_n_components, plot_variance=True)
        
        # Apply OneHotEncoding to categorical variables
        if categoric_features:
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categoric_features),
                    ('num', StandardScaler(), [col for col in X_train.columns if col not in categoric_features])
                ]
            )
        else:
            self.preprocessor = ColumnTransformer(
                transformers=[('num', StandardScaler(), X_train.columns)]
            )
        
        return X_train, X_test, y_train, y_test, self.preprocessor
                    

