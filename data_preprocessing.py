# data_preprocessing.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_regression, SelectKBest, RFECV
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor

class DataPreprocessor:
    def __init__(self, data):
        self.data = data.copy()
        self.preprocessor = None
        self.pca_model = None

    def preprocess_data(self, target_column, categoric_features=None, features_to_drop=None, drop_columns=None, y_imputation='zero', feature_selection_methods=None, pca=False, pca_n_components=None):
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
        
        # Handle categorical features
        if categoric_features:
            X_train[categoric_features] = X_train[categoric_features].astype(str)
            X_test[categoric_features] = X_test[categoric_features].astype(str)
            
        # Impute missing values for the target variable (y)
        if y_imputation == 'mean':
            y_train = y_train.fillna(y_train.mean())
            y_test = y_test.fillna(y_train.mean())
        elif y_imputation == 'median':
            y_train = y_train.fillna(y_train.median())
            y_test = y_test.fillna(y_train.median())
        elif y_imputation == 'ffill':
            y_train = y_train.fillna(method='ffill')
            y_test = y_test.fillna(method='ffill')
        elif y_imputation == 'zero':
            y_train = y_train.fillna(0)
            y_test = y_test.fillna(0)
            
        # Impute missing values for specific columns
        for col in ['Phosphorus concentration (weight%)', 'Sulphur concentration (weight%)']:
            if col in X_train.columns:
                mean_value = X_train[col].mean()
                X_train[col] = X_train[col].fillna(mean_value)
                X_test[col] = X_test[col].fillna(mean_value)
                
        # For all other columns, replace missing values with 0
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)
        
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
            
        # Apply feature selection methods if provided
        if feature_selection_methods:
            for method_info in feature_selection_methods:
                method = method_info['method']
                params = method_info.get('params', {})
                
                if method == 'SelectKBest':
                    k = params.get('k', 10)
                    select_kbest = SelectKBest(f_regression, k=k).fit(X_train, y_train)
                    X_train = select_kbest.transform(X_train)
                    X_test = select_kbest.transform(X_test)
                    
                elif method == 'RFECV':
                    estimator = params.get('estimator', RandomForestRegressor())
                    step = params.get('step', 1)
                    cv = params.get('cv', 5)
                    scoring = params.get('scoring', 'r2')
                    
                    rfecv = RFECV(estimator=estimator, step=step, cv=cv, scoring=scoring)
                    rfecv = rfecv.fit(X_train, y_train)
                    
                    X_train = rfecv.transform(X_train)
                    X_test = rfecv.transform(X_test)
                    
        # Apply PCA if specified
        if pca:
            pca_n_components = pca_n_components or min(X_train.shape[1], X_train.shape[0])
            self.pca_model = PCA(n_components=pca_n_components)
            
            X_train = self.pca_model.fit_transform(X_train)
            X_test = self.pca_model.transform(X_test)
            
            # Optionally, plot cumulative explained variance
            cumsum = np.cumsum(self.pca_model.explained_variance_ratio_) * 100
            plt.figure(figsize=(6, 6))
            plt.plot(cumsum, color='red', label='Cumulative Explained Variance')
            plt.title('Cumulative Explained Variance as a Function of the Number of Components')
            plt.ylabel('Cumulative Explained Variance (%)')
            plt.xlabel('Number of Principal Components')
            plt.axhline(y=95, color='k', linestyle='--', label='95% Explained Variance')
            plt.legend(loc='best')
            plt.show()
            
        return X_train, X_test, y_train, y_test, self.preprocessor
