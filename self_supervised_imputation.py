import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

class SelfSupervisedImputer:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    def _convert_y_values(self, y):
        """
        Converte valores não numéricos em y para um valor padrão e garante que todos sejam numéricos.
        """
        y = y.replace('<0.01', 0.01)
        y = pd.to_numeric(y, errors='coerce')
        return y

    def fit(self, X, y):
        """
        Treina o modelo para preencher valores ausentes em y baseado nos dados X.
        """
        y = self._convert_y_values(y)
        X_train = X[~y.isna()]
        y_train = y[~y.isna()]
        self.model.fit(X_train, y_train)
        
    def transform(self, X, y):
        """
        Preenche os valores ausentes em y baseado em X usando o modelo treinado.
        """
        y = self._convert_y_values(y)
        missing_indices = y[y.isna()].index

        if len(missing_indices) > 0:
            X_missing = X.loc[missing_indices]
            y_pred = self.model.predict(X_missing)
            y.loc[missing_indices] = y_pred

        return y

    def fit_transform(self, X, y):
        """
        Combina o treino e a imputação dos valores ausentes em y.
        """
        self.fit(X, y)
        return self.transform(X, y)

def impute_target(y_train, method='zero', X_train=None):
    """
    Impute missing values in the target variable using specified methods.
    """
    print(f'y_train shape: {y_train.shape} \n y_train number of NaN before imputing {y_train.isnull().sum()}')
    if method == 'mean':
        y_train_imputed = y_train.fillna(y_train.mean())
    elif method == 'median':
        y_train_imputed = y_train.fillna(y_train.median())
    elif method == 'ffill':
        y_train_imputed = y_train.fillna(method='ffill')
    elif method == 'zero':
        y_train_imputed = y_train.fillna(0)
    elif method == 'knn':
        # Assumindo que detect_type_columns é uma função que retorna apenas as colunas numéricas
        X_train_numeric = detect_type_columns(X_train)
        imputer = IterativeImputer(estimator=RandomForestRegressor(), max_iter=10, random_state=42)
        y_train_reshaped = y_train.values.reshape(-1, 1)
        X_and_y_train = np.hstack([X_train_numeric, y_train_reshaped])
        X_and_y_train_imputed = imputer.fit_transform(X_and_y_train)
        y_train_imputed = pd.Series(X_and_y_train_imputed[:, -1].ravel())
    
    print("Number of NaN after imputation:", y_train_imputed.isna().sum())
    return y_train_imputed

def detect_type_columns(df, type='numeric'):
    numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
    categoric_features = df.select_dtypes(include=['object', 'category']).columns
    return df[numeric_features] if type == 'numeric' else df[categoric_features]