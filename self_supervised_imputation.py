import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

class SelfSupervisedImputer:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    def _convert_y_values(self, y):
        """
        Converte valores não numéricos em y para um valor padrão e garante que todos sejam numéricos.
        """
        # Substituir valores como '<0.01' por um valor numérico adequado, como 0.01
        y = y.replace('<0.01', 0.01)

        # Garantir que todos os valores sejam convertidos para numérico; valores não convertíveis se tornam NaN
        y = pd.to_numeric(y, errors='coerce')
        
        return y

    def fit(self, X, y):
        """
        Treina o modelo para preencher valores ausentes em y baseado nos dados X.
        """
        # Converter valores de y para numérico
        y = self._convert_y_values(y)

        # Separar dados onde y não é NaN para treinamento
        X_train = X[~y.isna()]
        y_train = y[~y.isna()]

        # Treinar modelo com os dados disponíveis
        self.model.fit(X_train, y_train)
        
    def transform(self, X, y):
        """
        Preenche os valores ausentes em y baseado em X usando o modelo treinado.
        """
        # Converter valores de y para numérico
        y = self._convert_y_values(y)

        # Identificar os índices onde y é NaN
        missing_indices = y[y.isna()].index

        # Prever os valores ausentes
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
