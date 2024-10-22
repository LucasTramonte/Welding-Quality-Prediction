from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class NumericalDataAugmentor(BaseEstimator, TransformerMixin):
    def __init__(self, noise_level=0.01, random_state=None):
        self.noise_level = noise_level
        self.random_state = random_state
        self.numeric_columns = None

    def fit(self, X, y=None):
        self.numeric_columns = X.columns
        return self

    def transform(self, X, y=None):
        X_augmented = X.copy()
        rng = np.random.RandomState(self.random_state)
        noise = rng.normal(loc=0, scale=self.noise_level, size=X_augmented.shape)
        X_augmented += noise
        return X_augmented
    
class CategoricalDataAugmentor(BaseEstimator, TransformerMixin):
    def __init__(self, random_state=None):
        self.random_state = random_state
        self.categorical_columns = None
        self.category_distributions = {}

    def fit(self, X, y=None):
        self.categorical_columns = X.columns
        for col in self.categorical_columns:
            self.category_distributions[col] = X[col].value_counts(normalize=True)
        return self

    def transform(self, X, y=None):
        X_augmented = pd.DataFrame(index=X.index, columns=self.categorical_columns)
        rng = np.random.RandomState(self.random_state)
        n_samples = len(X)
        for col in self.categorical_columns:
            categories = self.category_distributions[col].index
            probabilities = self.category_distributions[col].values
            X_augmented[col] = rng.choice(categories, size=n_samples, p=probabilities)
        return X_augmented
 
class DataAugmentor(BaseEstimator, TransformerMixin):
    def __init__(self, noise_level=0.01, augmentation_fraction=0.5, random_state=None):
        self.noise_level = noise_level
        self.augmentation_fraction = augmentation_fraction
        self.random_state = random_state
        self.numeric_augmentor = None
        self.categorical_augmentor = None
        self.numeric_columns = None
        self.categorical_columns = None

    def fit(self, X, y=None):
        self.numeric_columns = X.select_dtypes(include=[np.number]).columns
        self.categorical_columns = X.select_dtypes(include=['object', 'category']).columns

        self.numeric_augmentor = NumericalDataAugmentor(
            noise_level=self.noise_level,
            random_state=self.random_state
        )
        self.numeric_augmentor.fit(X[self.numeric_columns])

        self.categorical_augmentor = CategoricalDataAugmentor(
            random_state=self.random_state
        )
        self.categorical_augmentor.fit(X[self.categorical_columns])

        return self

    def transform(self, X, y=None):
        X_numeric = X[self.numeric_columns]
        X_categorical = X[self.categorical_columns]
        n_samples = len(X)

        # Calcular o número de amostras aumentadas
        n_augmented_samples = int(n_samples * self.augmentation_fraction)

        rng = np.random.RandomState(self.random_state)
        # Selecionar índices aleatórios para aumentar
        indices_to_augment = rng.choice(n_samples, size=n_augmented_samples, replace=True)

        # Obter amostras para aumentar
        X_numeric_to_augment = X_numeric.iloc[indices_to_augment].reset_index(drop=True)
        X_categorical_to_augment = X_categorical.iloc[indices_to_augment].reset_index(drop=True)

        # Aumentar dados numéricos
        X_numeric_augmented = self.numeric_augmentor.transform(X_numeric_to_augment)

        # Aumentar dados categóricos
        X_categorical_augmented = self.categorical_augmentor.transform(X_categorical_to_augment)

        # Combinar dados numéricos e categóricos aumentados
        X_augmented = pd.concat([X_numeric_augmented, X_categorical_augmented], axis=1)

        # Concatenar dados originais com dados aumentados
        X_final = pd.concat([X, X_augmented], ignore_index=True)

        return X_final