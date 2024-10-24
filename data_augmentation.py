from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

# Class for augmenting numerical data by adding noise
class NumericalDataAugmentor(BaseEstimator, TransformerMixin):
    def __init__(self, noise_level=0.01, random_state=None):
        self.noise_level = noise_level
        self.random_state = random_state
        self.numeric_columns = None

    # Fit method to capture numeric columns
    def fit(self, X, y=None):
        self.numeric_columns = X.columns
        return self

    # Transform method to add noise to the numeric columns
    def transform(self, X, y=None):
        X_augmented = X.copy()
        rng = np.random.RandomState(self.random_state)  # Random number generator
        noise = rng.normal(loc=0, scale=self.noise_level, size=X_augmented.shape)  # Generate noise
        X_augmented += noise  # Add noise to the data
        return X_augmented
    
# Class for augmenting categorical data based on observed distributions
class CategoricalDataAugmentor(BaseEstimator, TransformerMixin):
    def __init__(self, random_state=None):
        self.random_state = random_state
        self.categorical_columns = None
        self.category_distributions = {}

    # Fit method to calculate category distributions
    def fit(self, X, y=None):
        self.categorical_columns = X.columns
        for col in self.categorical_columns:
            self.category_distributions[col] = X[col].value_counts(normalize=True)  # Compute category frequencies
        return self

    # Transform method to generate new samples based on category distributions
    def transform(self, X, y=None):
        X_augmented = pd.DataFrame(index=X.index, columns=self.categorical_columns)
        rng = np.random.RandomState(self.random_state)  # Random number generator
        n_samples = len(X)
        for col in self.categorical_columns:
            categories = self.category_distributions[col].index  # Get categories
            probabilities = self.category_distributions[col].values  # Get probabilities for each category
            X_augmented[col] = rng.choice(categories, size=n_samples, p=probabilities)  # Sample new data
        return X_augmented

# Main class for augmenting both numerical and categorical data
class DataAugmentor(BaseEstimator, TransformerMixin):
    def __init__(self, noise_level=0.01, augmentation_fraction=0.5, random_state=None):
        self.noise_level = noise_level
        self.augmentation_fraction = augmentation_fraction
        self.random_state = random_state
        self.numeric_augmentor = None
        self.categorical_augmentor = None
        self.numeric_columns = None
        self.categorical_columns = None

    # Fit method to initialize numeric and categorical augmentors
    def fit(self, X, y=None):
        self.numeric_columns = X.select_dtypes(include=[np.number]).columns  # Identify numeric columns
        self.categorical_columns = X.select_dtypes(include=['object', 'category']).columns  # Identify categorical columns

        # Initialize and fit numerical augmenter
        self.numeric_augmentor = NumericalDataAugmentor(
            noise_level=self.noise_level,
            random_state=self.random_state
        )
        self.numeric_augmentor.fit(X[self.numeric_columns])

        # Initialize and fit categorical augmenter
        self.categorical_augmentor = CategoricalDataAugmentor(
            random_state=self.random_state
        )
        self.categorical_augmentor.fit(X[self.categorical_columns])

        return self

    # Transform method to apply augmentation to the dataset
    def transform(self, X, y=None):
        X_numeric = X[self.numeric_columns]  # Extract numeric data
        X_categorical = X[self.categorical_columns]  # Extract categorical data
        n_samples = len(X)

        # Calculate the number of samples to augment
        n_augmented_samples = int(n_samples * self.augmentation_fraction)

        rng = np.random.RandomState(self.random_state)  # Random number generator
        # Select random indices for augmentation
        indices_to_augment = rng.choice(n_samples, size=n_augmented_samples, replace=True)

        # Get numeric and categorical data to augment
        X_numeric_to_augment = X_numeric.iloc[indices_to_augment].reset_index(drop=True)
        X_categorical_to_augment = X_categorical.iloc[indices_to_augment].reset_index(drop=True)

        # Augment numeric data by adding noise
        X_numeric_augmented = self.numeric_augmentor.transform(X_numeric_to_augment)

        # Augment categorical data by sampling new values
        X_categorical_augmented = self.categorical_augmentor.transform(X_categorical_to_augment)

        # Combine augmented numeric and categorical data
        X_augmented = pd.concat([X_numeric_augmented, X_categorical_augmented], axis=1)

        # Concatenate original data with augmented data
        X_final = pd.concat([X, X_augmented], ignore_index=True)

        return X_final

# Function to create a pipeline for data augmentation
def create_augmentation_pipeline(augmentation_fraction=0.3, noise_level=0.01, random_state=None):
    # Pipeline for data augmentation
    augmentation_pipeline = Pipeline(steps=[
        ('data_augmentor', DataAugmentor(
            noise_level=noise_level,
            augmentation_fraction=augmentation_fraction,
            random_state=random_state
        )),
    ])
    return augmentation_pipeline