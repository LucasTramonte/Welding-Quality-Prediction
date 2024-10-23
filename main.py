import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from data_preprocessing import create_preprocessing_pipeline
from data_imputation import create_imputation_pipeline
from data_load import load_data, train_test_split_balance
from data_augmentation import create_augmentation_pipeline
from data_pca import create_pca_preprocessor
from utils import timing_decorator
from model_training import ModelTrainer
from metrics_evaluation import MetricsEvaluator
from self_supervised_imputation import impute_target
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import time
from functools import wraps

def preprocess_data(X_train, X_test):
    preprocessing_pipeline = create_preprocessing_pipeline()
    X_train_preprocessed = preprocessing_pipeline.fit_transform(X_train)
    X_test_preprocessed = preprocessing_pipeline.transform(X_test)
    return X_train_preprocessed, X_test_preprocessed

def impute_data(X_train_preprocessed, X_test_preprocessed, include_outlier_removal=False):
    imputation_pipeline_train = create_imputation_pipeline(include_outlier_removal=include_outlier_removal)
    imputation_pipeline_test = create_imputation_pipeline(include_outlier_removal=include_outlier_removal)
    
    X_train_imputed = imputation_pipeline_train.fit_transform(X_train_preprocessed)
    imputation_pipeline_test.fit(X_train_preprocessed)
    X_test_imputed = imputation_pipeline_test.transform(X_test_preprocessed)
    
    return X_train_imputed, X_test_imputed

def augment_data(X_train_imputed, augmentation_fraction=0.4, noise_level=0.02, random_state=42):
    augmentation_pipeline = create_augmentation_pipeline(
        augmentation_fraction=augmentation_fraction, noise_level=noise_level, random_state=random_state)
    X_train_augmented = augmentation_pipeline.fit_transform(X_train_imputed)
    X_train_augmented.reset_index(drop=True, inplace=True)
    return X_train_augmented

def impute_target_data(y_train, X_train_augmented):
    n_original_samples = len(y_train)
    n_augmented_samples = len(X_train_augmented)

    y_train_extended = pd.concat([
        y_train,
        pd.Series([np.nan] * (n_augmented_samples - n_original_samples))
    ], ignore_index=True)

    y_train_augmented = impute_target(y_train_extended, method='knn', X_train=X_train_augmented)
    
    assert X_train_augmented.index.equals(y_train_augmented.index), "Os índices de X e y não estão alinhados após a imputação."
    
    return y_train_augmented

def apply_pca(X_train_augmented, y_train_augmented):
    preprocessor, high_corr_features, low_corr_features, binary_features = create_pca_preprocessor(
        X_train_augmented, y_train_augmented, correlation_threshold=0.075, n_components=2, plot=False)

    preprocessor.fit(X_train_augmented)
    X_train_final = preprocessor.transform(X_train_augmented)
    
    return preprocessor, X_train_final

def train_and_evaluate_model(X_train_final, y_train_augmented, X_test_final, y_test):
    model_trainer = ModelTrainer(preprocessor=None)
    model_trainer.train_models(X_train_final, y_train_augmented)
    
    pipelines = model_trainer.get_best_estimators()
    metrics_evaluator = MetricsEvaluator()
    results_df = metrics_evaluator.evaluate_models(pipelines, X_test_final, y_test)
    print(results_df)

# Flags para controle de etapas
RUN_INITIAL_PREPROCESS = True
RUN_IMPUTE = True
RUN_AUGMENTATION = False
RUN_IMPUTE_Y = True
RUN_PCA = True
RUN_TRAIN_MODEL = True
RUN_EVALUATE_MODEL = True

if __name__ == "__main__":
    # 1. Load the data
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split_balance(X, y, test_size=0.2, random_state=42)

    # 2. Apply initial preprocessing to X_train and X_test
    if RUN_INITIAL_PREPROCESS:
        X_train, X_test = preprocess_data(X_train, X_test)
        print("Initial preprocessing applied.")
    else:
        pass

    # 3. Apply imputation
    if RUN_IMPUTE:
        X_train, X_test = impute_data(X_train, X_test)
        print("Imputation applied.")
    else:
        pass

    # 4. Apply data augmentation
    if RUN_AUGMENTATION:
        X_train = augment_data(X_train)
        print("Data augmentation applied.")
    else:
        pass

    # 5. Impute missing values in y_train using preprocessed X_train
    if RUN_IMPUTE_Y:
        y_train= impute_target_data(y_train, X_train)
        print("Target imputation applied.")
    else:
        pass

    # 6. Apply PCA
    if RUN_PCA:
        print(X_train.shape)
        print(X_test.shape)
        preprocessor, X_train = apply_pca(X_train, y_train)
        X_test = preprocessor.transform(X_test)
        print("PCA applied.")
    else:
        pass

    # 7. Train the model and evaluate
    if RUN_TRAIN_MODEL and RUN_EVALUATE_MODEL:
        train_and_evaluate_model(X_train, y_train, X_test, y_test)
        print("Model training and evaluation completed.")