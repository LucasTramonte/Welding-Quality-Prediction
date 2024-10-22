import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from data_preprocessing import FixDatatypeTransformer, DropNanColsTransformer
from feature_engineering import RemoveHighCorrelation, LogTransformer
from data_imputation import ImputeDataFrame
from data_transformation import ScalerTransformer, CustomOneHotEncoder, OutlierRemover
from data_load import load_data, train_test_split_balance
from data_augmentation import DataAugmentor

def create_preprocessing_pipeline():
    # Pipeline para o pré-processamento (imputação, etc.)
    preprocessing_pipeline = Pipeline(steps=[
        ('fix_datatype', FixDatatypeTransformer()),
        ('drop_nan_cols', DropNanColsTransformer(threshold=0.6)),
        # A imputação será feita após a divisão dos dados
    ])
    return preprocessing_pipeline

def create_imputation_pipeline(include_outlier_removal=True):
    # Pipeline para a imputação e pré-processamento adicional
    steps = [
        ('imputer', ImputeDataFrame()),
        ('log_transform', LogTransformer()),
        ('scaler', ScalerTransformer()),
        ('one_hot_encoder', CustomOneHotEncoder()),
    ]
    if include_outlier_removal:
        # Inserir o OutlierRemover após a imputação
        steps.insert(1, ('outlier_removal', OutlierRemover()))

    imputation_pipeline = Pipeline(steps=steps)
    return imputation_pipeline

def create_augmentation_pipeline(augmentation_fraction=0.5, noise_level=0.01, random_state=None):
    # Pipeline para data augmentation
    augmentation_pipeline = Pipeline(steps=[
        ('data_augmentor', DataAugmentor(
            noise_level=noise_level,
            augmentation_fraction=augmentation_fraction,
            random_state=random_state
        )),
    ])
    return augmentation_pipeline


if __name__ == "__main__":
    # 1. Carregar os dados
    X, y = load_data()  # Supondo que a função load_data retorna X e y

    # 2. Dividir os dados em treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split_balance(X, y, test_size=0.2, random_state=42)

    # 3. Aplicar o pré-processamento inicial nos dados de treinamento
    preprocessing_pipeline = create_preprocessing_pipeline()
    X_train_preprocessed = preprocessing_pipeline.fit_transform(X_train)

    # 4. Aplicar o pré-processamento inicial nos dados de teste
    X_test_preprocessed = preprocessing_pipeline.transform(X_test)

    # 5. Criar pipelines de imputação separados
    # Pipeline de imputação para o treinamento (com remoção de outliers)
    imputation_pipeline_train = create_imputation_pipeline(include_outlier_removal=False)
    # Pipeline de imputação para o teste (sem remoção de outliers)
    imputation_pipeline_test = create_imputation_pipeline(include_outlier_removal=False)

    # 6. Aplicar a imputação e pré-processamento nos dados de treinamento
    X_train_imputed = imputation_pipeline_train.fit_transform(X_train_preprocessed)

    # 7. Sincronizar y_train após a remoção de outliers
    # Obter os índices das amostras restantes
    X_train_imputed_indices = X_train_imputed.index
    y_train_imputed = y_train.reset_index(drop=True).iloc[X_train_imputed_indices].reset_index(drop=True)

    # 8. Aplicar a imputação e pré-processamento nos dados de teste (sem remover outliers)
    imputation_pipeline_test.fit(X_train_preprocessed)
    X_test_imputed = imputation_pipeline_test.transform(X_test_preprocessed)


    # 8. Aplicar o data augmentation nos dados de treinamento
    augmentation_pipeline = create_augmentation_pipeline(
        augmentation_fraction=0.5, noise_level=0.05, random_state=42)
    X_train_augmented = augmentation_pipeline.fit_transform(X_train_imputed)

    # Aumentar y_train de forma correspondente
    n_augmented_samples = len(X_train_augmented) - len(X_train_imputed)
    y_train_augmented = pd.concat([
        y_train_imputed,
        y_train_imputed.sample(n=n_augmented_samples, replace=True, random_state=42).reset_index(drop=True)
    ], ignore_index=True)

    print(X_train_augmented.shape)
    print(y_train_augmented.shape)

    # 9. Treinar o modelo com X_train_augmented e y_train_augmented
    

    # 10. Avaliar o modelo usando X_test_imputed e y_test