import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from data_preprocessing import FixDatatypeTransformer, DropNanColsTransformer
from feature_engineering import RemoveHighCorrelation, LogTransformer
from data_imputation import ImputeDataFrame
from data_transformation import ScalerTransformer, CustomOneHotEncoder
from data_load import load_data

def create_full_pipeline():
    full_pipeline = Pipeline(steps=[
        ('fix_datatype', FixDatatypeTransformer()),
        ('drop_nan_cols', DropNanColsTransformer(threshold=0.6)),
        ('imputer', ImputeDataFrame()),
        ('log_transform', LogTransformer()),
        ('scaler', ScalerTransformer()),
        # ('one_hot_encoder', CustomOneHotEncoder())
    ])

    return full_pipeline

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()

    # Criar o pipeline completo
    pipeline = create_full_pipeline()

    # Aplicar o pipeline
    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed = pipeline.transform(X_test)
    
    # # Treinar e avaliar o modelo
    # model = train_model(X_train_processed, y_train)
    # evaluate_model(model, X_test_processed, y_test)