import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # Ativa o IterativeImputer
from sklearn.impute import IterativeImputer


def impute_data(X_train, X_test):
    """
    Impute missing values in the dataset.
    - X_train: training set
    - X_test: test set
    Returns the imputed X_train and X_test.
    """

    # Impute missing values for specific columns
    for col in ['Phosphorus concentration (weight%)', 'Sulphur concentration (weight%)']:
        if col in X_train.columns:
            mean_value = X_train[col].mean()
            X_train[col] = X_train[col].fillna(mean_value)
            X_test[col] = X_test[col].fillna(mean_value)
            
    # For all other columns, replace missing values with 0
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    return X_train, X_test

def impute_target(y_train, method='zero', X_train=None):
    """
    Impute missing values in the target variable.
    - y_train: target values for training set
    - y_test: target values for testing set
    - method: the imputation method ('mean', 'median', 'ffill', or 'zero')
    Returns the imputed y_train and y_test.
    """
    print(f'y_train shape: {y_train.shape} \n y_train number of NaN before imputing {y_train.isnull().sum()}')
    if method == 'mean':
        print(y_train.mean())
        y_train_imputed = y_train.fillna(y_train.mean())
        # y_test = y_test.fillna(y_train.mean())
    elif method == 'median':
        y_train_imputed = y_train.fillna(y_train.median())
        # y_test = y_test.fillna(y_train.median())
    elif method == 'ffill':
        y_train_imputed = y_train.fillna(method='ffill')
        # y_test = y_test.fillna(method='ffill')
    elif method == 'zero':
        y_train_imputed = y_train.fillna(0)
        # y_test = y_test.fillna(0)
    elif method == 'knn':
        X_train_mumeric = detect_type_columns(X_train)
       # Criar o IterativeImputer com um modelo base (RandomForestRegressor)
        imputer = IterativeImputer(estimator=RandomForestRegressor(), max_iter=10, random_state=42)

        # Ajustar o imputer aos dados de X_train e y_train (imputando os valores faltantes de y_train)
        # Aqui, X_train e y_train são combinados para o processo de imputação
        y_train_reshaped = y_train.reshape(-1, 1)  # Reshape necessário para o IterativeImputer
        X_and_y_train = np.hstack([X_train_mumeric, y_train_reshaped])  # Combinando X_train e y_train

        # Imputar os valores faltantes
        X_and_y_train_imputed = imputer.fit_transform(X_and_y_train)

        # Separar as features e os labels novamente após a imputação
        y_train_imputed = X_and_y_train_imputed[:, -1]  # Labels imputados (y_train)

        y_train_imputed = pd.Series(y_train_imputed.ravel())
    
    print("Number of NaN:", y_train_imputed.isna().sum())

    return y_train_imputed

def detect_type_columns(df, type='numeric'):
    numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
    categoric_features = df.select_dtypes(include=['object', 'category']).columns
    return df[numeric_features] if type == 'numeric' else df[categoric_features]

# Função para aplicar Random Perturbation em features numéricas
def random_perturbation(df, perturbation_factor=0.02):
    df_num = df.select_dtypes(include=['float64', 'int64']).copy()
    # Adicionar ruído gaussiano
    noise = np.random.normal(0, perturbation_factor, df_num.shape)
    df_num_augmented = df_num + noise
    return df_num_augmented

# Função para aplicar Sampling nas features categóricas
def sample_categories(df):
    df_cat = df.select_dtypes(include=['object', 'category']).copy()
    df_cat_augmented = df_cat.apply(lambda x: x.sample(frac=1, replace=True).reset_index(drop=True))
    return df_cat_augmented

# Função para combinar as duas estratégias e adicionar novos dados aleatórios
def data_augmentation_with_random_rows(df, augmentation_fraction=0.5, perturbation_factor=0.02):
    # Calcular o número de novas linhas a adicionar
    n_new_rows = int(len(df) * augmentation_fraction)
    
    # Selecionar aleatoriamente linhas do DataFrame original para aplicar augmentação
    random_rows = df.sample(n=n_new_rows, replace=True).reset_index(drop=True)
    
    # Aplicar o data augmentation nas linhas selecionadas
    df_num_augmented = random_perturbation(random_rows, perturbation_factor)
    df_cat_augmented = sample_categories(random_rows)
    
    # Combinar as features numéricas e categóricas aumentadas
    df_augmented = pd.concat([df_num_augmented, df_cat_augmented], axis=1)
    
    # Concatenar os dados originais com os novos dados aumentados
    df_final = pd.concat([df, df_augmented], axis=0).reset_index(drop=True)
    
    return df_final