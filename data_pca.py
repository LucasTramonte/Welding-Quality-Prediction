import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def create_pca_preprocessor(X_train, y_train, correlation_threshold=0.3, n_components=2, plot=False):
    """
    Cria um ColumnTransformer que aplica diferentes transformações a grupos de features com base em suas correlações com o alvo.

    Parâmetros:
    - X_train: DataFrame do pandas com as features de treinamento.
    - y_train: Série do pandas ou array com a variável alvo.
    - correlation_threshold: Limiar para separar high_corr_features e low_corr_features.
    - n_components: Número de componentes principais para o PCA nas low_corr_features.
    - scaler_type: Tipo de escalonador a ser usado ('standard' ou 'robust').

    Retorna:
    - preprocessor: ColumnTransformer configurado.
    - high_corr_features: Lista de features com alta correlação com o alvo.
    - low_corr_features: Lista de features com baixa correlação com o alvo.
    - binary_features: Lista de features binárias.
    """
    
    # 1. Calcular as correlações entre as variáveis e o alvo
    data = pd.concat([X_train, y_train], axis=1)
    corr_matrix = data.corr()
    target_corr = corr_matrix[y_train.name].drop(y_train.name)
    abs_target_corr = target_corr.abs()
    
    # 2. Identificar variáveis com alta e baixa correlação
    high_corr_features = abs_target_corr[abs_target_corr >= correlation_threshold].index.tolist()
    low_corr_features = abs_target_corr[abs_target_corr < correlation_threshold].index.tolist()
    
    # 3. Identificar as variáveis binárias (após one-hot encoding)
    binary_features = [col for col in X_train.columns if X_train[col].nunique() == 2 and X_train[col].dtype in ['int64', 'float64']]
    
    # 4. Remover as variáveis binárias das listas de high_corr_features e low_corr_features
    high_corr_features = [feat for feat in high_corr_features if feat not in binary_features]
    low_corr_features = [feat for feat in low_corr_features if feat not in binary_features]

    # 7. Criar os transformadores para cada grupo de variáveis
    high_corr_transformer = 'passthrough'  # Sem transformação
    
    low_corr_transformer = Pipeline(steps=[
        ('pca', PCA(n_components=n_components))
    ])
    
    binary_transformer = 'passthrough'  # Sem transformação
    
    # 8. Criar o ColumnTransformer
    preprocessor = ColumnTransformer(transformers=[
        ('high_corr', high_corr_transformer, high_corr_features),
        ('low_corr', low_corr_transformer, low_corr_features),
        ('binary', binary_transformer, binary_features)
    ])

    if plot:
        # Acessar o pipeline de baixa correlação
        low_corr_pipeline = preprocessor.named_transformers_['low_corr']
        pca = low_corr_pipeline.named_steps['pca']

        # Obter a variância explicada
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_explained_variance = np.cumsum(explained_variance_ratio)

        # Plotar a variância explicada
        n_components = len(explained_variance_ratio)
        components = np.arange(1, n_components + 1)

        plt.figure(figsize=(10, 6))
        plt.bar(components, explained_variance_ratio * 100, alpha=0.7, align='center',
                label='Variância explicada individual')
        plt.step(components, cumulative_explained_variance * 100, where='mid',
                label='Variância explicada acumulada', color='red')
        plt.xlabel('Componentes Principais')
        plt.ylabel('Variância Explicada (%)')
        plt.title('Análise da Variância Explicada pelo PCA')
        plt.legend(loc='best')
        plt.xticks(components)
        plt.grid(True)
        plt.show()
    
    return preprocessor, high_corr_features, low_corr_features, binary_features