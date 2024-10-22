import joblib
import pandas as pd

def save_model(model, filename):
    joblib.dump(model, filename)

def load_model(filename):
    return joblib.load(filename)

def to_dataframe_transformer(X, column_names=None):
    # Converter o np.array de volta para DataFrame
    if column_names is None:
        size_df = X.shape[1]
        col_names = [f'{i}' for i in range(size_df)]
        df = pd.DataFrame(X, columns=col_names)
    else:
        df = pd.DataFrame(X, columns=column_names)
    
    numeric_features_list = []
    categorical_features_list = []
    
    for col in column_names:
        try:
            # Tenta converter a coluna para numérico
            pd.to_numeric(df[col], errors='raise')  # Levanta erro se não puder converter
            numeric_features_list.append(col)  # Se for possível, adiciona como numérica
        except ValueError:
            # Se houver erro, classificamos como categórica
            categorical_features_list.append(col)
    
    # Convertendo colunas numéricas para float
    df[numeric_features_list] = df[numeric_features_list].astype('float')    
    return df