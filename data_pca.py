import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def create_pca_preprocessor(X_train, y_train, correlation_threshold=0.1, n_components=0.97, plot=False):
    
    # 1. Calculate correlations between features and the target
    data = pd.concat([X_train, y_train], axis=1)
    corr_matrix = data.corr()
    target_corr = corr_matrix[y_train.name].drop(y_train.name)
    abs_target_corr = target_corr.abs()
    
    # 2. Identify high and low correlation features
    high_corr_features = abs_target_corr[abs_target_corr >= correlation_threshold].index.tolist()
    low_corr_features = abs_target_corr[abs_target_corr < correlation_threshold].index.tolist()
    print(f'Number of high correlated features: {len(high_corr_features)}')
    print(f'Number of low correlated features: {len(low_corr_features)}')
    
    # 3. Identify binary features (after one-hot encoding)
    binary_features = [col for col in X_train.columns if X_train[col].nunique() == 2 and X_train[col].dtype in ['int64', 'float64']]
    
    # 4. Remove binary features from high_corr_features and low_corr_features lists
    high_corr_features = [feat for feat in high_corr_features if feat not in binary_features]
    low_corr_features = [feat for feat in low_corr_features if feat not in binary_features]

    # 5. Create transformers for each group of features
    high_corr_transformer = 'passthrough'  # No transformation
    
    low_corr_transformer = Pipeline(steps=[
        ('pca', PCA(n_components=n_components))
    ])
    
    binary_transformer = 'passthrough'  # No transformation
    
    # 6. Create the ColumnTransformer
    preprocessor = ColumnTransformer(transformers=[
        ('high_corr', high_corr_transformer, high_corr_features),
        ('low_corr', low_corr_transformer, low_corr_features),
        ('binary', binary_transformer, binary_features)
    ])

    if plot:
        # Temporary fit to calculate explained variance
        temp_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor)
        ])
        temp_pipeline.fit(X_train)
        
        # Get the explained variance
        pca = temp_pipeline.named_steps['preprocessor'].transformers_[1][1].named_steps['pca']
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_explained_variance = np.cumsum(explained_variance_ratio)

        # Create the plot
        components = np.arange(1, len(cumulative_explained_variance) + 1)

        plt.figure(figsize=(8, 6))
        
        # Plot the cumulative explained variance
        plt.plot(components, cumulative_explained_variance * 100, color='red', lw=2, label='Cumulative explained variance')

        # Plot the 95% explained variance line
        plt.axhline(y=95, color='black', linestyle='--', label='95% Explained Variance')

        # Add labels, title, and legend
        plt.xlabel('Principal components')
        plt.ylabel('Cumulative Explained Variance (%)')
        plt.title('Cumulative Explained Variance as a Function of the Number of Components')
        plt.legend(loc='best')
        plt.grid(True)

        # Show the plot
        plt.show()
    
    return preprocessor, high_corr_features, low_corr_features, binary_features