import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def apply_pca(X_train, X_test, n_components=None, plot_variance=False):
    """
    Apply PCA to the dataset.
    - X_train: training set
    - X_test: test set
    - n_components: number of principal components to keep (optional)
    - plot_variance: boolean flag to plot cumulative explained variance
    Returns the transformed X_train and X_test after PCA and the PCA model.
    """
    n_components = n_components or min(X_train.shape[1], X_train.shape[0])
    pca_model = PCA(n_components=n_components)
    
    X_train_pca = pca_model.fit_transform(X_train)
    X_test_pca = pca_model.transform(X_test)
    
    if plot_variance:
        # Optionally, plot cumulative explained variance
        cumsum = np.cumsum(pca_model.explained_variance_ratio_) * 100
        plt.figure(figsize=(6, 6))
        plt.plot(cumsum, color='red', label='Cumulative Explained Variance')
        plt.title('Cumulative Explained Variance as a Function of the Number of Components')
        plt.ylabel('Cumulative Explained Variance (%)')
        plt.xlabel('Number of Principal Components')
        plt.axhline(y=95, color='k', linestyle='--', label='95% Explained Variance')
        plt.legend(loc='best')
        plt.show()
    
    return X_train_pca, X_test_pca, pca_model
