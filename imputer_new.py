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

def impute_target(y_train, y_test, method='zero'):
    """
    Impute missing values in the target variable.
    - y_train: target values for training set
    - y_test: target values for testing set
    - method: the imputation method ('mean', 'median', 'ffill', or 'zero')
    Returns the imputed y_train and y_test.
    """
    if method == 'mean':
        y_train = y_train.fillna(y_train.mean())
        y_test = y_test.fillna(y_train.mean())
    elif method == 'median':
        y_train = y_train.fillna(y_train.median())
        y_test = y_test.fillna(y_train.median())
    elif method == 'ffill':
        y_train = y_train.fillna(method='ffill')
        y_test = y_test.fillna(method='ffill')
    elif method == 'zero':
        y_train = y_train.fillna(0)
        y_test = y_test.fillna(0)

    return y_train, y_test
