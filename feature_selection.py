from sklearn.feature_selection import f_regression, SelectKBest, RFECV
from sklearn.ensemble import RandomForestRegressor

def apply_feature_selection(X_train, X_test, y_train, feature_selection_methods):
    """
    Apply the feature selection methods to the given datasets.
    
    Parameters:
    - X_train: training set features
    - X_test: test set features
    - y_train: training set target
    - feature_selection_methods: a list of dictionaries with method information
      Each dictionary must contain the 'method' (e.g., 'SelectKBest' or 'RFECV') and optional 'params'.
    
    Returns the transformed X_train and X_test after feature selection.
    """
    for method_info in feature_selection_methods:
        method = method_info['method']  # Method name (e.g., 'SelectKBest' or 'RFECV')
        params = method_info.get('params', {})  # Parameters for the method, if any

        if method == 'SelectKBest':
            k = params.get('k', 10)  # Get the value of 'k' from parameters (default is 10)
            select_kbest = SelectKBest(f_regression, k=k).fit(X_train, y_train)
            # Transform the training and test sets using SelectKBest
            X_train = select_kbest.transform(X_train)
            X_test = select_kbest.transform(X_test)

        elif method == 'RFECV':
            # Set default parameters for RFECV if not provided in the 'params'
            estimator = params.get('estimator', RandomForestRegressor())
            step = params.get('step', 1)  # Step size for eliminating features
            cv = params.get('cv', 5)  # Number of cross-validation folds
            scoring = params.get('scoring', 'r2')  # Scoring metric (default is 'r2')

            rfecv = RFECV(estimator=estimator, step=step, cv=cv, scoring=scoring)
            rfecv = rfecv.fit(X_train, y_train)  # Fit RFECV to the training data

            # Transform the training and test sets using RFECV
            X_train = rfecv.transform(X_train)
            X_test = rfecv.transform(X_test)

    return X_train, X_test  # Return the transformed feature sets