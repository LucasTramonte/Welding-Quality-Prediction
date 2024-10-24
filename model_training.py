from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, learning_curve
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer, r2_score



class ModelTrainer:
    def __init__(self, preprocessor=None, search_type="random", n_iter=10):
        """
        Initializes the ModelTrainer with options for preprocessor, 
        search type (grid or random), and number of iterations for RandomizedSearchCV.
        """
        self.best_estimators = {}
        self.preprocessor = preprocessor
        self.search_type = search_type  # Choose between 'grid' or 'random'
        self.n_iter = n_iter  # For RandomizedSearchCV, number of iterations

    def _get_search(self, pipeline, param_grid):
        """
        Select the type of hyperparameter search (GridSearchCV or RandomizedSearchCV).
        Based on the search_type, it returns the appropriate search object.
        """
        if self.search_type == "grid":
            return GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
        else:
            return RandomizedSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error', n_iter=self.n_iter)

    def _define_pipelines(self):
        """
        Defines pipelines for multiple regression models: 
        Decision Tree, Random Forest, Gradient Boosting, and XGBoost.
        Each pipeline includes a preprocessor and a feature selection step.
        """
        return {
            "Decision Tree Regression": Pipeline([
                ("preprocessor", self.preprocessor),
                ("feature_selection", SelectKBest(score_func=f_regression)),
                ("regressor", DecisionTreeRegressor(random_state=42))
            ]),
            "Random Forest Regression": Pipeline([
                ("preprocessor", self.preprocessor),
                ("feature_selection", SelectKBest(score_func=f_regression)),
                ("regressor", RandomForestRegressor(random_state=42))
            ]),
            "Gradient Boosting Regression": Pipeline([
                ("preprocessor", self.preprocessor),
                ("feature_selection", SelectKBest(score_func=f_regression)),
                ("regressor", GradientBoostingRegressor(random_state=42))
            ]),
            "XGBoost Regression": Pipeline([
                ("preprocessor", self.preprocessor),
                ("feature_selection", SelectKBest(score_func=f_regression)),
                ("regressor", XGBRegressor(random_state=42, verbosity=0))
            ])
        }

    def _define_param_grids(self):
        """
        Defines parameter grids for hyperparameter tuning for each model.
        Parameters include feature selection 'k', number of estimators, depth, etc.
        """
        return {
            "Decision Tree Regression": {
                'feature_selection__k': [5, 10, 'all'],
                'regressor__max_depth': [None, 5, 10, 20],
                'regressor__min_samples_split': [2, 5, 10],
                'regressor__min_samples_leaf': [1, 2, 4],
            },
            "Random Forest Regression": {
                'feature_selection__k': [5, 10, 'all'],
                'regressor__n_estimators': [100, 200],
                'regressor__max_depth': [None, 10, 20],
                'regressor__min_samples_split': [2, 5],
                'regressor__min_samples_leaf': [1, 2],
                'regressor__bootstrap': [True, False]
            },
            "Gradient Boosting Regression": {
                'feature_selection__k': [5, 10, 'all'],
                'regressor__n_estimators': [100, 200],
                'regressor__learning_rate': [0.01, 0.05, 0.1],
                'regressor__max_depth': [3, 5, 10],
                'regressor__min_samples_split': [2, 5],
                'regressor__min_samples_leaf': [1, 2],
            },
            "XGBoost Regression": {
                'feature_selection__k': [5, 10, 'all'],
                'regressor__n_estimators': [100, 200],
                'regressor__learning_rate': [0.01, 0.05, 0.1],
                'regressor__max_depth': [3, 5, 10],
                'regressor__subsample': [0.8, 1.0],
                'regressor__colsample_bytree': [0.8, 1.0]
            }
        }

    def train_models(self, X_train, y_train):
        """
        Trains and tunes models using hyperparameter tuning (GridSearchCV or RandomizedSearchCV)
        and stores the best estimator for each model.
        """
        # Get pipelines and parameter grids
        pipelines = self._define_pipelines()
        param_grids = self._define_param_grids()

        # Perform hyperparameter tuning and training for each model
        for model_name, pipeline in pipelines.items():
            print(f"Training and tuning {model_name}...")
            param_grid = param_grids[model_name]
            search = self._get_search(pipeline, param_grid)
            search.fit(X_train, y_train)
            self.best_estimators[model_name] = search.best_estimator_
            print(f"Best parameters for {model_name}: {search.best_params_}")
            print(f"Best score for {model_name}: {search.best_score_}")

    def get_best_estimators(self):
        """
        Returns the best estimators found during hyperparameter tuning for each model.
        """
        return self.best_estimators

    def evaluate_best_model_with_cv(self, X_train, y_train, cv=10):
        """
        Evaluates the best models using cross-validation.
        It computes the Mean Squared Error (MSE) and R² score for each model.
        """
        cv_scores = {}
        
        for model_name, best_estimator in self.best_estimators.items():
            print(f"Evaluating {model_name} with cross-validation...")
            
            # Mean Squared Error
            mse_scores = cross_val_score(best_estimator, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
            mean_mse = np.mean(mse_scores)
            std_mse = np.std(mse_scores)
            
            # R² Score
            r2_scores = cross_val_score(best_estimator, X_train, y_train, cv=cv, scoring='r2')
            mean_r2 = np.mean(r2_scores)
            std_r2 = np.std(r2_scores)
            
            # Storing the results in a dictionary
            cv_scores[model_name] = {
                'mean_mse': -mean_mse,
                'std_mse': std_mse,
                'mean_r2': mean_r2,
                'std_r2': std_r2
            }
            
            print(f"{model_name} - Mean MSE: {-mean_mse}, Std MSE: {std_mse}")
            print(f"{model_name} - Mean R²: {mean_r2}, Std R²: {std_r2}")
        
        return cv_scores
    
    def plot_learning_curve(self, X_train, y_train, model_name, best_estimator, cv=5):
        """
        Plots the learning curve for the given model.
        """
        train_sizes, train_scores, test_scores = learning_curve(
            best_estimator, X_train, y_train, cv=cv, scoring="neg_mean_squared_error", 
            train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
        )

        # Compute the mean and standard deviation for training and test scores
        train_scores_mean = -train_scores.mean(axis=1)
        train_scores_std = train_scores.std(axis=1)
        test_scores_mean = -test_scores.mean(axis=1)
        test_scores_std = test_scores.std(axis=1)

        plt.figure(figsize=(8, 6))
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Validation score")

        plt.title(f'Learning Curve for {model_name}')
        plt.xlabel("Training examples")
        plt.ylabel("Mean Squared Error")
        plt.legend(loc="best")
        plt.grid(True)
        plt.show()