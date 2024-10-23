from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

class ModelTrainer:
    def __init__(self, preprocessor=None):
        self.best_estimators = {}
        self.preprocessor = preprocessor

    def train_models(self, X_train, y_train):
        """
        Build, tune, and train machine learning models using pipelines.
        """
        # Define pipelines with feature selection and the regressor
        pipelines = {
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
            ]),
            "LightGBM Regression": Pipeline([
                ("preprocessor", self.preprocessor),
                ("feature_selection", SelectKBest(score_func=f_regression)),
                ("regressor", LGBMRegressor(random_state=42))
            ])
        }

        # Define parameter grids for hyperparameter tuning
        param_grids = {
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
            },
            # "LightGBM Regression": {
            #     'feature_selection__k': [5, 10, 'all'],
            #     'regressor__n_estimators': [100, 200],
            #     'regressor__learning_rate': [0.01, 0.05, 0.1],
            #     'regressor__num_leaves': [31, 50, 100],
            #     'regressor__subsample': [0.8, 1.0],
            #     'regressor__colsample_bytree': [0.8, 1.0]
            # }
        }

        # Perform hyperparameter tuning and training for each model
        for model_name, pipeline in pipelines.items():
            print(f"Training and tuning {model_name}...")
            param_grid = param_grids[model_name]
            grid_search = GridSearchCV(
                pipeline, param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error'
            )
            grid_search.fit(X_train, y_train)
            self.best_estimators[model_name] = grid_search.best_estimator_
            print(f"Best parameters for {model_name}: {grid_search.best_params_}")
            print(f"Best score for {model_name}: {grid_search.best_score_}")

    def get_best_estimators(self):
        """
        Retrieve the best estimators after hyperparameter tuning.
        """
        return self.best_estimators
