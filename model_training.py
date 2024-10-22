from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

class ModelTrainer:
    def __init__(self, preprocessor=None):
        self.models = {}
        self.pipelines = {}
        self.preprocessor = preprocessor
        
    def train_models(self, X_train, y_train):
        """
        Build and train machine learning models using pipelines.
        """
        # Define models and pipelines
        self.models = {
            "Linear Regression": Pipeline([("preprocessor", self.preprocessor), ("Regressor", LinearRegression())]),
            "Ridge Regression": Pipeline([("preprocessor", self.preprocessor), ("Regressor", Ridge())]),
            "Lasso Regression": Pipeline([("preprocessor", self.preprocessor), ("Regressor", Lasso())]),
            "ElasticNet Regression": Pipeline([("preprocessor", self.preprocessor), ("Regressor", ElasticNet())]),
            "Decision Tree Regression": Pipeline([("preprocessor", self.preprocessor), ("Regressor", DecisionTreeRegressor(random_state=42))]),
            "Random Forest Regression": Pipeline([("preprocessor", self.preprocessor), ("Regressor", RandomForestRegressor(random_state=42))]),
            "Gradient Boosting Regression": Pipeline([("preprocessor", self.preprocessor), ("Regressor", GradientBoostingRegressor(random_state=42))])
        }
        
        # Train models
        for model_name, pipeline in self.models.items():
            pipeline.fit(X_train, y_train)
            self.pipelines[model_name] = pipeline
            
    def get_pipelines(self):
        """
        Retrieve the trained pipelines.
        """
        return self.pipelines
