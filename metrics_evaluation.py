# metrics_evaluation.py

from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve
import numpy as np

class MetricsEvaluator:
    def __init__(self):
        self.results_df = None
        self.models_predictions = {}
        
    def evaluate_models(self, pipelines, X_test, y_test):
        """
        Evaluate models using various metrics and store the results.
        """
        results = []
        for model_name, pipeline in pipelines.items():
            y_pred = pipeline.predict(X_test)
            self.models_predictions[model_name] = y_pred
            
            # print(np.isnan(y_pred).sum())
            # print(np.isnan(y_test).sum())
            mape = mean_absolute_percentage_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            results.append({
                "Model": model_name,
                "MAPE": mape,
                "R2 Score": r2,
                "MSE": mse
            })
        
        self.results_df = pd.DataFrame(results)
        self.results_df = self.results_df.sort_values(by="R2 Score", ascending=False).reset_index(drop=True)
        return self.results_df
    
    def plot_overestimation(self, y_test):
        """
        Generate histograms to visualize overestimation or underestimation by the models.
        """
        fig, axes = plt.subplots(2, 4, figsize=(20, 12)) 
        axes = axes.flatten()  
        fig.delaxes(axes[-1])
        
        for i, (model_name, y_pred) in enumerate(self.models_predictions.items()):
            df_model = pd.DataFrame({'true': y_test, 'predicted': y_pred})
            df_model['diff'] = df_model['predicted'] - df_model['true']
        
            sns.set(style="white")
            axes[i].hist(df_model['diff'], bins=26, color="pink", edgecolor='brown', linewidth=2)
            axes[i].set_title(f'True Values vs Predictions : {model_name}', fontsize=12)
            axes[i].set_xlabel('Difference (Prediction - True Value)', fontsize=12)
            axes[i].set_ylabel('Frequency', fontsize=12)
    
        plt.tight_layout()
        plt.show()
        
    def plot_learning_curves(self, pipelines, X_train, y_train):
        """
        Plot learning curves to identify overfitting or underfitting.
        """
        def plot_learning_curve_subplot(ax, model, title, X, y, cv=5):
            train_sizes, train_scores, test_scores = learning_curve(
                model, X, y, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1, 
                train_sizes=np.linspace(0.1, 1.0, 10)
            )
        
            # Calculate mean and standard deviation
            train_scores_mean = -train_scores.mean(axis=1)
            train_scores_std = train_scores.std(axis=1)
            test_scores_mean = -test_scores.mean(axis=1)
            test_scores_std = test_scores.std(axis=1)
        
            # Plot the learning curve
            ax.set_title(title)
            ax.set_xlabel("Number of samples in the training set")
            ax.set_ylabel("MSE")
            ax.grid()
        
            # Shaded area to represent standard deviation
            ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                            train_scores_mean + train_scores_std, alpha=0.1, color="r")
            ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                            test_scores_mean + test_scores_std, alpha=0.1, color="g")
        
            # Training and validation score curves
            ax.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
            ax.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Validation score")
        
            ax.legend(loc="best")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        model_names = list(pipelines.keys())
        
        # Loop through the models and plot learning curves
        for i, model_name in enumerate(model_names):
            plot_learning_curve_subplot(axes[i // 4, i % 4], 
                                        pipelines[model_name], 
                                        model_name, 
                                        X_train, 
                                        y_train)
        
        # Remove the last subplot if necessary
        if len(model_names) < 8:
            fig.delaxes(axes[1, 3])
        
        plt.tight_layout()
        plt.show()
