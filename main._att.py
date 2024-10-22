from dataset import load_and_label_data
from data_preprocessing_old import DataPreprocessor
from modeling import ModelTrainer
from metrics_evaluation import MetricsEvaluator

# Load and prepare data
file_path = 'Assets/Data/welddb.csv'
data = load_and_label_data(file_path)

# Instantiate DataPreprocessor
data_preprocessor = DataPreprocessor(data)

# Define parameters
categoric_features = ['AC or DC', 'Electrode positive or negative', 'Type of weld']
features_to_drop = ["Yield strength (MPa)", "Sulphur concentration (weight%)", 
                    "Chromium concentration (weight%)", "Molybdenum concentration (weight%)", "Elongation (%)"]
drop_columns = ["Charpy temperature (°C)", "50% FATT"]
target_column = "Yield strength (MPa)"

# Preprocess data
X_train, X_test, y_train, y_test, preprocessor = data_preprocessor.preprocess_data(
    target_column=target_column, 
    categoric_features=categoric_features, 
    features_to_drop=features_to_drop, 
    drop_columns=drop_columns, 
    y_imputation='mean',
    pca_n_components=None
)

# Instantiate ModelTrainer
model_trainer = ModelTrainer(preprocessor=preprocessor)
model_trainer.train_models(X_train, y_train)
pipelines = model_trainer.get_pipelines()

# Instantiate MetricsEvaluator
metrics_evaluator = MetricsEvaluator()
results_df = metrics_evaluator.evaluate_models(pipelines, X_test, y_test)

# Display results
print(results_df)

# # Plot overestimation
# metrics_evaluator.plot_overestimation(y_test)

# # Plot learning curves
# metrics_evaluator.plot_learning_curves(pipelines, X_train, y_train)
