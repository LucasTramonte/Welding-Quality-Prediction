import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import RobustScaler # it is not affected by outliers.
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.feature_selection import f_regression, SelectKBest, RFECV
import time

# Functions 

def evaluate_model(pipeline, X_train, y_train, X_test, y_test):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    return mape, r2, mse, y_pred

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

st.title("Predicting Steel Weld Quality Using Data-Driven Approaches")

st.markdown("""
The dataset contains the chemical composition of the steels studied and their room temperature mechanical properties.

## Purpose
This analysis provides chemical composition and mechanical property data for a collection of weld metal deposits. 

Here are some key questions we aim to address:

- **Is there a relationship between Ultimate Tensile Strength (MPa) and Yield Strength (MPa)?**
- **Is this relationship linear?**
- **How strong is the correlation between Ultimate Tensile Strength (MPa) and Reduction of Area (%)?**
- **What types of welds contribute to Yield Strength and Ultimate Tensile Strength?**
- **How accurately can we predict Ultimate Tensile Strength (MPa) and Yield Strength (MPa)?**

Our goal is to identify a function that predicts Ultimate Tensile Strength and Yield Strength based on the chemical composition of the steels. We will visualize the relationships between features and responses using scatter plots.

## Data Description
For more details, you can visit the [data source](https://www.phase-trans.msm.cam.ac.uk/map/data/materials/welddb-b.html).
""")

data = pd.read_csv('Assets/Data/welddb.csv', delimiter='\s+', header=None)

# Name the columns
data.columns = [
    'Carbon concentration (weight%)', 
    'Silicon concentration (weight%)', 
    'Manganese concentration (weight%)', 
    'Sulphur concentration (weight%)', 
    'Phosphorus concentration (weight%)', 
    'Nickel concentration (weight%)', 
    'Chromium concentration (weight%)', 
    'Molybdenum concentration (weight%)', 
    'Vanadium concentration (weight%)', 
    'Copper concentration (weight%)', 
    'Cobalt concentration (weight%)', 
    'Tungsten concentration (weight%)', 
    'Oxygen concentration (ppm by weight)', 
    'Titanium concentration (ppm by weight)', 
    'Nitrogen concentration (ppm by weight)', 
    'Aluminium concentration (ppm by weight)', 
    'Boron concentration (ppm by weight)', 
    'Niobium concentration (ppm by weight)', 
    'Tin concentration (ppm by weight)', 
    'Arsenic concentration (ppm by weight)', 
    'Antimony concentration (ppm by weight)', 
    'Current (A)', 
    'Voltage (V)', 
    'AC or DC', 
    'Electrode positive or negative', 
    'Heat input (kJ/mm)', 
    'Interpass temperature (°C)', 
    'Type of weld', 
    'Post weld heat treatment temperature (°C)', 
    'Post weld heat treatment time (hours)', 
    'Yield strength (MPa)', 
    'Ultimate tensile strength (MPa)', 
    'Elongation (%)', 
    'Reduction of Area (%)', 
    'Charpy temperature (°C)', 
    'Charpy impact toughness (J)', 
    'Hardness (kg/mm2)', 
    '50% FATT', 
    'Primary ferrite in microstructure (%)', 
    'Ferrite with second phase (%)', 
    'Acicular ferrite (%)', 
    'Martensite (%)', 
    'Ferrite with carbide aggregate (%)', 
    'Weld ID'
]

st.sidebar.title("Operations on the Dataset")

table = st.sidebar.checkbox("show table", False)

if table:
    st.write("Dataset")
    st.dataframe(data,width=2000,height=500)

# Replace 'N' with NaN
data.replace('N', pd.NA, inplace=True)

categoric_features = ['AC or DC', 'Electrode positive or negative','Type of weld'] # Weld ID isn't important
data_categoric = data[categoric_features] 

# Convert columns to numeric 
numeric_features = data.drop(columns = ['Weld ID','AC or DC', 'Electrode positive or negative','Type of weld']).columns
data_numeric = data[numeric_features].apply(pd.to_numeric, errors='coerce')

mean_phosphorus = data_numeric['Phosphorus concentration (weight%)'].dropna().astype(float).mean()
mean_sulphur = data_numeric['Sulphur concentration (weight%)'].dropna().astype(float).mean()

# Replace NaNs with average values for Phosphorus and Sulphur
data_numeric['Phosphorus concentration (weight%)'] = data_numeric['Phosphorus concentration (weight%)'].fillna(mean_phosphorus)
data_numeric['Sulphur concentration (weight%)'] = data_numeric['Sulphur concentration (weight%)'].fillna(mean_sulphur)

# Replace NaNs with 0 for the other columns
data_numeric = data_numeric.fillna(0)
data_categoric = data_categoric.fillna(0)

data_categoric["Electrode positive or negative"] = data_categoric["Electrode positive or negative"].replace([0, '0'], '0')
data_categoric["Electrode positive or negative"] = data_categoric["Electrode positive or negative"].astype("category")

# Concatenate dataframes

data_all = pd.concat([data_numeric, data_categoric], axis = 1)

data_numeric.drop(columns = ["Charpy temperature (°C)", "50% FATT"], inplace = True)
data_numeric = data_numeric.drop_duplicates(keep='last') #9 duplicated rows

data_all.drop(columns = ["Charpy temperature (°C)", "50% FATT"], inplace = True)
data_all = data_all.drop_duplicates(keep='last') #9 duplicated rows

selected_feature = st.selectbox('Choose a numeric feature to plot the histogram:', data_numeric.columns.to_list())

if selected_feature:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.hist(data_numeric[selected_feature], bins=50, color='blue', edgecolor='black')
    ax1.set_title(f'Histogram of {selected_feature}')
    ax1.set_xlabel(selected_feature)
    ax1.set_ylabel('Frequency')

    # Boxplot
    ax2.boxplot(data_numeric[selected_feature], vert=False)
    ax2.set_title(f'Boxplot of {selected_feature}')
    ax2.set_xlabel(selected_feature)

    plt.tight_layout()
    st.pyplot(fig)
    
selected_categoric_feature = st.selectbox('Choose a categoric feature to plot:', data_categoric.columns.to_list())

if selected_feature:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6)) 
    data_categoric[selected_categoric_feature].value_counts().plot(ax = ax1, kind = 'bar')
    ax1.set_title(f'Frequency of {selected_categoric_feature}')
    ax1.set_xlabel(selected_categoric_feature)
    ax1.set_ylabel('Frequency')

    # Boxplot
    sns.boxplot(x=data_categoric[selected_categoric_feature], y=data_numeric['Yield strength (MPa)'], ax=ax2)
    ax2.set_title(f'Boxplot of {selected_categoric_feature} vs Yield strength (MPa)')
    ax2.set_xlabel(selected_categoric_feature)
    ax2.set_ylabel('Yield strength (MPa)')

    plt.tight_layout()
    st.pyplot(fig)

# Reorder columns for better visualization
cols = data_numeric.columns.tolist()
cols.remove('Yield strength (MPa)')
cols.remove('Ultimate tensile strength (MPa)')

cols.append('Ultimate tensile strength (MPa)')
cols.append('Yield strength (MPa)')

data_numeric = data_numeric[cols]

st.markdown(""" ### Correlation """)

feature_x = st.selectbox('Choose the X feature for the jointplot:', data_numeric.columns.to_list())
feature_y = st.selectbox('Choose the Y feature for the jointplot:', data_numeric.columns.to_list())

if feature_x and feature_y:
    st.write(f"Plotting correlation between {feature_x} and {feature_y}")
    fig = sns.jointplot(data = data_numeric, x = feature_x,y = feature_y, kind="reg", color="#ce1414")
    st.pyplot(fig)
    
st.markdown("""
    ## Important Observations about Correlation:
    
    - **Sulphur concentration** and **Phosphorus concentration** are highly correlated, then we must keep only one of them.
    - **Chromium**, **Molybdenum**, and **Vanadium concentration** are highly correlated, then we must keep only one of them.
    - **Elongation (%)** and **Reduction of Area (%)** are highly correlated, then we must keep only one of them.
    - **Reduction of Area (%)** is highly correlated with the target variables **Yield Strength** and **Ultimate Tensile Strength**, so we should keep it for prediction!
    - **Yield Strength** and **Ultimate Tensile Strength** are highly correlated, so we should use **Yield Strength** to predict **Ultimate Tensile Strength**!
""")

# Modeling 

st.sidebar.title("Modeling")

categoric_features = ['AC or DC', 'Electrode positive or negative','Type of weld'] # Weld ID isn't important
data_categoric = data[categoric_features] 

# Convert columns to numeric 
numeric_features = data.drop(columns = ['Weld ID','AC or DC', 'Electrode positive or negative','Type of weld']).columns
data_numeric = data[numeric_features].apply(pd.to_numeric, errors='coerce')

# Concatenate dataframes

df_yield_strenght = pd.concat([data_numeric, data_categoric], axis = 1)
df_yield_strenght = df_yield_strenght.drop_duplicates(keep='last')
df_yield_strenght = df_yield_strenght.drop(columns = ['Charpy temperature (°C)', '50% FATT'])

df_uts = pd.concat([data_numeric, data_categoric], axis = 1)
df_uts = df_uts.drop_duplicates(keep='last')
df_uts = df_uts.drop(columns = ['Charpy temperature (°C)', '50% FATT'])

target_options = ['Yield strength (MPa)', 'Ultimate tensile strength (MPa)']
selected_target = st.sidebar.selectbox("Select the target variable:", target_options)

# Update feature matrix (X) and target variable (y) based on user selection
if selected_target == 'Yield strength (MPa)':
    st.title("Modeling: Predicting Yield Strength")
    X = df_yield_strenght.drop(columns=["Yield strength (MPa)", "Sulphur concentration (weight%)", 
                                        "Chromium concentration (weight%)", "Molybdenum concentration (weight%)", 
                                        "Elongation (%)"])
    y = df_yield_strenght['Yield strength (MPa)']
    y.fillna(0, inplace=True)
    
    # numeric features processing
    numeric_features_updated = [feature for feature in data_numeric.columns if feature not in ["Yield strength (MPa)", "Sulphur concentration (weight%)", "Chromium concentration (weight%)", "Molybdenum concentration (weight%)", "Elongation (%)", 'Charpy temperature (°C)', '50% FATT']]

else:
    st.title("Modeling: Predicting Ultimate tensile strength")
    X = df_uts.drop(columns=["Ultimate tensile strength (MPa)", "Sulphur concentration (weight%)", 
                                        "Chromium concentration (weight%)", "Molybdenum concentration (weight%)", 
                                        "Elongation (%)"])
    y = df_uts['Ultimate tensile strength (MPa)']
    y.fillna(0, inplace=True)
    
    # numeric features processing
    numeric_features_updated = [feature for feature in data_numeric.columns if feature not in ["Ultimate tensile strength (MPa)", "Sulphur concentration (weight%)", "Chromium concentration (weight%)", "Molybdenum concentration (weight%)", "Elongation (%)", 'Charpy temperature (°C)', '50% FATT']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state= 42)

## Pre Processing 

mean_phosphorus = X_train['Phosphorus concentration (weight%)'].dropna().astype(float).mean()

# Replace NaNs with average values for Phosphorus and Sulphur
X_train['Phosphorus concentration (weight%)'] = X_train['Phosphorus concentration (weight%)'].fillna(mean_phosphorus)
X_test['Phosphorus concentration (weight%)'] = X_test['Phosphorus concentration (weight%)'].fillna(mean_phosphorus)

# Replace NaNs with 0 for the other columns
X_train.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)

# Categoric features 
X_train["Electrode positive or negative"] = X_train["Electrode positive or negative"].astype("str")
X_test["Electrode positive or negative"] = X_test["Electrode positive or negative"].astype("str")

X_train['AC or DC'] = X_train['AC or DC'].astype("str")
X_test['AC or DC'] = X_test['AC or DC'].astype("str")

X_train['Type of weld'] = X_train['Type of weld'].astype("str")
X_test['Type of weld'] = X_test['Type of weld'].astype("str")

# Preprocessor: Scaling numeric features, encoding categorical
preprocessor_scaler = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features_updated),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categoric_features),
    ]
)

model_pipelines = {
    "Linear Regression": Pipeline([("preprocessor", preprocessor_scaler), ("Regressor", LinearRegression())]),
    "Ridge Regression": Pipeline([("preprocessor", preprocessor_scaler), ("Regressor", Ridge())]),
    "Lasso Regression": Pipeline([("preprocessor", preprocessor_scaler), ("Regressor", Lasso())]),
    "ElasticNet Regression": Pipeline([("preprocessor", preprocessor_scaler), ("Regressor", ElasticNet())]),
    "Decision Tree Regression": Pipeline([("preprocessor", preprocessor_scaler), ("Regressor", DecisionTreeRegressor())]),
    "Random Forest Regression": Pipeline([("preprocessor", preprocessor_scaler), ("Regressor", RandomForestRegressor())]),
    "Gradient Boosting Regression": Pipeline([("preprocessor", preprocessor_scaler), ("Regressor", GradientBoostingRegressor())]),
}

model_options = ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'ElasticNet Regression', 
                 'Decision Tree Regression', 'Random Forest Regression', 'Gradient Boosting Regression']

selected_model = st.sidebar.selectbox("Choose a model:", model_options)


if selected_model:
    latest_iteration = st.empty()

    # Initialize the progress bar
    bar = st.progress(0)

    st.write(f"### {selected_model} Results")

    # Start the timer for model evaluation
    start_time = time.time()
    
    # Evaluate the model (this may take time)
    pipeline = model_pipelines[selected_model]
    mape, r2, mse, y_pred = evaluate_model(pipeline, X_train, y_train, X_test, y_test)

    # Calculate the total time taken for the model evaluation
    elapsed_time = time.time() - start_time

    # Update the progress bar to 100%
    bar.progress(100)
    
    # Display the evaluation results
    st.write(f"**MAPE:** {mape:.4f}")
    st.write(f"**R² Score:** {r2:.4f}")
    st.write(f"**MSE:** {mse:.4f}")
    st.write(f"**Elapsed Time:** {elapsed_time:.2f} seconds")
    
    df_model = pd.DataFrame({'true': y_test, 'predicted': y_pred})
    df_model['diff'] = df_model['predicted'] - df_model['true']

    col1, col2 = st.columns(2)

    # Column 1: Prediction Difference Plot
    with col1:
        # Plotting the difference (prediction - true values) histogram
        fig_diff, ax_diff = plt.subplots(figsize=(10, 6))
        ax_diff.hist(df_model['diff'], bins=26, color="pink", edgecolor='brown', linewidth=2)
        ax_diff.set_title(f'True Values vs Predictions : {selected_model}', fontsize=12)
        ax_diff.set_xlabel('Difference (Prediction - True Value)', fontsize=12)
        ax_diff.set_ylabel('Frequency', fontsize=12)

        # Display the difference plot in the first column
        st.pyplot(fig_diff)

    # Column 2: Learning Curve Plot
    with col2:

        # Plotting the learning curve
        fig_lc, ax_lc = plt.subplots(figsize=(10, 6))
        plot_learning_curve_subplot(ax_lc, pipeline, f'Learning Curve: {selected_model}', X_train, y_train)

        # Display the learning curve plot in the second column
        st.pyplot(fig_lc)
        
    X_train_numeric = X_train.drop(columns=['AC or DC', 'Electrode positive or negative', 'Type of weld'])
    #Feature Importance
    regressor_test = RandomForestRegressor().fit(X_train_numeric, y_train)
    importances = regressor_test.feature_importances_

    std = np.std([tree.feature_importances_ for tree in regressor_test.estimators_],
                axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    st.write("**Feature ranking:**")

    fig, ax = plt.subplots(figsize=(14, 13))
    ax.set_title("Feature Importances")

    # Create the bar plot
    ax.bar(range(X_train_numeric.shape[1]), importances[indices], color="g", yerr=std[indices], align="center")
    ax.set_xticks(range(X_train_numeric.shape[1]))
    ax.set_xticklabels(X_train_numeric.columns[indices], rotation=90)
    ax.set_xlim([-1, X_train_numeric.shape[1]])

    st.pyplot(fig)
        
    