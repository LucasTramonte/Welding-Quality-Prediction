# %% [markdown]
# # Machine Learning Project - IA mention CentraleSupélec
# 
# Under the supervision of:
# 
# •⁠  ⁠Myriam TAMI
# 
# Students:
# 
# •⁠  ⁠Gabriel Souza Lima
# •⁠  ⁠Kiyoshi Araki
# •⁠  ⁠Lucas Tramonte
# •⁠  ⁠Rebeca Bayasari

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.impute import SimpleImputer
import math

# %% [markdown]
# EDA (Exploratory Data Analysis)

# %%
data_original = pd.read_csv('Assets/Data/welddb.csv', delimiter='\s+', header=None)

# %%
data = data_original.copy()

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

print(data)

# %% [markdown]
# Define categories

# %%
# Defining the categories
categories = {
    'Concentration of Elements': [
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
        'Antimony concentration (ppm by weight)'
    ],
    'Electrical and Welding Parameters': [
        'Current (A)', 
        'Voltage (V)', 
        'AC or DC', 
        'Electrode positive or negative', 
        'Heat input (kJ/mm)', 
        'Interpass temperature (°C)', 
        'Type of weld', 
        'Post weld heat treatment temperature (°C)', 
        'Post weld heat treatment time (hours)'
    ],
    'Mechanical Properties': [
        'Yield strength (MPa)', 
        'Ultimate tensile strength (MPa)', 
        'Elongation (%)', 
        'Reduction of Area (%)', 
        'Charpy temperature (°C)', 
        'Charpy impact toughness (J)', 
        'Hardness (kg/mm2)', 
        '50% FATT'
    ],
    'Microstructural Properties': [
        'Primary ferrite in microstructure (%)', 
        'Ferrite with second phase (%)', 
        'Acicular ferrite (%)', 
        'Martensite (%)', 
        'Ferrite with carbide aggregate (%)'
    ],
    'Identification': ['Weld ID']
}

# %% [markdown]
# Handle missing data as NaN and convert all columns except the 4 categorical ones to numeric

# %%
data_clean = data.replace({"N": np.nan})
categoric_features = ['AC or DC', 'Electrode positive or negative', 'Type of weld']  # Weld ID isn't important
# Convert columns to numeric
numeric_features = data_clean.drop(columns=['Weld ID', 'AC or DC', 'Electrode positive or negative', 'Type of weld']).columns
data_clean[numeric_features] = data_clean[numeric_features].apply(pd.to_numeric, errors='coerce')

# %% [markdown]
# Analyze the data distribution

# %%
# Function to calculate and plot the percentage of zeros, NaN, and non-zero values
def plot_zeros_nan_and_non_zero_percentage(df, categories):
    for category, columns in categories.items():
        zero_percentage = {}
        nan_percentage = {}
        non_zero_percentage = {}
        for column in columns:
            # Calculate the percentage of zeros
            zero_percentage[column] = (df[column] == 0).mean() * 100
            # Calculate the percentage of NaN
            nan_percentage[column] = df[column].isna().mean() * 100
            # Calculate the percentage of non-zero and non-NaN values
            non_zero_percentage[column] = 100 - zero_percentage[column] - nan_percentage[column]

        # Create stacked bar plot
        plt.figure(figsize=(12, 6))
        cols = list(zero_percentage.keys())
        zeros_vals = list(zero_percentage.values())
        nan_vals = list(nan_percentage.values())
        non_zero_vals = list(non_zero_percentage.values())

        # Bar positions
        indices = np.arange(len(cols))

        # Plot the stacked horizontal bars with a blue palette
        plt.barh(indices, zeros_vals, color='#add8e6', label='Zero values')  # Light blue
        plt.barh(indices, nan_vals, left=zeros_vals, color='#4682b4', label='NaN values')  # Steel blue
        plt.barh(indices, non_zero_vals, left=[i + j for i, j in zip(zeros_vals, nan_vals)], color='#00008b', label='Non-zero values')  # Dark blue

        # Plot settings
        plt.yticks(indices, cols)  # Bar labels
        plt.xlabel('Percentage (%)')
        plt.title(f'Percentage of Zeros, NaN, and Non-zero in {category}')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

# Call the function to plot the graphs for each category
plot_zeros_nan_and_non_zero_percentage(data_clean, categories)

# %% [markdown]
# The presence of an "N" indicates that the value was not reported in the publication. This is NOT meant to be an indication that the value is zero.
# 
# - It wouldn't make sense to set the phosphorus and sulfur concentrations to zero when their values aren't reported, because these impurity elements are inevitably present in welds in practice.
# - Therefore, we'll use the average values for these concentrations in such cases.
# - On the other hand, for other elements like manganese and nickel, it is reasonable to set their concentrations to zero if they are not reported. This is because they wouldn't be deliberate additions and their concentrations would be close to the detection limits of the commonly used analytical techniques [1].

# %%
# 1. Use the average values for Phosphorus and Sulphur concentrations when missing
data_clean["Phosphorus concentration (weight%)"] = data_clean["Phosphorus concentration (weight%)"].fillna(
    data_clean["Phosphorus concentration (weight%)"].mean())

data_clean["Sulphur concentration (weight%)"] = data_clean["Sulphur concentration (weight%)"].fillna(
    data_clean["Sulphur concentration (weight%)"].mean())

# 2. Set Manganese and Nickel concentrations to zero when missing
data_clean["Manganese concentration (weight%)"] = data_clean["Manganese concentration (weight%)"].fillna(0)
data_clean["Nickel concentration (weight%)"] = data_clean["Nickel concentration (weight%)"].fillna(0)

# %% [markdown]
# Remove columns with more than 60% missing data

# %%
# Define the threshold for removing columns with too many NaN values (60%)
threshold = 0.6

# Calculate the percentage of NaN values in each column
nan_percentage = data_clean.isna().mean()

# Select columns that have more than 60% missing values
columns_to_drop = nan_percentage[nan_percentage > threshold].index

# Remove those columns
data_clean = data_clean.drop(columns=columns_to_drop)

# Display the DataFrame without the columns that have more than 60% missing values
print(columns_to_drop)

# %%
print("Original data shape: ", data.shape, "\nCleaned data shape: ",data_clean.shape)

# %% [markdown]
# Analyzing the distribution of skewness and kurtosis allows identifying the distribution of numerical data and potential outliers. If the data has high skewness or kurtosis, it may indicate the need for transformations, such as a logarithmic transformation, to improve modeling, especially in algorithms sensitive to distribution. Positive or negative skewness reveals unbalanced tails, while high kurtosis indicates the presence of more outliers than expected. Transforming or adjusting this data can lead to better modeling results.
# 
# - Skewness measures the degree of symmetry in data distribution:
# - Skewness ≈ 0: The distribution is approximately symmetric (normal distribution).
# - Skewness > 0 (positive): The distribution has a longer tail on the right (many small values and a few large ones).
# - Skewness < 0 (negative): The distribution has a longer tail on the left (many large values and a few small ones).
# 
# Symmetrical distributions (skewness ≈ 0) are generally ideal for models that assume normality (such as linear regression).
# Asymmetrical distributions (skewness > 0 or < 0) may indicate that a transformation of the data (such as logarithmic) could be useful for modeling.
# 
# Kurtosis measures the "flatness" of the data distribution:
# Kurtosis ≈ 3: Indicates a normal distribution (mesokurtic).
# Kurtosis > 3 (leptokurtic): A distribution with heavier tails and a higher peak, indicating more extreme values (outliers).
# Kurtosis < 3 (platykurtic): A distribution with lighter tails and a less pronounced peak (flatter distribution).
# 
# High kurtosis (> 3) indicates that the data has more outliers than expected in a normal distribution, which may require attention in modeling, as extreme values can disproportionately influence the model.
# Low kurtosis (< 3) suggests that the data is more evenly distributed around the mean and has fewer outliers.

# %%
# Skewness and Kurtosis Analysis
skewness = data_clean.skew()
kurtosis = data_clean.kurt()

# Plot skewness (horizontal bar chart)
plt.figure(figsize=(10, 10))
sns.barplot(y=skewness.index, x=skewness.values)
plt.title("Skewness of Numeric Columns")
plt.xlabel("Skewness")
plt.ylabel("Columns")
plt.show()

# Plot kurtosis (horizontal bar chart)
plt.figure(figsize=(10, 10))
sns.barplot(y=kurtosis.index, x=kurtosis.values)
plt.title("Kurtosis of Numeric Columns")
plt.xlabel("Kurtosis")
plt.ylabel("Columns")
plt.show()

# %% [markdown]
# Analyzing outliers more carefully

# %%
# Initialize a DataFrame to store skewness, kurtosis, and outliers
analysis = pd.DataFrame(index=data_clean.columns, columns=['Skewness', 'Kurtosis', 'Outliers'])

# Loop over each numeric column to calculate skewness, kurtosis, and detect outliers
for column in data_clean.columns:
    # Calculate Skewness and Kurtosis
    skewness = data_clean[column].skew()
    kurtosis = data_clean[column].kurt()
    
    # Calculate Q1, Q3, and IQR for each column
    Q1 = data_clean[column].quantile(0.25)
    Q3 = data_clean[column].quantile(0.75)
    IQR = Q3 - Q1
    
    # Condition to identify outliers in the column
    outlier_condition = (data_clean[column] < (Q1 - 1.5 * IQR)) | (data_clean[column] > (Q3 + 1.5 * IQR))
    outliers = data_clean[column][outlier_condition]
    
    # Store the results in the analysis DataFrame
    analysis.at[column, 'Skewness'] = skewness
    analysis.at[column, 'Kurtosis'] = kurtosis
    analysis.at[column, 'Outliers'] = len(outliers)  # Count the number of outliers in the column

# Display the complete analysis
print("Complete Analysis of Skewness, Kurtosis, and Outliers:")
analysis.style.format(precision=4)  

# %% [markdown]
# Variables like "Sulphur" and "Phosphorus concentration" have extremely high skewness and kurtosis values, indicating that their data is highly skewed and contains many outliers.

# %% [markdown]
# Variables such as Sulphur and Phosphorus have extreme skewness and a significant number of outliers, suggesting that treatment of these values may be necessary before statistical analysis or modeling. Elements like Nickel and Oxygen also have a high number of outliers and high skewness. Variables like Molybdenum and Post weld heat treatment temperature have more stable distributions, with no outliers.

# %% [markdown]
# log

# %%
log_columns = ['Silicon concentration (weight%)',
               'Sulphur concentration (weight%)',
               'Phosphorus concentration (weight%)',
               'Nickel concentration (weight%)',
               'Titanium concentration (ppm by weight)',
               'Nitrogen concentration (ppm by weight)',
               'Oxygen concentration (ppm by weight)',
               'Voltage (V)',
               'Heat input (kJ/mm)']

# Apply logarithmic transformation (log(x + 1)) to avoid issues with zero values
data_clean[log_columns] = data_clean[log_columns].apply(lambda x: np.log(x + 1))

# Check the result
data_clean[log_columns].head()

# %% [markdown]
# Numerical features

# %%
imputer_median = SimpleImputer(strategy = "median")
imputer_mode = SimpleImputer(strategy = "most_frequent")

data_imputed = data_clean.copy()

# Ensure that the concentration_features exist in the DataFrame before filling them
concentration_features = [col for col in data_clean.columns if "concentration" in col and not("Phosphorus" in col or "Sulphur" in col)]
concentration_features = [col for col in concentration_features if col in data_imputed.columns]  # Check if they exist in the DataFrame

other_numeric_features = [col for col in data_clean.columns if col not in concentration_features]
other_numeric_features = [col for col in other_numeric_features if col in data_imputed.columns]  # Check if they exist in the DataFrame

# Fill missing values with zero for the concentration columns (except Phosphorus and Sulphur)
data_imputed[concentration_features] = data_imputed[concentration_features].fillna(0)

# Fill missing values with the median for other numeric columns
data_imputed[other_numeric_features] = imputer_median.fit_transform(data_imputed[other_numeric_features])

# Fill missing values with the most frequent value for categorical columns
data_imputed[categoric_features] = imputer_mode.fit_transform(data_imputed[categoric_features])

# %%
print("Number of missing values after imputing: ", data_imputed[data_imputed.isnull().any(axis=1)].shape[0])

# %% [markdown]
# Correlation

# %%
data_numeric = data_imputed.select_dtypes(include=[np.number])

# Display correlation matrix with a color gradient
data_numeric.corr().style.background_gradient(cmap="coolwarm")

# %% [markdown]
# - Sulphur concentration and Phosphorus concentration are highly correlated, then we must keep only one of them.
# - Chromium, Molybdenum, and Vanadium concentration are highly correlated, then we must keep only one of them.
# - Elongation (%) and Reduction of Area (%) are highly correlated, then we must keep only one of them.
# - Reduction of Area (%) is highly correlated with the target variables Yield strength and Ultimate tensile strength, then we must keep it in the prediction!!
# - Yield strength and Ultimate tensile strength are highly correlated, then we should keep the Yield strength for predicting Ultimate tensile strength!!

# %%
corr = data_numeric.corr()
corr_unstack = corr.unstack()

# Filter the diagonal (correlations of 1) and sort by absolute values
sorted_correlation_pairs = corr_unstack[corr_unstack != 1].abs().sort_values(ascending=False)

most_corr_pairs = sorted_correlation_pairs[sorted_correlation_pairs > 0.75]

# Print the most correlated pairs
print(most_corr_pairs.iloc[[i for i in range(0,len(most_corr_pairs),2)]])

# %% [markdown]
# Categorical features

# %%
data_categoric = data_imputed[categoric_features]
data_categoric["Electrode positive or negative"] = data_categoric["Electrode positive or negative"].replace([0, '0'], '0')
data_categoric["Electrode positive or negative"] = data_categoric["Electrode positive or negative"].astype("category")

fig, axs = plt.subplots(1, 3, figsize = (12,4))

for ax, col in zip(axs.flatten(), categoric_features):
    data_categoric[col].value_counts().plot(ax = ax, kind = 'bar')
    
plt.tight_layout()
plt.show()

# %% [markdown]
# Target variables

# %%
plt.figure(figsize=(10,6))
sns.boxplot(data = data_imputed, x = "Type of weld", y = "Yield strength (MPa)")
plt.show()

# %%
plt.figure(figsize=(10,6))
sns.boxplot(data = data_imputed, x = "Type of weld", y = "Ultimate tensile strength (MPa)")
plt.show()
