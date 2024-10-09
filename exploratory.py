#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Project - IA mention CentraleSupélec
# 
# Under the supervision of :
# 
# - Myriam TAMI
# 
# Students:
# 
# - Lucas Tramonte

# # Libraries
# 

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


# # EDA

# In[5]:


data_original = pd.read_csv('Assets/Data/welddb.csv', delimiter='\s+', header=None)


# In[6]:


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

data


# Percentage of 'N' values in each column

# In[7]:


percent_n = (data == 'N').mean() * 100
percent_n_sorted = percent_n.sort_values(ascending=False).apply(lambda x: f"{x:.2f}%")
percent_n_sorted


# - The presence of an ``N'' indicates that the value was not reported in the publication. This is NOT meant to be an indication that the value is zero.
# 
# - It wouldn't make sense to set the phosphorus and sulfur concentrations to zero when their values aren't reported, because these impurity elements are inevitably present in welds in practice. 
#     -   Therefore, we'll use the average values for these concentrations in such cases. 
#     -   On the other hand, for other elements like manganese and nickel, it is reasonable to set their concentrations to zero if they are not reported. This is because they wouldn't be deliberate additions and their concentrations would be close to the detection limits of the commonly used analytical techniques [1].

# In[8]:


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


# Numeric features

# In[9]:


# We need to remove the Charpy temperature (°C) and 50% FATT features
print("------------# of negative values : ------------ \n ")
print((data_numeric < 0).sum())


# In[10]:


data_numeric.drop(columns = ["Charpy temperature (°C)", "50% FATT"], inplace = True)
data_numeric = data_numeric.drop_duplicates(keep='last') #9 duplicated rows

data_all.drop(columns = ["Charpy temperature (°C)", "50% FATT"], inplace = True)
data_all = data_all.drop_duplicates(keep='last') #9 duplicated rows

# Reorder columns for better visualization
cols = data_numeric.columns.tolist()
cols.remove('Yield strength (MPa)')
cols.remove('Ultimate tensile strength (MPa)')

cols.append('Ultimate tensile strength (MPa)')
cols.append('Yield strength (MPa)')

data_numeric = data_numeric[cols]

# Summary
summary = pd.DataFrame({
    'Variable': data_numeric.columns,
    'Min': [data_numeric[col].min() for col in data_numeric.columns],
    'Max': [data_numeric[col].max() for col in data_numeric.columns],
    'Mean': [data_numeric[col].mean() for col in data_numeric.columns],
    'Standard Deviation': [data_numeric[col].std() for col in data_numeric.columns]
})

summary


# In[11]:


data_numeric.hist(bins = 50, figsize= (30,30))


# In[12]:


fig, axs = plt.subplots(5,4, figsize = (18,18))

for ax, col in zip(axs.flatten(), data_numeric.iloc[:,0:20].columns):
    data_numeric.iloc[:,0:20].boxplot(column = col, ax = ax)


# In[13]:


fig, axs = plt.subplots(3,6, figsize = (22,18))

for ax, col in zip(axs.flatten(), data_numeric.iloc[:,20::].columns):
    data_numeric.iloc[:,20::].boxplot(column = col, ax = ax)


# In[14]:


data_numeric.corr().style.background_gradient(cmap="coolwarm")


# - Sulphur concentration and Phosphorus concentration are highly correlated, then we must keep only one of them.
# - Chromium, Molybdenum and Vanadium concentration are highly correlated, then we must keep only one of them.
# - Elongation (%) and Reduction of Area (%) are highly correlated, then we must keep only one of them.
# - Reduction of Area (%) are highly correlated with the target variables Yield strenght and Ultimate tensile strenght, then we must keep it on the prediction!!
# - Yield strenght and Ultimate tensile strenght are highly correlated, then we should keep the Yield strenght on the prediction of the Ultimate tensile strenght !!

# In[15]:


sns.jointplot(data = data_numeric, x = "Sulphur concentration (weight%)",y = "Phosphorus concentration (weight%)", kind="reg", color="#ce1414")


# In[16]:


sns.set(style="white")
g = sns.PairGrid(data_numeric.loc[:,['Chromium concentration (weight%)','Molybdenum concentration (weight%)','Vanadium concentration (weight%)']], diag_sharey=False, height=5, aspect=1)
g.map_lower(sns.kdeplot, cmap="Blues_d")
g.map_upper(plt.scatter)
g.map_diag(sns.kdeplot, lw=3)


# In[17]:


sns.set(style="white")
g = sns.PairGrid(data_numeric.loc[:,['Reduction of Area (%)','Elongation (%)','Yield strength (MPa)', 'Ultimate tensile strength (MPa)']], diag_sharey=False, height=5, aspect=1)
g.map_lower(sns.kdeplot, cmap="Blues_d")
g.map_upper(plt.scatter)
g.map_diag(sns.kdeplot, lw=3)


# Categoric features

# In[18]:


data_categoric["Electrode positive or negative"] = data_categoric["Electrode positive or negative"].replace([0, '0'], '0')
data_categoric["Electrode positive or negative"] = data_categoric["Electrode positive or negative"].astype("category")

fig, axs = plt.subplots(1, 3, figsize = (12,4))

for ax, col in zip(axs.flatten(), categoric_features):
    data_categoric[col].value_counts().plot(ax = ax, kind = 'bar')
    
plt.tight_layout
plt.show()


# In[19]:


plt.figure(figsize=(10,6))
sns.boxplot(data = data_all, x = "Type of weld", y = "Yield strength (MPa)")
plt.show()


# In[20]:


plt.figure(figsize=(10,6))
sns.boxplot(data = data_all, x = "Type of weld", y = "Ultimate tensile strength (MPa)")
plt.show()

