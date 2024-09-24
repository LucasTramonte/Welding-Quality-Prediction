# Welding-Quality-Prediction
Predicting steel weld quality using data-driven approaches to extract, standardize, and explore welding expertise, with potential industrial applications.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Modeling](#modeling)
- [Streamlit App](#streamlit-app)
- [Conclusion](#conclusion)
- [References](#references)

## Project Overview

This project leverages machine learning models to predict weld quality, particularly focusing on **Yield Strength** and **Ultimate Tensile Strength** of steel welds. The dataset provides chemical composition and mechanical property data for a collection of all weld metal deposits. We explore various regression techniques to achieve accurate predictions.

## Installation

### Prerequisites
To run this project, ensure you have Python 3.7+ installed.

```bash
pip install -r requirements.txt
```

### Clone the Repository
```bash
git clone https://github.com/yourusername/Welding-Quality-Prediction.git
cd Welding-Quality-Prediction
```

The dataset and its description are both available at the following link:  ([WeldDB](https://www.phase-trans.msm.cam.ac.uk/map/data/materials/welddb-b.html))

More information can be found in chapter 5 of reference [1]

## Exploratory Data Analysis

The exploratory data analysis (EDA) was performed in the exploratory.ipynb script. This script includes:

- Visualizing the distribution of variables.
- Analyzing correlations between chemical compositions and mechanical properties.
- Identifying missing values and strategies for handling them.
- Basic statistics of the dataset such as mean, median, and standard deviations for different features.

## Modeling

The main.ipnyb script contains the machine learning modeling code. The following regression models were used:

- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet Regression
- Decision Tree Regression
- Random Forest Regression
- Gradient Boosting Regression

These models are used to predict both Yield Strength (MPa) and Ultimate Tensile Strength (MPa). We evaluate the models using metrics like:

- Mean Absolute Percentage Error (MAPE)
- R² Score
- Mean Squared Error (MSE)

## Streamlit App

A user-friendly web application built with Streamlit is available in the app.py script. This app allows users to:

- Select a target variable (either Yield Strength or Ultimate Tensile Strength).
- Choose between different models for prediction.
- Visualize model performance, including:
- Predicted vs Actual values.
- Feature importance (for tree-based models).
- Learning curves for different models.

## Conclusion
This project demonstrated the use of various machine learning models to predict key weld properties based on chemical composition and welding parameters. The predictive models can help optimize welding processes by providing insights into how different parameters impact the mechanical strength of welds. Further improvements could be made by:

- Fine-tuning hyperparameters.
- Testing more advanced models like XGBoost or Neural Networks.

## References
- [1] Tracey Cool, Design of Steel Weld Deposits, PhD Thesis, University of Cambridge, 1996.
- [2] Tracey Cool, H. K. D. H. Bhadeshia and David J. C. MacKay, Materials Science and Engineering, 1997.