#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 11:08:38 2024

@author: srilu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# To scale the data using z-score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Algorithms to use
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor

# Metrics to evaluate the model
from sklearn import metrics
from sklearn.metrics import r2_score, mean_squared_error

# For tuning the model
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('/Users/srilu/Documents/Financial Modelling/Case Study/Case 4 House Price_Decision Trees/13.5 House Price dataset.csv')
df.head()
df.info()
df.duplicated().sum()
round(df.isnull().sum() / df.isnull().count() * 100, 2)
df.nunique()
round(df.describe(),2)

# Creating numerical columns
num_cols = ['Sale_amount', 'Beds', 'Baths', 'Sqft_home', 'Sqft_lot', 'Age']

# Creating categorical variables
cat_cols = ['Type', 'Town', 'University']

# Exploratory Data Analysis

# Creating histograms
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

# Iterating over each column and plotting histograms
for ax, col in zip(axes.flatten(), num_cols):
    df[col].hist(ax=ax, grid=False,edgecolor='black')
    ax.set_title(col, fontsize=14)
    ax.set_xlabel('')
    ax.set_ylabel('Frequency', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()

# Printing the % sub categories of each category.
for i in cat_cols:
    print(df[i].value_counts(normalize = True))
    print('*' * 40)

# Correlation Analysis
sns.heatmap(data=df[['Sale_amount', 'Beds', 'Baths', 'Sqft_home', 'Sqft_lot', 'Age']].corr(), annot=True, fmt=".2f", cmap='PuBu')

# Data Preprocessing
# Dropping the columns
df = df.drop(columns=['Record','Sale_date','Build_year'],axis=1)

# Replacing Multi Family and Multiple Occupancy with OTHERS
df['Type'] = df['Type'].replace({'Multi Family': 'Others','Multiple Occupancy': 'Others'})

# Creating dummy variables for the categorical variables
df = pd.get_dummies(df, drop_first=True)

# Assign the feature data and target data to X and y, respectively.
X = df.iloc[:, 1: ]
y = df.iloc[:, 0]

fn = X.columns[0:]

# z_score normalize the data except the dummy variables
scaler = StandardScaler()
Xn = np.c_[scaler.fit_transform(X.iloc[:,:5].values), X.iloc[:, 5:].values] 

# Building the Decision Tree Model

# Divide the data for training and testing
Xn_train,Xn_test, y_train, y_test = train_test_split(Xn,y, test_size=.3,random_state=1234)

# Building decision tree model
dtr = DecisionTreeRegressor(random_state = 1234)

# Fitting decision tree model
dtr.fit(Xn_train, y_train)

# Checking performance on training set
y_train_pred = dtr.predict(Xn_train)

# Performance metrics
r2_train = r2_score(y_train, y_train_pred)
mse_train = mean_squared_error(y_train, y_train_pred)

print("R2 Score on Training Set:", r2_train)
print("Mean Squared Error on Training Set:", mse_train)

# Checking performance on test set
y_test_pred = dtr.predict(Xn_test)

# Performance metrics
r2_test = r2_score(y_test, y_test_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
print("R2 Score on Tesring Set:", r2_test)
print("Mean Squared Error on Testing Set:", mse_test)


# ================================================
# Tuning the model using GridSearchCV
# ================================================
dt = DecisionTreeRegressor(random_state = 1234)

# Grid of parameters to choose from
# Define the parameter grid to search over
param_grid = {
    'max_depth': [5, 10, 15, 20, 30, 50], # Maximum depth of the tree
    'min_samples_split': [5, 10, 15, 20, 30, 50],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4, 6, 8],  # Minimum number of samples required to be at a leaf node
    'max_features': [10, 15, 20,'auto', 'sqrt', 'log2']  # Number of features to consider when looking for the best split
}

# Instantiate the GridSearchCV object
grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)

# Fit the GridSearchCV object to the data
grid_search.fit(Xn_train, y_train)

# Get the best parameters and the best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best Score:", best_score)

# Set the regressor to the best combination of parameters
dt = grid_search.best_estimator_

# Fit the best estimator to the data
dt.fit(Xn_train, y_train)

# Checking performance on training set
y_train_pred_dt = dt.predict(Xn_train)

# Performance metrics
r2_test_dt = r2_score(y_train, y_train_pred_dt)
mse_test_dt = mean_squared_error(y_train, y_train_pred_dt)

print("R2 Score on Training Set:", r2_test_dt)
print("Mean Squared Error on Training Set:", mse_test_dt)

# Checking performance on test set
y_test_pred_dt = dt.predict(Xn_test)

# Performance metrics
r2_test_dt = r2_score(y_test, y_test_pred_dt)
mse_test_dt = mean_squared_error(y_test, y_test_pred_dt)
print("R2 Score on Testing Set:", r2_test_dt)
print("Mean Squared Error on Testing Set:", mse_test_dt)

# Decision Tree Plot
plt.figure(figsize=(25, 10)) 
tree.plot_tree(dt, max_depth=4, feature_names=fn, filled=True, fontsize=12, class_names=True)
plt.show()

# Plot the feature importances
importances = dt.feature_importances_
columns = X.columns
importance_df = pd.DataFrame(importances, index = columns, columns = ['Importance']).sort_values(by = 'Importance', ascending = False)

# Selecting the top 10 features
top_15_features = importance_df.head(15)

# Plotting the top 10 feature importances
plt.figure(figsize=(9, 9))
sns.barplot(x=top_15_features.Importance, y=top_15_features.index)

# Adding annotations
for i, v in enumerate(top_15_features.Importance):
    plt.text(v + 0.002, i, f'{v*100:.2f}%', va='center')  
    
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top 15 Feature Importances')
plt.show()

top_15_feature_names = top_15_features.index
X_reduced = X[top_15_feature_names]

dt_top_15_features = DecisionTreeRegressor()
dt_top_15_features.fit(X_reduced, y)

# Plot the decision tree of top 15 features
plt.figure(figsize=(25, 10))
tree.plot_tree(dt_top_15_features, max_depth=4, filled=True, feature_names=top_15_feature_names, fontsize=10, class_names=True)
plt.show()


