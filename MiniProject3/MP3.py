#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 14:04:40 2023

@author: stevekim
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
import matplotlib.pyplot as plt

# OLS MSE calculation

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=10) 

ls_results_train = sm.OLS(y_train, X_train).fit()

sum(np.square(y_train - ls_results_train.fittedvalues)) / len(y_train)

y_test_pred = ls_results_train.predict(X_test)

sum(np.square(y_test - y_test_pred)) / len(y_test)

# Lasso 

kf = KFold(n_splits=10, random_state = 25, shuffle=True)

tkf = kf.split(X, y)

lasso = Lasso()

alphacl = (np.linspace(0, 1, 101))

alpha_lgrid = [{'alpha': alphacl}]

def vector_values(grid_search, trials):
    mean_vec = np.zeros(trials)
    std_vec = np.zeros(trials)
    i = 0
    final = grid_search.cv_results_
    
   
    for mean_score, std_score in zip(final["mean_test_score"], final["std_test_score"]):
        mean_vec[i] = -mean_score
        std_vec[i] = std_score
        i = i+1

    return mean_vec, std_vec

grid_search_lasso = GridSearchCV(lasso, alpha_lgrid, cv = tkf, scoring = 'neg_mean_squared_error')

grid_search_lasso.fit(X, y)

mean_vec, std_vec = vector_values(grid_search_lasso, 101)

results_cv = pd.DataFrame(
    {
        "alphas": alphacl,
        "MSE": mean_vec,
    }
)

min_mse = min(results_cv['MSE'])

results_cv.loc[results_cv['MSE']==min_mse]

plt.plot(
    alphacl,
    mean_vec,
    linewidth=2,
)

# Ridge 

kf = KFold(n_splits=10, random_state = 25, shuffle=True)

tkf = kf.split(X, y)

ridge = Ridge()

alphacr = (np.linspace(0, 25, 251))

alpha_rgrid = [{'alpha': alphacr}]

grid_search_ridge = GridSearchCV(ridge, alpha_rgrid, cv = tkf, scoring = 'neg_mean_squared_error')

grid_search_ridge.fit(X, y)

mean_vec, std_vec = vector_values(grid_search_ridge, 251)

results_cv = pd.DataFrame(
    {
        "alphas": alphacr,
        "MSE": mean_vec,
    }
)

min_mse = min(results_cv['MSE'])

results_cv.loc[results_cv['MSE']==min_mse]

plt.plot(
    alphacr,
    mean_vec,
    linewidth=2,
)