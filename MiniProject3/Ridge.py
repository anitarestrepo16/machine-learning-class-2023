#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 21:35:01 2023

@author: stevekim
"""

Ridge 

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
import matplotlib.pyplot as plt

dat = pd.read_csv("/Users/stevekim/Desktop/Economics/2023 Winter/Machine Learning/Lab 4 (MP2)/winequality-red.csv", sep =';')

dat = dat.dropna()

y = dat['quality']

X = dat.drop(['quality'], axis = 1)

X = preprocessing.scale(X)

kf = KFold(n_splits=10, random_state = 55, shuffle=True)

tkf = kf.split(X, y)

ridge = Ridge()

alphac = (np.linspace(0, 1000, 201))

alpha_grid = [{'alpha': alphac }]

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

grid_search_ridge = GridSearchCV(ridge, alpha_grid, cv = tkf, scoring = 'neg_mean_squared_error')

grid_search_ridge.fit(X, y)

mean_vec, std_vec = vector_values(grid_search_ridge, 201)

results_cv = pd.DataFrame(
    {
        "alphas": alphac,
        "MSE": mean_vec,
    }
)

min_mse = min(results_cv['MSE'])

results_cv.loc[results_cv['MSE']==min_mse]

plt.plot(
    alphac,
    mean_vec,
    linewidth=2,
)

