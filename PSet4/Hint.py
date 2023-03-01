#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 18:59:41 2023

@author: stevekim
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier

# With cost complexity tuning 

tree = DecisionTreeClassifier(random_state=2)

path = tree.cost_complexity_pruning_path(X_train, y_train)

alphas = path.ccp_alphas

kf = KFold(n_splits=5, random_state = 13, shuffle=True)

tree_test_ac = []

for a in alphas:
    tree = DecisionTreeClassifier(ccp_alpha=a, random_state=2)
    cv_results = cross_val_score(tree, X_train, y_train, cv=kf)
    tree_test_ac.append(cv_results.mean())
    
# You are to write the rest on your own. You've already learned
# what you need to know. So if you feel lost, please review 
# code from answer keys, lab, and additional py.files I've
# uploaded 