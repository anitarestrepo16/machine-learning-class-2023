#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 21:16:10 2022

@author: stevekim
"""

import pandas as pd

dat = pd.read_csv('/Users/stevekim/Desktop/Spring 2022/Machine Learning/winequality-red.csv')

dat.head()

# Wow, what's going on? (CSV actually separated by ;)

dat = pd.read_csv('/Users/stevekim/Desktop/Spring 2022/Machine Learning/winequality-red.csv', sep = ';')

dat.head()

dat.dtypes

pd.set_option('display.max_columns', None)

dat.describe()

# For k-NN, whether we normalize the variance is an important decision

dat_std = dat.describe().iloc[2]

dat_std

# Looking at the predictors, should we normalize? Yes

x = dat.drop(['quality'], axis = 1)

from sklearn import preprocessing

x_norm = pd.DataFrame(preprocessing.scale(x))

y = dat['quality']

q = pd.DataFrame(np.where(y > 6, 1, 0)).squeeze()

# We just made wine quality binary: good if quality > 6; bad otherwise

# Now we try k-NN with k = 1

from sklearn.neighbors import KNeighborsClassifier

x_norm_train = x_norm.head(800)

q_train = q.head(800)

x_norm_test = x_norm.tail(799)

q_test = q.tail(799)

knn_1 = KNeighborsClassifier(n_neighbors = 1)

knn_1.fit(x_norm_train, q_train)

q_pred_knn1 = knn_1.predict(x_norm_test)

from sklearn.metrics import accuracy_score

accuracy_score(q_test, q_pred_knn1)

from sklearn.metrics import confusion_matrix

cm_knn1 = confusion_matrix(q_test, q_pred_knn1)

print(cm_knn1)

# Of 123 (= 72 + 51) wines classified as good, 51 are good 

knn_11 = KNeighborsClassifier(n_neighbors = 11)

knn_11.fit(x_norm_train, q_train)

q_pred_knn11 = knn_11.predict(x_norm_test)

accuracy_score(q_test, q_pred_knn11)

cm_knn11 = confusion_matrix(q_test, q_pred_knn11)

print(cm_knn11)

# Of 48 (14 + 34) wines classified as good, 34 are good

# Now we try finding the k that yields the lowest classification
# error rate with the first 800 as training set

def odd(n):
    return list(range(1, 2*n, 2))

ks = odd(400)

ac_rate = []

for i in ks:
     knn = KNeighborsClassifier(n_neighbors=i)
     knn.fit(x_norm_train, q_train)
     pred_i = knn.predict(x_norm_test)
     ac_rate.append(np.mean(pred_i == q_test))

max_value = max(ac_rate)

print(max_value)

# Looks like the best classification error we can get is around 14%

opt_k = ac_rate.index(max_value)

print(opt_k)

# The fifth odd number is the optimal k, which is 9

# We now try 5-fold cross-validation to estimate optimal k 

from sklearn.model_selection import GridSearchCV, KFold

knni = KNeighborsClassifier()

para = {'n_neighbors':ks}

knn_cv = GridSearchCV(knni, para, cv = KFold(5, random_state=40, shuffle=True))

knn_cv.fit(x_norm, q)

knn_cv.best_params_

knn_cv.best_score_

# Apparently, k = 1 is best. Let's redefine good wine as wine with
# quality strictly greater than 4

q2 = pd.DataFrame(np.where(y > 4, 1, 0)).squeeze()

knn_cv.fit(x_norm, q2)

knn_cv.best_params_

# With the new definition of good wine, optimal k is 19

knn_cv.best_score_

# We now manipulate the cost function. Say we only care about 
# classifying bad wine as good wine 

c10 = []
for i in ks:
     knn = KNeighborsClassifier(n_neighbors=i)
     knn.fit(x_norm_train, q_train)
     pred_i = knn.predict(x_norm_test)
     c10.append(sum(pred_i - q_test > 0))

cost = np.ndarray.tolist(np.array(c10))

min_value = min(cost)

opt_k2 = cost.index(min_value)

print(opt_k2)

# The 24th odd number is best; opt_k2 = 47
