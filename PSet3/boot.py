#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 10:44:51 2023

@author: stevekim
"""

# You are to replace the boot function provided in the lab with below
# Note line 19, which wasn't there in the lab

def boot(data, func, B):
    coef_intercept = []
    coef_balance = []
    coef_income = []
    coefs = ['intercept', 'balance', 'income']
    output = {coef: [] for coef in coefs}
    for i in range(B):
        np.random.seed(i)
        reg_out = func(data, get_indices(data, len(data)))
        for i, coef in enumerate(coefs):
            output[coef].append(reg_out[i])
    results = {}
    for coef in coefs:
        results[coef] = {
            'estimate': np.mean(output[coef]),
            'std_err': np.std(output[coef])
        }
    return results



