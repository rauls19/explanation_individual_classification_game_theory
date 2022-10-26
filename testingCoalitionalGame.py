"""
Usage: Testing Coalitional Game py
Python version: 3.9.X
Author: rauls19
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import random
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import os
from CoalitionalGame import miniCoalitionalGame
from sklearn.feature_selection import RFE
from sklearn.neural_network import MLPRegressor
# from LSTM import LSTM


# INI - LOAD DATA
X = pd.read_csv('dataset_X.csv')
Y = pd.read_csv('dataset_Y.csv')
X.drop(columns=['Unnamed: 0'], inplace=True)
Y.drop(columns=['Unnamed: 0'], inplace=True)
X.fillna(0, inplace=True)
Y.fillna(0, inplace=True)
# END

# INI - TESTING LINEAR MODEL
lm_model = LinearRegression()
minicg_lm = miniCoalitionalGame()
if not minicg_lm.check_model_availability(lm_model):
    print('Model is not available')
print('Linear Regression: ', minicg_lm.explainerContribution(lm_model, X, Y, X, 20, ['Age', 'Diastolic BP']))

# INI - TESTING MLP
nn_reg = MLPRegressor(hidden_layer_sizes=(60, 20),  activation='logistic', solver='adam', 
alpha=0.01, batch_size='auto', learning_rate='constant', learning_rate_init=0.01, max_iter=1000,
    shuffle=False, tol=0.0001, verbose=False, early_stopping= True, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
coalgame_nn_reg = miniCoalitionalGame()
print('MLP Regressor: ', coalgame_nn_reg.explainerContribution(nn_reg, X, Y, X, 5, ['Age']))

# INI - TESTING XGB
xgbm = xgb.XGBRegressor(learning_rate =0.01, n_estimators=215, max_depth=10, min_child_weight=0.8, subsample=1, nthread=4)
xgbm = xgb.XGBRegressor(learning_rate =0.01, n_estimators=215, max_depth=10, min_child_weight=0.8, subsample=1, nthread=4)
coalgame_xgb = miniCoalitionalGame()
print('XGB Regressor: ', coalgame_xgb.explainerContribution(xgbm, X, Y, X, 5, ['Age', 'Diastolic BP']))
