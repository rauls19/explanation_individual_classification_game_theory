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
X = pd.read_csv('D:\Projects\individual_classification_game_theory\dataset_X.csv')
Y = pd.read_csv('D:\Projects\individual_classification_game_theory\dataset_Y.csv')
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
print(minicg_lm.explainerContribution(lm_model, X, Y, 20, ['Age', 'Diastolic BP']))