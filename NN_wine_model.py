# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 11:04:18 2022

@author: chenl
"""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Import Libraries Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import datetime
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Load Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
wine = pd.read_csv("winequality-white.csv", delimiter = ";", header = 0)
wine = wine.replace(np.nan,0)
wine = wine.values
print(wine.shape)

x = wine[:, 0:11]
y = wine[:, 11]


start_time = datetime.datetime.now()
np.random.seed(4)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, \
                                                    random_state = 4)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Model Building
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Input_Nodes = [15,20,25,30,35]
Hidden_Nodes = [15,20,25,30,35]
for Num_Nodes_Input in Input_Nodes:
    for Num_Nodes_Hidden in Hidden_Nodes:
        print("First Layer Nodes =", Num_Nodes_Input, ",", "Second Layer Nodes =",\
                         Num_Nodes_Hidden)
        model = Sequential()
        model.add(Dense(Num_Nodes_Input, input_dim = 11, activation = 'relu'))
        model.add(Dense(Num_Nodes_Hidden, input_dim = 11, activation = 'relu'))
        model.add(Dense(1, kernel_initializer ='normal', activation = 'linear'))
        
        # Compile model
        model.compile(loss ='mean_squared_error', optimizer = 'adam', \
                          metrics = ['mse', 'mae'])
        # Fit model
        model.fit(x_train, y_train, epochs = 100, batch_size = 50, verbose = 0)

        scores = model.evaluate(x_test, y_test, verbose = 0)
        print("Total loss: ", scores[0])
        print("Test MAE: ", scores[2])
        stop_time = datetime.datetime.now()
        print("Time required for training model ",Num_Nodes_Input, Num_Nodes_Hidden, \
              ': ',\
              stop_time - start_time)
        print()
        print("===================================================")


        