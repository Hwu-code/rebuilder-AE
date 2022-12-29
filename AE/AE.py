# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 22:29:52 2022

@author: stran
"""

#%% import package
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import tensorflow.keras as keras
from keras.layers import Input, Dense, Lambda
from keras.models import Model, model_from_json
from keras import backend as K
from keras import losses

#%% data
data = pd.read_csv("C:/Users/stran/OneDrive/桌面/Python練習/論文/data/data2022 1-5(整理).csv")
data_train = data.iloc[0:25047]
data_test = data.iloc[25048:]
data_train = data_train.rename(columns = {'Unnamed: 0' : 'time'})
data_test = data_test.rename(columns = {'Unnamed: 0' : 'time'})
data_train = data_train.set_index('time')
data_test = data_test.set_index('time')
data_train_array = data_train.to_numpy()
data_test_array = data_test.to_numpy()

#%% model hyperparameters
batch_size, epochs, n_hidden, z_dim = 33, 100, 16, 2

#%% model architecture
x = Input(shape = data_train_array.shape[1])
encoder_layer1 = Dense(n_hidden, activation = 'relu')(x)
encoder_layer2 = Dense(n_hidden//2, activation = 'relu')(encoder_layer1)
latent = Dense(z_dim, activation = 'relu')(encoder_layer2)
decoder_layer1 = Dense(n_hidden//2, activation = 'relu')(latent)
decoder_layer2 = Dense(n_hidden, activation = 'relu')(decoder_layer1)
y = Dense(data_train_array.shape[1], activation= 'relu')(decoder_layer2)

#%% build model
AE = Model(x, y)
AE.compile(optimizer = 'rmsprop', loss = 'mse')
AE.summary()

#%% train
AE.fit(data_train_array,
       data_train_array,
       epochs=epochs,
       batch_size=batch_size)

# #%% save model.weight
# AE.save_weights("C:/Users/stran/OneDrive/桌面/Python練習/論文/AE/ae_model.weight")

# #%% call model.weight
# AE.load_weights("C:/Users/stran/OneDrive/桌面/Python練習/論文/AE/ae_model.weight", by_name=False)

#%% build encoder
encoder = Model(x, latent)
encoder.summary()

#%% plot the latent space
z = encoder.predict(data_test_array)
plt.figure(figsize=(6, 6))
plt.scatter(z[:, 0], z[:, 1], c='blue')
plt.show()

#%% reconstruct error
data_rec = AE.predict(data_test)
data_rec = pd.DataFrame(data_rec, columns = ['1','2','3','4','5','6','7','8','9','10'])
data_test_array = pd.DataFrame(data_test_array, columns = ['1','2','3','4','5','6','7','8','9','10'])
error = round((data_rec - data_test_array) / data_test_array, 4)

#%% visualize
error.plot()
plt.legend(fontsize = 'x-small')
