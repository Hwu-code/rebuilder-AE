# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 11:09:36 2022

@author: stran
"""
#%% import package
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import tensorflow.keras as keras
from keras.layers import Input, Dense, Lambda
from keras.models import Model, model_from_json
from keras import backend as K
from keras import losses

#%% import data
data_train = yf.download('^VIX', start="1990-01-01", end="2005-12-31")
data_test = yf.download('^VIX', start="2005-01-01", end="2020-12-31")
data_train = data_train.drop(columns = ['Volume'])
data_train_array = data_train.to_numpy()
data_test = data_test.drop(columns = ['Volume'])
data_test_array = data_test.to_numpy()
print(data_train_array.shape)
print(data_test_array.shape)

#%% model parameters
batch_size, epochs, n_hidden, z_dim = 1009, 100, 8, 2

# #%% model architecture
# class AE_model(keras.Model):
#     def __init__(self):
#         super().__init__()
#         self.encoder_layer1 = Dense(n_hidden, activation = 'relu')
#         self.encoder_layer2 = Dense(n_hidden//2, activation = 'relu')
#         self.encoder_layer3 = Dense(n_hidden//4, activation = 'relu')
#         self.latent = Dense(z_dim, activation = 'relu')
#         self.decoder_layer1 = Dense(n_hidden//4, activation = 'relu')
#         self.decoder_layer2 = Dense(n_hidden//2, activation = 'relu')
#         self.decoder_layer3 = Dense(n_hidden, activation = 'relu')
#         self.output_layer = Dense(inputs.shape[1], activation= 'relu')
        
#     def call(self, inputs):
#         x = self.encoder_layer1(inputs)
#         x = self.encoder_layer2(x)
#         x = self.encoder_layer3(x)
#         z = self.latent(x)
#         y = self.decoder_layer1(z)
#         y = self.decoder_layer2(z)
#         y = self.decoder_layer3(z)
#         predict = self.output_layer(y)
#         return predict
    
# AE = AE_model()
# y = AE(inputs)

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
AE.compile(optimizer = 'Adam', loss = 'mse')
AE.summary()

#%% train
AE.fit(data_train_array,
       data_train_array,
       epochs=epochs,
       batch_size=batch_size)

#%% build encoder
encoder = Model(x, latent)
encoder.summary()

#%% latent space
z = encoder.predict(data_train_array)

#%% plot the latent space
plt.figure(figsize=(6, 6))
plt.scatter(z[:, 0], z[:, 1], c='blue')
plt.show()

#%% build decoder
decoder = Model(latent, y)
decoder.summary()

#%% reconstruct
y_rec = decoder.predict(z)

#%% test
pre = AE.predict(data_train_array)

#%% display a predict plot
pre_data = data_train.copy()[[]] #copy df1 and erase all column
pre_data['Open'] = pre[:, 0]
pre_data['High'] = pre[:, 1]
pre_data['Low'] = pre[:, 2]
pre_data['Close'] = pre[:, 3]
pre_data['Adj Close'] = pre[:, 4]
pre_data = round(pre_data,2)

mc = mpf.make_marketcolors(up='r',down='g',inherit=True) 
s  = mpf.make_mpf_style(base_mpf_style='yahoo',marketcolors=mc) 
kwargs = dict(type='candle', mav=(5,20,60), figratio=(10,8), figscale=0.75, title='VIX index predicted', style=s) 
mpf.plot(pre_data, **kwargs)

#%% display a real plot
mc = mpf.make_marketcolors(up='r',down='g',inherit=True) 
s  = mpf.make_mpf_style(base_mpf_style='yahoo',marketcolors=mc) 
kwargs = dict(type='candle', mav=(5,20,60), figratio=(10,8), figscale=0.75, title='VIX index', style=s) 
mpf.plot(data_train, **kwargs)

#%% save model.weight
AE.save_weights("C:/Users/stran/OneDrive/桌面/Python練習/論文/AE test/ae_test_model.weight")

#%% call model.weight
AE.load_weights("C:/Users/stran/OneDrive/桌面/Python練習/論文/AE test/ae_test_model.weight", by_name=False)

test = decoder.predict(np.array([[0,50]]))
