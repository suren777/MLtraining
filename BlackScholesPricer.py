
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import tensorflow.keras as k
import numpy as np
from scipy.stats import norm


def blackScholesPriceTrue(F, K, T, vol):
    sqt = vol * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * vol* vol * T)/sqt
    d2 = d1 - sqt
    n1 = norm.cdf(d1)
    n2 = norm.cdf(d2)
    return n1 - K/F * n2

def createDataset(size):
    f = np.linspace(0.5,1.5, size)
    k = np.linspace(0.5,1.5, size)
    T = np.linspace(0.1,1,size)
    vol = np.linspace(0.01, 0.5, size)
    inputs = np.array(np.meshgrid(f,k,T,vol)).T.reshape(-1, 4)
    options = blackScholesPriceTrue(inputs[:,0],inputs[:,1], inputs[:,2], inputs[:,3])
    return inputs, options

class OptionsSequence(k.utils.Sequence):

    def __init__(self, dataSetSize, batch_size):
        self.x, self.y = createDataset(dataSetSize)
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size,:]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return batch_x, batch_y

def add_normalized_layer(h_units, activation='relu'):
    return [
        k.layers.Dense(h_units, use_bias=False),
        k.layers.BatchNormalization(),
        k.layers.Activation(activation)
            ]

def create_model(input_dim, h_layers, h_units, modelFile = None):
    layers = [k.layers.Dense(units=h_units,  input_dim=input_dim)]
#    layers +=[k.layers.Dropout(rate=0.8)]
    for _ in range(h_layers):
        layers += add_normalized_layer(h_units)
    layers += [k.layers.BatchNormalization()]
    layers += [k.layers.Dense(units=1, activation='elu')]
    model = k.models.Sequential(layers)
    optimizer = k.optimizers.RMSprop(lr=0.001)
    model.compile(optimizer=optimizer, loss='mse')
    if modelFile is not None:
        try:
            model.load_weights(modelFile)
        except:
            pass
    return model

generator = OptionsSequence(100)

model = create_model(4, 5, 64, 'bsPricerModel.hdf5')

# model.fit_generator(