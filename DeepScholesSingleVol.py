import tensorflow.keras as k
import numpy as np
import pickle
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def blackScholesPrice(F, K, T, vol):
    sqt = vol * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * vol* vol * T)/sqt
    d2 = d1 - sqt
    n1 = norm.cdf(d1)
    n2 = norm.cdf(d2)
    return F * n1 - K * n2

def mapHelper(a):
    F, K, T, vol = a
    return [blackScholesPrice(F, K, T, vol)]

class DataGenerator(k.utils.Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x[0]) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[0][idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return batch_x, batch_y


def dataGenerator(batchSize):
    np.random.seed(121)
    while True:
        vols = np.random.uniform(0.5, 1, batchSize)
        strikes = np.random.uniform(.5, 1.5, batchSize)
        forwards = np.random.uniform(.5, 1.5, batchSize)
        tenors = np.random.uniform(0.001, 2, batchSize)
        logMoneyness = np.log(np.divide(strikes, forwards))
        optionPrices = np.array(list(map(mapHelper, np.array([forwards, strikes, tenors, vols]).T)))
        inputs = np.array([forwards, strikes, tenors, logMoneyness, optionPrices.reshape(batchSize)]).T
        targets = vols.reshape((batchSize,1))
        yield inputs, targets

def dataFrameGenerator(batchSize):
    np.random.seed(121)
    while True:
        vols = np.random.uniform(0.5, 1, batchSize)
        strikes = np.random.uniform(.5, 1.5, batchSize)
        forwards = np.random.uniform(.5, 1.5, batchSize)
        tenors = np.random.uniform(0.001, 2, batchSize)
        logMoneyness = np.log(np.divide(strikes, forwards))
        optionPrices = np.array(list(map(mapHelper, np.array([forwards, strikes, tenors, vols]).T)))
        yield pd.DataFrame(np.array([forwards, strikes, tenors, logMoneyness, optionPrices.reshape(batchSize), vols]).T,
                          columns=['forwards,strikes,tenors,logMoneyness,optionPrices,vols'.split(',')])

def createDataframe(batchSize,df):
    np.random.seed(121)
    vols = np.random.uniform(0.5, 1, batchSize)
    strikes = np.random.uniform(.5, 1.5, batchSize)
    forwards = np.random.uniform(.5, 1.5, batchSize)
    tenors = np.random.uniform(0.001, 2, batchSize)
    logMoneyness = np.log(np.divide(strikes, forwards))
    optionPrices = np.array(list(map(mapHelper, np.array([forwards, strikes, tenors, vols]).T)))
    pd.DataFrame(np.array([forwards, strikes, tenors, logMoneyness, optionPrices.reshape(batchSize), vols]).T,
                      columns=['forwards,strikes,tenors,logMoneyness,optionPrices,vols'.split(',')]).to_csv(df, index=False)

def custom_activation(x):
    return k.backend.exp(x)

def loadData(file):
    with open(file, 'rb') as f:
        res = pickle.load(f)
    return res

def saveData(obj, file):
    with open(file, 'wb') as f:
        pickle.dump(obj, f)

def read_df(df, batch_size, position, dfSize = 256000):
    max_position = dfSize//batch_size
    if position>max_position:
        position = 0
    res = pd.read_csv(df, skiprows=int(batch_size*position), nrows=int(batch_size)).values
    return res[:, :-1], res[:,-1].reshape((int(batch_size),1))

def add_normalized_layer(h_units, activation='relu'):
    return [
        k.layers.Dense(h_units, use_bias=False),
        k.layers.BatchNormalization(),
        k.layers.Activation(activation)
            ]


def create_model(input_dim, h_layers, h_units, modelFile = None):
    layers = [k.layers.Dense(units=h_units,  input_dim=input_dim)]
#    layers +=[k.layers.Dropout(rate=0.8)]
    for _ in range(h_units):
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

h_layers = 8
h_units = 256
modelFile = 'bs_2_model.h5'
dataFile = 'bsDataset.pkl'
dfFile = 'bsFrame.csv'
dataFrameSize = 256000

if __name__=='__main__':

    import os
    if dfFile not in os.listdir():
        createDataframe(dataFrameSize, dfFile)

    model = create_model(input_dim=5, h_layers=h_layers, h_units=h_units, modelFile=modelFile)

    batch_size = 1024
    epochs = dataFrameSize//(batch_size*4)
    inner_epoch = 100
    for i in tqdm(range(epochs)):
        X, y = read_df(dfFile, batch_size*4, i, dataFrameSize)
        hist = model.fit(x=X, y=y, epochs=inner_epoch, batch_size=batch_size,
                  validation_split=0.05, shuffle=True, verbose=2)

        if i > 0:
            if hist.history['val_loss'][-1] <= 0:
                print("Overflow issue -- Terminating")
                break
            if hist.history['val_loss'][-1] < val_loss and hist.history['val_loss'][-1] > 0:
                val_loss = hist.history['val_loss'][-1]
                model.save_weights(modelFile)
                print("\n\tNew best val_loss:{0} \t on epoch: {1} ".format(val_loss, i*inner_epoch))
        else:
            val_loss = hist.history['val_loss'][0]

    model.save(modelFile)


    X_test, y_test = next(dataGenerator(1))
    model.evaluate(X_test, y_test)


    inputs, vols = next(dataGenerator(200))
    predictions = model.predict(inputs)

    df = pd.DataFrame(data=np.array([vols.flatten(), predictions.flatten()]).T, columns=['vols', 'pred_vols']).sort_values(by=['vols'])

    plt.figure(0)
    plt.plot(df.vols.values, color='r', marker='d', alpha=.4)
    plt.plot(df.pred_vols.values, color='b', marker='x', alpha=.4)
    plt.title('IV')
    plt.show()

