from sklearn.neural_network.multilayer_perceptron import MLPRegressor
import pandas as pd

n_units = 256
n_layers = 10
bsCalibrator = MLPRegressor(hidden_layer_sizes=tuple(n_units for _ in range(n_layers)), n_iter_no_change=10, max_iter=200)

dfFile = 'bsFrame.csv'
df = pd.read_csv(dfFile)
size = df.shape[0]
val=.2

Xtrain = df.iloc[:int(size*val), :-1].values
Ytrain = df.iloc[:int(size*val), -1].values
Xval = df.iloc[int(size*val):, :-1].values
Yval = df.iloc[int(size*val):, -1].values
del df

res = bsCalibrator.fit(Xtrain, Ytrain)
YPred = bsCalibrator.predict(Xval)

import matplotlib.pyplot as plt

plt.figure(0)
plt.plot((Yval-YPred)/Yval*100)
plt.show()

bsPricer = MLPRegressor(hidden_layer_sizes=tuple(n_units for _ in range(n_layers)), n_iter_no_change=10, max_iter=200)

dfFile = 'bsFrame.csv'
df = pd.read_csv(dfFile)
col = df.columns
col = col[:-2]+[col[-2]]+ [col[-1]]
size = df.shape[0]
val=.2

Xtrain = df.iloc[:int(size*val), :-1].values
Ytrain = df.iloc[:int(size*val), -1].values
Xval = df.iloc[int(size*val):, :-1].values
Yval = df.iloc[int(size*val):, -1].values
del df

res = bsPricer.fit(Xtrain, Ytrain)
YPred = bsPricer.predict(Xval)

import matplotlib.pyplot as plt

plt.figure(1)
plt.plot((Yval-YPred)/Yval*100)
plt.show()