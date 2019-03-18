from DeepScholesSingleVol import create_model, dataGenerator, read_df, h_layers, h_units, modelFile, dfFile
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

model = create_model(input_dim=5, h_layers=h_layers, h_units=h_units, modelFile=modelFile)


X, y = read_df(dfFile, 512, 0)

inputs, vols = next(dataGenerator(200))
predictions = model.predict(inputs)

df = pd.DataFrame(data=np.array([vols.flatten(), predictions.flatten()]).T, columns=['vols', 'pred_vols']).sort_values(by=['vols'])

plt.figure(0)
plt.plot(df.vols.values, color='r', marker='d', alpha=.4, linestyle="None")
plt.plot(df.pred_vols.values, color='b', marker='x', alpha=.4, linestyle="None")
plt.title('IV')
plt.show()