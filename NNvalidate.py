# Import pandas
import pandas as pd
from keras.models import load_model
import numpy as np


model = load_model('modelBP.h5')

# Read in white wine data
datos = pd.read_csv("pacientes200.csv", sep=',')
datas=200

dataset = np.array(datos)

y_test = dataset[0:datas,7:8]
X_test = dataset[0:datas,0:7]
print(dataset.shape)
print(X_test.shape)
print(y_test.shape)

yp= model.predict(X_test)
print(yp)

score = model.evaluate(X_test, y_test,verbose=1)

print(score)
