# Import pandas
import pandas as pd
from keras.models import load_model
import numpy as np


model = load_model('modelBP.h5')

# Read in white wine data
datos = pd.read_csv("pacientesreal.csv", sep=',')


dataset = np.array(datos)

y_test = dataset[0:14,7:8]
X_test = dataset[0:14,0:7]
print(dataset.shape)
print(X_test.shape)
print(y_test.shape)

yp= model.predict(X_test)
print(yp)

score = model.evaluate(X_test, y_test,verbose=1)

print(score)
