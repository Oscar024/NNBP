from keras.models import Sequential
from keras.layers import LSTM, Dense
import pandas as pd
import numpy as np


# Read in white wine data
datos = pd.read_csv("pacientestrain.csv", sep=',')


dataset = np.array(datos)


X_train = dataset[0:500,0:7]
y_train = dataset[0:500,7:8]
X_test = dataset[0:500,0:7]



# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(32, return_sequences=True, input_shape=(1,7)))
model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
#model.add(LSTM(32))  # return a single vector of dimension 32
model.add(Dense(1, activation='linear'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


model.fit(X_train, y_train,
          batch_size=1, epochs=1,verbose=1)