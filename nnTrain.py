# Import pandas
import pandas as pd
import numpy as np

# Read in white wine data
datos = pd.read_csv("pacientestrain.csv", sep=',')


dataset = np.array(datos)


X_train = dataset[0:500,0:7]
y_train = dataset[0:500,7:8]
X_test = dataset[0:500,0:7]
print(dataset.shape)
print(X_train.shape)
print(y_train.shape)


# Import `Sequential` from `keras.models`
from keras.models import Sequential

# Import `Dense` from `keras.layers`
from keras.layers import Dense

# Initialize the constructor
model = Sequential()

# Add an input layer
model.add(Dense(32, activation='sigmoid', input_shape=(7,)))

# Add one hidden layer
model.add(Dense(64, activation='sigmoid'))

model.add(Dense(7, activation='sigmoid'))




# Add an output layer
model.add(Dense(1, activation='sigmoid'))


# Model output shape
print(model.output_shape)

# Model summary
print(model.summary())

# Model config
print(model.get_config())

# List all weight tensors
print(model.get_weights())

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=200000, batch_size=10, verbose=1)



y_pred = model.predict(X_test)

model.save('modelBP.h5')  # creates a HDF5 file 'my_model.h5'

#print(y_pred)
print("end")