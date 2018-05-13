from keras.models import load_model
import numpy as np


model = load_model('modelBP.h5')

edad = 60
sexo = 0
BMI = 133
diastolica = 132
siastolica = 54
fuma = 1
padres =1
target = 0.99



X_test = np.array([[edad,sexo,BMI,diastolica,siastolica,fuma,padres]])
y_test = np.array([[target]])
print(X_test.shape)
yp= model.predict(X_test)
print(yp)



score = model.evaluate(X_test, y_test,verbose=1)

print(score)




