from keras.models import load_model
import numpy as np
import math

model = load_model('modelBP.h5')

edad = 60
sexo = 0
bmi = 133.3
dia = 54
sys = 132
fuma = 1
padres =1



target = 1-math.exp(-math.exp((math.log(4) - (22.949536 + (-0.156412*edad )+( -0.202933*sexo) + (-0.033881*bmi) + (-0.05933*sys) + (-0.128468*dia) + (-0.190731*fuma) +  (-0.166121*padres) + (0.001624*edad*dia))/0.876925)))

risk = target*100
print(risk)


X_test = np.array([[edad,sexo,bmi,sys,dia,fuma,padres]])
y_test = np.array([[target]])
print(X_test.shape)
yp= model.predict(X_test)
print(yp)



score = model.evaluate(X_test, y_test,verbose=1)

print(score)




