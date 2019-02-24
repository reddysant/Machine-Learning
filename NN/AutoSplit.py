from keras.models import Sequential
from keras.layers import Dense
import  numpy as np

dataset = np.loadtxt("C:/Santhosh/AIML/data/pima-indians-diabetes.csv", delimiter=",")
x = dataset[:,:8]
y = dataset[:,8]

def base_model():
    model= Sequential()
    model.add(Dense(8, kernel_initializer='normal', activation='relu', name='Input'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid', name='Output'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return  model
model = base_model()
model.fit(x,y, validation_split=0.30,  epochs=150, batch_size=10)

print('acc: ', model.evaluate(x,y)[1]*100, '%')






