from keras.models import Sequential
from keras.layers import  Dense
import numpy as np
from pandas import read_csv

dataframe = read_csv("C:\Santhosh\AIML\data\housing.csv", delim_whitespace=True)
dataset = dataframe.values
print(dataset.shape)

x= dataset[:,:13]
y = dataset[:,13]

print(x.shape)
print(y.shape)

def base_model():
    model = Sequential()
    #kernel_initializer --> Initial random varibales
    model.add(Dense(13, activation='relu', input_dim=13, name='input', kernel_initializer='normal'))
    # model.add(Dense(12, activation='relu', name='hidden1'))
    model.add(Dense(1, name='output', kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model

np.random.seed(7)
model = base_model()
# train - Fit
model.fit(x,y, epochs=500, batch_size=100, verbose=1, shuffle=False)

# get matrx - Evaluate
metrics = model.evaluate(x,y)
print('acc: ',metrics[1]*100, '%')

#predict
print('x=', x[1])
print('y=', y[1])
print(x[1])

print("==============")

prediction = model.predict(x[1].reshape(1,13))
print(prediction)

# #Build the model
# model = base_model()
# #Train the model
# model.fit(x, y, epochs=300, batch_size=100, verbose=1, shuffle=False)
#
# # evaluate the model
# scores = model.evaluate(x, y)
# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#
# #predict
# print(x[1])
# print(y[1])
# print("==============")
# prediction = model.predict(x[1].reshape(1,13))
# print(prediction)
