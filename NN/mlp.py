# using Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import time
# Makes random numbers predictable
np.random.seed(7)
start_time = time.time()

# reading the dataset, splitting it into input and output
#load dataset
dataset = np.loadtxt("C:/Santhosh/AIML/data/pima-indians-diabetes.csv", delimiter=",")
#print(dataset)
print(dataset.shape)
# shape (768, 9) -- has 9 columns, Output -- 9th column, input -- first 8 columns
# X- input, Y- Output
x = dataset[:,0:8]
y = dataset[:,8]

print(x.shape) # (768, 8) -- 768 samples, 8 features
print(y.shape) # (768, ) -- 768 samples, 1 feature

print(x.size) # 6144 records/values
print(y.size) # 768 records/values


# Model
model = Sequential()
model.add((Dense(12, activation='relu', input_dim=8, name='Input')))
# model.add(Dense(8, activation='relu', name='hidden'))
model.add((Dense(1, activation='sigmoid', name='output')))

# Plotting model
from keras.utils.vis_utils import plot_model
plot_model(model, to_file='layers.png', show_shapes=True, show_layer_names=True)


#learning - loss, optimizer
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#train - epochs, dataset
model.fit(x, y, epochs=150)

#Evaluate - Returns the loss value & metrics values
metrics = model.evaluate(x, y)


print("\n%s: %.2f%%" % (model.metrics_names[1], metrics[1]*100))
print('Elapsed TIme',time.time() - start_time)


