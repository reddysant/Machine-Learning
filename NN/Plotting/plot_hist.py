from keras.layers import  Dense
from keras.models import Sequential
from matplotlib import pyplot as plt
import numpy as np

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
dataset = np.loadtxt("C:/Santhosh/AIML/data/pima-indians-diabetes.csv", delimiter=",")

# split into input (X) and output (Y) variables
x = dataset[:,0:8]
y = dataset[:,8]

model = Sequential()
model.add(Dense(10, kernel_initializer='uniform', input_dim=8, name='input', activation='relu'))
model.add(Dense(200, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

hist = model.fit(x,y, validation_split=0.33, epochs=150,batch_size=10, verbose=0)

print(hist.history.keys())

#Plot - Accuracy
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Accuracy - Test vs Train')
plt.legend(['train','val'])
plt.show()

#plot - Loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Loss - Train vs Test')
plt.legend(['Train', 'Test'])
plt.show()

print('Acc: ', model.evaluate(x,y)[1]*100,'%')












