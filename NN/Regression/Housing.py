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

