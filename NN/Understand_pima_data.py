# Create your first MLP in Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy

# fix random seed for reproducibility
numpy.random.seed(7)

# load pima indians dataset
dataset = numpy.loadtxt("C:/Santhosh/AIML/data/pima-indians-diabetes.csv", delimiter=",")

x = dataset[:,];
print(x.shape)
# shape gives number of rows and colums, A touple
print(x)

# [row, columns] --> [:,:] --> all rows and columns  --> [:,:8]--> all rows and first 8 columns

