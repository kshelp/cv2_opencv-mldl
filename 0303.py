#0303.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

#1
def load_Iris():   
    label = {'setosa':0, 'versicolor':1, 'virginica':2}
    data = np.loadtxt("./data/iris.csv", skiprows = 1, delimiter = ',',
                      converters={4: lambda name: label[name.decode()]})
    return np.float32(data)
iris_data = load_Iris()

#2: split using TrainData
#2-1: 
X      = np.float32(iris_data[:,:-1])
y_true = np.float32(iris_data[:, -1])
data = cv2.ml.TrainData_create(samples=X,
                               layout=cv2.ml.ROW_SAMPLE,
                               responses=y_true)

#2-2:
data.setTrainTestSplitRatio(0.8) # train data: 80%, shuffle=True
#nSamples = data.getNSamples()   # 150
#data.setTrainTestSplit(int(nSamples*0.8)) # train data: 80%, shuffle=True
#data.shuffleTrainTest()

#2-3:
nTrain = data.getNTrainSamples() # 120
nTest = data.getNTestSamples()   #  30
#X = data.getSamples()
#y_true = data.getResponses()

#2-4:
x_train = data.getTrainSamples()
y_train = data.getTrainResponses()

x_test = data.getTestSamples()
y_test = data.getTestResponses() 

print("x_train.shape=", x_train.shape) # (120, 4)
print("y_train.shape=", y_train.shape) # (120, 1)
print("x_test.shape=",  x_test.shape)  # (30, 4)
print("y_test.shape=",  y_test.shape)  # (30, 1)

#2-5: print sample    
##print("x_train[:3]=", x_train[:3])
##print("y_train[:3]=", y_train[:3]) 
##print("x_test[:3]=",  x_test[:3])
##print("y_test[:3]=",  y_test[:3])
