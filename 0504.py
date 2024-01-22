#0504.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

#1:
TENSORFLOW, ONNX = 1, 2
net_type =  TENSORFLOW  # ONNX

if net_type == TENSORFLOW:
    net = cv2.dnn.readNet('./dnn/IRIS.pb')
else:
    net = cv2.dnn.readNet('./dnn/IRIS.onnx')
 
#2: load data
def load_Iris():   
    label = {'setosa':0, 'versicolor':1, 'virginica':2}
    data = np.loadtxt("./data/iris.csv", skiprows = 1, delimiter = ',',
                      converters={4: lambda name: label[name.decode()]})
    return np.float32(data)
iris_data = load_Iris()

np.random.seed(1)
def train_test_split(iris_data, ratio = 0.8, shuffle = True): # train: 0.8, test: 0.2   
    if shuffle:
        np.random.shuffle(iris_data)
        
    n = int(iris_data.shape[0] * ratio)
    x_train = iris_data[:n, :-1]
    y_train = iris_data[:n, -1]
    
    x_test = iris_data[n:, :-1]
    y_test = iris_data[n:, -1]
    return (x_train, y_train), (x_test, y_test)
    
(x_train, y_train), (x_test, y_test) = train_test_split(iris_data)

#3: classify and accuracy
#3-1
def calcAccuracy(y_true, y_pred, percent=True):
    N = y_true.shape[0] # number of data
    accuracy = np.sum(y_pred == y_true)/N
    if percent:
        accuracy *= 100
    return accuracy

#3-2: classify x_train
blob = x_train.reshape(-1, 1, 1, 4)
net.setInput(blob)
y_out = net.forward()
y_pred = np.argmax(y_out, axis=1)
train_accuracy = calcAccuracy(y_train, y_pred)
print('train_accuracy ={:.2f}%'.format(train_accuracy))

#3-3: classify x_test
blob = x_test.reshape(-1, 1, 1, 4)
net.setInput(blob)
y_out = net.forward()
y_pred = np.argmax(y_out, axis=1)
test_accuracy = calcAccuracy(y_test, y_pred)
print('test_accuracy ={:.2f}%'.format(test_accuracy))
