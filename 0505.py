#0505.py
import gzip
import cv2
import numpy as np
import matplotlib.pyplot as plt

#1:
TENSORFLOW, ONNX = 1, 2
net_type =  TENSORFLOW  # ONNX

if net_type == TENSORFLOW:
    net = cv2.dnn.readNet('./dnn/MNIST_DENSE.pb')
else:
    net = cv2.dnn.readNet('./dnn/MNIST_DENSE.onnx')
    
#2
import mnist # mnist.py
(x_train, y_train), (x_test, y_test)=mnist.load(flatten=True, normalize= True)

#3: classify and accuracy
#3-1
def calcAccuracy(y_true, y_pred, percent=True):
    N = y_true.shape[0] # number of data
    accuracy = np.sum(y_pred == y_true)/N
    if percent:
        accuracy *= 100
    return accuracy

#3-2: classify x_train
n = x_train.shape[1] # 784
blob = x_train.reshape(-1, 1, 1, n) 
net.setInput(blob)
y_out = net.forward()
y_pred = np.argmax(y_out, axis=1)
train_accuracy = calcAccuracy(y_train, y_pred)
print('train_accuracy ={:.2f}%'.format(train_accuracy))

#3-3: classify x_test
n = x_test.shape[1]
blob = x_test.reshape(-1, 1, 1, n)
net.setInput(blob)
y_out = net.forward()
y_pred = np.argmax(y_out, axis=1)
test_accuracy = calcAccuracy(y_test, y_pred)
print('test_accuracy ={:.2f}%'.format(test_accuracy))
