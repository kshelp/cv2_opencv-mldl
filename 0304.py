#0304.py
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

#2
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

#3
model = cv2.ml.KNearest_create()
ret = model.train(samples=x_train, layout=cv2.ml.ROW_SAMPLE, responses=y_train)

#4:
ret, train_pred = model.predict(x_train) # model.getDefaultK() = 10
train_pred = train_pred.flatten()
train_accuracy = np.sum(y_train == train_pred) / len(y_train)
print('train_accuracy=', train_accuracy)

#5:
ret, test_pred = model.predict(x_test) # model.getDefaultK() = 10
test_pred = test_pred.flatten()
test_accuracy = np.sum(y_test == test_pred) / len(y_test)
print('test_accuracy=', test_accuracy)


