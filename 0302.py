#0302.py
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

print("x_train.shape=", x_train.shape)
print("y_train.shape=", y_train.shape)
print("x_test.shape=",  x_test.shape)
print("y_test.shape=",  y_test.shape)

# print sample    
print("x_train[:3]=", x_train[:3])
print("y_train[:3]=", y_train[:3])
print("x_test[:3]=",  x_test[:3])
print("y_test[:3]=",  y_test[:3]) 
