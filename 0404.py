#0404.py
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
import numpy as np
import matplotlib.pyplot as plt
#1
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
print("x_train.shape:", x_train.shape)
print("y_train.shape:", y_train.shape)
print("x_test.shape:",  x_test.shape)
print("y_test.shape:",  y_test.shape)
  
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

#2
model = tf.keras.Sequential([
    Input(shape=(4,)),
    Dense(units=10, activation='sigmoid', name='layer1'),
    Dense(units=3,  activation='softmax', name = 'output')
    ])
#model.summary()

#3
opt = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

ret = model.fit(x_train, y_train, epochs=100, verbose=2) # batch_size=32

train_loss, train_acc = model.evaluate(x_train, y_train, verbose=2)
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2) 

#4 모델 동결(freezing)
import freeze_graph # freeze_graph.py
freeze_graph.freeze_model(model, "IRIS.pb")

