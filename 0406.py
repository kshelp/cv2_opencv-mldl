#0406.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers   import Input, Conv2D, MaxPool2D, Dense  
from tensorflow.keras.layers   import BatchNormalization, Dropout, Flatten
from tensorflow.keras.optimizers import Adam

#1
import mnist # mnist.py 
(x_train, y_train), (x_test, y_test) = mnist.load(flatten=False, one_hot=True, normalize = True)

#: expand data with channel = 1
x_train = np.expand_dims(x_train,axis=3) # (60000, 28, 28, 1) # (N, H, W, C)
x_test  = np.expand_dims(x_test, axis=3) # (10000, 28, 28, 1)

#2:  
model = tf.keras.Sequential([
    Input(x_train.shape[1:]),
    Conv2D(filters=16, kernel_size = (3,3), activation='relu'),
    BatchNormalization(),
    MaxPool2D(),
    Conv2D(filters=32, kernel_size=(3,3), activation='relu'),
    MaxPool2D(),
    Dropout( rate=0.1),
    Flatten(),
    Dense(units=10, activation='softmax')])
model.summary()

#3
opt = Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

ret = model.fit(x_train, y_train, epochs=10,  batch_size=64, verbose=2)

train_loss, train_acc = model.evaluate(x_train, y_train, verbose=2)
test_loss,  test_acc  = model.evaluate(x_test,  y_test, verbose=2) 

#4 모델 동결(freezing)
import freeze_graph # freeze_graph.py
freeze_graph.freeze_model(model, "MNIST_CNN.pb")
