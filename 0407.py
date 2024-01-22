#0407.py
'''
ref: https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence
'''

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers   import Input, Conv2D, MaxPool2D, Dense  
from tensorflow.keras.layers   import BatchNormalization, Dropout, Flatten
from tensorflow.keras.optimizers import Adam

#1            
import mnist  # mnist.py
(x_train, y_train), (x_test, y_test) = mnist.load(flatten=False, normalize=True)

#2
class Dataset(Sequence):
    def __init__(self, X, y, batch_size,
                       normalize=True, one_hot=True, shuffle=True):
        
        X = np.expand_dims(X, axis=3) #( , 28, 28, 1):NHWC

        if normalize:
            X = X.astype('float32')/255 #[0.0, 1.0]
        if one_hot:
            y = tf.keras.utils.to_categorical(y) 
            
        self.X = X
        self.y = y
        self.input_shape = X.shape[1:] # (28, 28, 1)
        self.batch_size = batch_size
        self.shuffle=shuffle
        self.indices = np.arange(len(self.X))

    def __len__(self):
        return int(np.ceil(len(self.X)/self.batch_size))

    def __getitem__(self, i):
        indices = self.indices[i*self.batch_size:(i+1)*self.batch_size]
        batch_X =  self.X[indices]
        batch_y =  self.y[indices]
        return batch_X, batch_y

    def on_epoch_end(self):       
        if self.shuffle:
            np.random.shuffle(self.indices)

#3: dataset
train_ds = Dataset(x_train, y_train, 64)
test_ds  = Dataset(x_test,  y_test, 64)

#4:  
model = tf.keras.Sequential([
    Input(train_ds.input_shape), # (28, 28, 1)
    Conv2D(filters=16, kernel_size = (3,3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPool2D(),
    Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu'),
    MaxPool2D(),
    Dropout( rate=0.2),
    Flatten(),
    Dense(units=10, activation='softmax')])
model.summary()

#5
opt = Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='categorical_crossentropy', #'mse'
                              metrics=['accuracy'])

ret = model.fit(train_ds, epochs=10, verbose=2)
test_loss,  test_acc  = model.evaluate(test_ds, verbose=2) 

#6 freezing
import freeze_graph # freeze_graph.py
freeze_graph.freeze_model(model, "MNIST_CNN2.pb")

