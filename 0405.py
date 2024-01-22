#0405.py
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense

###1-1
##from tensorflow.keras.datasets import mnist
##(x_train, y_train), (x_test, y_test) = mnist.load_data()
##
###flatten
##x_train = x_train.reshape(-1, 784)
##x_test  = x_test.reshape(-1, 784)
##
###normalize images
##x_train = x_train.astype('float32')
##x_test  = x_test.astype('float32')
##x_train /= 255.0 # [0.0, 1.0]
##x_test  /= 255.0
##
###one-hot encoding
##y_train = tf.keras.utils.to_categorical(y_train) # (60000, 10)
##y_test = tf.keras.utils.to_categorical(y_test)   # (10000, 10)

#1-2
import mnist # mnist.py 
(x_train, y_train), (x_test, y_test) = mnist.load(flatten=True, one_hot=True, normalize = True)


#2:  
model = tf.keras.Sequential([
     Input(shape=(784,)),
     Dense(units=20, activation='sigmoid', name='layer1'),
     Dense(units=20, activation='sigmoid', name='layer2'), 
     Dense(units=10, activation='softmax', name='output')])
#model.summary()

#3
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])


ret = model.fit(x_train, y_train, epochs=100, batch_size=64, verbose=2) 
train_loss, train_acc = model.evaluate(x_train, y_train, verbose=2)
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2) 
#print('train_acc=',train_acc )
#print('test_acc=', test_acc)

#4 모델 동결(freezing)
import freeze_graph # freeze_graph.py
freeze_graph.freeze_model(model, "MNIST_DENSE.pb")

