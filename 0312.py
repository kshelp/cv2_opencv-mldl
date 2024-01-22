#0312.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
 
#1: load data
import mnist  #mnist.py
##(x_train, y_train), (x_test, y_test) = mnist.load(flatten=True)
(x_train, y_train), (x_test, y_test) = mnist.load(flatten=True, normalize=True) 

#2:
model =  cv2.ml.SVM_create()
model.setKernel(cv2.ml.SVM_RBF) # default
model.setC(0.1)             #default = 1.0
model.setGamma(0.00015)  #default = 1.0

model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER+cv2.TERM_CRITERIA_EPS,100, 0.001))
ret = model.train(samples=x_train, layout=cv2.ml.ROW_SAMPLE, responses=y_train)
##ret = model.trainAuto(samples=x_train, layout=cv2.ml.ROW_SAMPLE, responses=y_train)

#3
ret, train_pred = model.predict(x_train)   
train_pred = np.int32(train_pred.flatten())
train_accuracy = np.sum(y_train == train_pred) / x_train.shape[0]
print('train_accuracy=', train_accuracy)

#4:
ret, test_pred = model.predict(x_test)   
test_pred = np.int32(test_pred.flatten())
test_accuracy = np.sum(y_test == test_pred) / x_test.shape[0]
print('test_accuracy=', test_accuracy)

#5: display 8 test images
fig = plt.figure(figsize = (8, 4))
plt.suptitle('x_test[8]: (test_pred[8]: y_test[:8])',fontsize=15) 
for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.title("true={}: pred={}".format(y_test[i], test_pred[i]))
    plt.imshow(x_test[i].reshape(28, 28), cmap = 'gray')
    plt.axis("off")
fig.tight_layout()
plt.show()
