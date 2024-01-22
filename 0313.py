#0313.py
import gzip
import cv2
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=2, suppress=True)

#1: MNIST
import mnist  # mnist.py
(x_train, y_train), (x_test, y_test) = mnist.load(flatten=True, normalize=True)

#2: PCA feature extraction
k= 10 # feature dimension, 10, 20, 100
##mean = cv2.reduce(x_train, dim = 0, rtype=cv2.REDUCE_AVG)
mean, eVects = cv2.PCACompute(x_train, mean=None, maxComponents=k)
x_train_pca =cv2.PCAProject(x_train, mean, eVects)
x_test_pca =cv2.PCAProject(x_test, mean, eVects)
print('x_train_pca.shape=', x_train_pca.shape) # (60000, k)
print('x_test_pca.shape=',  x_test_pca.shape)  # (10000, k)  

#3
model =  cv2.ml.SVM_create()
model.setKernel(cv2.ml.SVM_RBF) # default
model.setC(0.1)           #default = 1.0
model.setGamma(0.03375)   #default = 1.0

model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER+cv2.TERM_CRITERIA_EPS, 100, 0.0001))
ret = model.train(samples=x_train_pca, layout=cv2.ml.ROW_SAMPLE, responses=y_train)
##ret = model.trainAuto(samples=x_train_pca, layout=cv2.ml.ROW_SAMPLE, responses=y_train)
##print('model.getKernelType()=', model.getKernelType())
##print('model.getC()=', model.getC())
##print('model.getGamma()=', model.getGamma())

#4
ret, train_pred = model.predict(x_train_pca)  
train_pred = np.int32(train_pred.flatten())
train_accuracy = np.sum(y_train == train_pred) / x_train.shape[0]
print('train_accuracy=', train_accuracy)

#5:
ret, test_pred = model.predict(x_test_pca)  
test_pred = np.int32(test_pred.flatten())
test_accuracy = np.sum(y_test == test_pred) / x_test.shape[0]
print('test_accuracy=', test_accuracy)

#6:  display 8 test images
fig = plt.figure(figsize = (8, 4))
plt.suptitle('x_test[8]: (test_pred[8]: y_test[:8])',fontsize=15) 
for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.title("true={}: pred={}".format(y_test[i], test_pred[i]))
    plt.imshow(x_test[i].reshape(28, 28), cmap = 'gray')
    plt.axis("off")
fig.tight_layout()
plt.show()
 
