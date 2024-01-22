#0314.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

#1: MNIST
import mnist  # mnist.py
(x_train, y_train), (x_test, y_test) = mnist.load()
 
#2: HOG feature extraction
hog = cv2.HOGDescriptor(_winSize=(28, 28),
                         _blockSize=(14,14),
                         _blockStride=(7,7),
                         _cellSize=(7,7),
                         _nbins=9)   # _gammaCorrection=False
print("HOG feature size = ",  hog.getDescriptorSize())

def computeHOGfeature(X, hog):
  
  nSize = hog.getDescriptorSize()  #  (9x1)-> 4*(9x1) -> (3*3)(36x1)->(324x1)
  x_train_hog = np.zeros((X.shape[0], nSize), dtype=np.float32)
  for i, img in enumerate(X):
    x_train_hog[i] = hog.compute(np.uint8(img)).flatten()

  return x_train_hog

x_train_hog = computeHOGfeature(x_train, hog)
x_test_hog = computeHOGfeature(x_test, hog)
print('x_train_hog.shape=', x_train_hog.shape) # (60000, 324)
print('x_test_hog.shape=',  x_test_hog.shape)  # (60000, 324)  

#3
#3-1:
model =  cv2.ml.SVM_create()
model.setKernel(cv2.ml.SVM_RBF) # default
model.setC(12.5)     #default = 1.0
model.setGamma(0.5)  #default = 1.0
model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER+cv2.TERM_CRITERIA_EPS, 100, 0.001))

#3-2:
ret = model.train(samples=x_train_hog, layout=cv2.ml.ROW_SAMPLE, responses=y_train)
##ret= model.trainAuto(samples=x_train_hog, layout=cv2.ml.ROW_SAMPLE, responses=y_train)
##print('model.getKernelType()=', model.getKernelType())
##print('model.getC()=', model.getC())
##print('model.getGamma()=', model.getGamma())

#3-3:
model.save('./data/0314_HOG_SVM.train')

#4
ret, train_pred = model.predict(x_train_hog)  
train_pred = np.int32(train_pred.flatten())
train_accuracy = np.sum(y_train == train_pred) / x_train.shape[0]
print('train_accuracy=', train_accuracy)

#5:
ret, test_pred = model.predict(x_test_hog)  
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
 
