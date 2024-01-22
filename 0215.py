#0215.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=2, suppress=True)

#1: train data
#1-1
x_train = np.array([[0,0],
                    [0,1],
                    [1,0],
                    [1,1]], dtype = np.float32)

##y_train = np.array([0,0,0,1], dtype = np.float32)  # AND
##y_train = np.array([0,1,1,1], dtype = np.float32)  # OR
y_train = np.array([0,1,1,0], dtype = np.float32)    # XOR

#1-2: one-hot encoding
n_class = len(np.unique(y_train)) #  2 class
y_train_1hot = np.eye(n_class, dtype=np.float32)[np.int32(y_train)]
print("y_train_1hot=", y_train_1hot)

#2: Artificial Neural Networks
#2-1
model = cv2.ml.ANN_MLP_create()
model.setLayerSizes(np.array([2, 2, 2])) # 2-output neurons
model.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 2, 1) # [-1, 1]
model.setTermCriteria((cv2.TERM_CRITERIA_EPS+
                       cv2.TERM_CRITERIA_COUNT, 100, 1e-5))

#2-2: default 
##model.setTrainMethod(cv2.ml.ANN_MLP_RPROP, 0.1) # RpropDW0 = 0.1
##model.setRpropDW0(0.1)
##model.setRpropDWPlus(1.2)
##model.setRpropDWMinus(0.5)
##model.setRpropDWMin(1.19209e-07)
##model.setRpropDWMax(50.)

#2-3
##model.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP, 0.1, 0.9)
##model.setBackpropWeightScale(0.1)   # learning rate
##model.setBackpropMomentumScale(0.9) # momentum

#3: train using TrainData
data = cv2.ml.TrainData_create(samples=x_train,
                               layout=cv2.ml.ROW_SAMPLE,
                               responses=y_train_1hot) # one-hot

ret = model.train(data)


#4
ret, y_out = model.predict(x_train)
print("y_out=", y_out)
y_pred = np.argmax(y_out, axis = 1) # y_out.argmax(-1)
print("y_pred=", y_pred)

accuracy = np.sum(y_train == y_pred) / len(y_train)
print('accuracy=', accuracy)


#5
h = 0.01
xx, yy = np.meshgrid(np.arange(0-2*h, 1+2*h, h),
                     np.arange(0-2*h, 1+2*h, h))

sample = np.c_[xx.ravel(), yy.ravel()]
ret, out = model.predict(sample) 
pred = np.argmax(out, axis = 1)
pred = pred.reshape(xx.shape)

#6
ax = plt.gca()
ax.set_aspect('equal')

plt.contourf(xx, yy, pred, cmap=plt.cm.gray)
plt.contour(xx, yy, pred, colors='red', linewidths=1)

class_colors = ['blue', 'red']
for label in range(2): # 2 class
    plt.scatter(x_train[y_train==label, 0], x_train[y_train==label, 1],
                50, class_colors[label], 'o')

plt.show()
