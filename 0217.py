#0217.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=2, suppress=True)

#1: create data
N = 50
np.random.seed(123)
f1 = (0.6 + 0.6*np.random.rand(N), 0.5+0.6*np.random.rand(N))
f2 = (0.3 + 0.4*np.random.rand(N), 0.4+0.3*np.random.rand(N))
f3 = (0.8 + 0.4*np.random.rand(N), 0.3+0.3*np.random.rand(N))
f4 = (0.2*np.random.rand(N),       0.3*np.random.rand(N))
x = np.hstack((f1[0], f2[0], f3[0],f4[0])).astype(np.float32)
y = np.hstack((f1[1], f2[1], f3[1],f4[1])).astype(np.float32)
x_train = np.vstack((x, y)).T  # x_train.shape = (200, 2)

# assign label(0, 1, 2, 3)
y_train = np.zeros((4*N,), np.int32)
##y_train[:N]  = 0
y_train[N:2*N] = 1
y_train[2*N:3*N]=2
y_train[3*N:]  = 3

#1-2: one-hot encoding
n_class = len(np.unique(y_train)) #  4 class
y_train_1hot = np.eye(n_class, dtype=np.float32)[np.int32(y_train)]
##print("y_train_1hot=", y_train_1hot)


#2: Artificial Neural Networks
#2-1
model = cv2.ml.ANN_MLP_create()
model.setLayerSizes(np.array([2, 4]))       # 2-layers
model.setLayerSizes(np.array([2, 10, 4])) # 3-layers
model.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM) # [-1.7, 1.7]
model.setTermCriteria((cv2.TERM_CRITERIA_EPS+
                       cv2.TERM_CRITERIA_COUNT, 1000, 1e-5))

#2-2: default 
##model.setTrainMethod(cv2.ml.ANN_MLP_RPROP, 0.1) # RpropDW0 = 0.1

#2-3
model.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP, 0.01, 0.9)

#3: train using TrainData
data = cv2.ml.TrainData_create(samples=x_train,
                               layout=cv2.ml.ROW_SAMPLE,
                               responses=y_train_1hot) # one-hot
ret = model.train(data)

#4
ret, y_out = model.predict(x_train)
y_pred = np.argmax(y_out, axis = 1) # y_out.argmax(-1)

accuracy = np.sum(y_train == y_pred) / len(y_train)
print('accuracy=', accuracy)

#5
h = 0.01
x_min, x_max = x_train[:, 0].min()-h, x_train[:, 0].max()+h
y_min, y_max = x_train[:, 1].min()-h, x_train[:, 1].max()+h

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

sample = np.float32(np.c_[xx.ravel(), yy.ravel()])
ret, out = model.predict(sample) 
pred = np.argmax(out, axis = 1)
pred = pred.reshape(xx.shape)

#6
ax = plt.gca()
ax.set_aspect('equal')

plt.contourf(xx, yy, pred, cmap=plt.cm.gray)
plt.contour(xx, yy, pred, colors='red', linewidths=1)

markers= ('o','x','s','+','*','d')
colors = ('b','g','c','m','y','k')
labels = ('f1', 'f2', 'f3', 'f4')

for k in range(n_class): # 4 class
    plt.scatter(x_train[y_train==k, 0], x_train[y_train==k, 1],
                30, colors[k], markers[k], label=labels[k])
plt.legend(loc='best')
plt.show()
