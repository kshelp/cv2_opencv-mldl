#0214.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=2, suppress=True)

#1
x_train = np.array([[0,0],
                    [0,1],
                    [1,0],
                    [1,1]], dtype = np.float32)

##y_train = np.array([0,0,0,1], dtype = np.float32)  # AND
##y_train = np.array([0,1,1,1], dtype = np.float32)  # OR
y_train = np.array([0,1,1,0], dtype = np.float32)    # XOR

#2: Artificial Neural Networks
#2-1
##model = cv2.ml_ANN_MLP.load('./data/0213-and.train')
##model = cv2.ml_ANN_MLP.load('./data/0213-or.train')
model = cv2.ml_ANN_MLP.load('./data/0213-xor.train')

##mlp_net = cv2.ml.ANN_MLP_create()
##model = mlp_net.load('./data/0213-xor.train')


#2-2: weights
##layerSize = model.getLayerSizes()
##for i in range(layerSize.shape[0] + 2):
##    print("weights[{}] = {}".format(i, model.getWeights(i)))

#3
ret, y_out = model.predict(x_train)
y_pred = np.int32(y_out>0.5)
y_pred = y_pred.flatten()
accuracy = np.sum(y_train == y_pred) / len(y_train)
print('accuracy=', accuracy)

#4
h = 0.01
xx, yy = np.meshgrid(np.arange(0-2*h, 1+2*h, h),
                     np.arange(0-2*h, 1+2*h, h))

sample = np.c_[xx.ravel(), yy.ravel()]
ret, out = model.predict(sample) 
pred = np.int32(out>0.5)
pred = pred.reshape(xx.shape)

#5
ax = plt.gca()
ax.set_aspect('equal')

plt.contourf(xx, yy, pred, cmap=plt.cm.gray)
plt.contour(xx, yy, pred, colors='red', linewidths=1)

class_colors = ['blue', 'red']
for label in range(2): # 2 class
    plt.scatter(x_train[y_train==label, 0], x_train[y_train==label, 1],
                50, class_colors[label], 'o')

plt.show()
