#0503.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

#1: load model
TENSORFLOW, ONNX = 1, 2

net_type = TENSORFLOW #net_type = ONNX  

if net_type == TENSORFLOW:
    net = cv2.dnn.readNet('./dnn/XOR.pb')
else: # ONNX
    net = cv2.dnn.readNet('./dnn/XOR.onnx')
    #net = cv2.dnn.readNet('./dnn/XOR2.onnx') # by tf2onnx with SavedModel
    #net = cv2.dnn.readNet('./dnn/XOR3.onnx')

#2
#net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
#net.setPreferableTarget( cv2.dnn.DNN_TARGET_CUDA)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT) # DNN_BACKEND_OPENCV

##layersIds, inLayersShapes, outLayersShapes = net.getLayersShapes((4, 1,  1, 2))
print('net.getLayerNames() =', net.getLayerNames())
print('net.getUnconnectedOutLayersNames()=', net.getUnconnectedOutLayersNames())

#3: classify X
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]], dtype = np.float32)

y = np.array([0,1,1,0], dtype = np.float32) # XOR, set y for display 

blob = X.reshape(-1, 1, 1, 2) 
net.setInput(blob)
y_out = net.forward()
y_pred = np.argmax(y_out, axis=1)
print('y_pred =', y_pred)

#4
h = 0.01
xx, yy = np.meshgrid(np.arange(0-2*h, 1+2*h, h),
                     np.arange(0-2*h, 1+2*h, h))

sample = np.c_[xx.ravel(), yy.ravel()]

sample = sample.reshape((-1, 1, 1,  2))
net.setInput(sample)
out = net.forward()
 
pred = np.argmax(out, axis=1)
pred = pred.reshape(xx.shape)

#5
ax = plt.gca()
ax.set_aspect('equal')

plt.contourf(xx, yy, pred, cmap=plt.cm.gray)
plt.contour(xx, yy, pred, colors='red', linewidths=1)

class_colors = ['blue', 'red']
for label in range(2): # 2 class
    plt.scatter(X[y==label, 0], X[y==label, 1],
                50, class_colors[label], 'o')
plt.show()
