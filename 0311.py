#0311.py
''' ref1: http://yann.lecun.com/exdb/mnist/
    ref2: https://gist.github.com/ischlag/41d15424e7989b936c1609b53edd1390
    ref3: 텐서플로 프로그래밍, 가메출판사, 2020, 김동근
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt

#2
import mnist  # mnist.py
(x_train, y_train), (x_test, y_test) = mnist.load()
print('x_train.shape=', x_train.shape) # (60000, 28, 28)
print('y_train.shape=', y_train.shape) # (60000, ) 
print('x_test.shape=',  x_test.shape)  # (10000, 28, 28)
print('y_test.shape=',  y_test.shape)  # (10000, ) 

#3
nlabel, count = np.unique(y_train, return_counts = True)
print("nlabel:", nlabel) # [0 1 2 3 4 5 6 7 8 9]
print("count:", count)
print("# of Class:", len(nlabel) ) # 10

#4: display 8 images
print("y_train[:8]=", y_train[:8])
fig = plt.figure(figsize = (8, 4))
plt.suptitle('x_train[8], y_train[:8]',fontsize=15) 
for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.title("%d"%y_train[i])
    plt.imshow(x_train[i], cmap = 'gray')
    plt.axis("off")
fig.tight_layout()
plt.show()
 
