#0316.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

#1: MNIST
import mnist  # mnist.py
(x_train, y_train_1hot), (x_test, y_test_1hot) = mnist.load(flatten=True,
                                                            one_hot=True, normalize = True)

y_train = np.argmax(y_train_1hot, axis = 1) # category label: [0, 1, ..., 9]
y_test  = np.argmax(y_test_1hot, axis = 1)  
train_size = y_train.shape[0] # 60000
test_size  = y_test.shape[0]  # 10000

#2: TrainData
data = cv2.ml.TrainData_create(samples=x_train,
                               layout=cv2.ml.ROW_SAMPLE,
                               responses=y_train_1hot) 


#3: Artificial Neural Networks
#3-1:
model = cv2.ml.ANN_MLP_create()
model.setLayerSizes(np.array([784, 20, 20, 10]))
model.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 2, 1) # [-1, 1]
model.setTermCriteria((cv2.TERM_CRITERIA_EPS+
                       cv2.TERM_CRITERIA_COUNT, 1, 1e-5))

#3-2:
model.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP, 0.01, 0.1)
ret = model.train(data)  # initial training

#4: mini-batch training 
#4-1
iters_num =  100
batch_size=  1000
K = train_size//batch_size
train_loss_list = []

np.random.seed(111)

for epoch in range(iters_num):

#4-2
   train_loss = 0
   for i in range(K): # 60
      batch_mask = np.random.choice(train_size, batch_size)
      x_batch = x_train[batch_mask]
      y_batch_1hot = y_train_1hot[batch_mask]

      data = cv2.ml.TrainData_create(samples=x_batch,
                               layout=cv2.ml.ROW_SAMPLE,
                               responses=y_batch_1hot)    
#4-3
      ret = model.train(data, flags = cv2.ml.ANN_MLP_UPDATE_WEIGHTS)
      ret, train_out  = model.predict(x_batch)
      train_loss += np.sum((y_batch_1hot-train_out)**2)/batch_size  # MSE
      
#4-4     
   train_loss /= K
   train_loss_list.append(train_loss)
   print("epoch=", epoch, "train_loss=", train_loss)
             
#5: final loss, accuracy
#5-1
model.save('./data/0316_ANN_BACKPROP.train')

#5-2
ret, train_out  = model.predict(x_train)
train_pred = np.argmax(train_out, axis = 1)

train_loss    = np.sum((y_train_1hot-train_out)**2)/train_size  # MSE
train_accuracy = np.sum(train_pred==y_train)/train_size
print('train_loss={:.2f}'.format(train_loss))  
print('train_accuracy={:.2f}'.format(train_accuracy))

#5-3
ret, test_out  = model.predict(x_test)
test_pred = np.argmax(test_out, axis = 1)

test_loss    = np.sum((y_test_1hot-test_out)**2)/test_size  # MSE
test_accuracy = np.sum(test_pred==y_test)/test_size
print('test_loss={:.2f}'.format(test_loss))  
print('test_accuracy={:.2f}'.format(test_accuracy))

#6: display
#6-1: train_loss_list
x = np.linspace(0, iters_num, num=iters_num)
plt.title('train_loss')
plt.plot(x, train_loss_list, color='red', linewidth=2, label='train_loss')
plt.xlabel('iterations')
plt.ylabel('loss(MSE)')
plt.show()

#6-2:  display 8 test images
fig = plt.figure(figsize = (8, 4))
plt.suptitle('x_test[8]: (test_pred[8]: y_test[:8])',fontsize=15) 
for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.title("true={}: pred={}".format(y_test[i], test_pred[i]))
    plt.imshow(x_test[i].reshape(28, 28), cmap = 'gray')
    plt.axis("off")
fig.tight_layout()
plt.show()
 
