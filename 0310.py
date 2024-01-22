#0310.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

#1
def load_Iris():   
    label = {'setosa':0, 'versicolor':1, 'virginica':2}
    data = np.loadtxt("./data/iris.csv", skiprows = 1, delimiter = ',',
                      converters={4: lambda name: label[name.decode()]})
    return np.float32(data)

iris_data = load_Iris()

#2
X      = np.float32(iris_data[:,:-1])
y_true = np.float32(iris_data[:, -1])

n_class = len(np.unique(y_true)) #  3 class
y_true_1hot = np.eye(n_class, dtype=np.float32)[np.int32(y_true)]

data = cv2.ml.TrainData_create(samples=X,
                                    layout=cv2.ml.ROW_SAMPLE,
                                    responses=y_true_1hot)
data.setTrainTestSplitRatio(0.8) # train data: 80%, shuffle=True

x_train = data.getTrainSamples()
train_resp = data.getTrainResponses()     # one-hot
y_train = y_true[data.getTrainSampleIdx()]
##y_train = np.argmax(train_resp, axis = 1) # 0, 1, 2
nTrain = data.getNTrainSamples()

x_test  = data.getTestSamples()
test_resp = data.getTestResponses()     # one-hot
y_test = y_true[data.getTestSampleIdx()]
##y_test = np.argmax(test_resp, axis = 1) # 0, 1, 2
nTest = data.getNTestSamples()

#3: Artificial Neural Networks
model = cv2.ml.ANN_MLP_create()
model.setLayerSizes(np.array([4, 10, 3]))
model.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 2, 1) # [-1, 1]

model.setTermCriteria((cv2.TERM_CRITERIA_EPS+
                       cv2.TERM_CRITERIA_COUNT, 10, 1e-5))

##model.setTrainMethod(cv2.ml.ANN_MLP_RPROP, 0.1) # default method
##model.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)


ret = model.train(data)  # initial training

#4
train_loss_list = []
train_accuracy_list = []

iters_num = 100
for i in range(iters_num):
#4-1
    ret = model.train(data, flags = cv2.ml.ANN_MLP_UPDATE_WEIGHTS)
   
#4-2
    ret, train_out  = model.predict(x_train)
    train_pred = np.argmax(train_out, axis = 1)
    
#4-3
    train_loss     = np.sum((train_resp-train_out)**2)/nTrain  # MSE
    train_accuracy = np.sum(train_pred==y_train)/nTrain

#4-4
    train_loss_list.append(train_loss)
    train_accuracy_list.append(train_accuracy)
         
#5: final loss, accuracy
#5-1
print('train_loss={:.2f}'.format(train_loss))     
print('train_accuracy={:.2f}'.format(train_accuracy))

#5-2
ret, test_out  = model.predict(x_test)
test_pred = np.argmax(test_out, axis = 1)

test_loss     = np.sum((test_resp-test_out)**2)/nTest  # MSE
test_accuracy = np.sum(test_pred==y_test)/nTest
print('test_loss={:.2f}'.format(test_loss))     
print('test_accuracy={:.2f}'.format(test_accuracy))

#6: display
#6-1: train_loss_list
x = np.linspace(0, iters_num, num=iters_num)
plt.title('train_loss')
plt.plot(x, train_loss_list, color='red', linewidth=2, label='train_loss')
plt.xlabel('iterations')
plt.ylabel('loss(MSE)')
##plt.legend(loc=  'best')
plt.show()

#6-2: train_accuracy_list
plt.title('train_accuracy')
plt.plot(x, train_accuracy_list, color='blue', linewidth=2, label='train_accuracy')
plt.xlabel('iterations')
plt.ylabel('accuracy')
plt.xlim(0, iters_num)
plt.ylim(0, 1.1)
##plt.legend(loc='best')
plt.show()
