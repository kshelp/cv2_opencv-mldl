#0309.py
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

#3:  
model = cv2.ml.ANN_MLP_create()
model.setLayerSizes(np.array([4, 10, 3]))
model.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 2, 1) # [-1, 1]
model.setTermCriteria((cv2.TERM_CRITERIA_EPS+
                       cv2.TERM_CRITERIA_COUNT, 100, 1e-5))

##model.setTrainMethod(cv2.ml.ANN_MLP_RPROP, 0.1) # default method
ret = model.train(data)

#4
train_err, train_resp  = model.calcError(data, test=False)
train_accuracy1 = (100.0-train_err)
print('train_accuracy1={:.2f}%'.format(train_accuracy1))

test_err, test_resp  = model.calcError(data, test=True)
test_accuracy1 = (100.0-test_err)
print('test_accuracy1={:.2f}%'.format(test_accuracy1))

#5
#5-1
x_train = data.getTrainSamples()
ret, train_out  = model.predict(x_train)
train_pred = np.argmax(train_out, axis = 1) # train_out.argmax(-1) : 0, 1, 2

y_train = y_true[data.getTrainSampleIdx()]
##y_train = np.argmax(data.getTrainResponses(), axis = 1)
train_accuracy2 = np.sum(train_pred==y_train)/data.getNTrainSamples()
print('train_accuracy2={:.2f}%'.format(train_accuracy2*100))

#5-2
x_test = data.getTestSamples()
ret, test_out  = model.predict(x_test)
test_pred = np.argmax(test_out, axis = 1) # 0, 1, 2

y_test = y_true[data.getTestSampleIdx()]
##y_test = np.argmax(data.getTestResponses(), axis = 1)
test_accuracy2 = np.sum(test_pred==y_test)/data.getNTestSamples()
print('test_accuracy2={:.2f}%'.format(test_accuracy2*100))
