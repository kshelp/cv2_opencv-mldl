#0307.py
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
y_true = np.int32(iris_data[:, -1])  
data = cv2.ml.TrainData_create(samples=X,
                                    layout=cv2.ml.ROW_SAMPLE,
                                    responses=y_true)
data.setTrainTestSplitRatio(0.8) # train data: 80%, shuffle=True

#3: DTrees, RTrees, NormalBayesClassifier, SVM
#3-1
model = cv2.ml.DTrees_create()
model.setCVFolds(1) # If CVFolds > 1 then, tree pruning using cross-validation is not implemented
model.setMaxDepth(10)

#3-2
##model = cv2.ml.RTrees_create()

#3-3
##model =  cv2.ml.NormalBayesClassifier_create() # error

#3-4
##model =  cv2.ml.SVM_create()

#4
ret = model.train(data)

#5
train_err, train_resp  = model.calcError(data, test=False)
train_accuracy = (100.0-train_err)
print('train_err={:.2f}%'.format(train_err))
print('train_accuracy={:.2f}%'.format(train_accuracy))

test_err, test_resp  = model.calcError(data, test=True)
test_accuracy = (100.0-test_err)
print('test_err={:.2f}%'.format(test_err))
print('test_accuracy={:.2f}%'.format(test_accuracy))

#6
##ret, pred  = model.predict(data.getTrainSamples())
##train_accuracy2 = np.sum(pred==data.getTrainResponses())/data.getNTrainSamples()
##print('train_accuracy={:.2f}%'.format(train_accuracy2*100))
##
##ret, pred  = model.predict(data.getTestSamples())
##test_accuracy2 = np.sum(pred==data.getTestResponses())/data.getNTestSamples()
##print('test_accuracy2={:.2f}%'.format(test_accuracy2*100))

 
