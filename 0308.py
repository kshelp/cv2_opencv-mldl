#0308.py
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

#3
model = cv2.ml.EM_create()
model.setClustersNumber(3) # 3 class
ret = model.train(data)

#4
x_train = data.getTrainSamples()
y_train = data.getTrainResponses()
ret, train_prob = model.predict(x_train)
train_pred = np.argmax(train_prob, axis = 1)
train_pred = train_pred.reshape(y_train.shape)
##print("train_pred[:5]=", train_pred[:5])
##print("y_train[:5]=",     y_train[:5])

x_test = data.getTestSamples()
y_test = data.getTestResponses()
ret, test_prob = model.predict(x_test)
test_pred = np.argmax(test_prob, axis = 1)
test_pred = test_pred.reshape(y_test.shape)
##print("train_pred[:5]=", train_pred[:5])
##print("y_train[:5]=",     y_train[:5])

#5: EM re-labeling using matches which are calculated by true_label
#5-1
def findMatchingLabels(true_label, pred):
    matches = dict()
    nClass = np.unique(true_label).shape[0]
    for i in range(nClass):
        res = true_label[pred==i]
        labels, counts = np.unique(res, return_counts=True)
        k = labels[np.argmax(counts)]
        matches[i]= k
    return matches

matches = findMatchingLabels(y_train, train_pred)
print("matches=", matches)

#5-2 
##zeros = train_pred == 0
##ones  = train_pred == 1
##twos  = train_pred == 2
##train_pred[zeros] = matches[0]
##train_pred[ones] =  matches[1]
##train_pred[twos] =  matches[2]

def reLabeling(pred, matches):
    
    masks = []
    nClass = len(matches)
    for i in range(nClass):
        masks.append(pred == i)
        
    for i in range(nClass):
        pred[masks[i]] = matches[i]
    return pred

train_pred = reLabeling(train_pred, matches)      
##print("train_pred[:5]=", train_pred[:5])
##print("y_train[:5]=",     y_train[:5]) 
 
test_pred = reLabeling(test_pred, matches) 

#5-3 
def calcAccuracy(label, pred, percent=True):
    N = label.shape[0] # number of data
    accuracy = np.sum(pred == label)/N
    if percent:
        accuracy *= 100
    return accuracy
train_accuracy = calcAccuracy(y_train, train_pred)
print('train_accuracy={:.2f}%'.format(train_accuracy))

test_accuracy = calcAccuracy(y_test, test_pred)
print('test_accuracy={:.2f}%'.format(test_accuracy))


