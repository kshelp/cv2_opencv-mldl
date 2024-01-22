#0306.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=2, suppress=True)

#1
def load_Iris():   
    label = {'setosa':0, 'versicolor':1, 'virginica':2}
    data = np.loadtxt("./data/iris.csv", skiprows = 1, delimiter = ',',
                      converters={4: lambda name: label[name.decode()]})
    return np.float32(data)

iris_data = load_Iris()

#2: normalize 
def calulateStat(X):
    mu = np.mean(X, axis=0) 
    var= np.var(X,  axis=0)
    return mu, var

def normalizeScale(X, mu, var):
    eps = 0.00001
    X_hat = (X-mu)/(np.sqrt(var + eps))
    return X_hat
    
#3
#3-1:
X      = np.float32(iris_data[:,:-1])
y_true = np.float32(iris_data[:, -1])

mu, var = calulateStat(X)
X = normalizeScale(X, mu, var)

#3-2
data = cv2.ml.TrainData_create(samples=X,
                                    layout=cv2.ml.ROW_SAMPLE,
                                    responses=y_true)
data.setTrainTestSplitRatio(0.8) # train data: 80%, shuffle=True

#4 
model = cv2.ml.LogisticRegression_create()
model.setTrainMethod(cv2.ml.LogisticRegression_MINI_BATCH) # BATCH
ret = model.train(data)


#5:
#5-1
x_train = data.getTrainSamples()
y_train = data.getTrainResponses()
x_test = data.getTestSamples()
y_test = data.getTestResponses()

#5-2:
def calcAccuracy(label, pred, percent=True):
    N = label.shape[0] # number of data
    accuracy = np.sum(pred == label)/N
    if percent:
        accuracy *= 100
    return accuracy

#5-3
train_err, train_resp  = model.calcError(data, test=False)
train_accuracy1 = calcAccuracy(y_train, train_resp)
print("train_err = {:.2f}%".format(train_err))
print('train_accuracy1={:.2f}%'.format(train_accuracy1))

test_err, test_resp  = model.calcError(data, test=True)
test_accuracy1 = calcAccuracy(y_test, test_resp)
print('test_err={:.2f}%'.format(test_err))
print('test_accuracy1={:.2f}%'.format(test_accuracy1))
 
#6
def sigmoid(x):
    return  1./(1. + np.exp(-x))

#6-1
model_thetas = model.get_learnt_thetas() # model_thetas.shape = (3, 5)
print("model_thetas=", model_thetas)

x_train_t = cv2.hconcat([np.ones((x_train.shape[0], 1), dtype=x_train.dtype),
                         x_train]) # x_train_t.shape = (120, 5)

#6-2
prob = np.zeros((x_train_t.shape[0], model_thetas.shape[0]), dtype=np.float32) # (120, 3)
for i, thetas in enumerate(model_thetas):
    x = np.dot(x_train_t, thetas)
    y = sigmoid(x)
    prob[:, i] = y

#6-3
prob /= np.sum(prob, axis= 1).reshape(-1, 1)
train_pred = np.argmax(prob, axis = 1)
train_pred = train_pred.reshape(-1, 1)

print("np.sum(train_resp == train_pred)= ", np.sum(train_resp == train_pred)) # 120

#6-4
train_accuracy2 = calcAccuracy(y_train, train_pred)
print('train_accuracy2={:.2f}%'.format(train_accuracy2))
