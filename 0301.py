#0301.py
'''            
ref1: https://gist.github.com/curran/a08a1080b88344b0c8a7#file-iris-csv
ref2: 텐서플로 프로그래밍, 가메출판사, 2020, 김동근
''' 
import numpy as np
import matplotlib.pyplot as plt

#1
def load_Iris():   
    label = {'setosa':0, 'versicolor':1, 'virginica':2}
    data = np.loadtxt("./data/iris.csv", skiprows = 1, delimiter = ',',
                      converters={4: lambda name: label[name.decode()]})
    return np.float32(data)
 
iris_data = load_Iris()
X = iris_data[:,:-1]
y_true = iris_data[:, -1]  # the last column
    
print("X.shape=", X.shape)
print("y_true.shape=", y_true.shape)
print("X[:3]=", X[:3])
print("y_true[:3]=", y_true[:3])

#2
markers= "ox+*sd"
colors = "bgcmyk"
labels = ["Iris setosa", "Iris versicolor", "Iris virginica"]

fig = plt.gcf()
fig.set_size_inches(5, 5)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
for i, k in enumerate(np.unique(y_true)):
  plt.scatter(X[y_true == k, 0],     # Sepal Length
              X[y_true == k, 1],     # Sepal Width
              c = colors[i], marker = markers[i], label = labels[i])
plt.legend(loc = 'best')
plt.show()

#3
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
for i, k in enumerate(np.unique(y_true)):
  plt.scatter(X[y_true == k, 2], # Petal Length
              X[y_true == k, 3], # Petal Width
              c = colors[i], marker = markers[i], label = labels[i])
plt.legend(loc = 'best')
plt.show()
