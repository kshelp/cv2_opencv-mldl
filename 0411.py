#0411.py
import gzip
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import  TensorDataset, DataLoader
from torchsummary import summary  # pip install torchsummary

#1-1
import mnist # mnist.py  
(x_train, y_train), (x_test, y_test) = mnist.load(flatten=True, normalize = True)

#1-2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
x_train = torch.tensor(x_train, dtype=torch.float32).to(DEVICE)
y_train = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
y_train_1hot = F.one_hot(y_train.long(), num_classes=10).float()

x_test = torch.tensor(x_test, dtype=torch.float32).to(DEVICE)
y_test = torch.tensor(y_test, dtype=torch.float32).to(DEVICE)

ds = TensorDataset(x_train, y_train_1hot)
loader = DataLoader(dataset=ds, batch_size=64,  shuffle=True)


#2
class DenseNet(nn.Module):

    def __init__(self):
        super(DenseNet, self).__init__()
        self.fc1 = nn.Linear(in_features=784, out_features=20, bias=True)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 10)
                
    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        x = self.fc3(x)
        #x = torch.softmax(x, dim=1) # x = F.softmax(x, dim=1)
        return x

model = DenseNet().to(DEVICE)
summary(model, x_train.shape, device=DEVICE)

#3-1
loss_fn  = torch.nn.CrossEntropyLoss()
##def loss_fn(output, target):
##    #output = torch.softmax(output, dim=1) # output = F.softmax(output, dim=1)
##    #return (target * -torch.log(output)).sum(dim=1).mean()
##    return F.cross_entropy(output, target)
    
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#3-2
loss_list = []
iter_per_epoch = int(np.ceil(x_train.shape[0]/loader.batch_size)) # 60000/64= 938

model.train()
for epoch in range(100):
    batch_loss = 0.0
    for X, y in loader:
        optimizer.zero_grad()
        y_pred = model(X)
        
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        batch_loss += loss.item()*X.size(0) # mean -> sum
        
    batch_loss /= x_train.shape[0] # divide by train size
    loss_list.append(batch_loss)
    print("epoch=", epoch, "batch_loss=", batch_loss)

##from matplotlib import pyplot as plt
##plt.plot(loss_list)
##plt.show()

#3-3        
model.eval()
with torch.no_grad():   
    out = model(x_train)
    pred = torch.max(out.data, 1)[1]

    #accuracy = (pred == y_train).float().mean() # train accuracy
    accuracy = pred.eq(y_train.view_as(pred)).float().mean()
    print("train accuracy= ", accuracy.item())
    
    out = model(x_test)
    pred = torch.max(out.data, 1)[1]
    accuracy = pred.eq(y_test.view_as(pred)).float().mean() # test accuracy
    print("test accuracy= ", accuracy.item())
        
#4
dummy_input = torch.randn(100, 784).to(DEVICE)  # x_train
torch.onnx.export(model, dummy_input, "./dnn/MNIST_DENSE.onnx")

