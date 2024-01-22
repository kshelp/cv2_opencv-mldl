#0413.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import  Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms 
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary
import numpy as np

#1-1
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x/255), # [0, 1]
    ])

#1-2
num_class = 10
target_1hot = transforms.Lambda(
           lambda y: torch.zeros(num_class, dtype=torch.float).scatter_(0,
                                                        torch.tensor(y), value=1))
#1-3
DATA_PATH='./dnn'
train_ds = MNIST(root=DATA_PATH, train=True,  download=True, 
                  transform=data_transform,
                  target_transform= target_1hot)

test_ds  = MNIST(root=DATA_PATH, train=False,  download=True, 
                  transform=data_transform,
                  target_transform= target_1hot)

train_size = train_ds.data.shape[0]
test_size = test_ds.data.shape[0]
print('train_ds.data.shape= ', train_ds.data.shape) # torch.Size([60000, 28, 28])
print('test_set.data.shape= ',  test_ds.data.shape)  # torch.Size([10000, 28, 28])
print('train_ds.targets.shape= ', train_ds.targets.shape) # torch.Size([60000])
print('test_ds.targets.shape= ', test_ds.targets.shape)   # torch.Size([10000])

##for i in range(2):
##    image, label = train_ds[i]
##    image_arr = image.squeeze().numpy()
##    print("image_arr.shape=", image_arr.shape)
##    print("label=", label)

#1-4   
# if RuntimeError: CUDA out of memory, then reduce batch size
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False)
print('len(train_loader.dataset)=', len(train_loader.dataset))  # 60000
print('len(test_loader.dataset)=', len(test_loader.dataset))    # 10000

#2
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        
        self.layer1 = nn.Sequential( 
            # (, 1, 28, 28) : # NCHW
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            # (, 16, 28, 28 
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2))
            #(, 16, 14, 14)    
           
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
            #(,  32, 7,  7)

        self.layer3 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Flatten(),
            #(,  32*7*7)
            nn.Linear(32*7*7, 10), 
            #nn.Softmax(dim=1),
            )                    

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
    
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ConvNet().to(DEVICE)
summary(model, torch.Size([1, 28, 28]), device=DEVICE)

#3-1
loss_fn  = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

#3-2
loss_list = []
iter_per_epoch = int(np.ceil(train_size/train_loader.batch_size))

print('training.....')
model.train()
for epoch in range(10):
    correct = 0
    batch_loss = 0.0
    for i, (X, y) in enumerate(train_loader):
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(X)
        
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        #scheduler.step()
        
        batch_loss += loss.item()*X.size(0) # mean -> sum
        y_pred = out.max(1)[1]
        correct += y_pred.eq(y.max(1)[1]).sum().item()
        
    batch_loss /= train_size
    loss_list.append(batch_loss)
    train_accuracy = correct/train_size
    print("Epoch={}: batch_loss={}, train_accuracy={:.4f}".format(
                                                       epoch, batch_loss, train_accuracy))  
#3-3      
print('testing.....')  
model.eval()
with torch.no_grad():   
    correct = 0
    batch_loss = 0.0
    for i, (X, y) in enumerate(test_loader):  
        optimizer.zero_grad()
        X, y = X.to(DEVICE), y.to(DEVICE)
        out = model(X)
                
        loss = loss_fn(out, y)
        batch_loss += loss.item()*X.size(0) # mean -> sum

        y_pred = out.max(1)[1]
        correct += y_pred.eq(y.max(1)[1]).sum().item()
        
    batch_loss /= test_size
    test_accuracy = correct/test_size    
    print("test_set: batch_loss={}, test_accuracy={:.4f}".format(batch_loss, test_accuracy))
         
#4
dummy_input = torch.randn(1, 1, 28, 28).to(DEVICE)  # x_train
torch.onnx.export(model, dummy_input, "./dnn/MNIST_CNN2.onnx")
