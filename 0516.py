#0516.py
'''
https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
'''
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import models
from torchvision.datasets import CIFAR10
from torch.utils.data import  DataLoader
from torchsummary import summary

#1:
data_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

root_path='./dnn/cifar10'
train_ds = CIFAR10(root=root_path, train=True,  download=True, transform=data_transform)
test_ds  = CIFAR10(root=root_path, train=False, download=True, transform=data_transform)

train_size = train_ds.data.shape[0]
test_size = test_ds.data.shape[0]
print('train_ds.data.shape= ', train_ds.data.shape) # (50000, 32, 32, 3)
print('test_ds.data.shape= ',  test_ds.data.shape)  # (10000, 32, 32, 3)

# if RuntimeError: CUDA out of memory, then reduce batch size
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=128, shuffle=True)
print('len(train_loader.dataset)=', len(train_loader.dataset))
print('len(test_loader.dataset)=', len(test_loader.dataset))

#2:
class TransferVgg16(nn.Module):

    def __init__(self, freeze=True):
        super(TransferVgg16, self).__init__()

        vgg16 = models.vgg16(pretrained=True)
        self.features  = vgg16.features
        if freeze:
            for param in self.features.parameters(): # freeze the layers
                param.requires_grad = False
        self.avgpool = nn.AdaptiveAvgPool2d(output_size = (7,7))
        self.classifier = nn.Sequential(           
                    #nn.Flatten(),
                    nn.Linear(in_features=512*7*7, out_features=100),                    
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(100, 100),
                    nn.Dropout(0.2),
                    nn.Linear(100, 10),
                    nn.Softmax(dim=1)
                    )   
                          
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1) # nn.Flatten(),       
        x = self.classifier(x)
        return x

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'    
model = TransferVgg16(freeze=True).to(DEVICE)  
summary(model, torch.Size([3, 224, 224]), device=DEVICE)
 
#3
#3-1
loss_fn = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
loss_list = []
EPOCHS = 20
num_per_epoch = train_size//train_loader.batch_size

print("training.....")
model.train()
for epoch in range(EPOCHS):

    correct = 0
    batch_loss = 0.0
    for i, (X, y) in enumerate(train_loader): # 
        optimizer.zero_grad()
        X = X.to(DEVICE)
        y = y.to(DEVICE)
        out = model(X)
                
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()

        _, pred = out.max(1)
        correct += pred.eq(y).sum().item()
        
        batch_loss += loss.item()
        
    train_loss = batch_loss/len(train_loader) # len(train_loader)=391 if batch_size=128
    train_accuracy = correct/train_size
    loss_list.append(train_loss) 
    print("epoch={}, train_loss={}, train_accuracy={:.4f}".format(
           epoch, train_loss, train_accuracy))

#3-2
print("evaluating.....")
model.eval()
with torch.no_grad():   
    correct = 0
    batch_loss = 0.0
    for i, (X, y) in enumerate(test_loader):  
        optimizer.zero_grad()
        X = X.to(DEVICE)
        y = y.to(DEVICE)
        out = model(X)
                
        loss = loss_fn(out, y)
        batch_loss += loss.item()

        _, pred = out.max(1)
        correct += pred.eq(y).sum().item()
        
    test_loss = batch_loss/len(test_loader) # len(test_loader)=79 if batch_size=128
    test_accuracy = correct/test_size
    print("test_loss={}, test_accuracy={:.4f}".format(test_loss, test_accuracy))
        
#4
dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
torch.onnx.export(model, dummy_input, "./dnn/cifar10.onnx")

#5
from matplotlib import pyplot as plt
plt.plot(loss_list)
plt.show()

    
