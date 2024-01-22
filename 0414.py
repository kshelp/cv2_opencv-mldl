#0414.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import  DataLoader
from torchvision import transforms, datasets  
from torch.optim.lr_scheduler import StepLR 
from torchsummary import summary  

#1-1
data_transform  = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=5),
    
    transforms.Resize(256), 
    transforms.CenterCrop(224),
        
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x*2.0 -1.0) #[ -1, 1]
    #transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

#1-2
num_class = 2
target_1hot = transforms.Lambda(
        lambda y: torch.zeros(num_class, dtype=torch.float).scatter_(0,
                                                                     torch.tensor(y), value=1))

#1-3
TRAIN_PATH='./Petimages'
train_ds = datasets.ImageFolder(root=TRAIN_PATH, transform=data_transform,
                                                target_transform = target_1hot)

print("train_ds.class_to_idx = ", train_ds.class_to_idx)
print("train_ds.classes=", train_ds.classes)
##for i in range(2):
##    image, label = train_ds[i]
##    image_arr = image.squeeze().numpy()

    
#1-4    
# if RuntimeError: CUDA out of memory, then reduce batch size
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader  = DataLoader(train_ds, batch_size=64, shuffle=False)

train_size = len(train_loader.dataset)
test_size = len(test_loader.dataset)
print('train_size=', train_size)  # 24998
print('test_size=', test_size)    # 24998

##print("test_loader")
##for i in range(len(test_loader.dataset.imgs)):
##    print(i, test_loader.dataset.imgs[i])
##    
##print("test_loader")
##for i, (X, y) in enumerate(test_loader):
##    print(i, X.shape, y.shape),
    
#2
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        
        self.layer1 = nn.Sequential( 
            # (, 3, 224, 224) : # NCHW
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3), 
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2))
            # (, 16, 111, 111)       
           
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2))
            #(, 32, 54, 54)
        
        self.layer3 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Flatten(), 
            nn.Linear(32*54*54, 2),
            #nn.Softmax(dim=1),
            )                    

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x) 
        x = self.layer3(x)
        return x #F.log_softmax(x, dim=1)
    
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ConvNet().to(DEVICE)
summary(model, torch.Size([3, 224, 224]), device=DEVICE)

#3-1
loss_fn  = torch.nn.CrossEntropyLoss() # torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

#3-2
##iter_per_epoch = int(np.ceil(train_size/train_loader.batch_size))
##print(" iter_per_epoch=", iter_per_epoch)

loss_list = []
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
        X, y = X.to(DEVICE), y.to(DEVICE)
        out = model(X)    
        loss = loss_fn(out, y)
        
        b_loss = loss.item()*X.size(0) # mean -> sum
        batch_loss += b_loss
        
        y_pred = out.max(1)[1]
        correct += y_pred.eq(y.max(1)[1]).sum().item()
        
    batch_loss /= test_size    
    test_accuracy = correct/test_size
    print("test_set: batch_loss={}, test_accuracy={:.4f}".format(batch_loss, test_accuracy))
         
#4
dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)  # x_train
torch.onnx.export(model, dummy_input, "./dnn/Petimages.onnx")
