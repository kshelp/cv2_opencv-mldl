#0515.py
'''
ref: https://github.com/jbohnslav/opencv_transforms/tree/master/opencv_transforms
# pip install opencv_transforms
'''
import torch
from   torchvision import models
from   torchsummary import summary  
from   opencv_transforms import transforms
import numpy as np
import cv2

#1
image_name = ['./data/elephant.jpg', './data/eagle.jpg', './data/dog.jpg']
src = cv2.imread(image_name[0]) # BGR
src_rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB) # src[..., ::-1] : RGB

#2:
transform = transforms.Compose([
    transforms.Resize(226, cv2.INTER_AREA),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])])

img = transform(src_rgb)
img = torch.unsqueeze(img, dim=0) # img.shape = torch.Size([1, 3, 224, 224])

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
img = img.to(DEVICE)

#3:
vgg16 = models.vgg16(pretrained=True).to(DEVICE)
#summary(vgg16, img.shape[1:], device=DEVICE) # img.shape[1:]=[3, 224, 224]

vgg16.eval()
out = vgg16(img)
prob = torch.nn.functional.softmax(out, dim=1)[0]

top5_values, top5_indices = torch.topk(prob, k=5) # prob.topk(5)
top5_values    =  top5_values.cpu().detach().numpy()
top5_indices    =  top5_indices.cpu().detach().numpy()
print("top5_indices=", top5_indices)
print("top5_values=", top5_values)

#4:
import json
with open('./dnn/PreTrained/imagenet_class_index.json', 'r') as f:
    imagenet_class = json.load(f)
    
# top-5
for i, k in enumerate(top5_indices):
    label = imagenet_class[str(k)]
    print("top[{}]: ({}, {}, {:.4f})".format(i, label[0], label[1], top5_values[i]))  

top1 = imagenet_class[str(top5_indices[0])]
top1_name_value = top1[1]+" : "+ str(top5_values[0])

top1 = imagenet_class[str(top5_indices[0])]   
cv2.imshow(top1_name_value, src)
cv2.waitKey()
cv2.destroyAllWindows()
