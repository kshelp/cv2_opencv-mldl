#0514.py
'''
https://pytorch.org/vision/stable/models.html

# pip install torchvision
# pip install torchsummary
# pip install pillow

'''
import torch
from torchvision import transforms 
from torchvision import models
from torchsummary import summary  
from PIL import Image
import numpy as np

#print("dir(models)=", dir(models))    

#1:
image_name = ['./data/elephant.jpg', './data/eagle.jpg', './data/dog.jpg']
src = Image.open(image_name[0])  # mode=RGB

#2:
transform = transforms.Compose([
    transforms.Resize(226), # transforms.InterpolationMode.BILINEAR
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
    ])

img = transform(src) # img.shape = torch.Size([3, 224, 224])
img = torch.unsqueeze(img, dim=0) # img.shape = torch.Size([1, 3, 224, 224])

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
img = img.to(DEVICE)

#3:
vgg16 = models.vgg16(pretrained=True).to(DEVICE)
#summary(vgg16, img.shape[1:], device=DEVICE) # img.shape[1:]=[3, 224, 224]

#4:
dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)  
torch.onnx.export(vgg16, dummy_input, "./dnn/vgg_0514.onnx")

#5
vgg16.eval()
out = vgg16(img)
prob = torch.nn.functional.softmax(out, dim=1)[0]

top5_values, top5_indices = torch.topk(prob, k=5) # prob.topk(5)
top5_values    =  top5_values.cpu().detach().numpy()
top5_indices    =  top5_indices.cpu().detach().numpy()
print("top5_indices=", top5_indices)
print("top5_values=", top5_values)

#6:
import json
with open('./dnn/PreTrained/imagenet_class_index.json', 'r') as f:
    imagenet_class = json.load(f)
    
# top-5
for i, k in enumerate(top5_indices):
    label = imagenet_class[str(k)]
    print("top[{}]: ({}, {}, {:.4f})".format(i, label[0], label[1], top5_values[i]))

top1 = imagenet_class[str(top5_indices[0])]
top1_name_value = top1[1]+" : "+ str(top5_values[0])

#7
#7-1
import matplotlib.pyplot as plt
plt.gcf().canvas.manager.set_window_title(top1_name_value)
plt.title(top1_name_value)
plt.axis('off')
plt.imshow(src)
plt.tight_layout()
plt.show()

#7-2    
numpy_rgb = np.asarray(src)
src_bgr = numpy_rgb[..., ::-1] # BGR

import cv2
cv2.imshow(top1_name_value,  src_bgr)
cv2.waitKey()
cv2.destroyAllWindows()


