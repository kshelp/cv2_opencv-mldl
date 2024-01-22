#0510.py
'''
https://github.com/onnx/models/tree/master/vision/classification/resnet
'''
import cv2
import numpy as np

#1
net = cv2.dnn.readNet('./dnn/PreTrained/resnet152-v2-7.onnx')  

#2
image_name = ['./data/elephant.jpg', './data/eagle.jpg', './data/dog.jpg']
src = cv2.imread(image_name[0]) # 1, 2

def preprocess(img):
    X = img.copy()
    X = cv2.resize(X, dsize=(224, 224))        # (224, 224, 3)
    X = cv2.cvtColor(X, cv2.COLOR_BGR2RGB)

    X = X/255 # [0, 1]

    mean =[0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    X = (X - mean)/std
        
    return np.float32(X) 
    
img = preprocess(src)  # img.shape = (224, 224, 3)

#3
blob = cv2.dnn.blobFromImage(img)   # (1, 3, 224, 224)
net.setInput(blob)
out = net.forward() 
    
out = out.flatten()   # out.shape = (1000, )
top5 = out.argsort()[-5:][::-1]
top1 = top5[0] # np.argmax(out) 
print('top1=', top1)
print('top5=', top5)

#4:
#4-1
import json    
with open('./dnn/PreTrained/imagenet_labels.json', 'r') as f:
    imagenet_labels = json.load(f)

#4-2      
print('Top-1 prediction=', imagenet_labels[top1], out[top1])
print('Top-5 prediction:', [imagenet_labels[i] for i in top5])

 
