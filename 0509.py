#0509.py
'''
ref1: https://github.com/onnx/models/tree/master/vision/classification/vgg
ref2: https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
'''
import cv2
import numpy as np
import onnx
import onnxruntime as ort

#1
OPENCV, ONNX = 1, 2
net_type =  OPENCV  # ONNX
if net_type == OPENCV:
    net = cv2.dnn.readNet('./dnn/PreTrained/vgg16-7.onnx')     # 'vgg19-7.onnx’
else:
    net = ort.InferenceSession('./dnn/PreTrained/vgg16-7.onnx') # 'vgg19-7.onnx'

#2
image_name = ['./data/elephant.jpg', './data/eagle.jpg', './data/dog.jpg']
src = cv2.imread(image_name[0]) # BGR

def preprocess(img):  # "torch" mode in preprocess_input
    X = img.copy()
    X = cv2.resize(X, dsize=(224, 224))        # (224, 224, 3)
    X = cv2.cvtColor(X, cv2.COLOR_BGR2RGB)     # BRG to RGB
    #X = X[..., -1]
    X = X/255 # [0, 1]
    mean =[0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    X = (X - mean)/std       
    return np.float32(X) 
    
img  = preprocess(src)  # img.shape = (224, 224, 3), RGB
#img  = np.transpose(img, [2, 0, 1])# (3, 224, 224) 
#blob = np.expand_dims(img, axis=0) # (1, 3, 224, 224)
blob = cv2.dnn.blobFromImage(img)   # (1, 3, 224, 224) : NCHW

#3
if net_type == OPENCV:   
    net.setInput(blob)
    out = net.forward()  # out.shape = (1, 1000)

else:
    ort_inputs = {net.get_inputs()[0].name: blob}
    out = net.run(None, ort_inputs)[0]  # out.shape = (1, 1000)
    
out =out.flatten()   # out.shape = (1000, )
top5 = out.argsort()[-5:][::-1]
top1 = top5[0] # np.argmax(out) 
print('top1=', top1)
print('top5=', top5)

#4: Imagenet labels
#4-1
import json    
with open('./dnn/PreTrained/imagenet_labels.json', 'r') as f:
    imagenet_labels = json.load(f)

#4-2:      
print('Top-1 prediction=', imagenet_labels[top1], out[top1])
print('Top-5 prediction=', [imagenet_labels[i] for i in top5])
