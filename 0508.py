#0508.py
'''
ref1: https://github.com/onnx/models/tree/master/vision/classification/vgg
ref2: https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
'''
import cv2
import numpy as np

#1
net = cv2.dnn.readNet('./dnn/PreTrained/vgg16-7.onnx') # 'vgg19-7.onnx'

model = cv2.dnn_ClassificationModel(net)

#2
image_name = ['./data/elephant.jpg', './data/eagle.jpg', './data/dog.jpg']
src = cv2.imread(image_name[2]) 

#3
model.setInputParams(scale=1/255,
                     size=(224,224),
                     mean = (103.53, 116.28, 123.675), # ImageNet mean: BGR
                     swapRB=True) # RGB

#4
label, score = model.classify(src)
print("label={}, score={}".format(label, score))

#5: Imagenet labels
#5-1
##imagenet_labels = []
##file_name = './dnn/PreTrained/imagenet1000_clsidx_to_labels.txt'
##with open(file_name, 'r') as f:
##    for line in f.readlines():
##        line = line.replace("'", "")
##        key, value = line.split(':')
##        value = value.strip('}')
##        imagenet_labels.append(value.strip()[:-1]) # exclude last comma
##import json
##with open('./dnn/PreTrained/imagenet_labels.json', 'w') as f:
##    json.dump(imagenet_labels, f)

#5-2
import json    
with open('./dnn/PreTrained/imagenet_labels.json', 'r') as f:
    imagenet_labels = json.load(f)

#5-3     
print('prediction:', imagenet_labels[label])
cv2.imshow(imagenet_labels[label],  src)

cv2.waitKey()
cv2.destroyAllWindows()
