#0513.py
import cv2
import numpy as np

#1
TENSORFLOW, ONNX = 1, 2
net_type =  TENSORFLOW  # ONNX
if net_type == TENSORFLOW:
    net = cv2.dnn.readNet('./dnn/cats_vs_dogs.pb')
else:
    net = cv2.dnn.readNet('./dnn/cats_vs_dogs.onnx')
    
#2
image_name = ['./data/cat.jpg', './data/dog.jpg', './data/dog2.jpg']
src = cv2.imread(image_name[0])

def preprocess(img):
    X = img.copy()
    X = cv2.resize(X, dsize=(224, 224))        # (224, 224, 3)

    X = X/255 # [0, 1]

    mean =[0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    X = (X - mean)/std
        
    return np.float32(X) 
    
img = preprocess(src)  # img.shape = (224, 224, 3), BGR

#3
blob = cv2.dnn.blobFromImage(img)   # (1, 3, 224, 224) # NCHW
net.setInput(blob)
out = net.forward() 
pred = np.argmax(out, axis = 1)[0]
 
#4:
labels = ['cat', 'dog']
print('prediction:', labels[pred])

cv2.imshow(labels[pred],  src)
cv2.waitKey()
cv2.destroyAllWindows()
 
