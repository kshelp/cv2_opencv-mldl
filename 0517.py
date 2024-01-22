#0517.py
import cv2
import numpy as np
from   opencv_transforms import transforms  # pip install opencv_transforms
 
#1:
net = cv2.dnn.readNet('./dnn/cifar10.onnx') # cifar10-sgd-20.onnx


#2
image_name = ['./data/cat.jpg', './data/dog.jpg', './data/dog2.jpg',
              './data/eagle.jpg', './data/horses.jpg']
src = cv2.imread(image_name[0]) 
src_rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB) 

#3
#3-1
def preprocess(img):
    X = img.copy()
    X = cv2.resize(X, dsize=(224, 224)) 

    X = X/255 

    mean= (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    X = (X - mean)/std
        
    return np.float32(X)    
img = preprocess(src_rgb) 

#3-2
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda x: x/255),
    transforms.Lambda(lambda x:
                      np.float32((x-(0.485, 0.456, 0.406))/(0.229, 0.224, 0.225)))    
    ])
img2 = transform(src_rgb)

#4
blob = cv2.dnn.blobFromImage(img) # img2

net.setInput(blob)
out = net.forward()  
pred = np.argmax(out, axis = 1)[0]
print('out=', out)
print('pred=', pred)

#5
cifar10_labels = ('plane', 'car', 'bird', 'cat', 'deer',
                  'dog', 'frog', 'horse', 'ship', 'truck')

print('prediction:', cifar10_labels[pred])
cv2.imshow(cifar10_labels[pred] +" : " + str(out[0, pred]),  src)

cv2.waitKey()
cv2.destroyAllWindows()
