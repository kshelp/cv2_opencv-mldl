#0501.py
'''
ref: OpenCV(source) : dnn.cpp
'''
import cv2
import numpy as np
 
#1
src = cv2.imread('./data/elephant.jpg') # BGR
print("src.shape=", src.shape)

#2
def blobFromImage(src, scalefactor=1.0, size=None, mean=None, swapRB=False, crop=False):
#2-1
    size_w, size_h = size
    src_h,  src_w  = src.shape[:2]
    
#2-2       
    if crop:
        resizeFactor = max(size_w/src_w, size_h/src_h)
        resize_img = cv2.resize(src, None, fx=resizeFactor, fy=resizeFactor)

        x = round((resize_img.shape[1] - size_w)/2)
        y = round((resize_img.shape[0] - size_h)/2)

        img = resize_img[y:y+size_h, x:x+size_w]
    else:
        img = cv2.resize(src, size)
#2-3                
    if swapRB:
        mean = mean.tolist() # list(mean)
        mean[2], mean[0] = mean[0], mean[2]
#2-4        
    if mean:
        img = img - mean
#2-5        
    img = img*scalefactor
#2-6
    if swapRB:
        img = img[:, :, ::-1]
#2-7        
    img = img.transpose([2, 0, 1]) 
    img = np.expand_dims(img, axis = 0) 
    return np.float32(img)
 
#3-1
blob = blobFromImage(src, size=(300, 224), crop=True)

#3-2
blob2 = cv2.dnn.blobFromImage(src, size=(300, 224), crop=True)

 
print("blob.shape=", blob.shape)
print("blob2.shape=", blob2.shape)
print("np.allclose(blob, blob2)=", np.allclose(blob, blob2)) 

#4-1
blob= blob.squeeze() 
blob = blob.transpose([1, 2, 0]) 
blob = np.uint8(blob)
cv2.imshow("blob", blob)

#4-2
blob2= blob2.squeeze() 
blob2 = blob2.transpose([1, 2, 0]) 
blob2 = np.uint8(blob2)
cv2.imshow("blob2", blob2)           
 
cv2.waitKey()
cv2.destroyAllWindows()
