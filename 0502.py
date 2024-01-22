#0502.py
import cv2
import numpy as np
 
#1
src1 = cv2.imread('./data/elephant.jpg') # (533, 800, 3), BGR
src2 = cv2.imread('./data/eagle.jpg')    # (512, 773, 3), BGR
print("src1.shape=", src1.shape)
print("src2.shape=", src2.shape)

#2
blobs = cv2.dnn.blobFromImages([src1, src2], size=(300, 224), crop=True)
print("blobs.shape=", blobs.shape)

#3
for i in range(blobs.shape[0]):
    img = blobs[i].transpose([1, 2, 0])
    img = np.uint8(img)  
    cv2.imshow("blobs[%d]"%i, img)
    cv2.waitKey()
    
cv2.destroyAllWindows()
