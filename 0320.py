#0320.py
import cv2
import numpy as np

#1
src = cv2.imread('./data/lena.jpg')

#2-1
faceCascade= cv2.CascadeClassifier(
      './haarcascades/haarcascade_frontalface_default.xml')

#2-2
faces = faceCascade.detectMultiScale(src, scaleFactor=1.1, # caleFactor=2                                 
                                     minNeighbors = 0) 
if len(faces):
    print("faces.shape=", faces.shape)
#2-3
for (x, y, w, h) in faces:
    cv2.rectangle(src, (x,y),(x+w, y+h), (255,0,0), 2)

#3
faces = faces.tolist()
#faces.append([ 54, 276,  57,  57])
    
faces2,weights = cv2.groupRectangles(faces, groupThreshold=1) # eps=0.2 
print("faces2=", faces2)
print("weights=", weights)

for (x, y, w, h) in faces2:
    cv2.rectangle(src, (x,y),(x+w, y+h),(0,0,255), 2)
         
cv2.imshow('src', src)
cv2.waitKey()
cv2.destroyAllWindows()
