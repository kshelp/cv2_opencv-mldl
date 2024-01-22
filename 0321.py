#0321.py
# ref: https://docs.opencv.org/4.5.4/db/d28/tutorial_cascade_classifier.html
import cv2
import numpy as np

#1
src = cv2.imread('./data/lena.jpg')
faceCascade= cv2.CascadeClassifier(
      './haarcascades/haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('./haarcascades/haarcascade_eye.xml')


#2
#2-1
faces= faceCascade.detectMultiScale(src) # scaleFactor= 1.1, minNeighbors=3 

#2-2
for (x, y, w, h) in faces:

    cv2.rectangle(src, (x,y),(x+w, y+h),(255,0,0), 2)
    
    roi = src[y:y+h, x:x+w]
    eyes = eyeCascade.detectMultiScale(roi)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi, (ex,ey), (ex+ew,ey+eh),(0,255,0), 2)
        
cv2.imshow('src', src)
cv2.waitKey()
cv2.destroyAllWindows()
