# 0607.py
'''
ref: 0601.py
'''
import cv2
import pafy
import numpy as np
#1
with open("./dnn/PreTrained/coco.names", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

#2: random a list of colors
np.random.seed(1111)
COLORS = np.random.randint(0, 255, size=(len(class_names), 3), dtype="uint8").tolist()

#3:
net = cv2.dnn.readNet("./dnn/PreTrained/yolov4.cfg", "./dnn/PreTrained/yolov4.weights")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(scale=1/255, size=(416, 416), swapRB=True)

#4-1
#cap = cv2.VideoCapture(0)  # 0번 카메라
cap = cv2.VideoCapture('./data/vtest.avi')

while True:
#4-2    
    retval, frame = cap.read()
    if not retval: break
#4-3    
    labels, scores, boxes = model.detect(frame, confThreshold= 0.4, nmsThreshold=0.5)

#4-4
    for (label, score, box) in zip(labels, scores, boxes):  
        color = COLORS[label]
        text = "%s : %f" % (class_names[label], score)
        cv2.rectangle(frame, box, color, 2)
        cv2.putText(frame, text, (box[0], box[1]-5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow('frame',frame)
    
    key = cv2.waitKey(1)
    if key == 27: # Esc
        break
if cap.isOpened():
    cap.release()
cv2.destroyAllWindows()
