#0601.py
'''
https://github.com/pjreddie/darknet
https://pjreddie.com/media/files/yolov2.weights
https://pjreddie.com/media/files/yolov3.weights
https://github.com/AlexeyAB/darknet/wiki/YOLOv4-model-zoo
https://gist.github.com/YashasSamaga/e2b19a6807a13046e399f4bc3cca3a49
https://github.com/opencv/opencv/blob/master/samples/dnn/object_detection.py
'''
import cv2
import numpy as np

#1
with open("./dnn/PreTrained/coco.names", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0),
          (255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 128, 255)]

#2
image_name = ['./data/dog.jpg', './data/person.jpg',  './data/horses.jpg']
src = cv2.imread(image_name[0])

#3
net = cv2.dnn.readNet("./dnn/PreTrained/yolov2.cfg", "./dnn/PreTrained/yolov2.weights")
#net = cv2.dnn.readNet("./dnn/PreTrained/yolov3.cfg", "./dnn/PreTrained/yolov3.weights")
#net = cv2.dnn.readNet("./dnn/PreTrained/yolov4.cfg", "./dnn/PreTrained/yolov4.weights")
model = cv2.dnn_DetectionModel(net)

#4
#4-1
model.setInputParams(scale=1/255, size=(416, 416), swapRB=True)
#out = model.predict(src)

#4-2
labels, scores, boxes = model.detect(src, confThreshold= 0.5, nmsThreshold=0.4)

#5
for (label, score, box) in zip(labels, scores, boxes):   
    color = COLORS[label%len(COLORS)]
    text = "%s : %f" % (class_names[label], score)

    cv2.rectangle(src, box, color, 2)
    cv2.putText(src, text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
cv2.imshow("src", src)
cv2.waitKey(0)
cv2.destroyAllWindows()
            
