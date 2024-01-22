#0706.py
'''
https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD
model: MobileNet-SSD v3(weights, config)
'''
import cv2
import numpy as np

#1
with open("./dnn/PreTrained/coco90-2017.names", "r") as f:
    class_names = [line.strip() for line in f.readlines()]
class_names.insert(0, '')

COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 128, 255)]

#2
image_name = ['./data/dog.jpg', './data/person.jpg',  './data/horses.jpg']
src = cv2.imread(image_name[0])
H, W, C = src.shape

#3
#3-1
path = "./dnn/PreTrained/ssd_mobilenet_v3_large_coco_2020_01_14/"
##net = cv2.dnn.readNet(path + "frozen_inference_graph.pb",
##                      path + "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
##
##model = cv2.dnn_DetectionModel(net)

model = cv2.dnn_DetectionModel(path + "frozen_inference_graph.pb",
                      path + "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")    
#3-2
##model.setInputSize(320, 320)
##model.setInputScale(1.0 / 127.5)
##model.setInputMean((127.5, 127.5, 127.5))
##model.setInputSwapRB(True)
model.setInputParams(scale= 1.0/127.5,
                     size=(320, 320),
                     mean=(127.5, 127.5, 127.5),
                     swapRB=True)

#3-3
classes, scores, boxes = model.detect(src, confThreshold=0.6)

#4
for label, score, box in zip(classes.flatten(), scores.flatten(), boxes):

    color = COLORS[label%len(COLORS)] 
    cv2.rectangle(src, box, color, 2)

    text = "{}: {:.4f}".format(class_names[label], score)
    cv2.putText(src, text, (box[0], box[1] - 5),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
cv2.imshow('src', src)
cv2.waitKey()
cv2.destroyAllWindows()
