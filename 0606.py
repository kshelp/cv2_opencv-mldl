#0606.py
'''
ref1:https://github.com/ultralytics/yolov5/releases
ref2:https://github.com/ultralytics/yolov5/releases/tag/v5.0

'''
import torch 
import cv2
import numpy as np

#1
#1-1
with open("./dnn/PreTrained/coco.names", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0),
          (255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 128, 255)]

#1-2
image_name = ['./data/dog.jpg', './data/person.jpg',
              './data/horses.jpg', './data/eagle.jpg']
src = cv2.imread(image_name[0])
height, width, channels = src.shape

#2
#net = cv2.dnn.readNet("./dnn/yolov5n.onnx")       # [ref1]
#net = cv2.dnn.readNet("./dnn/yolov5n_0605.onnx") # [예제 6.5]
net = cv2.dnn.readNet("./dnn/yolov5n6_0605.onnx")# [예제 6.5] 
model = cv2.dnn_DetectionModel(net)

#3
input_size = 640, 640  # model width, height
model.setInputParams(scale=1/255, size=input_size, swapRB=True, crop=False)
outs = model.predict(src)[0]
print('outs.shape=', outs.shape)

#4
def detect(outs, input_size, img_size, confThreshold= 0.5, nmsThreshold=0.4):

    # size_w, size_h = input_size # model's input size
    # width, height = img_size
    
    ratio_w, ratio_h = width/input_size[0], height/input_size[1] 
   
    labels    = []
    class_scores = []
    boxes = [] 
     
    #for out in outs:
    outs = outs.squeeze()  #outs = outs[0]
    for detection in outs:
            scores = detection[5:] 
            label   = np.argmax(scores)
            class_score = scores[label]
            conf = detection[4] # confidence, objectness
            
            if conf < confThreshold: 
                continue            
           
            cx= int(detection[0] * ratio_w)
            cy= int(detection[1] * ratio_h)
            w = int(detection[2] * ratio_w)
            h = int(detection[3] * ratio_h)
                      
            x = int(cx - w/2)
            y = int(cy - h/2)
            
            boxes.append([x, y, w, h]) 
            class_scores.append(float(class_score))
            labels.append(label)

    indices = cv2.dnn.NMSBoxes(boxes, class_scores, confThreshold, nmsThreshold)
    if len(indices) == 0: # empty array
        boxes = np.array(boxes)
        class_scores = np.array(class_scores)
        labels = np.array(labels)
    else:
        boxes = np.array(boxes)[indices]
        class_scores = np.array(class_scores)[indices]
        labels = np.array(labels)[indices]        
    
    return labels, class_scores, boxes

labels, scores, boxes = detect(outs, input_size, (width, height))

#5
for (label, score, box) in zip(labels, scores, boxes):
    color = COLORS[label%len(COLORS)]
    text = "%s : %f" % (class_names[label], score)
    cv2.rectangle(src, box, color, 2)
    cv2.putText(src, text, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
cv2.imshow("src", src)
cv2.waitKey(0)
cv2.destroyAllWindows()
