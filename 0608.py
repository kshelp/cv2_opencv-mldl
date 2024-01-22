# 0608.py
'''
ref: 0606.py, 0607.py
'''
import cv2
import numpy as np
#1
with open("./dnn/PreTrained/coco.names", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

#2: random a list of colors
np.random.seed(1111)
COLORS = np.random.randint(0, 255, size=(len(class_names), 3), dtype="uint8").tolist()

#3:
net = cv2.dnn.readNet("./dnn/yolov5n.onnx") 
model = cv2.dnn_DetectionModel(net)

input_size = 640, 640  # model width, height
model.setInputParams(scale=1/255, size=input_size, swapRB=True)
#model.setInputParams(scale=1/127.5, size=input_size,
#                     mean = (127.5, 127.5, 127.5), swapRB=True)  

#4-1
#cap = cv2.VideoCapture(0)  # 0번 카메라
cap = cv2.VideoCapture('./data/vtest.avi')
width, height = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                 int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

#4-2
def detect(outs, input_size, img_size, confThreshold= 0.5, nmsThreshold=0.4):
    # size_w, size_h = input_size # model's input size
    # width, height = img_size
    
    ratio_w, ratio_h = width/input_size[0], height/input_size[1] 
   
    labels    = []
    class_scores = []
    boxes = [] 
     
    #for out in outs: # only 1 batch
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

while True:
#4-3    
    retval, frame = cap.read()
    if not retval: break
#4-4    
    outs = model.predict(frame)[0] # outs.shape= (1, 25200, 85)
    labels, scores, boxes = detect(outs, input_size, (width, height))

#4-5
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
