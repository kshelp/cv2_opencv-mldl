#0602.py
'''
https://pysource.com/2019/06/27/yolo-object-detection-using-opencv-with-python/
https://opensourcelibs.com/lib/yolov-4
https://medium.com/clique-org/panet-path-aggregation-network-in-yolov4-b1a6dd09d158
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
height, width, channels = src.shape

#3
net = cv2.dnn.readNet("./dnn/PreTrained/yolov2.cfg", "./dnn/PreTrained/yolov2.weights")
#net = cv2.dnn.readNet("./dnn/PreTrained/yolov3.cfg", "./dnn/PreTrained/yolov3.weights")
#net = cv2.dnn.readNet("./dnn/PreTrained/yolov4.cfg", "./dnn/PreTrained/yolov4.weights")
 
#4
#4-1
blob = cv2.dnn.blobFromImage(image = src, scalefactor=1/255, 
                               size= (416, 416),
                               #size= (608, 608),
                               mean=(0, 0, 0), swapRB=True)
#4-2
net.setInput(blob)
out_layer_names = net.getUnconnectedOutLayersNames()
print('out_layer_names=',out_layer_names)

#4-3
outs = net.forward(out_layer_names)  # out_layer_names[:1], out_layer_names[:2]
print('len(outs)=', len(outs))
for i in range(len(outs)):
    print(" outs[{}].shape={}".format(i, outs[i].shape))
    
#4-4
def detect(outs, img_size, confThreshold= 0.5, nmsThreshold=0.4):
    width, height = img_size
    
    labels    = []
    class_scores = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            label   = np.argmax(scores)
            class_score = scores[label]
            conf = detection[4] # confidence, objectness
            if conf < confThreshold:
                continue
            
            # conf >= confThreshold:  object detected 
            cx= int(detection[0] * width)
            cy= int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

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

labels, scores, boxes = detect(outs, (width, height))

#5
for (label, score, box) in zip(labels, scores, boxes):
    color = COLORS[label%len(COLORS)]
    text = "%s : %f" % (class_names[label], score)
    cv2.rectangle(src, box, color, 2)
    cv2.putText(src, text, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
cv2.imshow("src", src)
cv2.waitKey(0)
cv2.destroyAllWindows()
        
