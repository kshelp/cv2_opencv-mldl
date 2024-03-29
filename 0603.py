#0603.py
'''
ref1: https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/yolov2-coco
ref2: https://github.com/experiencor/keras-yolo2/tree/8bc3fcdc123a5d6b9624ed4188e07b876b79dac5

'''
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


#2
#2-1
net = cv2.dnn.readNet("./dnn/PreTrained/yolov2-coco-9.onnx")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(scale=1/255, size=(416, 416), swapRB=True)
outs = model.predict(src) 
print('outs[0].shape=', outs[0].shape)

#2-2
##blob = cv2.dnn.blobFromImage(image = src, scalefactor=1/255, 
##                             size= (416, 416),
##                             swapRB=True)
##net.setInput(blob)
##out_layer_names = net.getUnconnectedOutLayersNames()
##print('out_layer_names=',out_layer_names)
##outs = net.forward(out_layer_names)
##print('outs[0].shape=', outs[0].shape)

#2-3
outs = outs[0].squeeze()
outs = outs.reshape(5, 85, 13, 13)
outs = outs.transpose([2, 3, 0, 1])
print('outs.shape=', outs.shape) 

#3: ref2
#3-1
class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, c = None, classes = None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        
        self.c     = c
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        
        return self.label
    
    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
            
        return self.score
    
def _sigmoid(x):
    return 1. / (1. + np.exp(-x))

def _softmax(x, axis=-1):
    c = x - np.max(x)        
    e_x = np.exp(c)
    
    return e_x / e_x.sum(axis, keepdims=True)

def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
             return 0
        else:
            return min(x2,x4) - x3
        
def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])  
    
    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
    w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
    
    union = w1*h1 + w2*h2 - intersect
    
    return float(intersect) / union

#3-2
def decode_netout(netout, anchors, nb_class, obj_threshold=0.5, nms_threshold=0.4):
    grid_h, grid_w, nb_box = netout.shape[:3]

    boxes = []    
    # decode the output by the network
    netout[..., 4]  = _sigmoid(netout[..., 4])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * _softmax(netout[..., 5:])
    netout[..., 5:] *= netout[..., 5:] > obj_threshold
    
    for row in range(grid_h):
        for col in range(grid_w):
            for b in range(nb_box):
                # from 4th element onwards are confidence and class classes
                classes = netout[row,col,b,5:]
                
                if np.sum(classes) > 0:
                    # first 4 elements are x, y, w, and h
                    x, y, w, h = netout[row,col,b,:4]

                    x = (col + _sigmoid(x)) / grid_w # center position, unit: image width
                    y = (row + _sigmoid(y)) / grid_h # center position, unit: image height
                    w = anchors[2 * b + 0] * np.exp(w) / grid_w # unit: image width
                    h = anchors[2 * b + 1] * np.exp(h) / grid_h # unit: image height
                    confidence = netout[row,col,b,4]
                    
                    box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, confidence, classes)
                    
                    boxes.append(box)

    # suppress non-maximal boxes
    for c in range(nb_class):
        sorted_indices = list(reversed(np.argsort([box.classes[c] for box in boxes])))

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            
            if boxes[index_i].classes[c] == 0: 
                continue
            else:
                for j in range(i+1, len(sorted_indices)):
                    index_j = sorted_indices[j]
                    
                    if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_threshold:
                        boxes[index_j].classes[c] = 0
                        
    # remove the boxes which are less likely than a obj_threshold
    boxes = [box for box in boxes if box.get_score() > obj_threshold]
    
    return boxes

#3-3
ANCHORS = [0.57273, 0.677385,
           1.87446, 2.06253,
           3.33843, 5.47434,
           7.88282, 3.52778,
           9.77052, 9.16828]

boxes = decode_netout(outs, 
                      obj_threshold=0.5, 
                      nms_threshold=0.4,
                      anchors=ANCHORS,
                      nb_class=80)

#4
def draw_boxes(image, boxes, class_names):
    image_h, image_w, _ = image.shape

    for box in boxes:
        xmin = int(box.xmin*image_w)
        ymin = int(box.ymin*image_h)
        xmax = int(box.xmax*image_w)
        ymax = int(box.ymax*image_h)

        label= box.get_label()
        color = COLORS[label%len(COLORS)]

        cv2.rectangle(image, (xmin,ymin), (xmax,ymax), color, 2)
        cv2.putText(image, 
                    class_names[label] + ' ' + str(box.get_score()), 
                    (xmin, ymin - 13), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1e-3 * image_h, 
                    (0,255,0), 2)        
    return image

dst = draw_boxes(src, boxes, class_names)

cv2.imshow("dst", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
            
