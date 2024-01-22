#0604.py
'''
ref1:
https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/yolov3
ref2: https://github.com/qqwweee/keras-yolo3
'''
import cv2
import numpy as np
import onnxruntime as ort

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
image_shape = np.array([height, width], dtype=np.float32).reshape(1, 2)

#2
def preprocess(img, size=(416, 416)):
    X = img.copy()
    ih, iw = img.shape[:2]
    
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)
    #print("scale=", scale)
    #print("nh=", nh, "nw=", nw)
    
    X = cv2.resize(X, dsize=(nw, nh), interpolation=cv2.INTER_CUBIC)
    #cv2.imshow("X", X)

    new_image = np.ones(shape=(h, w, 3))*128
    #print("new_image.shape=", new_image.shape)

    x0 = (w-nw)//2
    y0 = (h-nh)//2
    #print("x0=", x0, "y0=", y0)
    new_image[y0:y0+nh, x0:x0+nw] = X     
    #cv2.imshow("new_image", np.uint8(new_image))
    
    new_image= new_image/255   # [0, 1] 
    new_image = new_image[..., ::-1]
    
    new_image= new_image.transpose([2, 0, 1,]) 
    new_image = np.expand_dims(new_image, 0)
      
    return np.float32(new_image)

image_data = preprocess(src) # image_data.shape= (1, 3, 416, 416)
 
#3
sess = ort.InferenceSession("./dnn/PreTrained/yolov3-10.onnx")
output_names = [output.name for output in sess._outputs_meta]
input_name = sess.get_inputs()[0].name
input_shape= sess.get_inputs()[1].name
outs = sess.run(output_names, # None
                input_feed={input_name:  image_data,
                            input_shape: image_shape})
#print("len(outs)=", len(outs))

'''
ref1:
The model has 3 outputs.
boxes: (1x'n_candidates'x4),the coordinates of all anchor boxes
scores: (1x80x'n_candidates'), the scores of all anchor boxes per class
indices:('nbox'x3), selected indices from the boxes tensor.
        The selected index format is (batch_index, class_index, box_index)
'''
#4
#4-1
boxes, scores, indices = outs
##print('boxes.shape=', boxes.shape)
##print('scores.shape=', scores.shape)
##print('indices.shape=', indices.shape)

out_boxes, out_scores, out_labels = [], [], []
for idx_ in indices:
    out_labels.append(idx_[1])
    out_scores.append(scores[tuple(idx_)])
    idx_1 = (idx_[0], idx_[2])
    out_boxes.append(boxes[idx_1])

#4-2: score_threshold=0.3, nms_threshold=0.45
for (label, score, box) in zip(out_labels, out_scores, out_boxes):   
    color = COLORS[label%len(COLORS)]
    text = "%s : %f" % (class_names[label], score)
    
    y1, x1, y2, x2 = np.int32(box)       
    cv2.rectangle(src, (x1, y1), (x2, y2), color, 2)
    cv2.putText(src, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
cv2.imshow("src", src)

#4-3:  
boxes = boxes.squeeze()
scores = scores.squeeze()
scores = scores.T  # transpose([1, 0])
pred = np.argmax(scores, axis = 1)

class_scores = []
class_boxes  = []
for i, k in enumerate(pred):
    class_scores.append(float(scores[i, k]))
    y1, x1, y2, x2 = boxes[i]
    class_boxes.append([int(x1), int(y1), int(x2-x1), int(y2-y1)])    
    
indices2 = cv2.dnn.NMSBoxes(class_boxes, class_scores,
                           score_threshold=0.5, nms_threshold=0.4)
#print("indices2=", indices2)

#4-4
dst = src.copy()
for k in indices2:
    label = pred[k]
    score   = class_scores[k]
    color = COLORS[label%len(COLORS)]
    text = "%s : %f" % (class_names[label], score)
    
    box = boxes[k]
    y1, x1, y2, x2 = np.int32(box)    
    cv2.rectangle(dst, (x1, y1), (x2, y2), color, 2)
    
    cv2.putText(dst, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
cv2.imshow("dst", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
 
