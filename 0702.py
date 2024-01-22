#0702.py
'''
ref: https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/faster-rcnn

'''
import cv2
import numpy as np
import onnxruntime as ort

#1
with open("./dnn/PreTrained/coco.names", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

COLORS = [(0, 0, 255),  (0, 255, 0),  (255, 0, 0),
          (255, 255, 0),(0, 255, 255),(255, 0, 255), (255, 128, 255)]

#2
image_name = ['./data/dog.jpg',   './data/person.jpg',
              './data/horses.jpg','./data/eagle.jpg']
src = cv2.imread(image_name[0])  # BGR
height, width, channels = src.shape

def preprocess(image): # BGR
    height, width, channels = image.shape
    
    # resize
    ratio = 800/min(image.shape[0], image.shape[1])
    image =  cv2.resize(image, dsize=(int(ratio * width), int(ratio * height)),
                        interpolation=cv2.INTER_LINEAR) # cv2.INTER_AREA, cv2.INTER_CUBIC  
    image = np.float32(image)

    # HWC -> CHW
    image = np.transpose(image, [2, 0, 1])

    # normalize by ImageNet BGR mean
    mean_vec = np.array([102.9801, 115.9465, 122.7717]) 
    for i in range(3):
        #image[i, :, :] = image[i, :, :] - mean_vec[i]
        image[i] = image[i] - mean_vec[i]     

    # pad to be divisible of 32
    import math
    padded_h = int(math.ceil(image.shape[1] / 32) * 32)
    padded_w = int(math.ceil(image.shape[2] / 32) * 32)

    padded_image = np.zeros((3, padded_h, padded_w), dtype=np.float32)
    padded_image[:, :image.shape[1], :image.shape[2]] = image
    image = padded_image
    return image
 
image_data = preprocess(src) 
 
#3
sess = ort.InferenceSession("./dnn/PreTrained/FasterRCNN-10.onnx")
output_names = [output.name for output in sess._outputs_meta]
input_name = sess.get_inputs()[0].name
 
outs = sess.run(output_names, # None
                input_feed= {input_name: image_data})
boxes, labels, scores = outs
print("boxes.shape=",  boxes.shape) 
print("labels.shape=", labels.shape) 
print("scores.shape=", scores.shape) 

#4
def display_detect_object(image, boxes, labels, scores, score_threshold=0.6):
    # resize boxes
    ratio = 800/min(image.shape[0], image.shape[1])    
    boxes /= ratio

    for box, label, score in zip(boxes, labels, scores):
        if score > score_threshold:
            x1, y1, x2, y2 = np.round(box).astype(np.int32)
            
            color = COLORS[label%len(COLORS)]
            
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            text = class_names[label-1] + ':' + str(np.round(score, 2))
            cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
  
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
       
display_detect_object(src, boxes, labels, scores)
