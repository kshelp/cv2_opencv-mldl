#0704.py
'''
ref: https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/mask-rcnn

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

def preprocess(image): # BGR
    H, W, C = image.shape
    
    # resize
    ratio = 800/min(H, W)
    image =  cv2.resize(image, dsize=(int(ratio * W), int(ratio * H)),
                        interpolation=cv2.INTER_LINEAR)  
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
sess = ort.InferenceSession("./dnn/PreTrained/MaskRCNN-10.onnx")
output_names = [output.name for output in sess._outputs_meta]
input_name = sess.get_inputs()[0].name
 
outs = sess.run(output_names, # None
                input_feed= {input_name: image_data})
boxes, labels, scores, masks = outs

print("boxes.shape=",  boxes.shape) 
print("labels.shape=", labels.shape) 
print("scores.shape=", scores.shape)   
print("masks.shape=",  masks.shape) 

#4
def display_detect_object(image, boxes, labels, scores, masks, score_threshold=0.6):
    dst = image.copy()   
    ratio = 800/min(image.shape[0], image.shape[1])    
    boxes /= ratio  # resize boxes

    for box, label, score, mask in zip(boxes, labels, scores, masks):
        if score > score_threshold:
            clone = image.copy()
            x1, y1, x2, y2 = np.round(box).astype(np.int32)
            color = COLORS[label%len(COLORS)]

            boxW = x2 - x1
            boxH = y2 - y1
            mask = cv2.resize(mask[0], (boxW, boxH),
			interpolation=cv2.INTER_NEAREST)
            mask = (mask > 0.1)

            contours, hierarchy = cv2.findContours(mask.astype(np.uint8),
                                      cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)            
            cv2.drawContours(clone[y1:y2, x1:x2], contours, -1, color, 3)

            roi = clone[y1:y2, x1:x2]
            roi = roi[mask]
            
            color = np.array(color)
            blended = ((0.4 * color) + (0.6 * roi)).astype("uint8")

            clone[y1:y2, x1:x2][mask] = blended
            dst[y1:y2, x1:x2][mask] = blended

            color = [int(c) for c in color]
            cv2.rectangle(clone, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(dst, (x1, y1), (x2, y2), color, 2)

            text = "{}: {:.4f}".format(class_names[label], score)
            cv2.putText(clone, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(dst, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.imshow("Output", clone)
            cv2.waitKey()
    return dst
              
dst = display_detect_object(src, boxes, labels, scores, masks)

cv2.imshow("dst", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()    
