#0708.py
'''
ref1:
 https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD # /ssd/utils.py
ref2: https://github.com/kuangliu/pytorch-ssd/blob/master/encoder.py
ref3: https://github.com/rbgirshick/fast-rcnn/blob/master/lib/utils/nms.py
ref4: https://learnopencv.com/non-maximum-suppression-theory-and-implementation-in-pytorch/
'''
import cv2
import numpy as np
import itertools

#1
#1-1
with open("./dnn/PreTrained/coco.names", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0),(255, 255, 0),
          (0, 255, 255), (255, 0, 255), (255, 128, 255)]

#1-2
image_name = ['./data/dog.jpg', './data/person.jpg',  './data/horses.jpg']
src = cv2.imread(image_name[0])
height, width, channels = src.shape

#2
#2-1
net = cv2.dnn.readNet("./dnn/ssd_model.onnx")  # [예제 7.7], #3-3
model = cv2.dnn_DetectionModel(net)


#2-2
#model.setInputParams(scale=1/255, size=(300, 300), swapRB=True)
model.setInputParams(scale=1/127.5, size=(300, 300), mean = (127.5, 127.5, 127.5), swapRB=True)  
outs = model.predict(src)

#2-3
##blob = cv2.dnn.blobFromImage(image = src, scalefactor=1/127.5, 
##                             size= (300, 300),
##                             mean=(127.5, 127.5, 127.5), swapRB=True)
##net.setInput(blob)
##
##out_layer_names = net.getUnconnectedOutLayersNames()
##print('out_layer_names=',out_layer_names) # ('645', '646')
##outs = net.forward(out_layer_names)  # out_layer_names[:1], out_layer_names[:2]

print('len(outs)=', len(outs)) #2
print("outs[0].shape=", outs[0].shape) # (1, 4, 8732)  
print("outs[1].shape=", outs[1].shape) # (1, 81, 8732)

#3: ref1(utils.py), ref2
#3-1
class DefaultBoxes(object):
    def __init__(self, fig_size, feat_size, steps, scales, aspect_ratios,
                       scale_xy=0.1, scale_wh=0.2):

        self.feat_size = feat_size
        self.fig_size = fig_size

        self.scale_xy_ = scale_xy
        self.scale_wh_ = scale_wh

        # According to https://github.com/weiliu89/caffe
        # Calculation method slightly different from paper
        self.steps = steps
        self.scales = scales

        fk = fig_size/np.array(steps)
        self.aspect_ratios = aspect_ratios

        self.default_boxes = []
        # size of feature and number of feature
        for idx, sfeat in enumerate(self.feat_size):

            sk1 = scales[idx]/fig_size
            sk2 = scales[idx+1]/fig_size
            sk3 = np.sqrt(sk1*sk2) 
            all_sizes = [(sk1, sk1), (sk3, sk3)]

            for alpha in aspect_ratios[idx]:
                w, h = sk1*np.sqrt(alpha), sk1/np.sqrt(alpha)
                all_sizes.append((w, h))
                all_sizes.append((h, w))
            for w, h in all_sizes:
                for i, j in itertools.product(range(sfeat), repeat=2):
                    cx, cy = (j+0.5)/fk[idx], (i+0.5)/fk[idx]
                    self.default_boxes.append((cx, cy, w, h))

        #self.dboxes = torch.tensor(self.default_boxes, dtype=torch.float)
        #self.dboxes.clamp_(min=0, max=1)            

        self.dboxes = np.float32(self.default_boxes)
        self.dboxes.clip(min=0, max=1)
        # For IoU calculation
        #self.dboxes_ltrb = self.dboxes.clone()
        self.dboxes_ltrb = self.dboxes.copy()
                
        self.dboxes_ltrb[:, 0] = self.dboxes[:, 0] - 0.5 * self.dboxes[:, 2]
        self.dboxes_ltrb[:, 1] = self.dboxes[:, 1] - 0.5 * self.dboxes[:, 3]
        self.dboxes_ltrb[:, 2] = self.dboxes[:, 0] + 0.5 * self.dboxes[:, 2]
        self.dboxes_ltrb[:, 3] = self.dboxes[:, 1] + 0.5 * self.dboxes[:, 3]

    @property
    def scale_xy(self):
        return self.scale_xy_

    @property
    def scale_wh(self):
        return self.scale_wh_

    def __call__(self, order="ltrb"):
        if order == "ltrb": return self.dboxes_ltrb
        if order == "xywh": return self.dboxes
#3-2        
def dboxes300_coco():
    figsize = 300
    feat_size = [38, 19, 10, 5, 3, 1]
    steps = [8, 16, 32, 64, 100, 300]
    # use the scales here:
    # https://github.com/amdegroot/ssd.pytorch/blob/master/data/config.py
    scales = [21, 45, 99, 153, 207, 261, 315]
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
    return dboxes

#4: 
#4-1
def _softmax(x, axis=-1):
    c = x - np.max(x)        
    e_x = np.exp(c)    
    return e_x / e_x.sum(axis, keepdims=True)

#4-2: ref1(utils.py), ref2
class Encoder(object):
    def __init__(self, dboxes):
        self.dboxes = dboxes(order="ltrb")
        #self.dboxes_xywh = dboxes(order="xywh").unsqueeze(dim=0) #KDK
        self.dboxes_xywh = np.expand_dims(dboxes(order="xywh"), axis=0)
        print("self.dboxes.shape=", self.dboxes.shape)
        #self.nboxes = self.dboxes.size(0)
        self.nboxes = self.dboxes.shape[0] #KDK
        self.scale_xy = dboxes.scale_xy
        self.scale_wh = dboxes.scale_wh

#4-3
    def scale_back_batch(self, bboxes_in, scores_in):
        """
            Do scale and transform from xywh to ltrb
            suppose input Nx4xnum_bbox Nxlabel_numxnum_bbox
        """
##KDK
        bboxes_in = bboxes_in.transpose([0, 2, 1]) #KDK
        scores_in = scores_in.transpose([0, 2, 1])  #KDK

        bboxes_in[:, :, :2] = self.scale_xy*bboxes_in[:, :, :2]
        bboxes_in[:, :, 2:] = self.scale_wh*bboxes_in[:, :, 2:]

        bboxes_in[:, :, :2] = bboxes_in[:, :, :2]*self.dboxes_xywh[:, :, 2:] \
                                             + self.dboxes_xywh[:, :, :2]
        bboxes_in[:, :, 2:] = np.exp(bboxes_in[:, :, 2:])*self.dboxes_xywh[:, :, 2:]
        #bboxes_in[:, :, 2:] = bboxes_in[:, :, 2:].exp()*self.dboxes_xywh[:, :, 2:] #KDK        

        # Transform format to ltrb
        l, t, r, b = bboxes_in[:, :, 0] - 0.5*bboxes_in[:, :, 2],\
                     bboxes_in[:, :, 1] - 0.5*bboxes_in[:, :, 3],\
                     bboxes_in[:, :, 0] + 0.5*bboxes_in[:, :, 2],\
                     bboxes_in[:, :, 1] + 0.5*bboxes_in[:, :, 3]

        bboxes_in[:, :, 0] = l
        bboxes_in[:, :, 1] = t
        bboxes_in[:, :, 2] = r
        bboxes_in[:, :, 3] = b

        #return bboxes_in, F.softmax(scores_in, dim=-1)
        return bboxes_in, _softmax(scores_in)  #KDK 

#4-4: KDK modify decode_batch() in ref2
    def decode(self, bboxes_in, scores_in, score_threshold=0.3, 
                                            nms_threshold = 0.7, nms=True):
        bboxes, probs = self.scale_back_batch(bboxes_in, scores_in)
        
        #print("bboxes.shape=", bboxes.shape) # (1, 8732, 4)
        #print("probs.shape=", probs.shape)   # (1, 8732, 81)
        bboxes_out = []
        scores_out = []
        labels_out = []
        
        bboxes = bboxes.squeeze(0) # (8732, 4)
        probs =  probs.squeeze(0)  # (8732, 81)
        for bbox, prob in zip(bboxes, probs):
            prob = prob[1:] # 0: back ground
            class_id    = np.argmax(prob)
            class_score = prob[class_id]
            if class_score > score_threshold:
                bboxes_out.append(bbox)
                scores_out.append(float(class_score))
                labels_out.append(class_id)
        bboxes_out= np.array(bboxes_out)
        scores_out= np.array(scores_out)
        labels_out= np.array(labels_out)
        if nms:
            indices = self.nms(bboxes_out, scores_out, nms_threshold)
            #print("indices=", indices)
            bboxes_out = bboxes_out[indices]
            scores_out = scores_out[indices]
            labels_out = labels_out[indices]
                       
        return bboxes_out, scores_out, labels_out
     
#4-5: ref3, ref4
    def nms(self, boxes, scores, thresh): # NMS(non-maximum-suppression)
        '''
        boxes : (n,  4)
        scores: (n, )
        '''
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1] # get boxes with more ious first

        keep = []
        while order.size > 0:
            i = order[0] # pick maxmum iou box
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1) # maximum width
            h = np.maximum(0.0, yy2 - yy1 + 1) # maxiumum height
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        
        return np.array(keep)

#4-6: 
dboxes = dboxes300_coco()
encoder = Encoder(dboxes)

#5: experiments
#5-1:
def display_detect_boxes(img, boxes, scores, labels):
    height, width, channels = img.shape
    dst = img.copy()
    
    for (classid, score, _box) in zip(labels, scores, boxes):
        box = _box.copy()
        box[0] = box[0]*width
        box[1] = box[1]*height
        box[2] = box[2]*width 
        box[3] = box[3]*height
        x1, y1, x2, y2 = np.int32(box)
       
        color = COLORS[classid%len(COLORS)]
        text = "%s : %f" % (class_names[classid], score)
        cv2.rectangle(dst, (x1, y1), (x2, y2), color, 2)
        cv2.putText(dst, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return dst

#5-2: nms=False
pred_boxes, pred_scores = model.predict(src) # outs
boxes, scores, labels = encoder.decode(pred_boxes, pred_scores, nms=False)
#print("boxes.shape=", boxes.shape)
#print("scores.shape=", scores.shape)
#print("labels.shape=", labels.shape)
   
dst = display_detect_boxes(src, boxes, scores, labels)        
cv2.imshow("nms=False", dst)

#5-3: nms=True
pred_boxes, pred_scores = model.predict(src) # outs
boxes, scores, labels = encoder.decode(pred_boxes, pred_scores) #  nms=True
dst = display_detect_boxes(src, boxes, scores, labels)        
cv2.imshow("nms=True", dst)

#5-4: nms=False, cv2.dnn.NMSBoxes()
pred_boxes, pred_scores = model.predict(src) # outs
boxes, scores, labels = encoder.decode(pred_boxes, pred_scores, nms=False)

indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.3, nms_threshold=0.7)
print("indices.shape=", indices.shape)
boxes = boxes[indices]
scores = scores[indices]
labels = labels[indices]
            
dst = display_detect_boxes(src, boxes, scores, labels)        
cv2.imshow("cv2.dnn.NMSBoxes", dst)

cv2.waitKey(0)
cv2.destroyAllWindows()    
