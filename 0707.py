#0707.py
'''
ref1: https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD
ref2: https://pytorch.org/docs/stable/hub.html
'''
import cv2
import numpy as np
import torch

#1
with open("./dnn/PreTrained/coco.names", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 128, 255)]

#2
image_name = ['./data/dog.jpg', './data/person.jpg',
              './data/horses.jpg', './data/eagle.jpg']
src = cv2.imread(image_name[0])
H, W, C = src.shape

#3: load from torch.hub
#3-1:
##entrypoints = torch.hub.list('pytorch/vision', force_reload=True)
##print(entrypoints) # entrypoints[0] = 'alexnet'
##model = torch.hub.load('pytorch/vision', entrypoints[0], pretrained=True)

##entrypoints=torch.hub.list('NVIDIA/DeepLearningExamples:torchhub', force_reload=True)
##print(entrypoints) # entrypoints[0] = 'nvidia_ssd'

               
#3-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd').to(device)
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')

#3-3
##print('torch.onnx.export: start...')
##dummy_input = torch.randn(1, 3, 300, 300).to(device)
##torch.onnx.export(ssd_model, dummy_input, "./dnn/ssd_model.onnx")
##print('torch.onnx.export: end')

#4: 
#4-1
##inputs = [utils.prepare_input(img) for img in image_name] #inputs[0].shape=(300, 300, 3)
##tensor = utils.prepare_tensor(inputs).to(device) 
##print('tensor.shape=', tensor.shape) # tensor.shape=(4, 3, 300, 300)

#4-2
inputs = [cv2.imread(img) for img in image_name]
blob = cv2.dnn.blobFromImages(inputs, size=(300, 300),
                             mean = (127.5, 127.5, 127.5),
                             scalefactor= 1.0 / 127.5, 
                             swapRB=True, crop=False)
tensor=torch.Tensor(blob).to(device)
print('tensor.shape=', tensor.shape) # tensor.shape=(4, 3, 300, 300)

#4-3
##blob = cv2.dnn.blobFromImage(src, size=(300, 300),
##                             mean = (127.5, 127.5, 127.5),
##                             scalefactor= 1.0 / 127.5, 
##                             swapRB=True, crop=False)
##tensor=torch.Tensor(blob).to(device) 
##print('tensor.shape=', tensor.shape) # tensor.shape=(1, 3, 300, 300)


#5
#5-1
ssd_model.eval()
with torch.no_grad():
    outs  = ssd_model(tensor) # outs = ssd_model.forward(tensor)

print("outs[0].shape=", outs[0].shape) # torch.Size([tensor.shape[0], 4, 8732])
print("outs[1].shape=", outs[1].shape) # torch.Size([tensor.shape[0], 81, 8732])

#5-2
results  = utils.decode_results(outs )
best_results = [utils.pick_best(detections, threshold=0.5) for detections  in results] 
classes_to_labels = utils.get_coco_object_dictionary() # class_names
print('len(best_results)=',len(best_results)) # tensor.shape[0]

#5-3
def display_detect_object(image, boxes, labels, scores):
    H, W, C = image.shape
    for box, label, score in zip(boxes, labels, scores):
        label -= 1 # 0, 1, ...

        color = COLORS[label%len(COLORS)]
        
        box =   box * np.array([W, H, W, H])
        x1, y1, x2, y2 = box.astype("int")
        cv2.rectangle(image,   (x1, y1), (x2, y2), color, 2)
        
        text = "{}: {:.4f}".format(class_names[label], score)
        cv2.putText(image, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#5-4: src in #4-3
##boxes, labels, scores = best_results[0]
##display_detect_object(src, boxes, labels, scores) 
##cv2.imshow('src', src)
##cv2.waitKey()

#5-5: inputs in (#4-1, #4-2)
for i in range(len(best_results)):
    boxes, labels, scores = best_results[i]
    display_detect_object(inputs[i], boxes, labels, scores)
    cv2.imshow('inputs[%d]'%i, inputs[i])
    cv2.waitKey()
    
cv2.destroyAllWindows()            

