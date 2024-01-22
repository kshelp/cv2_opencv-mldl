#0605.py
'''
ref1:https://github.com/ultralytics/yolov5/releases
ref2: https://github.com/ultralytics/yolov5/blob/master/detect.py
ref3: https://github.com/ultralytics/yolov5/blob/master/export.py
      C:/Users/kdk/.cache/torch/hub/ultralytics_yolov5_master/export.py
      
# pip install PyYAML
'''
import cv2
import numpy as np
import onnx
import torch

#torch.hub.set_dir('D:/.cache/torch/hub')
#1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True) #'yolov5n6'
model = model.to(DEVICE)

#2
#imgs = ['https://ultralytics.com/images/zidane.jpg'] 
image_names = ['./data/dog.jpg', './data/eagle.jpg']
imgs = []
for file_name in image_names:
    src = cv2.imread(file_name)
    imgs.append(src)

#3:inference
results = model(imgs)
results.print()

#results.imgs[0] = np.ascontiguousarray(results.imgs[0][..., ::-1]) # BGR->RGB
#results.imgs[1] = np.ascontiguousarray(results.imgs[1][..., ::-1])
results.save() # save_dir = 'runs/detect/exp'
##results.show()

#print("results.names=", results.names)  # COCO dataset 80 names
for i in range(results.n):
    print(f"results.xyxy[{i}]=",  results.xyxy[i])

for i in range(results.n):
    print(f"results.pandas().xyxy[{i}]=", results.pandas().xyxy[i]) #DataFrame 

#4: export ONNX
#4-1
# cmd> python export.py --weights yolov5n.pt --img 640 --batch 1

#4-2
import torch.nn as nn
from models.common import Conv
from models.yolo import Detect
from utils.activations import SiLU

model.eval()
# for k, m in model.named_modules():
#     if isinstance(m, Conv):  # assign export-friendly activations
#         if isinstance(m.act, nn.SiLU):
#             m.act = SiLU()
#     elif isinstance(m, Detect):
#         m.inplace = False
#         m.onnx_dynamic = False

#4-3             
dummy_input = torch.randn(1, 3, 640, 640).to(DEVICE)
y = model(dummy_input)# dry runs

# torch.onnx.export(model, dummy_input, "./dnn/yolov5s_0605.onnx", opset_version=12, #yolov5n6_0605.onnx
#                   #training= torch.onnx.TrainingMode.EVAL,
#                   #input_names=['images'],
#                   #output_names=['output'],
#                   #do_constant_folding=False,
#                   #dynamic_axes= None
#                   )

#5: check model
onnx_model=onnx.load("./dnn/yolov5n_0605.onnx")
onnx.checker.check_model(onnx_model)
print("model checked!")
