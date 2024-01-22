#0511.py
'''
ref: 텐서플로 딥러닝 프로그래밍, 가메출판사
'''
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image # Pillow

import cv2
import numpy as np 
import tf2onnx  # pip install tf2onnx

image_name = ['./data/elephant.jpg', './data/eagle.jpg', './data/dog.jpg']

#1: OpenCV
src = cv2.imread(image_name[0]) # BGR

def preprocess(img): #  "caffe" mode in preprocess_input, BGR
    X = img.copy()
    X = cv2.resize(X, dsize=(224, 224), interpolation=cv2.INTER_NEAREST_EXACT)

    mean = [103.939, 116.779, 123.68]
    X = X - mean  
    return np.float32(X)
    
img    = preprocess(src)   
blob = np.expand_dims(img, axis=0) # blob.shape = (1, 224, 224, 3), NHWC, BGR

#2: TensorFlow: vgg16.preprocess_input 
src2 = image.load_img(image_name[0], target_size=(224, 224)) # color_mode='rgb', interpolation='nearest'
src2 = image.img_to_array(src2)
img2 = np.expand_dims(src2, axis=0)
blob2  =  preprocess_input(img2)   # blob2.shape = (1, 224, 224, 3), NHWC, BGR

#3
#3-1
model  = VGG16(weights='imagenet') #include_top= True, input_shape = (224, 224, 3)) 
out = model.predict(blob)  # blob2
# (class_name, class_description, score) 
print('decode_predictions(top=5):', decode_predictions(out, top=5)) 

#3-2: decode using out 
import json    
with open('./dnn/PreTrained/imagenet_labels.json', 'r') as f:
    imagenet_labels = json.load(f)
out = out.flatten()

top5 = tf.math.top_k(out, k=5) # top5.values.numpy(), top5.indices.numpy()
print('Top-5(imagenet_labels):', [(imagenet_labels[i], out[i]) for i in top5.indices.numpy()])

#3-3
# https://github.com/raghakot/keras-vis
with open('./dnn/PreTrained/imagenet_class_index.json', 'r') as f:
    imagenet_class = json.load(f)
    
print('Top-5(imagenet_class):')
for i in top5.indices.numpy():
    value = imagenet_class[str(i)]
    print("({}, {}, {:.4f})".format(value[0], value[1], out[i]))

#4:
print("freezing pb....")
full_model = tf.function(lambda x: model(x))
full_model = full_model.get_concrete_function(
        tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
frozen_func = convert_variables_to_constants_v2(full_model)

tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                 logdir="./dnn",  name="vgg16.pb",  as_text=False)

#5:
##print("convert onnx ....")
##spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
##onnx_model, _ = tf2onnx.convert.from_keras(model,
##                     input_signature=spec, output_path="./dnn/vgg16.onnx")
