#0512.py
'''
ref: 텐서플로 딥러닝 프로그래밍, 가메출판사
'''
import tensorflow as tf
from tensorflow.keras.models  import Sequential
from tensorflow.keras.layers  import Dense, Flatten, Dropout
from tensorflow.keras.applications import VGG16
import tensorflow_datasets as tfds  # pip install tensorflow-datasets
import tf2onnx

#1:
#1-1:
(train_dataset, valid_dataset), info = tfds.load(name='cats_vs_dogs',
                                                 split=('train[:90%]', 'train[-10%:]'), with_info=True)
print("len(list(train_dataset))=",len(list(train_dataset))) 
print("len(list(valid_dataset))=",len(list(valid_dataset)))

#1-2:

def preprocess(ds):
    X = tf.image.resize(ds['image'], size=(224, 224))
    print("X.shape=", X.shape)
    X =X[..., ::-1]  # RGB -> BGR
    X   = tf.cast(X, tf.float32) / 255.0 # [0, 1]

    mean =[0.485, 0.456, 0.406] # ImageNet
    std = [0.229, 0.224, 0.225]
    X = (X - mean)/std

    X = tf.transpose(X, [2, 0, 1]) # HWC-> CHW
    label = tf.one_hot(ds['label'] , depth=2) 
      
    return X, label

BATCH_SIZE = 32
train_ds = train_dataset.map(preprocess).batch(BATCH_SIZE)
valid_ds = valid_dataset.map(preprocess).batch(BATCH_SIZE)

#1-3
for images, labels in train_ds.take(1): 
    for i in range(2):    
        print("i={}, images[i].shape={}, label[i]={}".format(i, images[i].shape, labels[i]))

#2: transfer learning using VGG16
#2-1
tf.keras.backend.set_image_data_format('channels_first') # NCHW
print("tf.keras.backend.image_data_format()=", tf.keras.backend.image_data_format())

vgg16 = VGG16(weights='imagenet', include_top= False, input_shape = (3, 224, 224)) 
vgg16.trainable=False

#2-2
num_classes = 2
model = Sequential([
    vgg16, 
    Flatten(), 
    Dense(100, activation='relu'),
    Dropout(0.2),
    Dense(100, activation='relu'),
    Dense(num_classes, activation='softmax') ]) # output layer
#model.summary()

#2-3
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt,
              loss = 'binary_crossentropy', #'categorical_crossentropy'
              metrics=['accuracy'])
#2-4
ret =model.fit(train_ds, batch_size=BATCH_SIZE, validation_data=valid_ds,
          epochs=10, verbose=2)
loss = ret.history['loss']

#3:
print("freezing pb....")
full_model = tf.function(lambda x: model(x))
full_model = full_model.get_concrete_function(
        tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
frozen_func = convert_variables_to_constants_v2(full_model)

tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                 logdir="./dnn",  name="cats_vs_dogs.pb",  as_text=False)

#4:
##print("convert onnx ....")
##spec = (tf.TensorSpec((None, 3, 224, 224), tf.float32, name="input"),)
##onnx_model, _ = tf2onnx.convert.from_keras(model,
##                                       input_signature=spec,                                       
##                                       output_path="./dnn/cats_vs_dogs.onnx")
